"""
Stage 3 (Offline): Offline Reinforcement Learning — Finetuning Policy Head
=============================================================================
Refines the policy head of the EndoscopeTrackingNetwork using IQL / TD3+BC.
To prevent representation collapse and the "moving target" problem, the 
feature encoders (ResNet18, BiLSTM, StateEncoder) are FROZEN by default.
RL only updates the final MLP action head and the Critic networks.
"""

import os
os.environ['QT_QPA_PLATFORM'] = 'offscreen'

import argparse
import copy
import json
import random
import re
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm


# ──────────────────────────────────────────────────────────────────────────────
# 0.  Config
# ──────────────────────────────────────────────────────────────────────────────

class OfflineRLConfig:
    # Data
    history_len: int  = 10
    cam_w: int        = 400
    cam_h: int        = 400

    # BC network fused feature dim (frozen)
    feature_dim: int  = 390
    action_dim: int   = 2

    max_motor_speed: float = 100.0

    # Training
    algo: str           = 'iql'
    batch_size: int     = 32
    num_epochs: int     = 200
    lr_actor: float     = 1e-4   
    lr_critic: float    = 1e-4
    lr_value: float     = 3e-4
    gamma: float        = 0.95
    tau: float          = 0.005
    grad_clip: float    = 1.0
    freeze_encoder: bool= True 

    # IQL
    iql_tau: float    = 0.6   
    iql_beta: float   = 3.0

    # TD3+BC
    td3bc_lambda: float     = 1.0   # was 2.5 — lower λ prevents Q-term overwhelming BC when |Q| is small
    td3_policy_noise: float = 0.2
    td3_noise_clip: float   = 0.5
    td3_policy_delay: int   = 2

    # Reward
    reward_alpha: float      = 1.0
    reward_beta: float       = 0.5
    reward_gamma: float      = 0.01
    reward_delta: float      = 1.0
    success_threshold: float = 0.1

    seed: int          = 42
    save_every: int    = 10


# ──────────────────────────────────────────────────────────────────────────────
# 1.  Dataset
# ──────────────────────────────────────────────────────────────────────────────

class OfflineDataset(Dataset):
    def __init__(self, csv_path: str, cfg: OfflineRLConfig):
        self.cfg = cfg
        df = pd.read_csv(csv_path).reset_index(drop=True)
        N  = len(df)
        H  = cfg.history_len

        self.image_paths = df['image_path'].tolist()
        self.rel   = df[['rel_x','rel_y','rel_z']].values.astype(np.float32)
        self.abs_  = df[['abs_x','abs_y','abs_z']].values.astype(np.float32)
        self.quat  = df[['qw','qx','qy','qz']].values.astype(np.float32)
        self.motor = df[['m1_spd','m2_spd','m1_pos','m2_pos']].values.astype(np.float32)
        self.actions = np.clip(
            df[['joy_vec_lr','joy_vec_ud']].values.astype(np.float32), -1.0, 1.0)

        if 'image_error_x' in df.columns and 'image_error_y' in df.columns:
            self.image_error = np.stack([
                df['image_error_x'].values.astype(np.float32),
                df['image_error_y'].values.astype(np.float32),
            ], axis=1)
        else:
            self.image_error = np.zeros((N, 2), dtype=np.float32)

        cam_cx, cam_cy   = cfg.cam_w / 2.0, cfg.cam_h / 2.0
        default_w = default_h = 20.0
        def _bbox(i):
            row = df.iloc[i]
            cx = float(row.get('bbox_cx', np.nan)); cx = cam_cx if np.isnan(cx) else cx
            cy = float(row.get('bbox_cy', np.nan)); cy = cam_cy if np.isnan(cy) else cy
            bw = float(row.get('bbox_w',  np.nan)); bw = default_w if np.isnan(bw) else bw
            bh = float(row.get('bbox_h',  np.nan)); bh = default_h if np.isnan(bh) else bh
            return [cx, cy, bw, bh]
        self.bbox = np.array([_bbox(i) for i in range(N)], dtype=np.float32)

        self.em_state = np.concatenate([self.rel, self.abs_, self.quat], axis=1)
        m = cfg.max_motor_speed
        self.motor_norm = np.clip(np.stack([
            self.motor[:,0] / m,
            self.motor[:,1] / m,
            self.motor[:,2] / 50.0,
            self.motor[:,3] / 50.0,
        ], axis=1).astype(np.float32), -3.0, 3.0)

        def _ep(path):
            match = re.search(r'(video_\d+)', str(path))
            return match.group(1) if match else 'unknown'
        self.episode_ids = [_ep(p) for p in self.image_paths]

        raw_rewards = self._compute_rewards()
        r_std = float(raw_rewards.std())
        r_std = r_std if r_std > 1e-6 else 1.0
        self.rewards = ((raw_rewards - raw_rewards.mean()) / r_std).astype(np.float32)
        print(f"[Dataset] Reward stats — mean: {raw_rewards.mean():.4f}, "
              f"std: {r_std:.4f}  (normalised to unit std)")

        active = (np.abs(self.motor[:,0]) > 0) | (np.abs(self.motor[:,1]) > 0)
        done   = np.zeros(N, dtype=np.float32)
        for i in range(1, N):
            if (active[i-1] and not active[i]) or \
               (self.episode_ids[i] != self.episode_ids[i-1]):
                done[i] = 1.0
        self.dones = done

        valid = []
        for i in range(H, N - 1):
            if all(self.episode_ids[i-H+j] == self.episode_ids[i] for j in range(H)):
                valid.append(i)
        self.valid_idx = valid
        print(f"[Dataset] {N} rows, {len(set(self.episode_ids))} episodes "
              f"-> {len(valid)} valid transitions")

    def _compute_rewards(self):
        cfg = self.cfg
        err = np.linalg.norm(self.image_error, axis=1)
        r   = -cfg.reward_alpha * err
        r  +=  cfg.reward_beta  * (err < 0.5).astype(float)
        r  -=  cfg.reward_gamma * np.linalg.norm(self.actions, axis=1)
        r  +=  cfg.reward_delta * (err < cfg.success_threshold).astype(float)
        return r.astype(np.float32)

    def _load_image(self, path: str) -> np.ndarray:
        img = cv2.imread(path)
        if img is None:
            return np.zeros((3, self.cfg.cam_h, self.cfg.cam_w), dtype=np.float32)
        img = cv2.resize(img, (self.cfg.cam_w, self.cfg.cam_h))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return img.transpose(2, 0, 1).astype(np.float32) / 255.0

    def _build_obs(self, i: int) -> Dict[str, np.ndarray]:
        H = self.cfg.history_len
        return {
            'image':       self._load_image(self.image_paths[i]),
            'trajectory':  self.bbox[i - H : i],
            'em_state':    self.em_state[i],
            'motor_state': self.motor_norm[i],
            'image_error': self.image_error[i],
        }

    def __len__(self):
        return len(self.valid_idx)

    def __getitem__(self, idx):
        i  = self.valid_idx[idx]
        i1 = i + 1 if i + 1 < len(self.episode_ids) else i
        obs  = self._build_obs(i)
        nobs = self._build_obs(i1)
        return {
            'image':            torch.from_numpy(obs['image']),
            'trajectory':       torch.from_numpy(obs['trajectory']),
            'em_state':         torch.from_numpy(obs['em_state']),
            'motor_state':      torch.from_numpy(obs['motor_state']),
            'image_error':      torch.from_numpy(obs['image_error']),
            'next_image':       torch.from_numpy(nobs['image']),
            'next_trajectory':  torch.from_numpy(nobs['trajectory']),
            'next_em_state':    torch.from_numpy(nobs['em_state']),
            'next_motor_state': torch.from_numpy(nobs['motor_state']),
            'next_image_error': torch.from_numpy(nobs['image_error']),
            'action':  torch.from_numpy(self.actions[i]),
            'reward':  torch.tensor(self.rewards[i], dtype=torch.float32),
            'done':    torch.tensor(self.dones[i],   dtype=torch.float32),
        }


# ──────────────────────────────────────────────────────────────────────────────
# 2.  Actor
# ──────────────────────────────────────────────────────────────────────────────

class BCNetworkActor(nn.Module):
    def __init__(self, bc_network: nn.Module, freeze_encoder: bool = True):
        super().__init__()
        self.net = bc_network
        
        if freeze_encoder:
            print("[Actor] Freezing feature encoders (ResNet/LSTM). Only training Policy Head.")
            for param in self.net.parameters():
                param.requires_grad = False
            
            if hasattr(self.net, 'policy'):
                for param in self.net.policy.parameters():
                    param.requires_grad = True
            else:
                print("[Warning] 'policy' module not found in BC network!")

    def encode(self, image, trajectory, em_state, motor_state, image_error) -> torch.Tensor:
        # 修正 RNN 權重警告
        if hasattr(self.net.trajectory_encoder, 'lstm'):
            self.net.trajectory_encoder.lstm.flatten_parameters()

        with torch.set_grad_enabled(not self.net.image_encoder.parameters().__iter__().__next__().requires_grad):
            img_feat, _          = self.net.image_encoder(image)
            traj_feat, dynamics  = self.net.trajectory_encoder(trajectory)
            state_feat           = self.net.state_encoder(em_state, motor_state)
            features = torch.cat([img_feat, traj_feat, state_feat, dynamics, image_error], dim=-1)
        return features

    def forward(self, image, trajectory, em_state, motor_state,
                image_error, deterministic: bool = True) -> torch.Tensor:
        out = self.net(image, trajectory, em_state, motor_state,
                       image_error, deterministic=deterministic)
        return out['action_mean']


# ──────────────────────────────────────────────────────────────────────────────
# 3.  Critics and Value network
# ──────────────────────────────────────────────────────────────────────────────

def _mlp(dims, act=nn.ReLU):
    layers = []
    for i in range(len(dims) - 1):
        layers.append(nn.Linear(dims[i], dims[i+1]))
        if i < len(dims) - 2:
            layers.append(act())
    return nn.Sequential(*layers)

class DoubleQCritic(nn.Module):
    def __init__(self, feature_dim, action_dim, hidden=None):
        super().__init__()
        hidden = hidden or [512, 256, 128]
        inp = feature_dim + action_dim
        self.q1 = _mlp([inp] + hidden + [1])
        self.q2 = _mlp([inp] + hidden + [1])

    def forward(self, features, action):
        x = torch.cat([features, action], dim=-1)
        return self.q1(x), self.q2(x)

    def q_min(self, features, action):
        q1, q2 = self.forward(features, action)
        return torch.min(q1, q2)

class ValueNetwork(nn.Module):
    def __init__(self, feature_dim, hidden=None):
        super().__init__()
        hidden = hidden or [512, 256, 128]
        self.net = _mlp([feature_dim] + hidden + [1])

    def forward(self, features):
        return self.net(features)


# ──────────────────────────────────────────────────────────────────────────────
# 4.  Helper
# ──────────────────────────────────────────────────────────────────────────────

def _to(batch, device, prefix=''):
    p = prefix
    return (batch[f'{p}image'].to(device),
            batch[f'{p}trajectory'].to(device),
            batch[f'{p}em_state'].to(device),
            batch[f'{p}motor_state'].to(device),
            batch[f'{p}image_error'].to(device))


# ──────────────────────────────────────────────────────────────────────────────
# 5.  IQL Trainer
# ──────────────────────────────────────────────────────────────────────────────

class IQLTrainer:
    def __init__(self, cfg, bc_network, device):
        self.cfg    = cfg
        self.device = device
        self.actor  = BCNetworkActor(bc_network, freeze_encoder=cfg.freeze_encoder).to(device)
        self.critic = DoubleQCritic(cfg.feature_dim, cfg.action_dim).to(device)
        self.value  = ValueNetwork(cfg.feature_dim).to(device)
        self.critic_target = copy.deepcopy(self.critic)
        self.value_target  = copy.deepcopy(self.value)   

        actor_params = [p for p in self.actor.parameters() if p.requires_grad]
        self.opt_actor  = torch.optim.Adam(actor_params, lr=cfg.lr_actor)
        self.opt_critic = torch.optim.Adam(self.critic.parameters(), lr=cfg.lr_critic)
        self.opt_value  = torch.optim.Adam(self.value.parameters(),  lr=cfg.lr_value)
        self._step = 0

    def _expectile(self, diff, tau):
        w = torch.where(diff > 0,
                        torch.full_like(diff, tau),
                        torch.full_like(diff, 1.0 - tau))
        return (w * diff.pow(2)).mean()

    def update(self, batch):
        a = batch['action'].to(self.device)
        r = batch['reward'].unsqueeze(-1).to(self.device)
        d = batch['done'].unsqueeze(-1).to(self.device)
        self._step += 1
        logs = {}

        img,  traj,  em,  mot,  err  = _to(batch, self.device)
        nimg, ntraj, nem, nmot, nerr = _to(batch, self.device, 'next_')

        with torch.no_grad():
            feat      = self.actor.encode(img,  traj,  em,  mot,  err)
            feat_next = self.actor.encode(nimg, ntraj, nem, nmot, nerr)

        # Value update
        with torch.no_grad():
            q1, q2 = self.critic_target(feat, a)
            q_min  = torch.min(q1, q2)
        v      = self.value(feat)
        v_loss = self._expectile(q_min - v, self.cfg.iql_tau)
        self.opt_value.zero_grad(); v_loss.backward()
        nn.utils.clip_grad_norm_(self.value.parameters(), self.cfg.grad_clip)
        self.opt_value.step()
        logs['loss/value'] = v_loss.item()

        # Critic update
        with torch.no_grad():
            td_target = r + self.cfg.gamma * (1 - d) * self.value_target(feat_next)  
            td_target = td_target.clamp(-10.0, 10.0)   
        q1, q2 = self.critic(feat, a)
        q_loss = F.mse_loss(q1, td_target) + F.mse_loss(q2, td_target)
        self.opt_critic.zero_grad(); q_loss.backward()
        nn.utils.clip_grad_norm_(self.critic.parameters(), self.cfg.grad_clip)
        self.opt_critic.step()
        logs['loss/critic'] = q_loss.item()

        # Actor update
        a_pred = self.actor(img, traj, em, mot, err)
        with torch.no_grad():
            q1_pi, q2_pi = self.critic_target(feat, a)
            adv     = torch.min(q1_pi, q2_pi) - self.value(feat)
            weights = torch.exp(adv / self.cfg.iql_beta).clamp(max=100.0)
            
        actor_loss = (weights * F.mse_loss(a_pred, a, reduction='none')
                      .sum(-1, keepdim=True)).mean()
        self.opt_actor.zero_grad(); actor_loss.backward()
        nn.utils.clip_grad_norm_([p for p in self.actor.parameters() if p.requires_grad], self.cfg.grad_clip)
        self.opt_actor.step()
        logs['loss/actor'] = actor_loss.item()

        for p, tp in zip(self.critic.parameters(), self.critic_target.parameters()):
            tp.data.mul_(1 - self.cfg.tau).add_(self.cfg.tau * p.data)
        for p, tp in zip(self.value.parameters(), self.value_target.parameters()):
            tp.data.mul_(1 - self.cfg.tau).add_(self.cfg.tau * p.data)

        return logs

    def save(self, path):
        torch.save({'actor':        self.actor.state_dict(),
                    'critic':       self.critic.state_dict(),
                    'value':        self.value.state_dict(),
                    'value_target': self.value_target.state_dict()}, path)


# ──────────────────────────────────────────────────────────────────────────────
# 6.  TD3+BC Trainer (Optimized)
# ──────────────────────────────────────────────────────────────────────────────

class TD3BCTrainer:
    def __init__(self, cfg, bc_network, device):
        self.cfg    = cfg
        self.device = device
        self.actor         = BCNetworkActor(bc_network, freeze_encoder=cfg.freeze_encoder).to(device)
        self.actor_target  = copy.deepcopy(self.actor)
        self.critic        = DoubleQCritic(cfg.feature_dim, cfg.action_dim).to(device)
        self.critic_target = copy.deepcopy(self.critic)

        actor_params = [p for p in self.actor.parameters() if p.requires_grad]
        self.opt_actor  = torch.optim.Adam(actor_params, lr=cfg.lr_actor)
        self.opt_critic = torch.optim.Adam(self.critic.parameters(), lr=cfg.lr_critic)
        self._step = 0
        
        self._last_actor_loss = 0.0  
        self._last_bc_loss = 0.0

    def update(self, batch):
        a = batch['action'].to(self.device)
        r = batch['reward'].unsqueeze(-1).to(self.device)
        d = batch['done'].unsqueeze(-1).to(self.device)
        self._step += 1
        logs = {}

        img,  traj,  em,  mot,  err  = _to(batch, self.device)
        nimg, ntraj, nem, nmot, nerr = _to(batch, self.device, 'next_')

        # ── Critic update ───────────────────────────────────────────────────
        with torch.no_grad():
            if hasattr(self.actor_target.net.trajectory_encoder, 'lstm'):
                self.actor_target.net.trajectory_encoder.lstm.flatten_parameters()

            feat      = self.actor.encode(img,  traj,  em,  mot,  err)
            feat_next = self.actor_target.encode(nimg, ntraj, nem, nmot, nerr)

            noise  = (torch.randn_like(a) * self.cfg.td3_policy_noise).clamp(
                -self.cfg.td3_noise_clip, self.cfg.td3_noise_clip)
            a_next = (self.actor_target(nimg, ntraj, nem, nmot, nerr) + noise).clamp(-1.0, 1.0)

            q1_t, q2_t = self.critic_target(feat_next, a_next)
            td_target  = r + self.cfg.gamma * (1 - d) * torch.min(q1_t, q2_t)
            td_target  = td_target.clamp(-10.0, 10.0)  

        q1, q2 = self.critic(feat, a)
        q_loss = F.mse_loss(q1, td_target) + F.mse_loss(q2, td_target)
        
        self.opt_critic.zero_grad()
        q_loss.backward()
        nn.utils.clip_grad_norm_(self.critic.parameters(), self.cfg.grad_clip)
        self.opt_critic.step()
        
        logs['loss/critic'] = q_loss.item()

        # ── Delayed actor update ────────────────────────────────────────────
        if self._step % self.cfg.td3_policy_delay == 0:
            feat_actor = self.actor.encode(img, traj, em, mot, err)
            a_pred = self.actor(img, traj, em, mot, err)

            # Fix 1: removed critic requires_grad toggling — opt_actor only holds
            # actor params so critic weights never receive gradients from actor_loss.
            # Toggling mid-step corrupts the autograd graph and destabilises the critic.
            q_val = self.critic.q_min(feat_actor, a_pred)

            lam = self.cfg.td3bc_lambda / (q_val.abs().mean().detach() + 1e-8)
            
            bc_loss = F.mse_loss(a_pred, a)
            actor_loss = -lam * q_val.mean() + bc_loss

            self.opt_actor.zero_grad()
            actor_loss.backward()
            nn.utils.clip_grad_norm_([p for p in self.actor.parameters() if p.requires_grad],
                                     self.cfg.grad_clip)
            self.opt_actor.step()
            
            self._last_actor_loss = actor_loss.item()
            self._last_bc_loss = bc_loss.item()

            for p, tp in zip(self.actor.parameters(), self.actor_target.parameters()):
                tp.data.mul_(1 - self.cfg.tau).add_(self.cfg.tau * p.data)
            for p, tp in zip(self.critic.parameters(), self.critic_target.parameters()):
                tp.data.mul_(1 - self.cfg.tau).add_(self.cfg.tau * p.data)

        logs['loss/actor'] = self._last_actor_loss
        logs['loss/bc_mse'] = self._last_bc_loss
        return logs

    def save(self, path):
        torch.save({'actor':        self.actor.state_dict(),
                    'critic':       self.critic.state_dict(),
                    'actor_target': self.actor_target.state_dict()}, path)


# ──────────────────────────────────────────────────────────────────────────────
# 7.  Offline Evaluation
# ──────────────────────────────────────────────────────────────────────────────

def evaluate_offline(trainer, val_loader, device):
    trainer.actor.eval()
    total_mse = 0.0
    total_q = 0.0
    n_batches = 0

    with torch.no_grad():
        for batch in val_loader:
            img, traj, em, mot, err = _to(batch, device)
            expert_a = batch['action'].to(device)

            feat = trainer.actor.encode(img, traj, em, mot, err)
            a_pred = trainer.actor(img, traj, em, mot, err)
            
            q1, q2 = trainer.critic(feat, a_pred)
            q_min = torch.min(q1, q2)

            mse = F.mse_loss(a_pred, expert_a).item()
            
            total_mse += mse
            total_q += q_min.mean().item()
            n_batches += 1

    trainer.actor.train()
    
    return {
        'eval/action_mse': total_mse / max(n_batches, 1),
        'eval/mean_q_value': total_q / max(n_batches, 1)
    }


# ──────────────────────────────────────────────────────────────────────────────
# 8.  Load BC network
# ──────────────────────────────────────────────────────────────────────────────

def load_bc_network(bc_path: str, device: torch.device) -> nn.Module:
    project_root = str(Path(bc_path).resolve().parent.parent)
    for root in [project_root, str(Path(bc_path).resolve().parent.parent.parent)]:
        for sub in ['', 'training', 'models', 'configs']:
            p = str(Path(root) / sub) if sub else root
            if p not in sys.path:
                sys.path.insert(0, p)

    from configs.config import get_config
    from models.network import EndoscopeTrackingNetwork

    config  = get_config()
    network = EndoscopeTrackingNetwork(config).to(device)
    ckpt    = torch.load(bc_path, map_location=device, weights_only=False)
    sd      = ckpt.get('model_state_dict', ckpt)
    missing, unexpected = network.load_state_dict(sd, strict=False)
    print(f"[BC] Loaded {bc_path}  missing={len(missing)}  unexpected={len(unexpected)}")
    return network


# ──────────────────────────────────────────────────────────────────────────────
# 9.  Training loop
# ──────────────────────────────────────────────────────────────────────────────

def train(args):
    cfg = OfflineRLConfig()
    cfg.algo       = args.algo
    cfg.num_epochs = args.epochs
    cfg.batch_size = args.batch

    random.seed(cfg.seed); np.random.seed(cfg.seed); torch.manual_seed(cfg.seed)

    is_cuda = torch.cuda.is_available()
    device  = torch.device('cuda' if is_cuda else 'cpu')
    print(f"[Train] Device: {device}" + (f" ({torch.cuda.get_device_name(0)})" if is_cuda else " (CPU)"))

    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    log_dir = Path('logs/offline_rl'); log_dir.mkdir(parents=True, exist_ok=True)
    writer  = SummaryWriter(log_dir=str(log_dir))

    print(f"\n[Train] Loading dataset: {args.data}")
    full_dataset = OfflineDataset(args.data, cfg)
    
    val_size = max(1, int(len(full_dataset) * 0.1)) 
    train_size = len(full_dataset) - val_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])
    
    train_loader = DataLoader(train_dataset, batch_size=cfg.batch_size, shuffle=True,
                              num_workers=4, pin_memory=is_cuda, drop_last=True)
    val_loader   = DataLoader(val_dataset, batch_size=cfg.batch_size, shuffle=False,
                              num_workers=4, pin_memory=is_cuda)

    print(f"[Train] Loading BC network: {args.bc_ckpt}")
    bc_network = load_bc_network(args.bc_ckpt, device)

    # 確保所有 RNN 層的權重在內存中連續
    bc_network.apply(lambda m: m.flatten_parameters() if hasattr(m, 'flatten_parameters') else None)

    if cfg.algo == 'iql':
        trainer = IQLTrainer(cfg, bc_network, device)
    else:
        trainer = TD3BCTrainer(cfg, bc_network, device)

    log_path    = log_dir / 'training_log.json'
    log_path.write_text('[]')
    best_eval_mse  = float('inf')
    global_step = 0

    print(f"\n[Train] Starting offline RL ({cfg.algo.upper()}) — "
          f"{cfg.num_epochs} epochs on {device}\n")

    for epoch in range(1, cfg.num_epochs + 1):
        epoch_logs: Dict[str, List[float]] = {}
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{cfg.num_epochs}", leave=False)

        for batch in pbar:
            logs = trainer.update(batch)
            global_step += 1
            for k, v in logs.items():
                epoch_logs.setdefault(k, []).append(v)
                writer.add_scalar(k, v, global_step)
            if 'loss/actor' in logs:
                pbar.set_postfix({'actor':  f"{logs['loss/actor']:.4f}",
                                  'critic': f"{logs['loss/critic']:.4f}"})

        mean_logs = {k: float(np.mean(v)) for k, v in epoch_logs.items()}
        print(f"Epoch {epoch:4d} | " + " | ".join(f"{k}: {v:.4f}" for k, v in mean_logs.items()))
        for k, v in mean_logs.items():
            writer.add_scalar(k, v, epoch)

        eval_metrics = evaluate_offline(trainer, val_loader, device)
        for k, v in eval_metrics.items():
            writer.add_scalar(k, v, epoch)
        print(f"          | eval/action_mse: {eval_metrics['eval/action_mse']:.4f} "
              f"| eval/mean_q: {eval_metrics['eval/mean_q_value']:.4f}")

        history = json.loads(log_path.read_text())
        history.append({"epoch": epoch, **mean_logs, **{k: float(v) for k, v in eval_metrics.items()}})
        log_path.write_text(json.dumps(history, indent=2))

        if eval_metrics['eval/action_mse'] < best_eval_mse:
            best_eval_mse = eval_metrics['eval/action_mse']
            trainer.save(str(out_dir / 'best.pt'))
            print(f"          ^ New best model (Action MSE = {best_eval_mse:.4f})")

        if epoch % cfg.save_every == 0:
            trainer.save(str(out_dir / f'epoch_{epoch:04d}.pt'))

    trainer.save(str(out_dir / 'final.pt'))
    writer.close()
    print(f"\n[Train] Done. Best Eval Action MSE: {best_eval_mse:.4f}. Checkpoints in {out_dir}/")


# ──────────────────────────────────────────────────────────────────────────────
# 10.  Entry point
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    ap = argparse.ArgumentParser(description='Offline RL — Finetuning BC policy head')
    ap.add_argument('--data',    required=True,
                    help='Path to master_bc_data_with_errors.csv')
    ap.add_argument('--bc_ckpt', default='checkpoints/bc/bc_model.pt',
                    help='Path to BC checkpoint (EndoscopeTrackingNetwork)')
    ap.add_argument('--algo',    default='iql', choices=['iql', 'td3bc'])
    ap.add_argument('--epochs',  type=int, default=200)
    ap.add_argument('--batch',   type=int, default=32)
    ap.add_argument('--out_dir', default='checkpoints/offline_rl')
    args = ap.parse_args()
    train(args)