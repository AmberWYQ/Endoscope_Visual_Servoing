"""
Stage 2: Behavior Cloning (BC) / Imitation Learning
Learn from expert demonstrations collected in real-world

This stage:
1. Loads pretrained PINN model
2. Fine-tunes on real expert demonstrations
3. Learns to handle real-world noise and disturbances
"""

import os
os.environ['QT_QPA_PLATFORM'] = 'offscreen'

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np
from pathlib import Path
import sys
import os
import json
import pandas as pd
from tqdm import tqdm
from datetime import datetime
from typing import Dict, List, Optional
import glob
import cv2

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from configs.config import get_config, Config
from models.network import EndoscopeTrackingNetwork
from simulator.endoscope_sim import make_env


class ExpertDemonstrationDataset(Dataset):
    """
    Dataset for loading expert demonstrations
    Compatible with LeRobot format and custom CSV format
    """
    
    def __init__(self, 
                 data_path: str,
                 config: Config,
                 history_length: int = 10,
                 transform=None):
        self.config = config
        self.history_length = history_length
        self.transform = transform
        
        # Load data based on format
        data_path = Path(data_path)
        
        if data_path.suffix == '.json':
            self._load_json(data_path)
        elif data_path.suffix == '.csv':
            self._load_csv(data_path)
        elif data_path.is_dir():
            self._load_directory(data_path)
        else:
            raise ValueError(f"Unsupported data format: {data_path}")
        
        print(f"Loaded {len(self)} samples from expert demonstrations")
    
    def _load_json(self, filepath: Path):
        """Load JSON format dataset"""
        with open(filepath, 'r') as f:
            data = json.load(f)
        
        self.samples = []
        
        for episode in data.get('episodes', [data]):
            frames = episode if isinstance(episode, list) else episode.get('frames', [episode])
            
            for i in range(self.history_length, len(frames)):
                # Collect history
                trajectory = []
                for j in range(i - self.history_length, i):
                    frame = frames[j]
                    # Extract target info from frame
                    if 'trajectory' in frame:
                        traj_point = frame['trajectory'][-1] if isinstance(frame['trajectory'][0], list) else frame['trajectory']
                    else:
                        traj_point = [320, 240, 20, 20]  # Default center
                    trajectory.append(traj_point)
                
                current_frame = frames[i]
                
                sample = {
                    'trajectory': np.array(trajectory, dtype=np.float32),
                    'em_state': np.array(current_frame.get('em_state', np.zeros(10)), dtype=np.float32),
                    'motor_state': np.array(current_frame.get('motor_state', np.zeros(4)), dtype=np.float32),
                    'image_error': np.array(current_frame.get('image_error', [0, 0]), dtype=np.float32),
                    'action': np.array(current_frame.get('action', [0, 0]), dtype=np.float32),
                }
                
                # Handle image
                if 'image' in current_frame:
                    img = np.array(current_frame['image'], dtype=np.uint8)
                    if img.ndim == 3:
                        sample['image'] = img.transpose(2, 0, 1).astype(np.float32) / 255.0
                    else:
                        sample['image'] = np.zeros((3, self.config.camera.height, self.config.camera.width), dtype=np.float32)
                else:
                    sample['image'] = np.zeros((3, self.config.camera.height, self.config.camera.width), dtype=np.float32)
                
                self.samples.append(sample)
    
    def _load_csv(self, filepath: Path):
        """Load CSV format (from your data collection)"""
        df = pd.read_csv(filepath)

        self.samples = []

        # Image centre used as fallback when bbox detection failed
        cam_cx = self.config.camera.width  / 2.0   # 320.0
        cam_cy = self.config.camera.height / 2.0   # 240.0
        # Neutral bbox size used when no detection is available
        default_w = 20.0
        default_h = 20.0

        for i in range(self.history_length, len(df)):

            # ── Trajectory: history of [bbox_cx, bbox_cy, bbox_w, bbox_h] ────
            # Uses real detector output when available; falls back to image
            # centre + default bbox size for frames where detection failed.
            trajectory = []
            for j in range(i - self.history_length, i):
                row = df.iloc[j]
                cx = row.get('bbox_cx', np.nan)
                cy = row.get('bbox_cy', np.nan)
                bw = row.get('bbox_w',  np.nan)
                bh = row.get('bbox_h',  np.nan)
                # Use image centre / default size if detection was missing
                if np.isnan(cx): cx = cam_cx
                if np.isnan(cy): cy = cam_cy
                if np.isnan(bw): bw = default_w
                if np.isnan(bh): bh = default_h
                trajectory.append([float(cx), float(cy), float(bw), float(bh)])

            current = df.iloc[i]

            # ── EM state ──────────────────────────────────────────────────────
            em_state = np.array([
                current.get('rel_x', 0), current.get('rel_y', 0), current.get('rel_z', 0),
                current.get('abs_x', 0), current.get('abs_y', 0), current.get('abs_z', 0),
                current.get('qw', 1),    current.get('qx', 0),
                current.get('qy', 0),    current.get('qz', 0),
            ], dtype=np.float32)

            # ── Motor state ───────────────────────────────────────────────────
            motor_state = np.array([
                current.get('m1_pos', 0), current.get('m2_pos', 0),
                current.get('m1_spd', 0), current.get('m2_spd', 0),
            ], dtype=np.float32)

            # ── Image error: from detector, NOT joystick proxy ────────────────
            image_error = np.array([
                current.get('image_error_x', 0.0),
                current.get('image_error_y', 0.0),
            ], dtype=np.float32)

            # ── Action: expert joystick input ─────────────────────────────────
            # joy_vec_lr/ud are the human demonstration signal [-1, 1].
            # m1_spd/m2_spd are the resulting motor speeds (not used as action
            # targets because they include motor dynamics / lag).
            action = np.array([
                current.get('joy_vec_lr', 0.0),
                current.get('joy_vec_ud', 0.0),
            ], dtype=np.float32)

            sample = {
                'trajectory':  np.array(trajectory, dtype=np.float32),
                'em_state':    em_state,
                'motor_state': motor_state,
                'image_error': image_error,
                'action':      action,
                'image_path':  str(current.get('image_path', '')),
            }

            self.samples.append(sample)
    
    def _load_directory(self, dirpath: Path):
        """Load from directory with multiple files"""
        self.samples = []
        
        # Look for CSV files
        csv_files = list(dirpath.glob("*.csv"))
        json_files = list(dirpath.glob("*.json"))
        
        for csv_file in csv_files:
            sub_dataset = ExpertDemonstrationDataset(str(csv_file), self.config, self.history_length)
            self.samples.extend(sub_dataset.samples)
        
        for json_file in json_files:
            sub_dataset = ExpertDemonstrationDataset(str(json_file), self.config, self.history_length)
            self.samples.extend(sub_dataset.samples)
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]

        # 实时读取图像文件
        img_path = sample.get('image_path', '')
        if img_path and os.path.exists(img_path):
            img = cv2.imread(img_path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, (self.config.camera.width, self.config.camera.height))
            img = img.transpose(2, 0, 1).astype(np.float32) / 255.0
        else:
            # 如果路径无效，回退到零矩阵
            img = np.zeros((3, self.config.camera.height, self.config.camera.width), dtype=np.float32)
        
        return {
            'image': torch.tensor(img, dtype=torch.float32), # 注意这里用的是局部变量 img
            'trajectory': torch.tensor(sample['trajectory'], dtype=torch.float32),
            'em_state': torch.tensor(sample['em_state'], dtype=torch.float32),
            'motor_state': torch.tensor(sample['motor_state'], dtype=torch.float32),
            'image_error': torch.tensor(sample['image_error'], dtype=torch.float32),
            'action': torch.tensor(sample['action'], dtype=torch.float32),
        }


class BCLoss(nn.Module):
    """
    Behavior Cloning Loss
    
    Components:
    1. Action MSE: Match expert actions
    2. Action Smoothness: Penalize jerky actions
    3. Feature Alignment: Align internal representations
    """
    
    def __init__(self, config: Config):
        super().__init__()
        self.config = config
        
    def forward(self, predictions: dict, targets: dict, prev_action: Optional[torch.Tensor] = None) -> dict:
        losses = {}
        
        # Action prediction loss (main BC loss)
        action_pred = predictions['action']
        action_target = targets['action']
        
        losses['action_mse'] = F.mse_loss(action_pred, action_target)
        
        # Also add L1 loss for robustness
        losses['action_l1'] = F.l1_loss(action_pred, action_target) * 0.5
        
        # Smoothness loss (if previous action available)
        if prev_action is not None:
            action_diff = action_pred - prev_action
            losses['smoothness'] = torch.mean(action_diff ** 2) * 0.1
        
        # Jacobian regularization (keep Jacobian reasonable)
        if 'jacobian' in predictions:
            jacobian = predictions['jacobian']
            # Frobenius norm regularization
            losses['jacobian_reg'] = torch.mean(torch.sum(jacobian ** 2, dim=[1, 2])) * 0.001
        
        # Total loss
        losses['total'] = sum(losses.values())
        
        return losses


class DAggerBuffer:
    """
    Buffer for DAgger (Dataset Aggregation) style training
    Aggregates expert corrections on policy rollouts
    """
    
    def __init__(self, max_size: int = 100000):
        self.max_size = max_size
        self.buffer = []
    
    def add(self, sample: dict):
        if len(self.buffer) >= self.max_size:
            self.buffer.pop(0)
        self.buffer.append(sample)
    
    def sample(self, batch_size: int) -> List[dict]:
        indices = np.random.choice(len(self.buffer), min(batch_size, len(self.buffer)), replace=False)
        return [self.buffer[i] for i in indices]
    
    def __len__(self):
        return len(self.buffer)


class BCTrainer:
    """
    Trainer for Stage 2: Behavior Cloning
    """
    
    def __init__(self, config: Config, 
                 pinn_checkpoint: Optional[str] = None,
                 checkpoint_dir: str = None):
        self.config = config
        self.device = torch.device(config.training.device)
        
        # Create model
        self.model = EndoscopeTrackingNetwork(config).to(self.device)
        
        # Load PINN pretrained weights if available
        if pinn_checkpoint and Path(pinn_checkpoint).exists():
            self._load_pinn_weights(pinn_checkpoint)
        
        # Optimizer (lower learning rate for fine-tuning)
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=config.training.bc_lr,
            weight_decay=config.training.bc_weight_decay
        )
        
        # Scheduler
        # CosineAnnealingLR: smooth single decay over all epochs, no restarts
        # Set T_max via config so it matches however many epochs you train for
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=self.config.training.bc_epochs,
            eta_min=1e-6
        )
        
        # Loss
        self.criterion = BCLoss(config)

        # Mixed precision scaler (AMP) — no-op on CPU, ~2x faster on T4
        self.scaler = torch.amp.GradScaler('cuda', enabled=self.device.type == 'cuda')
        
        # DAgger buffer
        self.dagger_buffer = DAggerBuffer()
        
        # Checkpointing
        self.checkpoint_dir = Path(checkpoint_dir or config.checkpoint_dir) / "bc"
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        # Logging
        self.log_dir = Path(config.log_dir) / "bc"
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        self.train_history = []
    
    def _load_pinn_weights(self, checkpoint_path: str):
        """Load pretrained PINN weights"""
        print(f"Loading PINN weights from {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
        
        # Load state dict (handle potential key mismatches)
        model_dict = self.model.state_dict()
        pretrained_dict = checkpoint['model_state_dict']
        
        # Filter out incompatible keys
        pretrained_dict = {k: v for k, v in pretrained_dict.items() 
                         if k in model_dict and v.shape == model_dict[k].shape}
        
        model_dict.update(pretrained_dict)
        self.model.load_state_dict(model_dict)
        
        print(f"Loaded {len(pretrained_dict)}/{len(model_dict)} layers from PINN checkpoint")
    
    def train(self, 
              data_path: str,
              num_epochs: int = None,
              use_dagger: bool = False):
        """Run BC training"""
        num_epochs = num_epochs or self.config.training.bc_epochs
        
        # Create dataset
        print(f"Loading expert demonstrations from {data_path}")
        dataset = ExpertDemonstrationDataset(data_path, self.config)
        
        # Split into train/val
        train_size = int(0.9 * len(dataset))
        val_size = len(dataset) - train_size
        train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
        
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.config.training.bc_batch_size,
            shuffle=True,
            num_workers=2,
            pin_memory=True,
            drop_last= True
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=self.config.training.bc_batch_size,
            shuffle=False,
            num_workers=2,
            drop_last= True
        )
        
        print(f"Training on {len(train_dataset)} samples, validating on {len(val_dataset)} samples")
        print(f"Starting BC training for {num_epochs} epochs...")
        
        best_val_loss = float('inf')
        
        for epoch in range(num_epochs):
            # Training (model.train() -- dropout active, used for gradients only)
            train_loss = self._train_epoch(train_loader, epoch)

            # Validation (model.eval() -- dropout off)
            val_loss = self._validate(val_loader)

            # Train eval loss: re-evaluate train set with model.eval() so it is
            # directly comparable to val_loss for visualization (no dropout noise)
            train_eval_loss = self._validate(train_loader)

            # Learning rate step
            self.scheduler.step()

            # Logging
            self.train_history.append({
                'epoch': epoch + 1,
                'train_loss': train_eval_loss,
                'val_loss': val_loss,
                'lr': self.optimizer.param_groups[0]['lr']
            })

            print(f"Epoch {epoch+1}: train_loss={train_eval_loss:.4f}, val_loss={val_loss:.4f}")
            
            # Save best model
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                self.save_checkpoint('best.pt')
            
            # Regular checkpoint
            if (epoch + 1) % self.config.training.save_every == 0:
                self.save_checkpoint(f'epoch_{epoch+1}.pt')
            
            # DAgger: collect more data with policy
            if use_dagger and (epoch + 1) % 10 == 0:
                self._dagger_collect()
        
        # Save final model
        self.save_checkpoint('final.pt')
        
        # Save training history
        with open(self.log_dir / 'train_history.json', 'w') as f:
            json.dump(self.train_history, f, indent=2)
        
        print(f"BC training complete! Best val loss: {best_val_loss:.4f}")
        
        return self.model
    
    def _train_epoch(self, dataloader: DataLoader, epoch: int) -> float:
        """Train for one epoch"""
        self.model.train()
        total_loss = 0.0
        core_loss  = 0.0   # action_mse + action_l1 + jacobian_reg (no smoothness)
        num_batches = 0
        prev_action = None
        
        pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}")
        
        for batch in pbar:
            # Move to device
            batch = {k: v.to(self.device) for k, v in batch.items()}

            # Forward pass with AMP
            with torch.autocast(device_type=self.device.type, dtype=torch.float16,
                                enabled=self.device.type == 'cuda'):
                predictions = self.model(
                    image=batch['image'],
                    trajectory=batch['trajectory'],
                    em_state=batch['em_state'],
                    motor_state=batch['motor_state'],
                    image_error=batch['image_error']
                )

                # Compute loss (with smoothness for gradient signal)
                targets = {'action': batch['action']}
                losses = self.criterion(predictions, targets, prev_action)

            # Backward pass with AMP scaler
            self.optimizer.zero_grad()
            self.scaler.scale(losses['total']).backward()
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.scaler.step(self.optimizer)
            self.scaler.update()
            
            # Update prev_action for smoothness
            prev_action = predictions['action'].detach()
            
            total_loss += losses['total'].item()
            # Core loss = total minus smoothness (comparable to val loss)
            smooth = losses.get('smoothness', torch.tensor(0.0)).item()
            core_loss += losses['total'].item() - smooth
            num_batches += 1
            
            pbar.set_postfix({
                'loss': f"{losses['total'].item():.4f}",
                'core': f"{(losses['total'].item() - smooth):.4f}"
            })
        
        # Return core loss so it's comparable to val_loss
        return core_loss / num_batches
    
    def _validate(self, dataloader: DataLoader) -> float:
        """Validate model"""
        self.model.eval()
        total_loss = 0.0
        num_batches = 0
        
        with torch.no_grad():
            for batch in dataloader:
                batch = {k: v.to(self.device) for k, v in batch.items()}
                
                predictions = self.model(
                    image=batch['image'],
                    trajectory=batch['trajectory'],
                    em_state=batch['em_state'],
                    motor_state=batch['motor_state'],
                    image_error=batch['image_error']
                )
                
                targets = {'action': batch['action']}
                losses = self.criterion(predictions, targets)
                
                total_loss += losses['total'].item()
                num_batches += 1
        
        return total_loss / num_batches
    
    def _dagger_collect(self, num_episodes: int = 10):
        """Collect data using DAgger"""
        print("Collecting DAgger data...")
        
        env = make_env(self.config)
        self.model.eval()
        
        for _ in range(num_episodes):
            obs, info = env.reset()
            trajectory_buffer = []
            
            for step in range(100):
                # Convert observation to tensors
                image = torch.tensor(obs['image'].transpose(2, 0, 1)[None] / 255.0, 
                                   dtype=torch.float32, device=self.device)
                trajectory = torch.tensor(obs['trajectory'][None], 
                                        dtype=torch.float32, device=self.device)
                em_state = torch.tensor(obs['em_state'][None], 
                                       dtype=torch.float32, device=self.device)
                motor_state = torch.tensor(obs['motor_state'][None], 
                                          dtype=torch.float32, device=self.device)
                image_error = torch.tensor(obs['image_error'][None], 
                                          dtype=torch.float32, device=self.device)
                
                # Get policy action
                with torch.no_grad():
                    predictions = self.model(image, trajectory, em_state, motor_state, image_error)
                    policy_action = predictions['action'].cpu().numpy()[0]
                
                # Get expert action (ground truth from simulator)
                jacobian_gt = info['jacobian_gt']
                try:
                    J_inv = np.linalg.pinv(jacobian_gt)
                    expert_action = -0.3 * J_inv @ obs['image_error']
                    expert_action = np.clip(expert_action, -30, 30)
                except:
                    expert_action = policy_action
                
                # Add to DAgger buffer
                sample = {
                    'image': obs['image'].transpose(2, 0, 1).astype(np.float32) / 255.0,
                    'trajectory': obs['trajectory'].astype(np.float32),
                    'em_state': obs['em_state'].astype(np.float32),
                    'motor_state': obs['motor_state'].astype(np.float32),
                    'image_error': obs['image_error'].astype(np.float32),
                    'action': expert_action.astype(np.float32)
                }
                self.dagger_buffer.add(sample)
                
                # Execute policy action (with some probability of expert action)
                if np.random.random() < 0.3:  # 30% expert
                    action = expert_action
                else:
                    action = policy_action
                
                obs, _, term, trunc, info = env.step(action)
                
                if term or trunc:
                    break
        
        print(f"DAgger buffer size: {len(self.dagger_buffer)}")
    
    def save_checkpoint(self, filename: str):
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'config': self.config,
            'train_history': self.train_history
        }
        torch.save(checkpoint, self.checkpoint_dir / filename)
        print(f"Saved checkpoint: {self.checkpoint_dir / filename}")
    
    def load_checkpoint(self, filepath: str):
        checkpoint = torch.load(filepath, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        if 'scheduler_state_dict' in checkpoint:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        print(f"Loaded checkpoint: {filepath}")


def generate_bc_toy_dataset(config: Config, output_dir: str = "./data/bc_toy"):
    """Generate toy dataset for BC stage"""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    env = make_env(config)
    
    episodes = []
    
    print("Generating BC toy dataset with expert demonstrations...")
    
    for ep_idx in range(50):  # 50 episodes
        obs, info = env.reset()
        episode_data = []
        
        for step in range(200):
            # Expert policy (smoother, more realistic)
            error = obs['image_error']
            jacobian_gt = info['jacobian_gt']
            
            # Add realistic delays and smoothing
            try:
                J_inv = np.linalg.pinv(jacobian_gt)
                action = -0.25 * J_inv @ error  # Slightly slower response
                # Add small human-like noise
                action += np.random.randn(2) * 2
                action = np.clip(action, -40, 40)
            except:
                action = np.zeros(2)
            
            frame_data = {
                'time_sec': step * config.simulator.dt,
                'image': obs['image'].tolist(),
                'em_state': obs['em_state'].tolist(),
                'motor_state': obs['motor_state'].tolist(),
                'trajectory': obs['trajectory'].tolist(),
                'image_error': obs['image_error'].tolist(),
                'action': action.tolist(),
            }
            episode_data.append(frame_data)
            
            obs, reward, term, trunc, info = env.step(action)
            
            if term or trunc:
                break
        
        episodes.append(episode_data)
        
        if (ep_idx + 1) % 10 == 0:
            print(f"Episode {ep_idx + 1}/50 collected")
    
    dataset = {
        'metadata': {
            'stage': 'bc',
            'num_episodes': len(episodes),
            'created': datetime.now().isoformat(),
        },
        'episodes': episodes
    }
    
    with open(output_dir / 'bc_toy_dataset.json', 'w') as f:
        json.dump(dataset, f)
    
    print(f"Saved BC toy dataset to {output_dir}")
    
    return output_dir


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Stage 2: Behavior Cloning Training")
    parser.add_argument('--data', type=str, default='./data/bc_toy/bc_toy_dataset.json', help='Path to expert data')
    parser.add_argument('--pinn_checkpoint', type=str, default='./checkpoints/pinn/best.pt', help='PINN checkpoint')
    parser.add_argument('--epochs', type=int, default=100, help='Number of epochs')
    parser.add_argument('--generate-data', action='store_true', help='Generate toy dataset only')
    parser.add_argument('--use-dagger', action='store_true', help='Use DAgger during training')
    args = parser.parse_args()
    
    config = get_config()
    
    if args.generate_data:
        generate_bc_toy_dataset(config)
    else:
        trainer = BCTrainer(config, pinn_checkpoint=args.pinn_checkpoint)
        trainer.train(args.data, num_epochs=args.epochs, use_dagger=args.use_dagger)

"""
python /home/ET/jinchenhan/kit/low_level_controller/training/train_bc.py \
--data /home/ET/jinchenhan/kit/low_level_controller/ds/testing/batch_extracted_packages/pkg_data_20251211-182653/sampled_dataset.csv \
--pinn_checkpoint /home/ET/jinchenhan/kit/low_level_controller/checkpoints/pinn/best.pt \
--epochs 1 

"""
