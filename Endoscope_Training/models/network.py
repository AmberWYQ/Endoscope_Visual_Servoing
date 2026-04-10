"""
Neural Network Architecture for Endoscope Visual Servoing
Unified architecture for PINN/IL/RL with modular Jacobian learning

Key Components:
1. ImageEncoder: Extract visual features from endoscopic images
2. TrajectoryEncoder: Learn temporal dynamics from target trajectory
3. StateEncoder: Process EM tracker + motor states
4. JacobianEstimator: Learn local image-to-actuation Jacobian
5. PolicyNetwork: Output optimal motor commands

Input Modalities:
- Endoscopic images (640x480x3)
- Target center trajectory (history of pixel coordinates)
- Target bounding box size (temporal, for depth estimation)
- EM tracker 6-DoF (position + quaternion)
- Motor positions and velocities
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from typing import Dict, Tuple, Optional
import math


class ImageEncoder(nn.Module):
    """
    Encode endoscopic images to feature vectors
    Uses pretrained backbone with attention mechanism
    """
    def __init__(self, feature_dim: int = 256, backbone: str = "resnet18", pretrained: bool = True):
        super().__init__()
        self.feature_dim = feature_dim
        
        # Load backbone
        if backbone == "resnet18":
            base = models.resnet18(weights=models.ResNet18_Weights.DEFAULT if pretrained else None)
            backbone_out_dim = 512
            self.backbone = nn.Sequential(*list(base.children())[:-2])  # Remove avgpool and fc
        elif backbone == "efficientnet_b0":
            base = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT if pretrained else None)
            backbone_out_dim = 1280
            self.backbone = base.features
        else:
            raise ValueError(f"Unknown backbone: {backbone}")
        
        # Spatial attention
        self.attention = nn.Sequential(
            nn.Conv2d(backbone_out_dim, 64, 1),
            nn.ReLU(),
            nn.Conv2d(64, 1, 1),
            nn.Sigmoid()
        )
        
        # Global pooling and projection
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.projection = nn.Sequential(
            nn.Linear(backbone_out_dim, feature_dim),
            nn.LayerNorm(feature_dim),
            nn.ReLU()
        )
        
        # Target-aware attention (learns to focus on target region)
        self.target_attention = nn.Sequential(
            nn.Conv2d(backbone_out_dim, 64, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 1, 1),
            nn.Sigmoid()
        )
        
    def forward(self, x: torch.Tensor, target_hint: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: Images [B, 3, H, W]
            target_hint: Optional target location hint [B, 2] (normalized coordinates)
        Returns:
            features: [B, feature_dim]
            attention_map: [B, 1, H', W'] for visualization
        """
        # Extract features
        features = self.backbone(x)  # [B, C, H', W']
        
        # Compute attention
        attention = self.attention(features)
        
        # Apply attention
        attended_features = features * attention
        
        # Global pool and project
        pooled = self.global_pool(attended_features).flatten(1)
        output = self.projection(pooled)
        
        return output, attention


class TrajectoryEncoder(nn.Module):
    """
    Encode temporal trajectory of target in image space
    Learns velocity, acceleration, and predicts future motion
    """
    def __init__(self, 
                 input_dim: int = 4,  # (center_x, center_y, bbox_w, bbox_h)
                 hidden_dim: int = 128,
                 output_dim: int = 64,
                 num_layers: int = 2,
                 history_len: int = 10):
        super().__init__()
        self.history_len = history_len
        
        # Input projection (normalize and embed)
        self.input_proj = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU()
        )
        
        # Bidirectional LSTM for temporal encoding
        self.lstm = nn.LSTM(
            hidden_dim, hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=0.1 if num_layers > 1 else 0
        )
        
        # Temporal attention
        self.temporal_attention = nn.MultiheadAttention(
            hidden_dim * 2, num_heads=4, batch_first=True
        )
        
        # Output projection
        self.output_proj = nn.Sequential(
            nn.Linear(hidden_dim * 2, output_dim),
            nn.LayerNorm(output_dim),
            nn.ReLU()
        )
        
        # Velocity/acceleration estimator
        self.dynamics_head = nn.Sequential(
            nn.Linear(hidden_dim * 2, 64),
            nn.ReLU(),
            nn.Linear(64, 4)  # (vel_x, vel_y, acc_x, acc_y)
        )
        
    def forward(self, trajectory: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            trajectory: [B, T, 4] - history of (center_x, center_y, bbox_w, bbox_h)
        Returns:
            features: [B, output_dim]
            dynamics: [B, 4] - estimated (vel_x, vel_y, acc_x, acc_y)
        """
        B, T, _ = trajectory.shape
        
        # Project input
        x = self.input_proj(trajectory)  # [B, T, hidden_dim]
        
        # LSTM encoding
        lstm_out, _ = self.lstm(x)  # [B, T, hidden_dim * 2]
        
        # Self-attention over time
        attended, _ = self.temporal_attention(lstm_out, lstm_out, lstm_out)
        
        # Take the last timestep's features
        last_features = attended[:, -1, :]  # [B, hidden_dim * 2]
        
        # Output projection
        features = self.output_proj(last_features)
        
        # Estimate dynamics
        dynamics = self.dynamics_head(last_features)
        
        return features, dynamics


class StateEncoder(nn.Module):
    """
    Encode robot state from EM tracker and motor readings
    """
    def __init__(self, 
                 em_dim: int = 10,  # rel_xyz(3) + abs_xyz(3) + quat(4)
                 motor_dim: int = 4,  # m1_pos, m2_pos, m1_spd, m2_spd
                 output_dim: int = 64):
        super().__init__()
        
        # Position encoder
        self.position_encoder = nn.Sequential(
            nn.Linear(6, 32),  # rel + abs position
            nn.LayerNorm(32),
            nn.ReLU()
        )
        
        # Orientation encoder (quaternion)
        self.orientation_encoder = nn.Sequential(
            nn.Linear(4, 32),
            nn.LayerNorm(32),
            nn.ReLU()
        )
        
        # Motor state encoder
        self.motor_encoder = nn.Sequential(
            nn.Linear(motor_dim, 32),
            nn.LayerNorm(32),
            nn.ReLU()
        )
        
        # Fusion
        self.fusion = nn.Sequential(
            nn.Linear(32 + 32 + 32, output_dim),
            nn.LayerNorm(output_dim),
            nn.ReLU()
        )
        
    def forward(self, em_state: torch.Tensor, motor_state: torch.Tensor) -> torch.Tensor:
        """
        Args:
            em_state: [B, 10] - EM tracker reading
            motor_state: [B, 4] - motor positions and velocities
        Returns:
            features: [B, output_dim]
        """
        # Split EM state
        position = em_state[:, :6]  # rel + abs xyz
        orientation = em_state[:, 6:10]  # quaternion
        
        # Encode components
        pos_feat = self.position_encoder(position)
        ori_feat = self.orientation_encoder(orientation)
        motor_feat = self.motor_encoder(motor_state)
        
        # Fuse
        combined = torch.cat([pos_feat, ori_feat, motor_feat], dim=-1)
        return self.fusion(combined)


class JacobianEstimator(nn.Module):
    """
    Estimate local Jacobian matrix from image space to actuation space
    J: d(image_error) / d(motor_action)
    
    This learns the relationship:
    - A: Image space -> Task space (target tracking)
    - B: Task space -> Actuation space (motor commands)
    Combined: J = B @ A (composition of mappings)
    """
    def __init__(self,
                 image_feat_dim: int = 256,
                 state_feat_dim: int = 64,
                 hidden_dims: list = [128, 64],
                 image_error_dim: int = 2,  # (error_x, error_y) in image
                 action_dim: int = 2):  # (m1_spd, m2_spd)
        super().__init__()
        
        self.image_error_dim = image_error_dim
        self.action_dim = action_dim
        
        # Compute Jacobian dimensions
        self.jacobian_size = image_error_dim * action_dim  # 2x2 = 4
        
        # Network to predict Jacobian elements
        layers = []
        in_dim = image_feat_dim + state_feat_dim
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(in_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.1)
            ])
            in_dim = hidden_dim
        
        self.feature_extractor = nn.Sequential(*layers)
        
        # Output heads
        # Main Jacobian prediction
        self.jacobian_head = nn.Linear(hidden_dims[-1], self.jacobian_size)
        
        # Jacobian condition number regularizer (for numerical stability)
        self.condition_head = nn.Linear(hidden_dims[-1], 1)
        
    def forward(self, image_features: torch.Tensor, state_features: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            image_features: [B, image_feat_dim]
            state_features: [B, state_feat_dim]
        Returns:
            jacobian: [B, image_error_dim, action_dim] - Local Jacobian matrix
            condition_estimate: [B, 1] - Estimated condition number
        """
        # Combine features
        combined = torch.cat([image_features, state_features], dim=-1)
        
        # Extract features
        features = self.feature_extractor(combined)
        
        # Predict Jacobian
        jacobian_flat = self.jacobian_head(features)
        jacobian = jacobian_flat.view(-1, self.image_error_dim, self.action_dim)
        
        # Estimate condition number
        condition = F.softplus(self.condition_head(features))
        
        return jacobian, condition
    
    def compute_action_from_jacobian(self, jacobian: torch.Tensor, image_error: torch.Tensor, 
                                      gain: float = 1.0) -> torch.Tensor:
        """
        Compute action using Jacobian pseudo-inverse
        action = -gain * J^+ @ error
        
        Args:
            jacobian: [B, 2, 2]
            image_error: [B, 2] - (error_x, error_y)
            gain: Control gain
        Returns:
            action: [B, 2]
        """
        # Compute pseudo-inverse: J^+ = J^T @ (J @ J^T)^-1
        # For 2x2, use direct inverse when possible
        try:
            jacobian_pinv = torch.linalg.pinv(jacobian)
            action = -gain * torch.bmm(jacobian_pinv, image_error.unsqueeze(-1)).squeeze(-1)
        except:
            # Fallback to damped pseudo-inverse
            damping = 0.01
            JJT = torch.bmm(jacobian, jacobian.transpose(1, 2))
            JJT_damped = JJT + damping * torch.eye(2, device=jacobian.device).unsqueeze(0)
            JJT_inv = torch.linalg.inv(JJT_damped)
            jacobian_pinv = torch.bmm(jacobian.transpose(1, 2), JJT_inv)
            action = -gain * torch.bmm(jacobian_pinv, image_error.unsqueeze(-1)).squeeze(-1)
        
        return action


class PolicyNetwork(nn.Module):
    """
    Policy network that combines all modalities to output motor commands
    Can use Jacobian-based control or learned end-to-end policy
    """
    def __init__(self,
                 image_feat_dim: int = 256,
                 trajectory_feat_dim: int = 64,
                 state_feat_dim: int = 64,
                 hidden_dims: list = [256, 128, 64],
                 action_dim: int = 2,
                 use_jacobian: bool = True):
        super().__init__()
        
        self.use_jacobian = use_jacobian
        self.action_dim = action_dim
        
        # Feature fusion
        total_feat_dim = image_feat_dim + trajectory_feat_dim + state_feat_dim
        
        # Add dynamics (vel, acc) and error features
        total_feat_dim += 4 + 2  # dynamics (4) + image_error (2)
        
        # Policy MLP
        layers = []
        in_dim = total_feat_dim
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(in_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.1)
            ])
            in_dim = hidden_dim
        
        self.policy_net = nn.Sequential(*layers)
        
        # Output head (mean and log_std for stochastic policy)
        self.mean_head = nn.Linear(hidden_dims[-1], action_dim)
        self.log_std_head = nn.Linear(hidden_dims[-1], action_dim)
        
        # Initialize output layer with small weights
        nn.init.xavier_uniform_(self.mean_head.weight, gain=0.01)
        nn.init.zeros_(self.mean_head.bias)
        
    def forward(self, 
                image_features: torch.Tensor,
                trajectory_features: torch.Tensor,
                state_features: torch.Tensor,
                dynamics: torch.Tensor,
                image_error: torch.Tensor,
                jacobian_action: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            image_features: [B, image_feat_dim]
            trajectory_features: [B, trajectory_feat_dim]  
            state_features: [B, state_feat_dim]
            dynamics: [B, 4] - (vel_x, vel_y, acc_x, acc_y)
            image_error: [B, 2] - current tracking error
            jacobian_action: Optional[B, 2] - action from Jacobian controller
        Returns:
            action_mean: [B, action_dim]
            action_log_std: [B, action_dim]
        """
        # Concatenate all features
        features = torch.cat([
            image_features,
            trajectory_features,
            state_features,
            dynamics,
            image_error
        ], dim=-1)
        
        # Policy network
        policy_features = self.policy_net(features)
        
        # Output
        action_mean = self.mean_head(policy_features)
        action_log_std = self.log_std_head(policy_features)
        action_log_std = torch.clamp(action_log_std, -5, 2)
        
        # If using Jacobian, blend with Jacobian-based action
        if self.use_jacobian and jacobian_action is not None:
            # Learn a residual on top of Jacobian action
            action_mean = jacobian_action + action_mean
        
        return action_mean, action_log_std
    
    def sample_action(self, action_mean: torch.Tensor, action_log_std: torch.Tensor, 
                      deterministic: bool = False) -> torch.Tensor:
        """Sample action from Gaussian policy"""
        if deterministic:
            return action_mean
        else:
            std = torch.exp(action_log_std)
            noise = torch.randn_like(action_mean)
            return action_mean + std * noise


class EndoscopeTrackingNetwork(nn.Module):
    """
    Complete network for endoscope visual servoing
    Unified architecture for PINN/IL/RL stages
    """
    def __init__(self, config):
        super().__init__()
        self.config = config
        nc = config.network
        
        # Encoders
        self.image_encoder = ImageEncoder(
            feature_dim=nc.image_feature_dim,
            backbone=nc.image_backbone
        )
        
        self.trajectory_encoder = TrajectoryEncoder(
            input_dim=4,  # bbox + target_center + emtracker
            hidden_dim=nc.lstm_hidden_dim,
            output_dim=nc.trajectory_feature_dim,
            num_layers=nc.num_lstm_layers,
            history_len=nc.trajectory_history
        )
        
        self.state_encoder = StateEncoder(
            em_dim=config.em_tracker.state_dim,
            motor_dim=4,
            output_dim=nc.state_feature_dim
        )
        
        # Jacobian estimator
        self.jacobian_estimator = JacobianEstimator(
            image_feat_dim=nc.image_feature_dim,
            state_feat_dim=nc.state_feature_dim,
            hidden_dims=nc.jacobian_hidden_dims,
            image_error_dim=2,
            action_dim=nc.action_dim
        )
        
        # Policy network
        self.policy = PolicyNetwork(
            image_feat_dim=nc.image_feature_dim,
            trajectory_feat_dim=nc.trajectory_feature_dim,
            state_feat_dim=nc.state_feature_dim,
            hidden_dims=nc.policy_hidden_dims,
            action_dim=nc.action_dim,
            use_jacobian=True
        )
        
    def forward(self, 
                image: torch.Tensor,
                trajectory: torch.Tensor,
                em_state: torch.Tensor,
                motor_state: torch.Tensor,
                image_error: torch.Tensor,
                deterministic: bool = False) -> Dict[str, torch.Tensor]:
        """
        Forward pass through complete network
        
        Args:
            image: [B, 3, H, W] - current endoscopic image
            trajectory: [B, T, 4] - trajectory history (center_x, center_y, bbox_w, bbox_h)
            em_state: [B, 10] - EM tracker state
            motor_state: [B, 4] - motor positions and velocities
            image_error: [B, 2] - current tracking error (target - center)
            deterministic: Whether to sample deterministically
            
        Returns:
            Dictionary with:
                - action: [B, 2] - motor speed commands
                - jacobian: [B, 2, 2] - estimated Jacobian
                - dynamics: [B, 4] - estimated target dynamics
                - attention: [B, 1, H', W'] - attention map
        """
        # Encode image
        image_features, attention = self.image_encoder(image)
        
        # Encode trajectory
        trajectory_features, dynamics = self.trajectory_encoder(trajectory)
        
        # Encode state
        state_features = self.state_encoder(em_state, motor_state)
        
        # Estimate Jacobian
        jacobian, condition = self.jacobian_estimator(image_features, state_features)
        
        # Compute Jacobian-based action
        jacobian_action = self.jacobian_estimator.compute_action_from_jacobian(
            jacobian, image_error, gain=0.5
        )
        
        # Policy output
        action_mean, action_log_std = self.policy(
            image_features,
            trajectory_features,
            state_features,
            dynamics,
            image_error,
            jacobian_action
        )
        
        # Sample action
        action = self.policy.sample_action(action_mean, action_log_std, deterministic)
        
        return {
            'action': action,
            'action_mean': action_mean,
            'action_log_std': action_log_std,
            'jacobian': jacobian,
            'jacobian_condition': condition,
            'jacobian_action': jacobian_action,
            'dynamics': dynamics,
            'attention': attention,
            'image_features': image_features,
            'trajectory_features': trajectory_features,
            'state_features': state_features
        }
    
    def get_action(self, obs: Dict, deterministic: bool = False) -> torch.Tensor:
        """
        Convenience method to get action from observation dictionary
        """
        output = self.forward(
            image=obs['image'],
            trajectory=obs['trajectory'],
            em_state=obs['em_state'],
            motor_state=obs['motor_state'],
            image_error=obs['image_error'],
            deterministic=deterministic
        )
        return output['action']


class ValueNetwork(nn.Module):
    """
    Value network for RL (critic)
    Estimates Q(s, a) or V(s)
    """
    def __init__(self, config, num_critics: int = 2):
        super().__init__()
        self.num_critics = num_critics
        nc = config.network
        
        # Shared feature dimensions
        state_dim = nc.image_feature_dim + nc.trajectory_feature_dim + nc.state_feature_dim + 4 + 2
        action_dim = nc.action_dim
        
        # Create multiple Q-networks for twin Q-learning
        self.q_networks = nn.ModuleList([
            nn.Sequential(
                nn.Linear(state_dim + action_dim, 256),
                nn.LayerNorm(256),
                nn.ReLU(),
                nn.Linear(256, 256),
                nn.ReLU(),
                nn.Linear(256, 1)
            ) for _ in range(num_critics)
        ])
        
    def forward(self, 
                image_features: torch.Tensor,
                trajectory_features: torch.Tensor,
                state_features: torch.Tensor,
                dynamics: torch.Tensor,
                image_error: torch.Tensor,
                action: torch.Tensor) -> torch.Tensor:
        """
        Returns Q-values from all critics
        """
        state = torch.cat([
            image_features, trajectory_features, state_features, dynamics, image_error
        ], dim=-1)
        
        state_action = torch.cat([state, action], dim=-1)
        
        q_values = torch.stack([q(state_action) for q in self.q_networks], dim=0)
        return q_values  # [num_critics, B, 1]


if __name__ == "__main__":
    # Test the network
    from config import get_config
    
    config = get_config()
    model = EndoscopeTrackingNetwork(config)
    
    # Create dummy inputs
    B = 4
    T = 10
    H, W = 480, 640
    
    obs = {
        'image': torch.randn(B, 3, H, W),
        'trajectory': torch.randn(B, T, 14),
        'em_state': torch.randn(B, 10),
        'motor_state': torch.randn(B, 4),
        'image_error': torch.randn(B, 2)
    }
    
    output = model.forward(**obs)
    
    print("Output shapes:")
    for k, v in output.items():
        if isinstance(v, torch.Tensor):
            print(f"  {k}: {v.shape}")
    
    print(f"\nTotal parameters: {sum(p.numel() for p in model.parameters()):,}")
