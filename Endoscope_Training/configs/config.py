"""
Configuration for Endoscope Visual Servoing System
Three-stage training: PINN -> Behavior Cloning -> Reinforcement Learning
"""

from dataclasses import dataclass, field
from typing import List, Tuple, Optional
import torch

 
@dataclass
class SerialConfig:
    """Serial parameters"""
    baud_rate: int = 115200


@dataclass
class CameraConfig:
    """Camera parameters"""
    width: int = 400 # 640
    height: int = 400 # 480
    fps: int = 20
    fov: float = 70.0  # Field of view in degrees
    
@dataclass
class RobotConfig:
    """Continuum robot endoscope parameters"""
    num_motors: int = 2
    max_motor_speed: float = 100.0  # Max speed units
    min_motor_speed: float = -100.0
    
    # Physical parameters for PINN
    segment_length: float = 30.0  # mm, bendable segment
    backbone_length: float = 200.0  # mm, total length
    outer_diameter: float = 5.0  # mm
    
    # Bending limits (radians)
    max_bending_angle: float = 1.57  # ~90 degrees
    
    # Motor to bending mapping (simplified)
    motor_to_curvature_gain: List[float] = field(default_factory=lambda: [0.01, 0.01])
    
@dataclass
class EMTrackerConfig:
    """EM tracker configuration"""
    state_dim: int = 10  # rel_xyz(3) + abs_xyz(3) + quaternion(4)
    position_range: Tuple[float, float] = (-0.5, 0.5)  # meters
    fps: int  = 20
    raw_buffer_seconds: float = 2.0
    
@dataclass 
class NetworkConfig:
    """Neural network architecture configuration"""
    # Image encoder
    image_feature_dim: int = 256
    image_backbone: str = "resnet18"  # or "efficientnet_b0"
    
    # Trajectory encoder (temporal)
    trajectory_history: int = 10  # Number of past frames
    trajectory_dim: int = 14
    trajectory_feature_dim: int = 64
    lstm_hidden_dim: int = 128
    num_lstm_layers: int = 2
    
    # State encoder
    state_feature_dim: int = 64
    
    # Jacobian estimator
    jacobian_hidden_dims: List[int] = field(default_factory=lambda: [128, 64])
    
    # Policy head
    policy_hidden_dims: List[int] = field(default_factory=lambda: [256, 128, 64])
    
    hidden_dim: int = 256

    # Input
    input_dim: int = 110  # 10*11

    # Output
    action_dim: int = 2  # m1_spd, m2_spd
    
    # Dropout
    dropout: float = 0.1
    
@dataclass
class PINNConfig:
    """Physics-Informed Neural Network configuration"""
    # Physics loss weights
    lambda_kinematics: float = 1.0
    lambda_jacobian: float = 0.5
    lambda_dynamics: float = 0.3
    lambda_boundary: float = 0.2
    
    # Training
    num_collocation_points: int = 1000
    num_boundary_points: int = 200
    
@dataclass
class TrainingConfig:
    """Training hyperparameters"""
    # Common
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    seed: int = 42
    
    # Stage 1: PINN
    pinn_epochs: int = 500
    pinn_lr: float = 1e-3
    pinn_batch_size: int = 64
    
    # Stage 2: Behavior Cloning
    bc_epochs: int = 200
    bc_lr: float = 1e-4
    bc_batch_size: int = 128
    bc_weight_decay: float = 1e-5
    
    # Stage 3: Reinforcement Learning
    rl_total_timesteps: int = 500000
    rl_lr: float = 3e-4
    rl_batch_size: int = 256
    rl_gamma: float = 0.99
    rl_tau: float = 0.005  # Soft update coefficient
    rl_buffer_size: int = 100000
    
    # Checkpointing
    save_every: int = 50
    eval_every: int = 10
    
@dataclass
class SimulatorConfig:
    """Simulator configuration for PINN training"""
    dt: float = 0.05  # 20 Hz
    max_episode_steps: int = 500
    
    # Target dynamics
    target_motion_type: str = "sinusoidal"  # "random", "circular", "sinusoidal"
    target_speed_range: Tuple[float, float] = (1.0, 5.0)  # pixels/frame
    
    # Noise
    em_noise_std: float = 0.001
    motor_noise_std: float = 0.5
    target_detection_noise_std: float = 2.0  # pixels
    
    # Reward shaping
    reward_center_weight: float = 1.0
    reward_smooth_weight: float = 0.1
    reward_velocity_weight: float = 0.05
    
@dataclass
class Config:
    """Master configuration"""
    camera: CameraConfig = field(default_factory=CameraConfig)
    robot: RobotConfig = field(default_factory=RobotConfig)
    em_tracker: EMTrackerConfig = field(default_factory=EMTrackerConfig)
    network: NetworkConfig = field(default_factory=NetworkConfig)
    pinn: PINNConfig = field(default_factory=PINNConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    simulator: SimulatorConfig = field(default_factory=SimulatorConfig)
    serial_setting: SerialConfig = field(default_factory=SerialConfig)
    
    # Paths
    data_dir: str = "./data"
    checkpoint_dir: str = "./checkpoints"
    log_dir: str = "./logs"
    
def get_config() -> Config:
    """Get default configuration"""
    return Config()
