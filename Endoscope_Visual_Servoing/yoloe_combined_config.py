"""
Combined Configuration for YOLO-E Visual Servoing Pipeline

Unifies:
- YOLO-E perception model configuration (instruction-based detection)
- Low-level control model configuration
- Hardware interface settings
- Safety parameters
- Visualization settings
"""

import os
from dataclasses import dataclass, field
from typing import List, Tuple, Optional
import torch


@dataclass
class CameraConfig:
    """Camera and image parameters"""
    width: int = 640
    height: int = 480
    fps: float = 20.0
    camera_id: int = 0
    
    @property
    def center_x(self) -> int:
        return self.width // 2
    
    @property
    def center_y(self) -> int:
        return self.height // 2


@dataclass
class YOLOEPerceptionConfig:
    """YOLO-E based detection model configuration"""
    model_path: str = 'yolov8x-worldv2.pt'
    default_target: str = 'polyp'
    confidence_threshold: float = 0.25
    iou_threshold: float = 0.45
    max_lost_frames: int = 10
    detection_rate_hz: float = 20.0


@dataclass
class LowLevelControlConfig:
    """Low-level neural network control configuration"""
    image_feature_dim: int = 256
    trajectory_feature_dim: int = 64
    state_feature_dim: int = 64
    lstm_hidden_dim: int = 128
    num_lstm_layers: int = 2
    policy_hidden_dims: List[int] = field(default_factory=lambda: [256, 128, 64])
    jacobian_hidden_dims: List[int] = field(default_factory=lambda: [128, 64])
    trajectory_history: int = 10
    em_state_dim: int = 10
    motor_state_dim: int = 4
    action_dim: int = 2
    use_jacobian: bool = True
    jacobian_blend_ratio: float = 0.3
    checkpoint_path: str = './checkpoints/low_level_model.pth'
    control_rate_hz: float = 20.0


@dataclass
class RobotConfig:
    """Robot and motor parameters"""
    num_motors: int = 2
    max_motor_speed: float = 100.0
    min_motor_speed: float = -100.0
    baud_rate: int = 115200
    motor_port: str = '/dev/ttyUSB0'
    motor_cmd_min: int = -10000
    motor_cmd_max: int = 10000


@dataclass
class SafetyConfig:
    """Safety and fallback behavior configuration"""
    no_detection_action: str = 'hold'
    low_confidence_threshold: float = 0.5
    low_confidence_gain_reduction: float = 0.5
    enable_smoothing: bool = True
    smoothing_alpha: float = 0.3
    max_velocity_change: float = 20.0
    pixel_deadzone: int = 10
    emergency_stop_on_error: bool = True
    max_consecutive_errors: int = 50
    search_pattern_enabled: bool = True
    search_velocity: float = 10.0
    search_period: float = 2.0


@dataclass
class EMTrackerConfig:
    """EM tracker configuration (optional)"""
    enabled: bool = False
    port: str = '/dev/ttyUSB1'
    fps: int = 20
    raw_buffer_seconds: float = 2.0
    state_dim: int = 10


@dataclass
class VisualizationConfig:
    """Visualization and UI settings"""
    enabled: bool = True
    ui_width: int = 1000
    ui_height: int = 600
    video_x: int = 340
    video_y: int = 80
    color_bbox_tracking: Tuple[int, int, int] = (0, 200, 255)
    color_bbox_lost: Tuple[int, int, int] = (255, 255, 0)
    color_bbox_no_detect: Tuple[int, int, int] = (255, 0, 0)
    color_error_arrow: Tuple[int, int, int] = (255, 0, 255)
    color_command_arrow: Tuple[int, int, int] = (0, 255, 0)
    color_center_cross: Tuple[int, int, int] = (0, 255, 0)
    arrow_thickness: int = 2
    arrow_tip_length: float = 0.3
    error_arrow_scale: float = 1.0
    command_arrow_scale: float = 50.0


@dataclass
class LoggingConfig:
    """Data logging configuration"""
    enabled: bool = True
    log_dir: str = './control_logs'
    save_video: bool = True
    save_csv: bool = True
    save_detections: bool = True
    save_actions: bool = True
    video_fps: float = 20.0
    video_codec: str = 'mp4v'


@dataclass
class IntegratedConfig:
    """Master configuration for YOLO-E visual servoing pipeline"""
    camera: CameraConfig = field(default_factory=CameraConfig)
    perception: YOLOEPerceptionConfig = field(default_factory=YOLOEPerceptionConfig)
    low_level: LowLevelControlConfig = field(default_factory=LowLevelControlConfig)
    robot: RobotConfig = field(default_factory=RobotConfig)
    safety: SafetyConfig = field(default_factory=SafetyConfig)
    em_tracker: EMTrackerConfig = field(default_factory=EMTrackerConfig)
    visualization: VisualizationConfig = field(default_factory=VisualizationConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
    simulation_mode: bool = True
    use_mock_perception: bool = False
    use_mock_control: bool = False
    
    def get_device(self) -> torch.device:
        return torch.device(self.device)


def get_config() -> IntegratedConfig:
    """Get default configuration"""
    return IntegratedConfig()


def get_config_from_args(args) -> IntegratedConfig:
    """Build configuration from command line arguments"""
    config = IntegratedConfig()
    
    if hasattr(args, 'mode'):
        config.simulation_mode = (args.mode == 'simulation')
    if hasattr(args, 'yoloe_model') and args.yoloe_model:
        config.perception.model_path = args.yoloe_model
    if hasattr(args, 'target') and args.target:
        config.perception.default_target = args.target
    if hasattr(args, 'control_checkpoint') and args.control_checkpoint:
        config.low_level.checkpoint_path = args.control_checkpoint
    if hasattr(args, 'confidence_threshold'):
        config.perception.confidence_threshold = args.confidence_threshold
    if hasattr(args, 'camera_id'):
        config.camera.camera_id = args.camera_id
    if hasattr(args, 'mock_perception'):
        config.use_mock_perception = args.mock_perception
    if hasattr(args, 'mock_control'):
        config.use_mock_control = args.mock_control
    if hasattr(args, 'no_display'):
        config.visualization.enabled = not args.no_display
    if hasattr(args, 'output_dir') and args.output_dir:
        config.logging.log_dir = args.output_dir
    
    return config


if __name__ == '__main__':
    config = get_config()
    print("YOLO-E Visual Servoing Configuration")
    print("=" * 50)
    print(f"Device: {config.device}")
    print(f"Camera: {config.camera.width}x{config.camera.height} @ {config.camera.fps}Hz")
    print(f"YOLO-E model: {config.perception.model_path}")
    print(f"Default target: {config.perception.default_target}")
    print(f"Confidence threshold: {config.perception.confidence_threshold}")
