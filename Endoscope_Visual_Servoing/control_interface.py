"""
Control Interface for Visual Servoing Pipeline

Provides a standardized interface for the low-level neural network control model.
Handles:
- Model loading and inference
- Observation construction from detection results
- Trajectory buffer management
- Action output in standardized format
- Mock model for testing

Input: DetectionResult from perception + sensor states
Output: 2-DoF bending commands [m1_spd, m2_spd] in [-1, 1]
"""

import time
from dataclasses import dataclass, field
from typing import Optional, Dict, Any, Tuple, List
import numpy as np
import torch
import torch.nn as nn


@dataclass
class ControlAction:
    """Standardized control action output"""
    # Motor speeds [m1, m2] in normalized range [-1, 1]
    action: np.ndarray
    
    # Mean action (before noise for stochastic policies)
    action_mean: np.ndarray
    
    # Whether action is valid
    valid: bool
    
    # Jacobian matrix [2, 2] if available
    jacobian: Optional[np.ndarray] = None
    
    # Jacobian-based action component
    jacobian_action: Optional[np.ndarray] = None
    
    # Attention map if available
    attention: Optional[np.ndarray] = None
    
    # Target dynamics (vel_x, vel_y, acc_x, acc_y)
    dynamics: Optional[np.ndarray] = None
    
    # Inference time
    inference_time: float = 0.0
    
    def scale_to_motor(self, max_speed: float = 100.0) -> np.ndarray:
        """Scale normalized action to motor speed units"""
        return self.action * max_speed


class TrajectoryBuffer:
    """
    Circular buffer for target trajectory history.
    
    Stores (center_x, center_y, bbox_w, bbox_h) for each timestep.
    """
    
    def __init__(self, history_length: int = 10):
        self.history_length = history_length
        self.buffer = np.zeros((history_length, 4))  # [center_x, center_y, bbox_w, bbox_h]
        self.timestamps = np.zeros(history_length)
        self.write_idx = 0
        self.count = 0
    
    def add(self, center: np.ndarray, bbox_size: np.ndarray, timestamp: float):
        """
        Add new observation.
        
        Args:
            center: [cx, cy] target center
            bbox_size: [w, h] bounding box size
            timestamp: Time of observation
        """
        self.buffer[self.write_idx] = np.array([
            center[0], center[1], bbox_size[0], bbox_size[1]
        ])
        self.timestamps[self.write_idx] = timestamp
        self.write_idx = (self.write_idx + 1) % self.history_length
        self.count = min(self.count + 1, self.history_length)
    
    def add_from_detection(self, detection_result, timestamp: float):
        """Add from DetectionResult object"""
        if detection_result.bbox is not None:
            center = detection_result.center
            bbox_size = np.array([detection_result.bbox[2], detection_result.bbox[3]])
            self.add(center, bbox_size, timestamp)
        elif self.count > 0:
            # Repeat last observation if no detection
            last_idx = (self.write_idx - 1) % self.history_length
            self.buffer[self.write_idx] = self.buffer[last_idx]
            self.timestamps[self.write_idx] = timestamp
            self.write_idx = (self.write_idx + 1) % self.history_length
    
    def get_trajectory(self) -> np.ndarray:
        """Get trajectory in temporal order [history_length, 4]"""
        if self.count < self.history_length:
            # Pad with last observation or zeros
            traj = np.zeros((self.history_length, 4))
            if self.count > 0:
                for i in range(self.history_length):
                    idx = min(i, self.count - 1)
                    traj[i] = self.buffer[idx]
            return traj
        else:
            # Full buffer - reorder to temporal sequence
            indices = np.arange(self.write_idx, self.write_idx + self.history_length) % self.history_length
            return self.buffer[indices]
    
    def get_latest(self) -> np.ndarray:
        """Get the most recent observation"""
        if self.count == 0:
            return np.zeros(4)
        last_idx = (self.write_idx - 1) % self.history_length
        return self.buffer[last_idx]
    
    def clear(self):
        """Clear the buffer"""
        self.buffer[:] = 0
        self.timestamps[:] = 0
        self.write_idx = 0
        self.count = 0


class ControlInterface:
    """
    Standardized interface for the low-level control model.
    
    Takes detection results and sensor states, outputs motor commands.
    """
    
    def __init__(
        self,
        checkpoint_path: str,
        config,
        device: Optional[torch.device] = None,
    ):
        """
        Initialize the control interface.
        
        Args:
            checkpoint_path: Path to model checkpoint
            config: Configuration object (IntegratedConfig or LowLevelControlConfig)
            device: Torch device
        """
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Get config
        if hasattr(config, 'low_level'):
            self.config = config.low_level
            self.full_config = config
        else:
            self.config = config
            self.full_config = None
        
        # Image dimensions
        self.img_width = 640
        self.img_height = 480
        
        # Trajectory buffer
        self.trajectory_buffer = TrajectoryBuffer(self.config.trajectory_history)
        
        # State tracking
        self.motor_state = np.zeros(4)  # [m1_pos, m2_pos, m1_spd, m2_spd]
        self.em_state = np.zeros(10)  # EM tracker state
        
        # Build and load model
        print(f"[Control] Loading model from: {checkpoint_path}")
        self._build_model(checkpoint_path)
        
        # Statistics
        self.frame_count = 0
        self.total_time = 0.0
        
        print(f"[Control] Initialized on device: {self.device}")
    
    def _build_model(self, checkpoint_path: str):
        """Build and load the control model with robust config handling"""
        import os
        
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
        
        # First, try to load checkpoint to inspect its contents
        try:
            checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
        except TypeError:
            # Older PyTorch version
            checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        # ── IQL checkpoint unwrapping ─────────────────────────────────────────
        # IQL checkpoints are saved as:
        #   { "actor": <state_dict>, "critic": ..., "value": ... }
        # where every actor key has a "net." prefix (e.g. "net.image_encoder...").
        # We extract "actor" and strip the "net." prefix so the existing loading
        # strategies see bare keys matching EndoscopeTrackingNetwork directly.
        if isinstance(checkpoint, dict) and "actor" in checkpoint and "critic" in checkpoint:
            algo = "TD3-BC" if "actor_target" in checkpoint else "IQL"
            print(f"[Control] Detected {algo} checkpoint format — extracting and remapping 'actor' weights")
            raw_actor = checkpoint["actor"]
            stripped = {}
            for k, v in raw_actor.items():
                new_key = k[len("net."):] if k.startswith("net.") else k
                stripped[new_key] = v
            checkpoint = {"model_state_dict": stripped}
            print(f"[Control] Remapped {len(stripped)} actor keys (stripped 'net.' prefix)")

        # Try multiple loading strategies
        model_loaded = False

        # Strategy 1: Try loading with configs module available (original method)
        if not model_loaded:
            try:
                from low_level_network import EndoscopeTrackingNetwork
                from configs.config import get_config as get_ll_config
                
                # Get config for network architecture
                if self.full_config is not None:
                    ll_config = get_ll_config()
                    ll_config.network.trajectory_history = self.config.trajectory_history
                    ll_config.network.image_feature_dim = self.config.image_feature_dim
                    ll_config.network.trajectory_feature_dim = self.config.trajectory_feature_dim
                    ll_config.network.state_feature_dim = self.config.state_feature_dim
                else:
                    ll_config = get_ll_config()
                
                self.model = EndoscopeTrackingNetwork(ll_config)
                self._load_state_dict_from_checkpoint(checkpoint)
                self.model = self.model.to(self.device)
                self.model.eval()
                model_loaded = True
                print("[Control] Loaded model using configs module")
                
            except ImportError as e:
                print(f"[Control] configs module not available: {e}")
            except Exception as e:
                print(f"[Control] Strategy 1 failed: {e}")
        
        # Strategy 2: Build model from checkpoint's saved config (portable method)
        if not model_loaded:
            try:
                from low_level_network import EndoscopeTrackingNetwork
                
                # Extract config from checkpoint
                saved_config = checkpoint.get('config', {})
                network_config = checkpoint.get('network_config', {})
                
                # If config is a dict, create a minimal config object
                if isinstance(saved_config, dict) or network_config:
                    config_obj = self._create_config_from_dict(saved_config, network_config)
                    self.model = EndoscopeTrackingNetwork(config_obj)
                    self._load_state_dict_from_checkpoint(checkpoint)
                    self.model = self.model.to(self.device)
                    self.model.eval()
                    model_loaded = True
                    print("[Control] Loaded model from checkpoint's saved config (portable)")
                    
            except Exception as e:
                print(f"[Control] Strategy 2 failed: {e}")
        
        # Strategy 3: Build model with default config
        if not model_loaded:
            try:
                from low_level_network import EndoscopeTrackingNetwork
                
                # Create default config
                config_obj = self._create_default_config()
                self.model = EndoscopeTrackingNetwork(config_obj)
                self._load_state_dict_from_checkpoint(checkpoint)
                self.model = self.model.to(self.device)
                self.model.eval()
                model_loaded = True
                print("[Control] Loaded model with default config")
                
            except Exception as e:
                print(f"[Control] Strategy 3 failed: {e}")
        
        if not model_loaded:
            raise RuntimeError("Failed to load control model with any strategy")
    
    def _load_state_dict_from_checkpoint(self, checkpoint):
        """Load state dict from checkpoint with flexible key handling"""
        if isinstance(checkpoint, dict):
            if 'model_state_dict' in checkpoint:
                self.model.load_state_dict(checkpoint['model_state_dict'])
            elif 'state_dict' in checkpoint:
                self.model.load_state_dict(checkpoint['state_dict'])
            else:
                # Assume the checkpoint is the state dict itself
                self.model.load_state_dict(checkpoint)
        else:
            # checkpoint is already a state dict
            self.model.load_state_dict(checkpoint)
        print("[Control] Model state dict loaded successfully")
    
    def _create_config_from_dict(self, saved_config: dict, network_config: dict = None):
        """Create a config object from saved dictionary"""
        
        class MinimalNetworkConfig:
            """Minimal network config for model instantiation"""
            def __init__(self, cfg_dict: dict, net_cfg: dict = None):
                # Default values
                self.image_feature_dim = 256
                self.image_backbone = "resnet18"
                self.trajectory_history = 10
                self.trajectory_feature_dim = 64
                self.lstm_hidden_dim = 128
                self.num_lstm_layers = 2
                self.state_feature_dim = 64
                self.jacobian_hidden_dims = [128, 64]
                self.policy_hidden_dims = [256, 128, 64]
                self.action_dim = 2
                self.dropout = 0.1
                
                # Override from network_config if provided
                if net_cfg:
                    for key, value in net_cfg.items():
                        if hasattr(self, key):
                            setattr(self, key, value)
                
                # Override from saved_config['network'] if it exists
                if cfg_dict and 'network' in cfg_dict:
                    net_dict = cfg_dict['network']
                    for key, value in net_dict.items():
                        if hasattr(self, key):
                            setattr(self, key, value)
        
        class MinimalCameraConfig:
            def __init__(self, cfg_dict: dict):
                self.width = 640
                self.height = 480
                self.fps = 20
                self.fov = 70.0
                
                if cfg_dict and 'camera' in cfg_dict:
                    for key, value in cfg_dict['camera'].items():
                        if hasattr(self, key):
                            setattr(self, key, value)
        
        class MinimalRobotConfig:
            def __init__(self, cfg_dict: dict):
                self.num_motors = 2
                self.max_motor_speed = 100.0
                self.min_motor_speed = -100.0
                
                if cfg_dict and 'robot' in cfg_dict:
                    for key, value in cfg_dict['robot'].items():
                        if hasattr(self, key):
                            setattr(self, key, value)
        
        class MinimalEMTrackerConfig:
            def __init__(self, cfg_dict: dict):
                self.state_dim = 10
                self.position_range = (-0.5, 0.5)
                self.fps = 20
                
                if cfg_dict and 'em_tracker' in cfg_dict:
                    for key, value in cfg_dict['em_tracker'].items():
                        if hasattr(self, key):
                            setattr(self, key, value)
        
        class MinimalTrainingConfig:
            def __init__(self, cfg_dict: dict):
                self.device = "cuda" if torch.cuda.is_available() else "cpu"
                self.seed = 42
                
                if cfg_dict and 'training' in cfg_dict:
                    for key, value in cfg_dict['training'].items():
                        if hasattr(self, key):
                            setattr(self, key, value)
        
        class MinimalConfig:
            """Minimal config that mimics the structure expected by EndoscopeTrackingNetwork"""
            def __init__(self, cfg_dict: dict, net_cfg: dict = None):
                self.network = MinimalNetworkConfig(cfg_dict, net_cfg)
                self.camera = MinimalCameraConfig(cfg_dict)
                self.robot = MinimalRobotConfig(cfg_dict)
                self.em_tracker = MinimalEMTrackerConfig(cfg_dict)
                self.training = MinimalTrainingConfig(cfg_dict)
        
        return MinimalConfig(saved_config, network_config)
    
    def _create_default_config(self):
        """Create a default config object"""
        return self._create_config_from_dict({}, {})
    
    def _preprocess_image(self, frame: np.ndarray) -> torch.Tensor:
        """Preprocess frame for network input"""
        import cv2
        
        # Ensure correct size
        if frame.shape[:2] != (self.img_height, self.img_width):
            frame = cv2.resize(frame, (self.img_width, self.img_height))
        
        # Convert BGR to RGB if needed (assume input is RGB)
        # Normalize
        img = frame.astype(np.float32) / 255.0
        img = (img - np.array([0.485, 0.456, 0.406])) / np.array([0.229, 0.224, 0.225])
        
        # To tensor [1, 3, H, W]
        tensor = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0).float()
        return tensor.to(self.device)
    
    def update_trajectory(self, detection_result, timestamp: float):
        """Update trajectory buffer with new detection"""
        self.trajectory_buffer.add_from_detection(detection_result, timestamp)
    
    def set_motor_state(self, positions: np.ndarray, speeds: np.ndarray):
        """Update motor state"""
        self.motor_state = np.concatenate([positions, speeds])
    
    def set_em_state(self, em_state: np.ndarray):
        """Update EM tracker state"""
        self.em_state = em_state
    
    @torch.no_grad()
    def compute_action(
        self,
        frame: np.ndarray,
        detection_result,
        timestamp: float,
        deterministic: bool = True,
    ) -> ControlAction:
        """
        Compute control action from current observation.
        
        Args:
            frame: Current camera frame (RGB)
            detection_result: DetectionResult from perception
            timestamp: Current timestamp
            deterministic: Whether to use deterministic action
        
        Returns:
            ControlAction with motor commands
        """
        start_time = time.time()
        
        # Check if we have a valid detection
        if detection_result.no_detection or detection_result.center is None:
            # Return zero action for no detection
            return ControlAction(
                action=np.zeros(2),
                action_mean=np.zeros(2),
                valid=False,
                inference_time=time.time() - start_time,
            )
        
        # Update trajectory buffer
        self.update_trajectory(detection_result, timestamp)
        
        # Compute image error
        image_error = detection_result.get_pixel_error(self.img_width, self.img_height)

        # Prepare network inputs
        image_tensor = self._preprocess_image(frame)
        
        trajectory = torch.from_numpy(
            self.trajectory_buffer.get_trajectory()
        ).unsqueeze(0).float().to(self.device)
        
        em_state = torch.from_numpy(
            self.em_state
        ).unsqueeze(0).float().to(self.device)
        
        motor_state = torch.from_numpy(
            self.motor_state
        ).unsqueeze(0).float().to(self.device)
        
        image_error_tensor = torch.from_numpy(
            image_error
        ).unsqueeze(0).float().to(self.device)
        
        # Forward pass
        outputs = self.model(
            image=image_tensor,
            trajectory=trajectory,
            em_state=em_state,
            motor_state=motor_state,
            image_error=image_error_tensor,
            deterministic=deterministic,
        )
        
        # Extract action
        action = outputs['action'].cpu().numpy()[0]
        action_mean = outputs['action_mean'].cpu().numpy()[0]        
        # Optionally blend with Jacobian-based action
        jacobian = None
        jacobian_action = None
        
        if self.config.use_jacobian and 'jacobian' in outputs:
            jacobian = outputs['jacobian'].cpu().numpy()[0]
            if 'jacobian_action' in outputs:
                jacobian_action = outputs['jacobian_action'].cpu().numpy()[0]
                # Blend: (1-ratio) * learned + ratio * jacobian
                blend_ratio = self.config.jacobian_blend_ratio
                action = (1 - blend_ratio) * action_mean + blend_ratio * jacobian_action
        
        # Normalize by vector magnitude — preserves direction and axis ratio,
        # only scales down when magnitude > 1, leaves small outputs untouched
        norm = np.linalg.norm(action)
        action = action / norm if norm > 1.0 else action
        # --- Experimental: rotate action vector to align with P-controller direction ---
        # Adjust ACTION_ROTATION_DEG to tune (positive = counter-clockwise)
        ACTION_ROTATION_DEG = 60.0
        if ACTION_ROTATION_DEG != 0.0:
            theta = np.radians(ACTION_ROTATION_DEG)
            c, s = np.cos(theta), np.sin(theta)
            R = np.array([[c, -s], [s, c]])
            action = R @ action 
        # print(f"[Control] Model command — action: {action}")

        
        # Extract optional outputs
        attention = None
        if 'attention' in outputs:
            attention = outputs['attention'].cpu().numpy()[0]
        
        dynamics = None
        if 'dynamics' in outputs:
            dynamics = outputs['dynamics'].cpu().numpy()[0]
        
        # Timing
        elapsed = time.time() - start_time
        self.frame_count += 1
        self.total_time += elapsed
        
        return ControlAction(
            action=action,
            action_mean=action_mean,
            valid=True,
            jacobian=jacobian,
            jacobian_action=jacobian_action,
            attention=attention,
            dynamics=dynamics,
            inference_time=elapsed,
        )
    
    def reset(self):
        """Reset control state"""
        self.trajectory_buffer.clear()
        self.motor_state = np.zeros(4)
        self.em_state = np.zeros(10)
        self.frame_count = 0
        self.total_time = 0.0
    
    def get_stats(self) -> Dict[str, Any]:
        """Get control statistics"""
        return {
            'frame_count': self.frame_count,
            'total_time': self.total_time,
            'avg_fps': self.frame_count / self.total_time if self.total_time > 0 else 0,
            'trajectory_count': self.trajectory_buffer.count,
        }


class MockControlInterface:
    """
    Mock control interface for testing without the neural network model.
    
    Implements a simple proportional controller based on pixel error.
    """
    
    def __init__(
        self,
        config=None,
        k_p: float = 0.005,
        **kwargs
    ):
        """
        Initialize mock control interface.
        
        Args:
            config: Optional configuration
            k_p: Proportional gain for mock controller
        """
        print("[MockControl] Initializing mock control interface")
        
        self.k_p = k_p
        self.img_width = 640
        self.img_height = 480
        
        # Trajectory buffer
        self.trajectory_buffer = TrajectoryBuffer(10)
        
        # State
        self.motor_state = np.zeros(4)
        self.em_state = np.zeros(10)
        
        # Statistics
        self.frame_count = 0
        self.total_time = 0.0
    
    def update_trajectory(self, detection_result, timestamp: float):
        """Update trajectory buffer"""
        self.trajectory_buffer.add_from_detection(detection_result, timestamp)
    
    def set_motor_state(self, positions: np.ndarray, speeds: np.ndarray):
        """Update motor state"""
        self.motor_state = np.concatenate([positions, speeds])
    
    def set_em_state(self, em_state: np.ndarray):
        """Update EM tracker state"""
        self.em_state = em_state
    
    def compute_action(
        self,
        frame: np.ndarray,
        detection_result,
        timestamp: float,
        deterministic: bool = True,
    ) -> ControlAction:
        """
        Compute control action using simple proportional control.
        
        Args:
            frame: Current camera frame (RGB) - ignored by mock
            detection_result: DetectionResult from perception
            timestamp: Current timestamp
            deterministic: Whether to use deterministic action
        
        Returns:
            ControlAction with motor commands
        """
        start_time = time.time()
        
        # Check for valid detection
        if detection_result.no_detection or detection_result.center is None:
            return ControlAction(
                action=np.zeros(2),
                action_mean=np.zeros(2),
                valid=False,
                inference_time=time.time() - start_time,
            )
        
        # Update trajectory
        self.update_trajectory(detection_result, timestamp)
        
        # Compute pixel error
        error = detection_result.get_pixel_error(self.img_width, self.img_height)
        
        # Simple proportional control
        # Map error to action: positive ex -> need to move left -> positive m1
        action = np.array([
            self.k_p * error[0],  # ex -> m1
            self.k_p * error[1],  # ey -> m2
        ])
        
        # Clip to valid range
        action = np.clip(action, -1.0, 1.0)
        
        # Add small noise for non-deterministic
        if not deterministic:
            action += np.random.randn(2) * 0.05
            action = np.clip(action, -1.0, 1.0)
        
        elapsed = time.time() - start_time
        self.frame_count += 1
        self.total_time += elapsed
        
        return ControlAction(
            action=action,
            action_mean=action,
            valid=True,
            inference_time=elapsed,
        )
    
    def reset(self):
        """Reset control state"""
        self.trajectory_buffer.clear()
        self.motor_state = np.zeros(4)
        self.em_state = np.zeros(10)
        self.frame_count = 0
        self.total_time = 0.0
    
    def get_stats(self) -> Dict[str, Any]:
        """Get control statistics"""
        return {
            'frame_count': self.frame_count,
            'total_time': self.total_time,
            'avg_fps': self.frame_count / self.total_time if self.total_time > 0 else 0,
            'trajectory_count': self.trajectory_buffer.count,
        }


class ProportionalController:
    """
    Position-based proportional controller for visual servoing.

    Generates 2-DoF motor commands directly from the target's pixel
    position in the image, without using a learned neural network.
    Useful as a baseline to compare against bc_model.

    Command law:
        m1 = kp_x * error_x   (horizontal centering)
        m2 = kp_y * error_y   (vertical   centering)

    where error = (target_center - image_center) / image_half_size,
    giving a normalised error in [-1, 1].
    """

    def __init__(
        self,
        kp_x: float = 0.6,
        kp_y: float = 0.6,
        img_width: int = 640,
        img_height: int = 480,
        deadzone: float = 0.02,
        max_cmd: float = 1.0,
    ):
        """
        Args:
            kp_x: Proportional gain for horizontal axis.
            kp_y: Proportional gain for vertical axis.
            img_width: Expected frame width in pixels.
            img_height: Expected frame height in pixels.
            deadzone: Normalised error below which command is zeroed
                      (avoids motor chatter near centre).
            max_cmd: Clip output to [-max_cmd, max_cmd].
        """
        self.kp_x = kp_x
        self.kp_y = kp_y
        self.img_width = img_width
        self.img_height = img_height
        self.deadzone = deadzone
        self.max_cmd = max_cmd

        # Shared trajectory buffer (for display / recording parity)
        self.trajectory_buffer = TrajectoryBuffer(10)
        self.motor_state = np.zeros(4)
        self.em_state = np.zeros(10)

        self.frame_count = 0
        self.total_time = 0.0

        print(f"[ProportionalController] kp_x={kp_x}, kp_y={kp_y}, "
              f"deadzone={deadzone}, max_cmd={max_cmd}")

    # ------------------------------------------------------------------
    # Interface methods (mirror ControlInterface / MockControlInterface)
    # ------------------------------------------------------------------

    def update_trajectory(self, detection_result, timestamp: float):
        self.trajectory_buffer.add_from_detection(detection_result, timestamp)

    def set_motor_state(self, positions: np.ndarray, speeds: np.ndarray):
        self.motor_state = np.concatenate([positions, speeds])

    def set_em_state(self, em_state: np.ndarray):
        self.em_state = em_state

    def compute_action(
        self,
        frame: np.ndarray,
        detection_result,
        timestamp: float,
        deterministic: bool = True,
    ) -> ControlAction:
        """
        Compute P-control action from detection result.

        Args:
            frame: Current camera frame (RGB) — not used, kept for API parity.
            detection_result: DetectionResult from perception.
            timestamp: Current timestamp.
            deterministic: Unused; kept for API parity.

        Returns:
            ControlAction with motor commands in [-1, 1].
        """
        start_time = time.time()

        if detection_result.no_detection or detection_result.center is None:
            return ControlAction(
                action=np.zeros(2),
                action_mean=np.zeros(2),
                valid=False,
                inference_time=time.time() - start_time,
            )

        self.update_trajectory(detection_result, timestamp)

        # Normalised pixel error in [-1, 1]
        # get_pixel_error returns (ex, ey) in raw pixels relative to centre;
        # divide by half-dimensions to normalise.
        error_px = detection_result.get_pixel_error(self.img_width, self.img_height)
        print(f"[ProportionalController] Raw pixel error: {error_px}")
        norm_ex = error_px[0] / (self.img_width  / 2.0)
        norm_ey = error_px[1] / (self.img_height / 2.0)

        # Apply deadzone
        if abs(norm_ex) < self.deadzone:
            norm_ex = 0.0
        if abs(norm_ey) < self.deadzone:
            norm_ey = 0.0

        # P-control
        action = np.array([norm_ex * self.kp_x, norm_ey * self.kp_y])

        # Normalize by vector magnitude — same as ControlInterface
        norm = np.linalg.norm(action)
        action = action / norm if norm > 0.0 else action
        print(f"[Control] Baseline command — action: {action}")


        elapsed = time.time() - start_time
        self.frame_count += 1
        self.total_time += elapsed

        return ControlAction(
            action=action,
            action_mean=action,
            valid=True,
            inference_time=elapsed,
        )

    def reset(self):
        self.trajectory_buffer.clear()
        self.motor_state = np.zeros(4)
        self.em_state = np.zeros(10)
        self.frame_count = 0
        self.total_time = 0.0

    def get_stats(self) -> Dict[str, Any]:
        return {
            'frame_count': self.frame_count,
            'total_time': self.total_time,
            'avg_fps': self.frame_count / self.total_time if self.total_time > 0 else 0,
            'trajectory_count': self.trajectory_buffer.count,
        }


def create_control_interface(
    config,
    use_mock: bool = False,
) -> ControlInterface:
    """
    Factory function to create control interface.
    
    Args:
        config: IntegratedConfig or LowLevelControlConfig
        use_mock: Whether to use mock interface for testing
    
    Returns:
        ControlInterface or MockControlInterface
    """
    if use_mock:
        return MockControlInterface(config=config)
    else:
        # Get checkpoint path
        if hasattr(config, 'low_level'):
            checkpoint_path = config.low_level.checkpoint_path
            device = config.get_device() if hasattr(config, 'get_device') else None
        else:
            checkpoint_path = config.checkpoint_path
            device = None
        
        return ControlInterface(
            checkpoint_path=checkpoint_path,
            config=config,
            device=device,
        )


if __name__ == '__main__':
    # Test mock control
    print("Testing MockControlInterface...")
    
    from perception_interface import MockPerceptionInterface, DetectionResult
    
    # Create mock perception
    ref_img = np.random.randint(0, 255, (60, 80, 3), dtype=np.uint8)
    mock_perc = MockPerceptionInterface(ref_img)
    
    # Create mock control
    mock_ctrl = MockControlInterface()
    
    # Test loop
    frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    
    for i in range(10):
        detection = mock_perc.detect(frame)
        action = mock_ctrl.compute_action(frame, detection, timestamp=i * 0.05)
        print(f"Frame {i}: error={detection.get_pixel_error()}, action={action.action}, valid={action.valid}")
    
    print("\nMock control test complete!")