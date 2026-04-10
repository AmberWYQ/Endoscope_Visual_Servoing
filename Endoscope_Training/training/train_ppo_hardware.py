"""
Stage 3: PPO Hardware On-Policy Training
Train robotic endoscope control using Proximal Policy Optimization

Features:
1. On-policy training (no replay buffer needed)
2. YOLOv8x-worldv2 for black point detection
3. Hardware-safe training with action limits
4. GAE (Generalized Advantage Estimation)
5. Headless operation support for Linux servers
"""
import rospy
from trakstar.msg import TrakstarMsg

import os
# Fix Qt platform plugin error for headless Linux servers
os.environ['QT_QPA_PLATFORM'] = 'offscreen'
os.environ['OPENCV_VIDEOIO_PRIORITY_MSMF'] = '0'

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal
import numpy as np
from pathlib import Path
import sys
import json
from collections import deque
from tqdm import tqdm
from datetime import datetime
from typing import Dict, Tuple, Optional, List
from dataclasses import dataclass
import cv2
import time
import threading
import queue
import serial

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from configs.config import get_config, Config
from models.network import EndoscopeTrackingNetwork

# Try to import ultralytics for YOLOv8
try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except ImportError:
    print("Warning: ultralytics not installed. Run: pip install ultralytics --break-system-packages")
    YOLO_AVAILABLE = False


@dataclass
class PPOConfig:
    """PPO-specific hyperparameters"""
    # PPO hyperparameters
    clip_epsilon: float = 0.2
    value_loss_coef: float = 0.5
    entropy_coef: float = 0.01
    max_grad_norm: float = 0.5
    
    # GAE parameters
    gamma: float = 0.99
    gae_lambda: float = 0.95
    
    # Training parameters
    lr: float = 3e-4
    num_epochs: int = 10
    num_minibatches: int = 4
    rollout_steps: int = 256  # Steps before update
    total_timesteps: int = 100000
    
    # Hardware safety
    max_action_magnitude: float = 40.0  # Max motor command (deg/s)
    action_smoothing: float = 0.3  # Exponential smoothing factor
    emergency_stop_threshold: float = 100.0  # Error threshold for emergency stop
    
    # Camera settings
    camera_width: int = 640
    camera_height: int = 480
    camera_fps: int = 20

    # Serial settings
    baud_rate: int = 115200

    # EMTracker settings
    emtracker_fps: int = 20
    emtracker_raw_buffer_seconds: float = 2.0


class BlackPointDetector:
    """
    YOLOv8-World based black point detector for endoscope images
    Uses open-vocabulary detection with text prompt "black point"
    """
    
    def __init__(self, model_path: str = "yolov8x-worldv2.pt", device: str = "cuda"):
        self.device = device
        self.model = None
        self.prompt = ["black point", "black dot", "marker", "target point"]
        
        if YOLO_AVAILABLE:
            print(f"Loading YOLOv8-World model: {model_path}")
            self.model = YOLO(model_path)
            # Set the text prompts for open-vocabulary detection
            self.model.set_classes(self.prompt)
            print(f"YOLOv8-World initialized with prompts: {self.prompt}")
        else:
            print("Warning: Using fallback detection (centroid-based)")
    
    def detect(self, image: np.ndarray, conf_threshold: float = 0.3) -> Dict:
        """
        Detect black point in endoscope image
        
        Args:
            image: BGR image from camera (H, W, 3)
            conf_threshold: Confidence threshold for detection
            
        Returns:
            Dict with detection results:
                - detected: bool
                - bbox: [x1, y1, x2, y2] or None
                - center: [cx, cy] or None  
                - confidence: float
                - image_error: [ex, ey] error from image center
        """
        h, w = image.shape[:2]
        image_center = np.array([w / 2, h / 2])
        
        result = {
            'detected': False,
            'bbox': None,
            'center': None,
            'confidence': 0.0,
            'image_error': np.zeros(2)
        }
        
        if self.model is not None:
            # YOLOv8-World detection
            detections = self.model.predict(
                image, 
                conf=conf_threshold,
                verbose=False,
                device=self.device
            )
            
            if len(detections) > 0 and len(detections[0].boxes) > 0:
                # Get highest confidence detection
                boxes = detections[0].boxes
                best_idx = boxes.conf.argmax().item()
                
                bbox = boxes.xyxy[best_idx].cpu().numpy()
                conf = boxes.conf[best_idx].item()
                
                # Calculate center of bounding box
                cx = (bbox[0] + bbox[2]) / 2
                cy = (bbox[1] + bbox[3]) / 2
                center = np.array([cx, cy])
                
                # Calculate error from image center (normalized)
                image_error = (center - image_center) / np.array([w/2, h/2])
                
                result.update({
                    'detected': True,
                    'bbox': bbox.tolist(),
                    'center': center.tolist(),
                    'confidence': conf,
                    'image_error': image_error
                })
        else:
            # Fallback: simple color-based detection
            result = self._fallback_detection(image, image_center)
        
        return result
    
    def _fallback_detection(self, image: np.ndarray, image_center: np.ndarray) -> Dict:
        """Fallback detection using color thresholding"""
        h, w = image.shape[:2]
        
        # Convert to grayscale and threshold for black points
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        _, binary = cv2.threshold(gray, 50, 255, cv2.THRESH_BINARY_INV)
        
        # Find contours
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        result = {
            'detected': False,
            'bbox': None,
            'center': None,
            'confidence': 0.0,
            'image_error': np.zeros(2)
        }
        
        if contours:
            # Find largest contour (assumed to be the target)
            largest = max(contours, key=cv2.contourArea)
            area = cv2.contourArea(largest)
            
            # Filter by reasonable area
            if 10 < area < 10000:
                M = cv2.moments(largest)
                if M["m00"] > 0:
                    cx = M["m10"] / M["m00"]
                    cy = M["m01"] / M["m00"]
                    center = np.array([cx, cy])
                    
                    x, y, bw, bh = cv2.boundingRect(largest)
                    bbox = [x, y, x + bw, y + bh]
                    
                    image_error = (center - image_center) / np.array([w/2, h/2])
                    
                    result.update({
                        'detected': True,
                        'bbox': bbox,
                        'center': center.tolist(),
                        'confidence': min(area / 1000, 1.0),
                        'image_error': image_error
                    })
        
        return result

class EMTrackerInterface:
    """Interface for EM tracker (placeholder - implement for your hardware)"""
    
    def __init__(self, config: Config, port: str = "None"):
        self.config = config  # PPOConfig
        self.port = port
        self.connected = False
        self.last_state = np.zeros(10)

        self.lock = threading.Lock()
        self.raw_times = deque()
        self.raw_pos = deque()
        self.raw_quat = deque()
        self.csv_file = None
        self.csv_writer = None
        self.start_time = 0
        self.origin_pos = None
        self.origin_set = False
        self.rec_counter = 0
        self.last_stat_time = time.time()
        
        # Try to connect
        self._connect()
        
    def _connect(self):
        """Connect to EM tracker"""
        try:
            # ROS Init
            self.sub = rospy.Subscriber("/trakstar_msg", TrakstarMsg, self._on_raw)
            self.timer = rospy.Timer(rospy.Duration(1.0/self.config.emtracker_fps), self._on_tick)
            self.connected = True  # Set True when implemented
            print("[EMTracker] Connected")
        except Exception as e:
            print(f"[EMTracker] Connection failed: {e}")
            self.connected = False

    def _on_raw(self, msg):
        t_now = msg.header.stamp.to_sec() if hasattr(msg, "header") else rospy.get_time()
        if getattr(msg, "n_tracker", 0) <= 0: return
        t0 = msg.transform[0]
        with self.lock:
            self.raw_times.append(t_now)
            self.raw_pos.append([t0.translation.x, t0.translation.y, t0.translation.z])
            self.raw_quat.append([t0.rotation.w, t0.rotation.x, t0.rotation.y, t0.rotation.z])
            while len(self.raw_times) > int(self.config.emtracker_fps * self.config.emtracker_raw_buffer_seconds):
                self.raw_times.popleft(); self.raw_pos.popleft(); self.raw_quat.popleft()

    def _on_tick(self, evt):
        with self.lock:
            n = len(self.raw_times)
            if n == 0: return
            t_arr = np.fromiter(self.raw_times, dtype=float, count=n)
            p_arr = np.array(self.raw_pos, dtype=float)
            q_arr = np.array(self.raw_quat, dtype=float)
            
        t_des = evt.current_real.to_sec()
        
        # 插值逻辑
        if t_des <= t_arr[0]: pos_u, quat_u = p_arr[0], q_arr[0]
        elif t_des >= t_arr[-1]: pos_u, quat_u = p_arr[-1], q_arr[-1]
        else:
             j = int(np.searchsorted(t_arr, t_des) - 1); j = max(0, min(j, n-2))
             t0, t1 = t_arr[j], t_arr[j+1]
             a = 0.0 if t1 == t0 else float(np.clip((t_des - t0)/(t1 - t0), 0.0, 1.0))
             pos_u = (1.0 - a) * p_arr[j] + a * p_arr[j+1]
             quat_u = self.quat_normalize(self.quat_slerp(q_arr[j], q_arr[j+1], a))

        if not self.origin_set:
            self.origin_pos = np.array(pos_u); self.origin_set = True
            print(f"[EMTracker] Origin Set: {self.origin_pos}")
            
        rel_pos = np.array(pos_u) - self.origin_pos 
            
        self.last_state = np.concatenate([
            # np.atleast_1d(t_des - self.start_time),
            np.atleast_1d(rel_pos),
            np.atleast_1d(pos_u),
            np.atleast_1d(quat_u)
        ])

    # 四元数插值辅助函数
    def quat_normalize(self, q):
        q = np.asarray(q, dtype=float)
        n = np.linalg.norm(q)
        return q / (n if n > 0 else 1.0)

    def quat_slerp(self, q0, q1, alpha):
        q0 = self.quat_normalize(q0); q1 = self.quat_normalize(q1)
        dot = float(np.dot(q0, q1))
        if dot < 0.0: q1 = -q1; dot = -dot
        if dot > 0.9995: return self.quat_normalize(q0 + alpha*(q1 - q0))
        theta = np.arccos(np.clip(dot, -1.0, 1.0))
        s0 = np.sin((1.0 - alpha) * theta) / np.sin(theta)
        s1 = np.sin(alpha * theta) / np.sin(theta)
        return s0*q0 + s1*q1
          
    def read(self) -> np.ndarray:
        """
        Read current EM tracker state
        
        Returns:
            state: [rel_x, rel_y, rel_z, abs_x, abs_y, abs_z, qw, qx, qy, qz]
        """
        return self.last_state
    
    def close(self):
        """Close connection"""
        if self.connected:
            # self.serial.close()
            self.connected = False

class HardwareInterface:
    """
    Interface for real hardware control
    Handles camera capture and motor commands
    """
    
    def __init__(self, config: PPOConfig, 
                 camera_id: int = 0,
                 motor_port: str = "/dev/ttyUSB0"):
        self.config = config
        self.camera_id = camera_id
        self.motor_port = motor_port
        self.connected = False
        self.lock = threading.Lock()
        
        # Camera setup
        self.cap = None
        self.current_frame = None
        self.frame_lock = threading.Lock()
        
        # Motor state
        self.motor_state = np.zeros(2)  # [motor1_pos, motor2_pos]
        self.last_action = np.zeros(2)
        
        # EM tracker state (if available)
        self.emtracker = EMTrackerInterface(config)
        self.em_state = np.zeros(10, dtype=np.float32)  # [rel_x, rel_y, rel_z, abs_x, abs_y, abs_z, qw, qx, qy, qz]
        
        # Safety flags
        self.emergency_stop = False
        
        # Initialize detector
        self.detector = BlackPointDetector()

        # Trajectory
        self.traj_history_size = 10
        self.state_dim = 10
        self.traj_dim = 14
        self.traj_buffer = deque(
            [np.zeros(self.traj_dim, dtype=np.float32) for _ in range(self.traj_history_size)],
            maxlen=self.traj_history_size
            )

        
    def connect(self) -> bool:
        """Connect to hardware"""
        try:
            # Initialize camera
            self.cap = cv2.VideoCapture(self.camera_id)
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.config.camera_width)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.config.camera_height)
            
            if not self.cap.isOpened():
                print(f"Error: Could not open camera {self.camera_id}")
                return False
            
            print(f"Camera connected: {self.config.camera_width}x{self.config.camera_height}")
            
            # TODO: Initialize motor serial connection
            self.serial = serial.Serial(self.motor_port, self.config.baud_rate, timeout=0.1)
            self.connected = True
            print(f"Motor interface initialized (port: {self.motor_port})")
            
            return True
            
        except Exception as e:
            print(f"Hardware connection error: {e}")
            return False
    
    def disconnect(self):
        """Safely disconnect hardware"""
        if self.cap is not None:
            self.cap.release()
        # TODO: Close motor serial connection
        if self.connected:
            self.serial.close()
            self.connected = False
            print("Hardware disconnected")
    
    def get_observation(self) -> Dict:
        """
        Get current observation from hardware
        
        Returns:
            Dict with observation components
        """

        # Capture frame
        ret, frame = self.cap.read()
        if not ret:
            raise RuntimeError("Failed to capture frame")
        
        with self.frame_lock:
            self.current_frame = frame.copy()
        
        # Detect black point
        detection = self.detector.detect(frame)
        print(f"detection is {detection}")
        if detection['detected'] is False:
            target = np.zeros(4)
        else:
            print(f"center is {detection['center']}")
            target = np.concatenate([detection['bbox']])
        print(f"target is {target}")

        # Get em state
        self.em_state = self.emtracker.read()
        print(f"em_state: {self.em_state}")

        # Update trajectory (detection + em_state)
        self.traj_buffer.append(np.concatenate([target, self.em_state]))
        print(f"traj_buffer is {self.traj_buffer}")

        # Build observation
        obs = {
            'image': frame,  # (H, W, 3) BGR
            'image_error': detection['image_error'].astype(np.float32),
            'motor_state': self.motor_state.copy().astype(np.float32),
            'em_state': self.em_state.copy().astype(np.float32),
            'trajectory': np.array(self.traj_buffer, dtype=np.float32), 
            'detection': detection
        }
        
        return obs

    def write_raw(self, data):
        try:
            with self.lock:
                self.serial.write(data)
                self.last_send_time = time.time()
            return True
        except Exception as e:
            print(f"[Motors] Send error: {e}")
            return False
    
    def send_action(self, action: np.ndarray) -> bool:
        """
        Send action to motors with safety checks
        
        Args:
            action: [motor1_velocity, motor2_velocity] in deg/s
            
        Returns:
            Success flag
        """
        if self.emergency_stop:
            print("EMERGENCY STOP ACTIVE - action blocked")
            return False
        
        # Clip action to safe range
        action = np.clip(action, -self.config.max_action_magnitude, 
                        self.config.max_action_magnitude)
        
        # Apply smoothing for hardware safety
        smoothed_action = (self.config.action_smoothing * action + 
                         (1 - self.config.action_smoothing) * self.last_action)
        
        # TODO: Send to actual motors via serial
        # command = f"M1:{smoothed_action[0]:.2f},M2:{smoothed_action[1]:.2f}\n"
        # self.motor_serial.write(command.encode())
        cmd_str = "{:>6d}{:>6d}{:>6d}{:>6d}".format(int(smoothed_action[0]),int(smoothed_action[1]),0,0)
        self.write_raw(cmd_str.encode('ascii'))

        # Update state (simulated for now)
        dt = 1.0 / self.config.camera_fps
        self.motor_state += smoothed_action * dt  # Approximate position update
        self.last_action = smoothed_action.copy()
        
        return True
    
    def compute_reward(self, obs: Dict, action: np.ndarray) -> Tuple[float, Dict]:
        """
        Compute reward for current state
        
        Reward components:
        1. Tracking error (negative distance from center)
        2. Action smoothness penalty
        3. Detection bonus
        """
        detection = obs['detection']
        image_error = obs['image_error']
        
        # Base tracking reward (negative L2 error)
        error_magnitude = np.linalg.norm(image_error)
        tracking_reward = -error_magnitude
        
        # Detection bonus
        detection_bonus = 0.5 if detection['detected'] else -0.5
        
        # Action smoothness penalty
        action_magnitude = np.linalg.norm(action)
        action_penalty = -0.01 * action_magnitude
        
        # Success bonus for being close to center
        success_bonus = 1.0 if error_magnitude < 0.1 else 0.0
        
        # Total reward
        reward = tracking_reward + detection_bonus + action_penalty + success_bonus
        
        info = {
            'tracking_error': error_magnitude,
            'detected': detection['detected'],
            'tracking_reward': tracking_reward,
            'detection_bonus': detection_bonus,
            'action_penalty': action_penalty,
            'success_bonus': success_bonus
        }
        
        return reward, info
    
    def check_done(self, obs: Dict, step: int, max_steps: int) -> Tuple[bool, bool]:
        """
        Check if episode should end
        
        Returns:
            (terminated, truncated)
        """
        error_magnitude = np.linalg.norm(obs['image_error'])
        
        # Emergency stop check
        if error_magnitude > self.config.emergency_stop_threshold:
            self.emergency_stop = True
            return True, False
        
        # Success termination
        if error_magnitude < 0.05:
            return True, False
        
        # Max steps truncation
        if step >= max_steps:
            return False, True
        
        return False, False

    def auto_homing(self, speed_limit=None, tolerance=10):
        if not self.connected:
            print("[Motors] not connected, homing cannot be executed.")
            return
        
        if speed_limit is None:
            speed_limit = self.config.max_action_magnitude * 0.5
        print("[Motors] is homing...")

        while True:
            error = -self.motor_state

            # Motors are within tolerance
            if np.all(np.abs(error) < tolerance):
                break

            # Compute homing speed
            kp = 0.5
            homing_speeds = error * kp
            homing_speeds = np.clip(homing_speeds, -speed_limit, speed_limit)

            # Send command
            self.send_action(homing_speeds)

            time.sleep(0.05)
        
        self.stop() # Stop the motor
        print("[Motors] homing completed.")

        if np.any(self.motor_state != np.zeros(2)):
            with self.lock:
                self.motor_state = np.zeros(2)

    def stop(self):
        self.send_action(np.zeros(2))
    
    def reset(self) -> Dict:
        """Reset environment for new episode"""
        self.emergency_stop = False
        self.last_action = np.zeros(2)
        
        # TODO: Command motors to home position
        # self.motor_serial.write(b"HOME\n")
        self.auto_homing()
        time.sleep(0.5)  # Wait for motors to settle
        
        return self.get_observation()


class RolloutBuffer:
    """
    Buffer for storing on-policy rollout data
    """
    
    def __init__(self, buffer_size: int, obs_shapes: Dict, action_dim: int, device: torch.device):
        self.buffer_size = buffer_size
        self.device = device
        self.ptr = 0
        self.full = False
        
        # Allocate buffers
        img_shape = obs_shapes.get('image', (3, 480, 640))
        self.images = np.zeros((buffer_size, *img_shape), dtype=np.float32)
        self.trajectories = np.zeros((buffer_size, *obs_shapes['trajectory']), dtype=np.float32)
        self.em_states = np.zeros((buffer_size, *obs_shapes['em_state']), dtype=np.float32)
        self.motor_states = np.zeros((buffer_size, *obs_shapes['motor_state']), dtype=np.float32)
        self.image_errors = np.zeros((buffer_size, *obs_shapes['image_error']), dtype=np.float32)
        
        self.actions = np.zeros((buffer_size, action_dim), dtype=np.float32)
        self.rewards = np.zeros(buffer_size, dtype=np.float32)
        self.values = np.zeros(buffer_size, dtype=np.float32)
        self.log_probs = np.zeros(buffer_size, dtype=np.float32)
        self.dones = np.zeros(buffer_size, dtype=np.float32)
        
        # Computed during finalization
        self.advantages = np.zeros(buffer_size, dtype=np.float32)
        self.returns = np.zeros(buffer_size, dtype=np.float32)
    
    def add(self, obs: Dict, action: np.ndarray, reward: float, 
            value: float, log_prob: float, done: bool):
        """Add a transition to the buffer"""
        self.images[self.ptr] = obs['image'].transpose(2, 0, 1) / 255.0
        self.trajectories[self.ptr] = obs['trajectory']
        self.em_states[self.ptr] = obs['em_state']
        self.motor_states[self.ptr] = obs['motor_state']
        self.image_errors[self.ptr] = obs['image_error']
        
        self.actions[self.ptr] = action
        self.rewards[self.ptr] = reward
        self.values[self.ptr] = value
        self.log_probs[self.ptr] = log_prob
        self.dones[self.ptr] = done
        
        self.ptr += 1
        if self.ptr >= self.buffer_size:
            self.full = True
    
    def compute_returns_and_advantages(self, last_value: float, gamma: float, gae_lambda: float):
        """
        Compute GAE advantages and returns
        """
        last_gae = 0
        
        for t in reversed(range(self.ptr)):
            if t == self.ptr - 1:
                next_value = last_value
                next_non_terminal = 1.0 - self.dones[t]
            else:
                next_value = self.values[t + 1]
                next_non_terminal = 1.0 - self.dones[t]
            
            delta = self.rewards[t] + gamma * next_value * next_non_terminal - self.values[t]
            last_gae = delta + gamma * gae_lambda * next_non_terminal * last_gae
            self.advantages[t] = last_gae
        
        self.returns[:self.ptr] = self.advantages[:self.ptr] + self.values[:self.ptr]
    
    def get_batches(self, num_minibatches: int) -> List[Dict[str, torch.Tensor]]:
        """Generate minibatches for training"""
        indices = np.random.permutation(self.ptr)
        batch_size = self.ptr // num_minibatches
        
        batches = []
        for start in range(0, self.ptr, batch_size):
            end = start + batch_size
            if end > self.ptr:
                break
            
            batch_indices = indices[start:end]
            
            batch = {
                'image': torch.tensor(self.images[batch_indices], device=self.device),
                'trajectory': torch.tensor(self.trajectories[batch_indices], device=self.device),
                'em_state': torch.tensor(self.em_states[batch_indices], device=self.device),
                'motor_state': torch.tensor(self.motor_states[batch_indices], device=self.device),
                'image_error': torch.tensor(self.image_errors[batch_indices], device=self.device),
                'action': torch.tensor(self.actions[batch_indices], device=self.device),
                'old_log_prob': torch.tensor(self.log_probs[batch_indices], device=self.device),
                'advantage': torch.tensor(self.advantages[batch_indices], device=self.device),
                'return': torch.tensor(self.returns[batch_indices], device=self.device)
            }
            batches.append(batch)
        
        return batches
    
    def reset(self):
        """Reset buffer for new rollout"""
        self.ptr = 0
        self.full = False


class PPOTrainer:
    """
    Proximal Policy Optimization trainer for hardware on-policy training
    """
    
    def __init__(self, 
                 config: Config,
                 ppo_config: PPOConfig = None,
                 bc_checkpoint: Optional[str] = None,
                 checkpoint_dir: str = None):
        
        self.config = config
        self.ppo_config = ppo_config or PPOConfig()
        self.device = torch.device(config.training.device if torch.cuda.is_available() else "cpu")
        
        print(f"Using device: {self.device}")
        
        # Actor-Critic network (shared backbone)
        self.actor = EndoscopeTrackingNetwork(config).to(self.device)
        
        # Value head (separate network or can be part of actor)
        combined_dim = (config.network.image_feature_dim + 
                config.network.trajectory_feature_dim + 
                config.network.state_feature_dim)
        
        self.value_head = nn.Sequential(
            nn.Linear(combined_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        ).to(self.device)
        
        # Load BC pretrained weights if available
        if bc_checkpoint and Path(bc_checkpoint).exists():
            self._load_bc_weights(bc_checkpoint)
        
        # Optimizer for both actor and value head
        self.optimizer = torch.optim.Adam([
            {'params': self.actor.parameters(), 'lr': self.ppo_config.lr},
            {'params': self.value_head.parameters(), 'lr': self.ppo_config.lr}
        ])
        
        # Hardware interface
        self.hardware = HardwareInterface(self.ppo_config)
        
        # Rollout buffer
        obs_shapes = {
            'trajectory': (self.config.network.trajectory_history, self.config.network.trajectory_dim),
            'em_state': (self.config.em_tracker.state_dim,),
            'motor_state': (self.config.robot.num_motors,),
            'image_error': (2,)
        }
        self.buffer = RolloutBuffer(
            self.ppo_config.rollout_steps,
            obs_shapes,
            config.network.action_dim,
            self.device
        )
        
        # Checkpointing
        self.checkpoint_dir = Path(checkpoint_dir or config.checkpoint_dir) / "ppo"
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        # Logging
        self.log_dir = Path(config.log_dir) / "ppo"
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        self.train_history = []
        self.episode_rewards = []
    
    def _load_bc_weights(self, checkpoint_path: str):
        """Load pretrained BC weights"""
        print(f"Loading BC weights from {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
        model_dict = self.actor.state_dict()
        pretrained_dict = checkpoint['model_state_dict']
        
        pretrained_dict = {k: v for k, v in pretrained_dict.items() 
                         if k in model_dict and v.shape == model_dict[k].shape}
        
        model_dict.update(pretrained_dict)
        self.actor.load_state_dict(model_dict)
        
        print(f"Loaded {len(pretrained_dict)}/{len(model_dict)} layers from BC checkpoint")
    
    def get_action_and_value(self, obs: Dict, deterministic: bool = False) -> Tuple[np.ndarray, float, float]:
        """
        Get action, value, and log probability from current observation
        """
        with torch.no_grad():
            # Prepare observation tensors
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
            
            # Forward pass through actor
            output = self.actor(image, trajectory, em_state, motor_state, image_error,
                              deterministic=deterministic)
            
            action_mean = output['action_mean']
            action_log_std = output['action_log_std']
            
            # Sample action from distribution
            if deterministic:
                action = action_mean
            else:
                std = torch.exp(action_log_std)
                dist = Normal(action_mean, std)
                action = dist.sample()
            
            # Compute log probability
            std = torch.exp(action_log_std)
            dist = Normal(action_mean, std)
            log_prob = dist.log_prob(action).sum(dim=-1)
            
            # Get value estimate
            # Use the fused features from actor for value estimation
            features = torch.cat([
                output['image_features'],
                output['trajectory_features'],
                output['state_features']
            ], dim=-1)
            
            # Simple pooling if features are 3D
            if features.dim() == 3:
                features = features.mean(dim=1)
            
            # Ensure correct input size for value head
            
            value = self.value_head(features)
            
            action_np = action.cpu().numpy()[0]
            value_np = value.cpu().numpy()[0, 0]
            log_prob_np = log_prob.cpu().numpy()[0]
        
        return action_np, value_np, log_prob_np
    
    def evaluate_actions(self, batch: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Evaluate actions for PPO update
        
        Returns:
            values, log_probs, entropy
        """
        # Forward pass
        output = self.actor(
            batch['image'],
            batch['trajectory'],
            batch['em_state'],
            batch['motor_state'],
            batch['image_error']
        )
        
        action_mean = output['action_mean']
        action_log_std = output['action_log_std']
        std = torch.exp(action_log_std)
        
        # Distribution
        dist = Normal(action_mean, std)
        
        # Log probabilities
        log_probs = dist.log_prob(batch['action']).sum(dim=-1)
        
        # Entropy
        entropy = dist.entropy().sum(dim=-1)
        
        # Value estimate
        features = torch.cat([
            output['image_features'],
            output['trajectory_features'],
            output['state_features']
        ], dim=-1)
        
        if features.dim() == 3:
            features = features.mean(dim=1)
        
        values = self.value_head(features).squeeze(-1)
        
        return values, log_probs, entropy
    
    def update(self) -> Dict[str, float]:
        """
        PPO update step
        """
        # Get last value for GAE computation
        # (This should be done with the last observation, simplified here)
        last_value = 0.0
        
        # Compute advantages and returns
        self.buffer.compute_returns_and_advantages(
            last_value,
            self.ppo_config.gamma,
            self.ppo_config.gae_lambda
        )
        
        # Training stats
        total_policy_loss = 0.0
        total_value_loss = 0.0
        total_entropy_loss = 0.0
        num_updates = 0
        
        # Multiple epochs over the data
        for epoch in range(self.ppo_config.num_epochs):
            batches = self.buffer.get_batches(self.ppo_config.num_minibatches)
            
            for batch in batches:
                # Normalize advantages
                advantages = batch['advantage']
                advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
                
                # Get current policy values
                values, log_probs, entropy = self.evaluate_actions(batch)
                
                # Policy loss (clipped surrogate objective)
                ratio = torch.exp(log_probs - batch['old_log_prob'])
                surr1 = ratio * advantages
                surr2 = torch.clamp(ratio, 
                                   1 - self.ppo_config.clip_epsilon,
                                   1 + self.ppo_config.clip_epsilon) * advantages
                policy_loss = -torch.min(surr1, surr2).mean()
                
                # Value loss
                value_loss = F.mse_loss(values, batch['return'])
                
                # Entropy loss (for exploration)
                entropy_loss = -entropy.mean()
                
                # Total loss
                loss = (policy_loss + 
                       self.ppo_config.value_loss_coef * value_loss +
                       self.ppo_config.entropy_coef * entropy_loss)
                
                # Optimization step
                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    list(self.actor.parameters()) + list(self.value_head.parameters()),
                    self.ppo_config.max_grad_norm
                )
                self.optimizer.step()
                
                total_policy_loss += policy_loss.item()
                total_value_loss += value_loss.item()
                total_entropy_loss += entropy_loss.item()
                num_updates += 1
        
        return {
            'policy_loss': total_policy_loss / num_updates,
            'value_loss': total_value_loss / num_updates,
            'entropy_loss': total_entropy_loss / num_updates
        }
    
    def train(self, total_timesteps: int = None, use_simulation: bool = True):
        """
        Main training loop
        
        Args:
            total_timesteps: Total environment steps
            use_simulation: If True, use simulated environment; else use real hardware
        """
        total_timesteps = total_timesteps or self.ppo_config.total_timesteps
        
        print(f"Starting PPO training for {total_timesteps} timesteps...")
        print(f"Mode: {'Simulation' if use_simulation else 'Hardware'}")
        
        if not use_simulation:
            # Connect to hardware
            if not self.hardware.connect():
                print("Failed to connect to hardware. Exiting.")
                return
        
        try:
            if use_simulation:
                self._train_simulation(total_timesteps)
            else:
                self._train_hardware(total_timesteps)
        finally:
            if not use_simulation:
                self.hardware.disconnect()
        
        # Save final model
        self.save_checkpoint('ppo_final.pt')
        
        # Save training history
        with open(self.log_dir / 'train_history.json', 'w') as f:
            json.dump(self.train_history, f, indent=2)
        
        print(f"PPO training complete!")
        if self.episode_rewards:
            print(f"Final avg reward (last 100): {np.mean(self.episode_rewards[-100:]):.2f}")
    
    def _train_hardware(self, total_timesteps: int):
        """Hardware training loop"""
        obs = self.hardware.reset()
        episode_reward = 0.0
        episode_length = 0
        episode_num = 0
        
        pbar = tqdm(range(total_timesteps))
        
        for t in pbar:
            # Get action from policy
            action, value, log_prob = self.get_action_and_value(obs)
            
            # Execute action on hardware
            self.hardware.send_action(action)
            
            # Small delay for hardware response
            time.sleep(1/self.config.camera.fps)  # 50Hz control frequency
            
            # Get new observation
            next_obs = self.hardware.get_observation()
            
            # Compute reward
            reward, info = self.hardware.compute_reward(next_obs, action)
            
            # Check if done
            terminated, truncated = self.hardware.check_done(
                next_obs, episode_length, 
                max_steps=self.ppo_config.rollout_steps
            )
            done = terminated or truncated
            
            # Store transition
            self.buffer.add(obs, action, reward, value, log_prob, done)
            
            episode_reward += reward
            episode_length += 1
            
            # Update policy when buffer is full
            if self.buffer.ptr >= self.ppo_config.rollout_steps:
                update_info = self.update()
                self.buffer.reset()
                
                pbar.set_postfix({
                    'ep_rew': f"{episode_reward:.2f}",
                    'policy': f"{update_info['policy_loss']:.4f}",
                    'value': f"{update_info['value_loss']:.4f}"
                })
            
            # Handle episode end
            if done:
                self.episode_rewards.append(episode_reward)
                
                self.train_history.append({
                    'episode': episode_num,
                    'reward': episode_reward,
                    'length': episode_length,
                    'timestep': t,
                    'tracking_error': info['tracking_error']
                })
                
                if episode_num % 10 == 0:
                    avg_reward = np.mean(self.episode_rewards[-100:]) if self.episode_rewards else 0
                    print(f"\nEpisode {episode_num}: reward={episode_reward:.2f}, "
                          f"avg_100={avg_reward:.2f}, error={info['tracking_error']:.3f}")
                
                if episode_num % 50 == 0:
                    self.save_checkpoint(f'episode_{episode_num}.pt')
                
                # Reset for new episode
                obs = self.hardware.reset()
                episode_reward = 0.0
                episode_length = 0
                episode_num += 1
            else:
                obs = next_obs
    
    def _train_simulation(self, total_timesteps: int):
        """Simulation training loop (for testing)"""
        from simulator.endoscope_sim import make_env
        
        env = make_env(self.config)
        obs, info = env.reset()
        
        episode_reward = 0.0
        episode_length = 0
        episode_num = 0
        
        pbar = tqdm(range(total_timesteps))
        
        for t in pbar:
            # Get action
            action, value, log_prob = self.get_action_and_value(obs)
            
            # Step environment
            next_obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            
            # Store transition
            self.buffer.add(obs, action, reward, value, log_prob, done)
            
            episode_reward += reward
            episode_length += 1
            
            # Update when buffer is full
            if self.buffer.ptr >= self.ppo_config.rollout_steps:
                update_info = self.update()
                self.buffer.reset()
                
                pbar.set_postfix({
                    'ep_rew': f"{episode_reward:.2f}",
                    'policy': f"{update_info['policy_loss']:.4f}"
                })
            
            if done:
                self.episode_rewards.append(episode_reward)
                self.train_history.append({
                    'episode': episode_num,
                    'reward': episode_reward,
                    'length': episode_length,
                    'timestep': t
                })
                
                if episode_num % 10 == 0:
                    avg_reward = np.mean(self.episode_rewards[-100:]) if self.episode_rewards else 0
                    print(f"\nEpisode {episode_num}: reward={episode_reward:.2f}, avg_100={avg_reward:.2f}")
                
                if episode_num % 100 == 0:
                    self.save_checkpoint(f'episode_{episode_num}.pt')
                
                obs, info = env.reset()
                episode_reward = 0.0
                episode_length = 0
                episode_num += 1
            else:
                obs = next_obs
    
    def save_checkpoint(self, filename: str):
        """Save training checkpoint"""
        checkpoint = {
            'actor_state_dict': self.actor.state_dict(),
            'value_head_state_dict': self.value_head.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'config': self.config,
            'ppo_config': self.ppo_config,
            'train_history': self.train_history,
            'episode_rewards': self.episode_rewards
        }
        torch.save(checkpoint, self.checkpoint_dir / filename)
        print(f"Saved checkpoint: {self.checkpoint_dir / filename}")
    
    def load_checkpoint(self, filepath: str):
        """Load training checkpoint"""
        checkpoint = torch.load(filepath, map_location=self.device, weights_only=False)
        self.actor.load_state_dict(checkpoint['actor_state_dict'])
        self.value_head.load_state_dict(checkpoint['value_head_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.train_history = checkpoint.get('train_history', [])
        self.episode_rewards = checkpoint.get('episode_rewards', [])
        print(f"Loaded checkpoint: {filepath}")
    
    def evaluate(self, num_episodes: int = 10, use_simulation: bool = True) -> Dict:
        """Evaluate trained policy"""
        print(f"Evaluating policy over {num_episodes} episodes...")
        
        if use_simulation:
            from simulator.endoscope_sim import make_env
            env = make_env(self.config)
        else:
            if not self.hardware.connect():
                print("Failed to connect to hardware")
                return {}
        
        rewards = []
        errors = []
        
        try:
            for ep in range(num_episodes):
                if use_simulation:
                    obs, info = env.reset()
                else:
                    obs = self.hardware.reset()
                
                episode_reward = 0.0
                episode_errors = []
                
                for step in range(500):
                    action, _, _ = self.get_action_and_value(obs, deterministic=True)
                    
                    if use_simulation:
                        obs, reward, term, trunc, info = env.step(action)
                        episode_errors.append(info.get('tracking_error', 0))
                        done = term or trunc
                    else:
                        self.hardware.send_action(action)
                        time.sleep(0.02)
                        obs = self.hardware.get_observation()
                        reward, info = self.hardware.compute_reward(obs, action)
                        episode_errors.append(info['tracking_error'])
                        term, trunc = self.hardware.check_done(obs, step, 500)
                        done = term or trunc
                    
                    episode_reward += reward
                    
                    if done:
                        break
                
                rewards.append(episode_reward)
                errors.append(np.mean(episode_errors))
                print(f"Episode {ep+1}: reward={episode_reward:.2f}, avg_error={np.mean(episode_errors):.4f}")
        
        finally:
            if not use_simulation:
                self.hardware.disconnect()
        
        results = {
            'mean_reward': np.mean(rewards),
            'std_reward': np.std(rewards),
            'mean_error': np.mean(errors),
            'std_error': np.std(errors)
        }
        
        print(f"\nEvaluation Results:")
        print(f"  Mean reward: {results['mean_reward']:.2f} ± {results['std_reward']:.2f}")
        print(f"  Mean error: {results['mean_error']:.4f} ± {results['std_error']:.4f}")
        
        return results


def main():
    import argparse

    try:         
        rospy.init_node('endoscope_ppo_agent', anonymous=True, disable_signals=True)
    except rospy.exceptions.ROSException:          
        print("[System] ROS Node already initialized")
    
    parser = argparse.ArgumentParser(description="PPO Hardware Training for Robotic Endoscope")
    parser.add_argument('--bc-checkpoint', type=str, default='./checkpoints/bc/best.pt',
                       help='BC checkpoint path')
    parser.add_argument('--timesteps', type=int, default=50000,
                       help='Total training timesteps')
    parser.add_argument('--hardware', action='store_true',
                       help='Use real hardware (default: simulation)')
    parser.add_argument('--evaluate', action='store_true',
                       help='Evaluate only (no training)')
    parser.add_argument('--checkpoint', type=str,
                       help='Checkpoint to load for evaluation')
    parser.add_argument('--camera-id', type=int, default=0,
                       help='Camera device ID')
    parser.add_argument('--motor-port', type=str, default='/dev/ttyUSB0',
                       help='Motor serial port')
    
    # PPO hyperparameters
    parser.add_argument('--lr', type=float, default=3e-4, help='Learning rate')
    parser.add_argument('--clip-epsilon', type=float, default=0.2, help='PPO clip epsilon')
    parser.add_argument('--rollout-steps', type=int, default=256, help='Steps per rollout')
    parser.add_argument('--num-epochs', type=int, default=10, help='PPO epochs per update')
    
    args = parser.parse_args()
    
    # Load config
    config = get_config()
    
    # Setup PPO config
    ppo_config = PPOConfig(
        lr=args.lr,
        clip_epsilon=args.clip_epsilon,
        rollout_steps=args.rollout_steps,
        num_epochs=args.num_epochs,
        total_timesteps=args.timesteps
    )
    
    # Update hardware interface settings
    ppo_config.camera_id = args.camera_id
    ppo_config.motor_port = args.motor_port

    ppo_config.camera_height = config.camera.height
    ppo_config.camera_width = config.camera.width
    ppo_config.camera_fps = config.camera.fps
    ppo_config.baud_rate = config.serial_setting.baud_rate
    
    # Create trainer
    trainer = PPOTrainer(
        config, 
        ppo_config=ppo_config,
        bc_checkpoint=args.bc_checkpoint if not args.evaluate else None
    )

    

    
    if args.evaluate:
        if args.checkpoint:
            trainer.load_checkpoint(args.checkpoint)
        trainer.evaluate(num_episodes=20, use_simulation=not args.hardware)
    else:
        trainer.train(
            total_timesteps=args.timesteps,
            use_simulation=not args.hardware
        )


if __name__ == "__main__":
    main()


"""
Usage Examples:

# Simulation training (for testing):
python train_ppo_hardware.py --timesteps 50000

# Hardware training:
python training/train_ppo_hardware.py --hardware --timesteps 10000 --camera-id 0 --motor-port /dev/ttyUSB0

# Load from BC checkpoint:
python train_ppo_hardware.py --bc-checkpoint ./checkpoints/bc/bc_model.pt --timesteps 50000

# Evaluate trained model:
python train_ppo_hardware.py --evaluate --checkpoint ./checkpoints/ppo/final.pt

# Hardware evaluation:
python train_ppo_hardware.py --evaluate --hardware --checkpoint ./checkpoints/ppo/final.pt
"""
