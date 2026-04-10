"""
Hardware Control Mode for Endoscope Visual Servoing
Real-time control of robotic endoscope using trained neural network

Based on user's Xbox control pattern from lerobot_dataset_2motor.py

Supports:
- Real-time inference with trained model
- EM tracker integration  
- Motor control via serial/CAN
- Live visualization
- Data logging for further training
"""
import rospy
from trakstar.msg import TrakstarMsg

import os
import sys
import time
import json
import threading
from collections import deque
import queue
from pathlib import Path
from datetime import datetime
from typing import Optional, Dict, Tuple, Any
from dataclasses import dataclass, field

import cv2
import numpy as np
import torch
import pandas as pd
import serial
import re

# Add parent to path
# sys.path.insert(0, str(Path(__file__).parent.parent))

from configs.config import Config, get_config
from models.network import EndoscopeTrackingNetwork

# Set Qt to offscreen for headless servers
# os.environ['QT_QPA_PLATFORM'] = 'offscreen'


@dataclass
class ControlState:
    """Current control state"""
    em_state: np.ndarray = field(default_factory=lambda: np.zeros(10))
    motor_state: np.ndarray = field(default_factory=lambda: np.zeros(4))
    target_center: np.ndarray = field(default_factory=lambda: np.zeros(2))
    target_bbox: np.ndarray = field(default_factory=lambda: np.zeros(4))
    image_error: np.ndarray = field(default_factory=lambda: np.zeros(2))
    action: np.ndarray = field(default_factory=lambda: np.zeros(2))
    timestamp: float = 0.0


class TargetDetector:
    """Detect black target point in endoscopic images"""
    
    def __init__(self, config: Config):
        self.config = config
        self.center_x = config.camera.width // 2
        self.center_y = config.camera.height // 2
        
        # Detection parameters
        self.min_area = 50
        self.max_area = 50000
        self.threshold = 50  # For black point
        
    def detect(self, frame: np.ndarray) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """
        Detect black target in frame
        
        Returns:
            center: [x, y] target center or None
            bbox: [x, y, w, h] bounding box or None
        """
        if frame is None:
            return None, None
            
        # Convert to grayscale
        if len(frame.shape) == 3:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        else:
            gray = frame
            
        # Threshold for black point (inverse - black becomes white)
        _, binary = cv2.threshold(gray, self.threshold, 255, cv2.THRESH_BINARY_INV)
        
        # Morphological operations to clean up
        kernel = np.ones((3, 3), np.uint8)
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
        
        # Find contours
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            return None, None
            
        # Find largest valid contour
        valid_contours = []
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if self.min_area < area < self.max_area:
                valid_contours.append((area, cnt))
                
        if not valid_contours:
            return None, None
            
        # Get largest
        valid_contours.sort(key=lambda x: x[0], reverse=True)
        best_contour = valid_contours[0][1]
        
        # Get bounding box and center
        x, y, w, h = cv2.boundingRect(best_contour)
        
        # Compute centroid
        M = cv2.moments(best_contour)
        if M["m00"] > 0:
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
        else:
            cx = x + w // 2
            cy = y + h // 2
            
        return np.array([cx, cy]), np.array([x, y, w, h])
    
    def compute_error(self, center: Optional[np.ndarray]) -> np.ndarray:
        """Compute error from image center"""
        if center is None:
            return np.zeros(2)
        return np.array([
            center[0] - self.center_x,
            center[1] - self.center_y
        ])


class EMTrackerInterface:
    """Interface for EM tracker (placeholder - implement for your hardware)"""
    
    def __init__(self, config: Config, port: str = "None"):
        self.config = config
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
            self.timer = rospy.Timer(rospy.Duration(1.0/self.config.em_tracker.fps), self._on_tick)
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
            while len(self.raw_times) > int(self.config.em_tracker.fps * self.config.em_tracker.raw_buffer_seconds):
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
            np.atleast_1d(t_des - self.start_time),
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
        """
        if not self.connected:
            # Return simulated data with small noise
            noise = np.random.randn(10) * 0.001
            self.last_state = noise
            self.last_state[6] = 1.0  # qw = 1 for identity quaternion
            return self.last_state
            
        try:
            # Placeholder - implement actual read
            # data = self.serial.readline()
            # Parse data into state vector
            pass
        except Exception as e:
            print(f"[EMTracker] Read error: {e}")
        """
            
        return self.last_state
    
    def close(self):
        """Close connection"""
        if self.connected:
            # self.serial.close()
            self.connected = False


class MotorController:
    """Control endoscope bending motors (placeholder - implement for your hardware)"""
    
    def __init__(self, config: Config, port: str = "/dev/ttyUSB0"):
        self.config = config
        self.port = port
        self.connected = False
        self.serial = None
        self.lock = threading.Lock()
        self.last_send_time = 0
        self.t_read = None
        
        # Motor state
        self.motor_actual_positions = np.zeros(2)
        self.motor_positions = np.zeros(2)
        self.motor_speeds = np.zeros(2)
        
        # Safety limits
        self.max_speed = config.robot.max_motor_speed
        self.min_speed = config.robot.min_motor_speed
        
        self._connect()
        
    def _connect(self):
        """Connect to motor controller"""
        print(f"[Motors] Attempting connection to {self.port}...")
        try:
            # Placeholder - implement actual connection
            # CAN bus, serial, or other interface
            self.serial = serial.Serial(self.port, self.config.serial_setting.baud_rate, timeout=0.1)
            self.connected = True
            print(f"[Motors] Connected to {self.port}")

            self.t_read = threading.Thread(target=self.serial_read_loop, daemon=True)
            self.t_read.start()

        except Exception as e:
            print(f"[Motors] Connection failed: {str(e)}")
            self.connected = False

    def write_raw(self, data):
        try:
            with self.lock:
                self.serial.write(data)
                self.last_send_time = time.time()
            return True
        except Exception as e:
            print(f"[Motors] Send error: {e}")
            return False
            
    def send_command(self, speeds: np.ndarray) -> bool:
        """
        Send speed commands to motors
        
        Args:
            speeds: [m1_spd, m2_spd]
            
        Returns:
            success: True if command sent
        """
        # Clip to limits
        clamped = np.clip(speeds, self.min_speed, self.max_speed)
        self.motor_speeds = clamped
        cmd_str = "{:>6d}{:>6d}{:>6d}{:>6d}".format(int(self.motor_speeds[0]),int(self.motor_speeds[1]),0,0)
        
        if not self.connected:
            # Simulate motor response
            dt = 0.05  # 20 Hz
            self.motor_positions += speeds * dt * 0.01  # Simplified model
            # print(f"[Motors] Connection failed, using simulated response.")
            return True
        
        # Send speed command
        self.write_raw(cmd_str.encode('ascii'))

        # Update position
        dt = 1.0 /  self.config.camera.fps
        self.motor_positions[0] += self.motor_speeds[0] * dt
        self.motor_positions[1] += self.motor_speeds[1] * dt

        return True

    def serial_read_loop(self):
        pattern = re.compile(r"FB:(-?\d+),(-?\d+),(-?\d+),(-?\d+),(-?\d+),(-?\d+),(-?\d+),(-?\d+)")
        while self.connected:
            if self.serial and self.serial.is_open:
                try:
                    line = self.serial.readline().decode('ascii', errors='ignore').strip()
                    if line.startswith("FB:"):
                        match = pattern.search(line)
                        if match:
                            p = [int(match.group(i)) for i in range(5, 7)]
                            self.motor_actual_positions = p
                except:pass
            time.sleep(0.002)
            
    def read_state(self) -> np.ndarray:
        """
        Read motor positions and speeds
        
        Returns:
            state: [m1_pos, m2_pos, m1_spd, m2_spd]
        """
        return np.concatenate([self.motor_positions, self.motor_speeds])
    
    def stop(self):
        """Emergency stop"""
        self.send_command(np.zeros(2))
        
    def close(self):
        """Close connection"""
        self.stop()
        if self.connected:
            self.serial.close()
            self.connected = False


class TrajectoryBuffer:
    """Circular buffer for target trajectory"""
    
    def __init__(self, history_length: int = 10):
        self.history_length = history_length
        self.buffer = np.zeros((history_length, 4))  # [center_x, center_y, bbox_w, bbox_h]
        self.timestamps = np.zeros(history_length)
        self.write_idx = 0
        self.count = 0
        
    def add(self, center: np.ndarray, bbox: np.ndarray, timestamp: float):
        """Add new observation"""
        self.buffer[self.write_idx] = np.array([
            center[0], center[1], bbox[2], bbox[3]
        ])
        self.timestamps[self.write_idx] = timestamp
        self.write_idx = (self.write_idx + 1) % self.history_length
        self.count = min(self.count + 1, self.history_length)
        
    def get_trajectory(self) -> np.ndarray:
        """Get trajectory in temporal order"""
        if self.count < self.history_length:
            # Pad with last observation
            traj = np.zeros((self.history_length, 4))
            if self.count > 0:
                for i in range(self.history_length):
                    idx = min(i, self.count - 1)
                    traj[i] = self.buffer[idx]
            return traj
        else:
            # Full buffer - reorder
            idx = np.arange(self.write_idx, self.write_idx + self.history_length) % self.history_length
            return self.buffer[idx]
            
    def clear(self):
        """Clear buffer"""
        self.buffer[:] = 0
        self.timestamps[:] = 0
        self.write_idx = 0
        self.count = 0


class DataLogger:
    """Log control data for later training"""
    
    def __init__(self, output_dir: str, config: Config):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.config = config
        
        # Create session directory
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        self.session_dir = self.output_dir / f"session_{timestamp}"
        self.session_dir.mkdir(parents=True, exist_ok=True)
        
        # Data storage
        self.data = []
        self.video_writer = None
        self.frame_count = 0
        
        # Start video writer
        self._init_video()
        
    def _init_video(self):
        """Initialize video writer"""
        video_path = self.session_dir / "video.mp4"
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        self.video_writer = cv2.VideoWriter(
            str(video_path),
            fourcc,
            self.config.camera.fps,
            (self.config.camera.width, self.config.camera.height)
        )
        
    def log(self, state: ControlState, frame: np.ndarray):
        """Log a single timestep"""
        # Log data
        entry = {
            "time_sec": state.timestamp,
            "rel_x": state.em_state[0],
            "rel_y": state.em_state[1],
            "rel_z": state.em_state[2],
            "abs_x": state.em_state[3],
            "abs_y": state.em_state[4],
            "abs_z": state.em_state[5],
            "qw": state.em_state[6],
            "qx": state.em_state[7],
            "qy": state.em_state[8],
            "qz": state.em_state[9],
            "m1_pos": state.motor_state[0],
            "m2_pos": state.motor_state[1],
            "m1_spd": state.action[0],
            "m2_spd": state.action[1],
            "target_x": state.target_center[0],
            "target_y": state.target_center[1],
            "bbox_w": state.target_bbox[2] if len(state.target_bbox) > 2 else 0,
            "bbox_h": state.target_bbox[3] if len(state.target_bbox) > 3 else 0,
            "error_x": state.image_error[0],
            "error_y": state.image_error[1]
        }
        self.data.append(entry)
        
        # Write video frame
        if self.video_writer is not None and frame is not None:
            if frame.shape[1] != self.config.camera.width or frame.shape[0] != self.config.camera.height:
                frame = cv2.resize(frame, (self.config.camera.width, self.config.camera.height))
            self.video_writer.write(frame)
            
        self.frame_count += 1
        
    def save(self):
        """Save logged data"""
        # Save CSV
        df = pd.DataFrame(self.data)
        csv_path = self.session_dir / "data.csv"
        df.to_csv(csv_path, index=False)
        
        # Close video
        if self.video_writer is not None:
            self.video_writer.release()
            
        # Save metadata
        meta = {
            "num_frames": self.frame_count,
            "duration_sec": self.data[-1]["time_sec"] if self.data else 0,
            "fps": self.config.camera.fps,
            "config": {
                "camera": {"width": self.config.camera.width, "height": self.config.camera.height},
                "robot": {"num_motors": self.config.robot.num_motors}
            }
        }
        with open(self.session_dir / "metadata.json", "w") as f:
            json.dump(meta, f, indent=2)
            
        print(f"[Logger] Saved {self.frame_count} frames to {self.session_dir}")


class EndoscopeController:
    """
    Main controller for real-time endoscope visual servoing
    
    Usage:
        controller = EndoscopeController(checkpoint_path="checkpoints/rl/final.pt")
        controller.run()
    """
    
    def __init__(
        self,
        checkpoint_path: str,
        config: Optional[Config] = None,
        camera_id: int = 0,
        em_tracker_port: str = "/dev/ttyUSB0",
        motor_port: str = "/dev/ttyUSB0",
        enable_logging: bool = True,
        log_dir: str = "./control_logs",
        visualize: bool = True
    ):
        self.config = config or get_config()
        self.visualize = visualize
        self.enable_logging = enable_logging
        
        # Device
        self.device = torch.device(self.config.training.device)
        
        # Load model
        print(f"[Controller] Loading model from {checkpoint_path}")
        self.model = self._load_model(checkpoint_path)
        self.model.eval()
        
        # Initialize hardware interfaces
        print("[Controller] Initializing hardware...")
        self.camera = cv2.VideoCapture(camera_id)
        self.camera.set(cv2.CAP_PROP_FRAME_WIDTH, self.config.camera.width)
        self.camera.set(cv2.CAP_PROP_FRAME_HEIGHT, self.config.camera.height)
        self.camera.set(cv2.CAP_PROP_FPS, self.config.camera.fps)
        
        self.em_tracker = EMTrackerInterface(self.config, em_tracker_port)
        self.motors = MotorController(self.config, motor_port)
        
        # Initialize components
        self.detector = TargetDetector(self.config)
        self.trajectory_buffer = TrajectoryBuffer(self.config.network.trajectory_history)
        
        # State
        self.running = False
        self.state = ControlState()
        self.start_time = 0
        
        # Data logging
        if enable_logging:
            self.logger = DataLogger(log_dir, self.config)
        else:
            self.logger = None
            
        # Control parameters
        self.control_rate = self.config.camera.fps  # Hz
        self.use_jacobian = True  # Use learned Jacobian for control
        
        print("[Controller] Initialization complete")
        
    def _load_model(self, checkpoint_path: str) -> EndoscopeTrackingNetwork:
        """Load trained model"""
        model = EndoscopeTrackingNetwork(self.config)
        
        if os.path.exists(checkpoint_path):
            checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
            if "model_state_dict" in checkpoint:
                model.load_state_dict(checkpoint["model_state_dict"])
            else:
                model.load_state_dict(checkpoint)
            print(f"[Controller] Loaded checkpoint from {checkpoint_path}")
        else:
            print(f"[Controller] WARNING: Checkpoint not found at {checkpoint_path}")
            print("[Controller] Using randomly initialized model")
            
        return model.to(self.device)
    
    def _preprocess_image(self, frame: np.ndarray) -> torch.Tensor:
        """Preprocess image for network"""
        # Resize if needed
        if frame.shape[:2] != (self.config.camera.height, self.config.camera.width):
            frame = cv2.resize(frame, (self.config.camera.width, self.config.camera.height))
            
        # Convert BGR to RGB
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Normalize
        frame = frame.astype(np.float32) / 255.0
        frame = (frame - np.array([0.485, 0.456, 0.406])) / np.array([0.229, 0.224, 0.225])
        
        # To tensor [1, 3, H, W]
        tensor = torch.from_numpy(frame).permute(2, 0, 1).unsqueeze(0).float()
        return tensor.to(self.device)
    
    def _get_action(self, frame: np.ndarray) -> Tuple[np.ndarray, Dict]:
        """
        Get action from neural network
        
        Returns:
            action: [m1_spd, m2_spd]
            info: Additional information (Jacobian, attention, etc.)
        """
        # Detect target
        center, bbox = self.detector.detect(frame)
        
        if center is None:
            # No target - stop or continue with last action
            return np.zeros(2), {"target_found": False}
            
        # Update state
        self.state.target_center = center
        self.state.target_bbox = bbox if bbox is not None else np.zeros(4)
        self.state.image_error = self.detector.compute_error(center)
        
        # Update trajectory buffer
        current_time = time.time() - self.start_time
        self.trajectory_buffer.add(center, self.state.target_bbox, current_time)
        
        # Read sensor states
        self.state.em_state = self.em_tracker.read()
        self.state.motor_state = self.motors.read_state()
        
        # Prepare network inputs
        with torch.no_grad():
            image_tensor = self._preprocess_image(frame)
            
            trajectory = torch.from_numpy(
                self.trajectory_buffer.get_trajectory()
            ).unsqueeze(0).float().to(self.device)
            
            em_state = torch.from_numpy(
                np.array(self.state.em_state)
            ).unsqueeze(0).float().to(self.device)
            
            motor_state = torch.from_numpy(
                np.array(self.state.motor_state)
            ).unsqueeze(0).float().to(self.device)
            
            image_error = torch.from_numpy(
                self.state.image_error
            ).unsqueeze(0).float().to(self.device)
            
            # Forward pass
            outputs = self.model(
                image=image_tensor,
                trajectory=trajectory,
                em_state=em_state,
                motor_state=motor_state,
                image_error=image_error
            )
            
            # Get action
            action_mean = outputs["action_mean"].cpu().numpy()[0]
            
            # Optionally use Jacobian-based control
            if self.use_jacobian and "jacobian" in outputs:
                jacobian = outputs["jacobian"].cpu().numpy()[0]
                # Combine learned action with Jacobian-based correction
                j_action = outputs.get("jacobian_action", outputs["action_mean"])
                j_action = j_action.cpu().numpy()[0]
                # Blend (0.7 learned + 0.3 Jacobian)
                action = 0.7 * action_mean + 0.3 * j_action
            else:
                action = action_mean
                
        # Clip to limits
        action = np.clip(action, self.config.robot.min_motor_speed, self.config.robot.max_motor_speed)
        
        info = {
            "target_found": True,
            "error": self.state.image_error,
            "jacobian": outputs.get("jacobian", None),
            "attention": outputs.get("attention", None)
        }
        
        return action, info
    
    def _draw_overlay(self, frame: np.ndarray, action: np.ndarray, info: Dict) -> np.ndarray:
        """Draw visualization overlay"""
        vis = frame.copy()
        
        # Image center crosshair
        cx, cy = self.config.camera.width // 2, self.config.camera.height // 2
        cv2.line(vis, (cx - 20, cy), (cx + 20, cy), (0, 255, 0), 2)
        cv2.line(vis, (cx, cy - 20), (cx, cy + 20), (0, 255, 0), 2)
        
        if info.get("target_found", False):
            # Target center
            tx, ty = int(self.state.target_center[0]), int(self.state.target_center[1])
            cv2.circle(vis, (tx, ty), 10, (0, 0, 255), 2)
            
            # Bounding box
            if self.state.target_bbox is not None and len(self.state.target_bbox) == 4:
                x, y, w, h = self.state.target_bbox.astype(int)
                cv2.rectangle(vis, (x, y), (x + w, y + h), (255, 0, 0), 2)
                
            # Error vector
            cv2.arrowedLine(vis, (cx, cy), (tx, ty), (255, 255, 0), 2)
            
            # Error text
            error = self.state.image_error
            cv2.putText(vis, f"Error: ({error[0]:.1f}, {error[1]:.1f})",
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        else:
            cv2.putText(vis, "Target not found", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            
        # Action
        cv2.putText(vis, f"Action: [{action[0]:.1f}, {action[1]:.1f}]",
                   (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # FPS
        elapsed = time.time() - self.start_time
        if elapsed > 0:
            fps = self.logger.frame_count / elapsed if self.logger else 0
            cv2.putText(vis, f"FPS: {fps:.1f}", (10, 90),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
        return vis
    
    def run(self, duration: float = None):
        """
        Run control loop
        
        Args:
            duration: Run for this many seconds (None = indefinitely)
        """
        print("[Controller] Starting control loop (press 'q' to quit)")
        
        self.running = True
        self.start_time = time.time()
        
        try:
            while not rospy.is_shutdown() and self.running:
                loop_start = time.time()
                
                # Check duration
                if duration is not None and (loop_start - self.start_time) > duration:
                    print("[Controller] Duration reached, stopping")
                    break
                    
                # Capture frame
                ret, frame = self.camera.read()
                if not ret:
                    print("[Controller] Failed to read camera frame")
                    continue
                    
                # Get action
                action, info = self._get_action(frame)
                
                # Update state
                self.state.action = action
                self.state.timestamp = loop_start - self.start_time
                
                # Send to motors
                self.motors.send_command(-action)
                
                # Log data
                if self.logger:
                    self.logger.log(self.state, frame)
                    
                # Visualization
                if self.visualize:
                    vis_frame = self._draw_overlay(frame, action, info)
                    cv2.imshow("Endoscope Controller", vis_frame)
                    
                    key = cv2.waitKey(1) & 0xFF
                    if key == ord('q'):
                        print("[Controller] Quit requested")
                        break
                    elif key == ord('s'):
                        # Save snapshot
                        cv2.imwrite(f"snapshot_{int(time.time())}.png", frame)
                        print("[Controller] Snapshot saved")
                    elif key == ord('j'):
                        # Toggle Jacobian control
                        self.use_jacobian = not self.use_jacobian
                        print(f"[Controller] Jacobian control: {self.use_jacobian}")
                        
                # Rate limiting
                elapsed = time.time() - loop_start
                sleep_time = 1.0 / self.control_rate - elapsed
                if sleep_time > 0:
                    time.sleep(sleep_time)
                
                time.sleep(0.001)
                    
        except KeyboardInterrupt:
            print("[Controller] Interrupted")
        finally:
            self.stop()
            
    def stop(self):
        """Stop controller and cleanup"""
        print("[Controller] Stopping...")
        
        self.running = False
        
        # Stop motors
        self.motors.stop()
        
        # Save log
        if self.logger:
            self.logger.save()
            
        # Cleanup
        self.camera.release()
        self.motors.close()
        self.em_tracker.close()
        cv2.destroyAllWindows()
        
        print("[Controller] Stopped")


class XboxTeleopController:
    """
    Xbox controller teleoperation for data collection
    Based on user's lerobot_dataset_2motor.py pattern
    """
    
    def __init__(
        self,
        config: Optional[Config] = None,
        camera_id: int = 0,
        em_tracker_port: str = "/dev/ttyUSB0",
        motor_port: str = "/dev/ttyUSB0",
        log_dir: str = "./teleop_data"
    ):
        self.config = config or get_config()
        
        # Initialize hardware
        self.camera = cv2.VideoCapture(camera_id)
        self.camera.set(cv2.CAP_PROP_FRAME_WIDTH, self.config.camera.width)
        self.camera.set(cv2.CAP_PROP_FRAME_HEIGHT, self.config.camera.height)
        
        self.em_tracker = EMTrackerInterface(self.config, em_tracker_port)
        self.motors = MotorController(self.config, motor_port)
        self.detector = TargetDetector(self.config)
        
        # Logging
        self.logger = DataLogger(log_dir, self.config)
        
        # Xbox controller (requires inputs library)
        self.joystick = None
        self._init_joystick()
        
        # State
        self.running = False
        self.state = ControlState()
        self.start_time = 0
        
    def _init_joystick(self):
        """Initialize Xbox controller"""
        try:
            import pygame
            pygame.init()
            pygame.joystick.init()
            
            if pygame.joystick.get_count() > 0:
                self.joystick = pygame.joystick.Joystick(0)
                self.joystick.init()
                print(f"[Xbox] Controller connected: {self.joystick.get_name()}")
            else:
                print("[Xbox] No controller found, using keyboard fallback")
        except ImportError:
            print("[Xbox] pygame not installed, using keyboard fallback")
            
    def _read_joystick(self) -> Tuple[float, float]:
        """Read joystick values"""
        if self.joystick is None:
            return 0.0, 0.0
            
        try:
            import pygame
            pygame.event.pump()
            
            # Left stick for motor control
            # Axis 0: Left/Right (-1 to 1)
            # Axis 1: Up/Down (-1 to 1)
            lr = self.joystick.get_axis(0)  # -1 (left) to 1 (right)
            ud = -self.joystick.get_axis(1)  # Invert: -1 (down) to 1 (up)
            
            # Deadzone
            if abs(lr) < 0.1:
                lr = 0.0
            if abs(ud) < 0.1:
                ud = 0.0
                
            return lr, ud
        except:
            return 0.0, 0.0
            
    def run(self, duration: float = None):
        """Run teleoperation loop"""
        print("[Teleop] Starting (press 'q' to quit, stick to control)")
        
        self.running = True
        self.start_time = time.time()
        
        try:
            while self.running:
                loop_start = time.time()
                
                if duration is not None and (loop_start - self.start_time) > duration:
                    break
                    
                # Read frame
                ret, frame = self.camera.read()
                if not ret:
                    continue
                    
                # Read joystick
                lr, ud = self._read_joystick()
                
                # Map to motor speeds (scaled by max speed)
                action = np.array([
                    lr * self.config.robot.max_motor_speed,
                    ud * self.config.robot.max_motor_speed
                ])
                
                # Detect target
                center, bbox = self.detector.detect(frame)
                
                # Update state
                self.state.timestamp = loop_start - self.start_time
                self.state.em_state = self.em_tracker.read()
                self.state.motor_state = self.motors.read_state()
                self.state.target_center = center if center is not None else np.zeros(2)
                self.state.target_bbox = bbox if bbox is not None else np.zeros(4)
                self.state.image_error = self.detector.compute_error(center)
                self.state.action = action
                
                # Send to motors
                self.motors.send_command(-action)
                
                # Log
                self.logger.log(self.state, frame)
                
                # Visualize
                vis = frame.copy()
                cx, cy = self.config.camera.width // 2, self.config.camera.height // 2
                cv2.line(vis, (cx - 20, cy), (cx + 20, cy), (0, 255, 0), 2)
                cv2.line(vis, (cx, cy - 20), (cx, cy + 20), (0, 255, 0), 2)
                
                if center is not None:
                    tx, ty = int(center[0]), int(center[1])
                    cv2.circle(vis, (tx, ty), 8, (0, 0, 255), -1)
                    
                cv2.putText(vis, f"Joy: ({lr:.2f}, {ud:.2f})", (10, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                cv2.putText(vis, f"Motor: [{action[0]:.1f}, {action[1]:.1f}]", (10, 60),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                           
                cv2.imshow("Teleoperation", vis)
                
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                    
                # Rate limit
                elapsed = time.time() - loop_start
                sleep_time = 1.0 / self.config.camera.fps - elapsed
                if sleep_time > 0:
                    time.sleep(sleep_time)
                    
        except KeyboardInterrupt:
            pass
        finally:
            self.stop()
            
    def stop(self):
        """Stop and cleanup"""
        self.running = False
        self.motors.stop()
        self.logger.save()
        self.camera.release()
        cv2.destroyAllWindows()


def main():
    """Command-line interface for hardware control"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Endoscope Hardware Control")
    parser.add_argument("--mode", type=str, choices=["control", "teleop"], default="control",
                       help="control: neural network control, teleop: Xbox teleoperation")
    parser.add_argument("--checkpoint", type=str, default="./checkpoints/rl/final.pt",
                       help="Path to model checkpoint (for control mode)")
    parser.add_argument("--camera", type=int, default=0, help="Camera device ID")
    parser.add_argument("--em-port", type=str, default="/dev/ttyUSB0", help="EM tracker port")
    parser.add_argument("--motor-port", type=str, default="/dev/ttyUSB1", help="Motor controller port")
    parser.add_argument("--duration", type=float, default=None, help="Run duration in seconds")
    parser.add_argument("--log-dir", type=str, default="./control_logs", help="Data logging directory")
    parser.add_argument("--no-visualize", action="store_true", help="Disable visualization")
    parser.add_argument("--no-logging", action="store_true", help="Disable data logging")
    
    args = parser.parse_args()
    
    config = get_config()

    rospy.init_node("integrated_controller", anonymous=True)
    
    if args.mode == "control":
        # Neural network control
        controller = EndoscopeController(
            checkpoint_path=args.checkpoint,
            config=config,
            camera_id=args.camera,
            em_tracker_port=args.em_port,
            motor_port=args.motor_port,
            enable_logging=not args.no_logging,
            log_dir=args.log_dir,
            visualize=not args.no_visualize
        )
        controller.run(duration=args.duration)
        
    elif args.mode == "teleop":
        # Xbox teleoperation
        teleop = XboxTeleopController(
            config=config,
            camera_id=args.camera,
            em_tracker_port=args.em_port,
            motor_port=args.motor_port,
            log_dir=args.log_dir
        )
        teleop.run(duration=args.duration)


if __name__ == "__main__":
    main()
