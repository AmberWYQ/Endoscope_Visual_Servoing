"""
Physics-Based Simulator for Continuum Robot Endoscope
Used for PINN (Physics-Informed Neural Network) training

This simulator models:
1. Continuum robot kinematics (constant curvature model)
2. Motor-to-bending mapping
3. Camera projection
4. Target dynamics
5. EM tracker simulation
"""

import numpy as np
import torch
from typing import Dict, Tuple, Optional, List
from dataclasses import dataclass
import gymnasium as gym
from gymnasium import spaces
import cv2
import os

# Fix Qt platform issue for headless servers
os.environ['QT_QPA_PLATFORM'] = 'offscreen'
cv2.setNumThreads(0)


@dataclass
class EndoscopeState:
    """Complete state of the endoscope"""
    # EM tracker readings
    rel_position: np.ndarray  # [3] relative position
    abs_position: np.ndarray  # [3] absolute position
    orientation: np.ndarray   # [4] quaternion (w, x, y, z)
    
    # Motor states
    motor_positions: np.ndarray  # [2] current positions
    motor_velocities: np.ndarray  # [2] current velocities
    
    # Derived states
    tip_position: np.ndarray  # [3] tip position in world frame
    bending_angles: np.ndarray  # [2] bending angles (theta, phi)


class ContinuumRobotKinematics:
    """
    Kinematics model for continuum robot using constant curvature assumption
    The bendable segment can bend in 2 DoF (controlled by 2 motors)
    """
    def __init__(self, 
                 segment_length: float = 30.0,  # mm
                 backbone_length: float = 200.0,  # mm
                 motor_to_curvature: List[float] = [0.01, 0.01]):
        self.L = segment_length
        self.total_length = backbone_length
        self.rigid_length = backbone_length - segment_length
        self.k_motor = np.array(motor_to_curvature)
        
    def motor_to_bending(self, motor_positions: np.ndarray) -> np.ndarray:
        """
        Map motor positions to bending angles
        
        Args:
            motor_positions: [2] motor positions
        Returns:
            bending_angles: [2] (theta, phi) where:
                - theta: bending magnitude
                - phi: bending plane angle
        """
        # Simple linear mapping (can be made more complex)
        curvatures = self.k_motor * motor_positions
        
        # Convert to theta, phi representation
        kappa = np.linalg.norm(curvatures)
        phi = np.arctan2(curvatures[1], curvatures[0]) if kappa > 1e-6 else 0.0
        theta = kappa * self.L
        
        return np.array([theta, phi])
    
    def forward_kinematics(self, bending_angles: np.ndarray, 
                           base_position: np.ndarray = None,
                           base_orientation: np.ndarray = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute tip position and orientation using constant curvature model
        
        Args:
            bending_angles: [2] (theta, phi)
            base_position: [3] base position
            base_orientation: [4] base quaternion
        Returns:
            tip_position: [3]
            tip_orientation: [4] quaternion
        """
        if base_position is None:
            base_position = np.array([0.0, 0.0, 0.0])
        if base_orientation is None:
            base_orientation = np.array([1.0, 0.0, 0.0, 0.0])
            
        theta, phi = bending_angles
        
        # Handle small theta (nearly straight)
        if abs(theta) < 1e-6:
            # Tip position (along z-axis from base)
            tip_local = np.array([0, 0, self.L])
        else:
            # Constant curvature model
            kappa = theta / self.L
            
            # Tip position in local frame
            x = (1 - np.cos(theta)) / kappa * np.cos(phi)
            y = (1 - np.cos(theta)) / kappa * np.sin(phi)
            z = np.sin(theta) / kappa
            
            tip_local = np.array([x, y, z])
        
        # Transform by base orientation
        tip_position = base_position + self._rotate_by_quaternion(tip_local, base_orientation)
        
        # Compute tip orientation (apply bending rotation)
        bending_quat = self._axis_angle_to_quaternion(
            np.array([np.cos(phi), np.sin(phi), 0]) * theta
        )
        tip_orientation = self._quaternion_multiply(base_orientation, bending_quat)
        
        return tip_position, tip_orientation
    
    def compute_jacobian(self, motor_positions: np.ndarray, 
                         delta: float = 0.01) -> np.ndarray:
        """
        Compute numerical Jacobian: d(tip_position) / d(motor_position)
        
        Args:
            motor_positions: [2] current motor positions
            delta: perturbation for numerical differentiation
        Returns:
            J: [3, 2] Jacobian matrix
        """
        J = np.zeros((3, 2))
        
        # Current tip position
        bending = self.motor_to_bending(motor_positions)
        pos0, _ = self.forward_kinematics(bending)
        
        for i in range(2):
            motor_perturbed = motor_positions.copy()
            motor_perturbed[i] += delta
            bending_perturbed = self.motor_to_bending(motor_perturbed)
            pos_perturbed, _ = self.forward_kinematics(bending_perturbed)
            
            J[:, i] = (pos_perturbed - pos0) / delta
            
        return J
    
    def compute_image_jacobian(self, motor_positions: np.ndarray,
                               camera_matrix: np.ndarray,
                               tip_depth: float = 50.0,
                               delta: float = 0.01) -> np.ndarray:
        """
        Compute Jacobian from motor positions to image coordinates
        J_image = d(pixel_xy) / d(motor_position)
        
        Args:
            motor_positions: [2] current motor positions
            camera_matrix: [3, 3] camera intrinsics
            tip_depth: depth of tip from camera
            delta: perturbation
        Returns:
            J_image: [2, 2] image Jacobian
        """
        fx, fy = camera_matrix[0, 0], camera_matrix[1, 1]
        cx, cy = camera_matrix[0, 2], camera_matrix[1, 2]
        
        J_image = np.zeros((2, 2))
        
        # Current projection
        bending = self.motor_to_bending(motor_positions)
        pos0, _ = self.forward_kinematics(bending)
        uv0 = np.array([fx * pos0[0] / tip_depth + cx, fy * pos0[1] / tip_depth + cy])
        
        for i in range(2):
            motor_perturbed = motor_positions.copy()
            motor_perturbed[i] += delta
            bending_perturbed = self.motor_to_bending(motor_perturbed)
            pos_perturbed, _ = self.forward_kinematics(bending_perturbed)
            uv_perturbed = np.array([
                fx * pos_perturbed[0] / tip_depth + cx,
                fy * pos_perturbed[1] / tip_depth + cy
            ])
            
            J_image[:, i] = (uv_perturbed - uv0) / delta
            
        return J_image
    
    @staticmethod
    def _rotate_by_quaternion(v: np.ndarray, q: np.ndarray) -> np.ndarray:
        """Rotate vector v by quaternion q"""
        w, x, y, z = q
        # Quaternion rotation: q * v * q^-1
        t = 2 * np.cross(np.array([x, y, z]), v)
        return v + w * t + np.cross(np.array([x, y, z]), t)
    
    @staticmethod
    def _quaternion_multiply(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
        """Multiply two quaternions"""
        w1, x1, y1, z1 = q1
        w2, x2, y2, z2 = q2
        return np.array([
            w1*w2 - x1*x2 - y1*y2 - z1*z2,
            w1*x2 + x1*w2 + y1*z2 - z1*y2,
            w1*y2 - x1*z2 + y1*w2 + z1*x2,
            w1*z2 + x1*y2 - y1*x2 + z1*w2
        ])
    
    @staticmethod
    def _axis_angle_to_quaternion(axis_angle: np.ndarray) -> np.ndarray:
        """Convert axis-angle to quaternion"""
        angle = np.linalg.norm(axis_angle)
        if angle < 1e-6:
            return np.array([1.0, 0.0, 0.0, 0.0])
        axis = axis_angle / angle
        return np.array([
            np.cos(angle / 2),
            axis[0] * np.sin(angle / 2),
            axis[1] * np.sin(angle / 2),
            axis[2] * np.sin(angle / 2)
        ])


class TargetDynamics:
    """
    Simulate dynamic target motion in 3D space
    """
    def __init__(self, motion_type: str = "sinusoidal",
                 speed_range: Tuple[float, float] = (1.0, 5.0),
                 bounds: Tuple[float, float] = (-20.0, 20.0)):
        self.motion_type = motion_type
        self.speed_range = speed_range
        self.bounds = bounds
        self.reset()
        
    def reset(self):
        """Reset target to random initial state"""
        self.position = np.random.uniform(-10, 10, size=3)
        self.position[2] = np.random.uniform(40, 80)  # Keep depth positive
        self.velocity = np.zeros(3)
        self.time = 0.0
        
        # Motion parameters
        self.frequency = np.random.uniform(0.1, 0.5, size=3)
        self.amplitude = np.random.uniform(5, 15, size=3)
        self.phase = np.random.uniform(0, 2*np.pi, size=3)
        
    def step(self, dt: float) -> Tuple[np.ndarray, np.ndarray]:
        """
        Update target position
        Returns: (position, velocity)
        """
        self.time += dt
        
        if self.motion_type == "sinusoidal":
            # Sinusoidal motion
            new_pos = self.amplitude * np.sin(2 * np.pi * self.frequency * self.time + self.phase)
            new_pos[2] += 60  # Base depth
            self.velocity = (new_pos - self.position) / dt
            self.position = new_pos
            
        elif self.motion_type == "circular":
            # Circular motion in XY plane
            radius = 15.0
            omega = 0.5
            self.position[0] = radius * np.cos(omega * self.time)
            self.position[1] = radius * np.sin(omega * self.time)
            self.velocity[0] = -radius * omega * np.sin(omega * self.time)
            self.velocity[1] = radius * omega * np.cos(omega * self.time)
            
        elif self.motion_type == "random":
            # Random walk with momentum
            self.velocity += np.random.randn(3) * 0.5
            self.velocity = np.clip(self.velocity, -self.speed_range[1], self.speed_range[1])
            self.position += self.velocity * dt
            self.position = np.clip(self.position, self.bounds[0], self.bounds[1])
            self.position[2] = np.clip(self.position[2], 30, 100)
            
        return self.position.copy(), self.velocity.copy()


class EndoscopeSimulator(gym.Env):
    """
    Gymnasium environment for endoscope visual servoing simulation
    """
    metadata = {'render_modes': ['rgb_array', 'human'], 'render_fps': 20}
    
    def __init__(self, config, render_mode: str = "rgb_array"):
        super().__init__()
        self.config = config
        self.render_mode = render_mode
        
        # Initialize components
        self.kinematics = ContinuumRobotKinematics(
            segment_length=config.robot.segment_length,
            backbone_length=config.robot.backbone_length,
            motor_to_curvature=config.robot.motor_to_curvature_gain
        )
        
        self.target = TargetDynamics(
            motion_type=config.simulator.target_motion_type,
            speed_range=config.simulator.target_speed_range
        )
        
        # Camera parameters
        self.img_width = config.camera.width
        self.img_height = config.camera.height
        self.camera_matrix = np.array([
            [500, 0, self.img_width / 2],
            [0, 500, self.img_height / 2],
            [0, 0, 1]
        ], dtype=np.float32)
        
        # State
        self.motor_positions = np.zeros(2)
        self.motor_velocities = np.zeros(2)
        self.base_position = np.zeros(3)
        self.base_orientation = np.array([1.0, 0.0, 0.0, 0.0])
        
        # Trajectory history
        self.trajectory_history = []
        self.history_length = config.network.trajectory_history
        
        # Timing
        self.dt = config.simulator.dt
        self.max_steps = config.simulator.max_episode_steps
        self.current_step = 0
        
        # Action and observation spaces
        self.action_space = spaces.Box(
            low=-config.robot.max_motor_speed,
            high=config.robot.max_motor_speed,
            shape=(2,),
            dtype=np.float32
        )
        
        # Observation: image + state
        self.observation_space = spaces.Dict({
            'image': spaces.Box(0, 255, (self.img_height, self.img_width, 3), dtype=np.uint8),
            'em_state': spaces.Box(-np.inf, np.inf, (10,), dtype=np.float32),
            'motor_state': spaces.Box(-np.inf, np.inf, (4,), dtype=np.float32),
            'trajectory': spaces.Box(-np.inf, np.inf, (self.history_length, 4), dtype=np.float32),
            'image_error': spaces.Box(-np.inf, np.inf, (2,), dtype=np.float32),
            'target_position': spaces.Box(-np.inf, np.inf, (3,), dtype=np.float32),
        })
        
        # For rendering
        self.last_image = None
        
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        # Reset state
        self.motor_positions = np.zeros(2)
        self.motor_velocities = np.zeros(2)
        self.base_position = np.zeros(3)
        self.base_orientation = np.array([1.0, 0.0, 0.0, 0.0])
        
        # Reset target
        self.target.reset()
        
        # Reset history
        self.trajectory_history = []
        
        # Reset timing
        self.current_step = 0
        
        obs = self._get_observation()
        info = self._get_info()
        
        return obs, info
    
    def step(self, action: np.ndarray):
        # Clip action
        action = np.clip(action, -self.config.robot.max_motor_speed, 
                        self.config.robot.max_motor_speed)
        
        # Apply motor dynamics
        self.motor_velocities = action.copy()
        self.motor_positions += self.motor_velocities * self.dt
        
        # Clip motor positions
        self.motor_positions = np.clip(self.motor_positions, -1000, 1000)
        
        # Add noise
        if self.config.simulator.motor_noise_std > 0:
            self.motor_velocities += np.random.randn(2) * self.config.simulator.motor_noise_std
        
        # Update target
        self.target.step(self.dt)
        
        # Get observation
        obs = self._get_observation()
        
        # Compute reward
        reward = self._compute_reward(action)
        
        # Check termination
        self.current_step += 1
        terminated = False
        truncated = self.current_step >= self.max_steps
        
        info = self._get_info()
        
        return obs, reward, terminated, truncated, info
    
    def _get_observation(self) -> Dict:
        # Update kinematics
        bending = self.kinematics.motor_to_bending(self.motor_positions)
        tip_position, tip_orientation = self.kinematics.forward_kinematics(
            bending, self.base_position, self.base_orientation
        )
        
        # EM tracker state (with noise)
        rel_position = tip_position - self.base_position
        abs_position = tip_position
        
        if self.config.simulator.em_noise_std > 0:
            rel_position += np.random.randn(3) * self.config.simulator.em_noise_std
            abs_position += np.random.randn(3) * self.config.simulator.em_noise_std
        
        em_state = np.concatenate([rel_position, abs_position, tip_orientation]).astype(np.float32)
        
        # Motor state
        motor_state = np.concatenate([
            self.motor_positions, self.motor_velocities
        ]).astype(np.float32)
        
        # Render image
        image = self._render_image(tip_position)
        
        # Compute target in image
        target_uv, target_size = self._project_target()
        
        # Update trajectory history
        self.trajectory_history.append([
            target_uv[0], target_uv[1], target_size, target_size
        ])
        if len(self.trajectory_history) > self.history_length:
            self.trajectory_history.pop(0)
        
        # Pad trajectory if needed
        while len(self.trajectory_history) < self.history_length:
            self.trajectory_history.insert(0, self.trajectory_history[0] if self.trajectory_history else [320, 240, 20, 20])
        
        trajectory = np.array(self.trajectory_history, dtype=np.float32)
        
        # Image error (target - center)
        center = np.array([self.img_width / 2, self.img_height / 2])
        image_error = (target_uv - center).astype(np.float32)
        
        return {
            'image': image,
            'em_state': em_state,
            'motor_state': motor_state,
            'trajectory': trajectory,
            'image_error': image_error,
            'target_position': self.target.position.astype(np.float32)
        }
    
    def _project_target(self) -> Tuple[np.ndarray, float]:
        """Project 3D target to image coordinates"""
        pos = self.target.position
        
        # Add detection noise
        if self.config.simulator.target_detection_noise_std > 0:
            pos = pos + np.random.randn(3) * self.config.simulator.target_detection_noise_std * 0.1
        
        # Project
        if pos[2] > 0:
            u = self.camera_matrix[0, 0] * pos[0] / pos[2] + self.camera_matrix[0, 2]
            v = self.camera_matrix[1, 1] * pos[1] / pos[2] + self.camera_matrix[1, 2]
            
            # Size based on depth (closer = larger)
            size = 500 / pos[2]
            
            # Add pixel noise
            if self.config.simulator.target_detection_noise_std > 0:
                u += np.random.randn() * self.config.simulator.target_detection_noise_std
                v += np.random.randn() * self.config.simulator.target_detection_noise_std
        else:
            u, v = self.img_width / 2, self.img_height / 2
            size = 10
            
        return np.array([u, v]), size
    
    def _render_image(self, tip_position: np.ndarray) -> np.ndarray:
        """Render synthetic endoscopic image"""
        # Create background (tissue-like texture)
        image = np.random.randint(40, 80, (self.img_height, self.img_width, 3), dtype=np.uint8)
        image[:, :, 0] = np.clip(image[:, :, 0] + 40, 0, 255)  # Reddish tint
        
        # Add some texture
        for _ in range(20):
            cx = np.random.randint(0, self.img_width)
            cy = np.random.randint(0, self.img_height)
            radius = np.random.randint(5, 30)
            color = (np.random.randint(60, 100), np.random.randint(30, 60), np.random.randint(30, 60))
            cv2.circle(image, (cx, cy), radius, color, -1)
        
        # Apply slight blur for realism
        image = cv2.GaussianBlur(image, (5, 5), 1)
        
        # Draw target (black circle)
        target_uv, target_size = self._project_target()
        target_uv = np.clip(target_uv, 0, [self.img_width-1, self.img_height-1]).astype(int)
        target_size = int(np.clip(target_size, 5, 50))
        
        cv2.circle(image, tuple(target_uv), target_size, (10, 10, 10), -1)
        
        # Draw center crosshair
        cx, cy = self.img_width // 2, self.img_height // 2
        cv2.line(image, (cx - 10, cy), (cx + 10, cy), (0, 255, 0), 1)
        cv2.line(image, (cx, cy - 10), (cx, cy + 10), (0, 255, 0), 1)
        
        self.last_image = image
        return image
    
    def _compute_reward(self, action: np.ndarray) -> float:
        """Compute reward based on tracking error and smoothness"""
        # Get current error
        target_uv, _ = self._project_target()
        center = np.array([self.img_width / 2, self.img_height / 2])
        error = np.linalg.norm(target_uv - center)
        
        # Normalize error
        max_error = np.sqrt(self.img_width**2 + self.img_height**2) / 2
        normalized_error = error / max_error
        
        # Center reward (exponential decay)
        reward_center = np.exp(-5 * normalized_error) * self.config.simulator.reward_center_weight
        
        # Smoothness penalty
        action_magnitude = np.linalg.norm(action)
        reward_smooth = -action_magnitude / 100 * self.config.simulator.reward_smooth_weight
        
        # Velocity matching (try to match target velocity)
        target_vel = self.target.velocity[:2]  # XY velocity
        reward_velocity = -np.linalg.norm(target_vel) * 0.001 * self.config.simulator.reward_velocity_weight
        
        return reward_center + reward_smooth + reward_velocity
    
    def _get_info(self) -> Dict:
        """Get additional info"""
        target_uv, target_size = self._project_target()
        center = np.array([self.img_width / 2, self.img_height / 2])
        
        # Compute Jacobian
        jacobian = self.kinematics.compute_image_jacobian(
            self.motor_positions, self.camera_matrix, self.target.position[2]
        )
        
        return {
            'tracking_error': np.linalg.norm(target_uv - center),
            'target_uv': target_uv,
            'target_size': target_size,
            'jacobian_gt': jacobian,
            'target_velocity': self.target.velocity.copy(),
            'motor_positions': self.motor_positions.copy()
        }
    
    def render(self):
        if self.render_mode == "rgb_array":
            return self.last_image
        return None
    
    def get_ground_truth_jacobian(self) -> np.ndarray:
        """Get ground truth Jacobian for PINN supervision"""
        return self.kinematics.compute_image_jacobian(
            self.motor_positions, self.camera_matrix, self.target.position[2]
        )


class PINNDataCollector:
    """
    Collect data for PINN training from simulator
    """
    def __init__(self, env: EndoscopeSimulator, num_points: int = 10000):
        self.env = env
        self.num_points = num_points
        
    def collect(self) -> Dict[str, np.ndarray]:
        """Collect physics data for PINN training"""
        data = {
            'motor_positions': [],
            'bending_angles': [],
            'tip_positions': [],
            'jacobians': [],
            'image_jacobians': [],
            'em_states': [],
            'target_uvs': [],
            'target_sizes': []
        }
        
        for _ in range(self.num_points):
            # Random motor positions
            motor_pos = np.random.uniform(-500, 500, size=2)
            
            # Compute kinematics
            bending = self.env.kinematics.motor_to_bending(motor_pos)
            tip_pos, tip_orient = self.env.kinematics.forward_kinematics(bending)
            jacobian = self.env.kinematics.compute_jacobian(motor_pos)
            
            # Random target depth
            depth = np.random.uniform(30, 100)
            image_jacobian = self.env.kinematics.compute_image_jacobian(
                motor_pos, self.env.camera_matrix, depth
            )
            
            # EM state
            em_state = np.concatenate([tip_pos, tip_pos, tip_orient])
            
            # Random target
            target_xy = np.random.uniform(-30, 30, size=2)
            target_uv = self.env.camera_matrix[:2, :2] @ target_xy / depth + self.env.camera_matrix[:2, 2]
            target_size = 500 / depth
            
            data['motor_positions'].append(motor_pos)
            data['bending_angles'].append(bending)
            data['tip_positions'].append(tip_pos)
            data['jacobians'].append(jacobian)
            data['image_jacobians'].append(image_jacobian)
            data['em_states'].append(em_state)
            data['target_uvs'].append(target_uv)
            data['target_sizes'].append(target_size)
        
        return {k: np.array(v) for k, v in data.items()}


def make_env(config, render_mode="rgb_array"):
    """Factory function to create environment"""
    return EndoscopeSimulator(config, render_mode=render_mode)


if __name__ == "__main__":
    # Test the simulator
    import sys
    sys.path.append('/home/claude/endoscope_tracking')
    from configs.config import get_config
    
    config = get_config()
    env = make_env(config)
    
    obs, info = env.reset()
    print("Observation shapes:")
    for k, v in obs.items():
        if isinstance(v, np.ndarray):
            print(f"  {k}: {v.shape}")
    
    print("\nInfo:")
    for k, v in info.items():
        print(f"  {k}: {v}")
    
    # Run a few steps
    for i in range(10):
        action = env.action_space.sample()
        obs, reward, term, trunc, info = env.step(action)
        print(f"Step {i}: reward={reward:.4f}, error={info['tracking_error']:.2f}")
    
    print("\nSimulator test passed!")
