#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
YOLO-E Visual Servoing Control Main

Chains two pre-trained models:
1. YOLO-E (YOLO-World) for instruction-based target detection
2. Low-level autonomous control (action): Predicts 2-DoF bending commands from detection

Pipeline:
    Camera Frame → YOLO-E Detection (with text prompt) → BBox/Confidence → 
    Observation Construction → Low-Level Policy → 2-DoF Actions →
    Safety Filter → Motor Commands

Usage:
    # Simulation mode with mock models (for testing)
    python yoloe_control_main.py --mode simulation --mock-all --target "polyp"
    
    # Simulation mode with real YOLO-E model, bc_model controller (default)
    python yoloe_control_main.py --mode simulation \
        --yoloe-model ./checkpoints/best_blackpoint_base.pt \
        --target "black spot" \
        --confidence-threshold 0.01 \
        --control-checkpoint ./checkpoints/bc_model.pt

    # Use position-based proportional controller instead of bc_model
    python yoloe_control_main.py --mode simulation \
        --yoloe-model ./checkpoints/best_blackpoint_base.pt \
        --target "black spot" \
        --confidence-threshold 0.01 \
        --control-mode proportional \
        --p-gain-x 0.6 --p-gain-y 0.6

    # Run BOTH controllers simultaneously and compare outputs (bc_model is sent to robot)
    python yoloe_control_main.py --mode robot \
        --yoloe-model ./checkpoints/best_blackpoint_base.pt \
        --target "black spot" \
        --confidence-threshold 0.01 \
        --control-checkpoint ./checkpoints/bc_model.pt \
        --control-mode both \
        --p-gain-x 0.6 --p-gain-y 0.6

Controls:
    - 't' - Change target (enter new text prompt)
    - 'h' - Toggle heatmap visualization
    - 'Space' - Pause/resume control (robot mode)
    - 's' - Start/stop recording
    - 'q' - Quit
"""

import argparse
import sys
import os
import time
import threading
from datetime import datetime
from pathlib import Path
from typing import Optional, Tuple

import cv2
import numpy as np

# Try to import pygame for visualization
try:
    import pygame
    PYGAME_AVAILABLE = True
except ImportError:
    PYGAME_AVAILABLE = False
    print("[Warning] pygame not available, using OpenCV for display")

# Local imports
from yoloe_combined_config import IntegratedConfig, get_config, get_config_from_args
from yoloe_perception_interface import (
    YOLOEPerceptionInterface, MockYOLOEPerceptionInterface,
    DetectionResult, create_yoloe_perception_interface
)
from control_interface import (
    ControlInterface, MockControlInterface, ProportionalController,
    ControlAction, TrajectoryBuffer, create_control_interface
)
from safety_manager import SafetyManager, SafetyState, SafetyOutput
from data_recorder import DataRecorder, VisualizationRecorder


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description='YOLO-E Visual Servoing Control',
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    
    # Mode
    parser.add_argument('--mode', type=str, choices=['simulation', 'robot'], default='simulation',
                        help='Run mode: simulation (no robot) or robot (sends commands)')
    
    # YOLO-E model
    parser.add_argument('--yoloe-model', type=str, default='best_blackpoint_base.pt',
                        help='Path to YOLO-E model (pretrained or finetuned)')
    
    # Target prompt
    parser.add_argument('--target', type=str, default='polyp',
                        help='Text prompt for target detection (e.g., "polyp", "lesion")')
    
    # Control model checkpoint
    parser.add_argument('--control-checkpoint', type=str, default=None,
                        help='Path to low-level control model checkpoint')
    
    # Mock models for testing
    parser.add_argument('--mock-perception', action='store_true',
                        help='Use mock perception model')
    parser.add_argument('--mock-control', action='store_true',
                        help='Use mock control model')
    parser.add_argument('--mock-all', action='store_true',
                        help='Use all mock models (for testing)')
    
    # Control mode for comparison
    parser.add_argument('--control-mode', type=str,
                        choices=['bc_model', 'proportional', 'both'],
                        default='bc_model',
                        help=(
                            'Which controller generates the executed command. '
                            '"bc_model" uses the neural network (default). '
                            '"proportional" uses the position-based P-controller. '
                            '"both" runs both and sends the bc_model command, '
                            'but shows both outputs side-by-side for comparison.'
                        ))
    
    # Proportional controller gains (used when --control-mode is proportional or both)
    parser.add_argument('--p-gain-x', type=float, default=0.6,
                        help='Proportional gain on horizontal error (default: 0.6)')
    parser.add_argument('--p-gain-y', type=float, default=0.6,
                        help='Proportional gain on vertical error (default: 0.6)')
    
    # Detection configuration
    parser.add_argument('--confidence-threshold', type=float, default=0.25,
                        help='Detection confidence threshold')
    
    # Camera
    parser.add_argument('--camera-id', type=int, default=0,
                        help='Camera device ID')
    
    # Visualization
    parser.add_argument('--no-display', action='store_true',
                        help='Run without visualization (headless)')
    parser.add_argument('--show-heatmap', action='store_true',
                        help='Show attention heatmap (if available)')
    
    # Logging
    parser.add_argument('--output-dir', type=str, default='./control_logs',
                        help='Output directory for logs')
    parser.add_argument('--no-logging', action='store_true',
                        help='Disable data logging')
    
    return parser.parse_args()


class CameraCapture:
    """Threaded camera capture for smooth frame acquisition"""
    
    def __init__(self, camera_id: int = 0, width: int = 640, height: int = 480, fps: float = 30.0):
        self.camera_id = camera_id
        self.width = width
        self.height = height
        self.fps = fps
        self.cap = None
        self.frame = None
        self.running = False
        self.lock = threading.Lock()
    
    def start(self) -> bool:
        """Start camera capture"""
        self.cap = cv2.VideoCapture(self.camera_id)
        if not self.cap.isOpened():
            print(f"[Camera] Failed to open camera {self.camera_id}")
            return False
        
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        self.cap.set(cv2.CAP_PROP_FPS, self.fps)
        self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'))
        
        self.running = True
        self.thread = threading.Thread(target=self._capture_loop, daemon=True)
        self.thread.start()
        
        print(f"[Camera] Started: {self.width}x{self.height} @ {self.fps}Hz")
        return True
    
    def _capture_loop(self):
        """Continuous frame capture loop"""
        while self.running:
            ret, frame = self.cap.read()
            if ret:
                h, w, _ = frame.shape
                frame = frame[35:35 + 410, 140:140 + 480]
                frame = cv2.resize(frame, (self.width, self.height))
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                with self.lock:
                    self.frame = frame_rgb
            else:
                time.sleep(0.001)
    
    def get_frame(self) -> Optional[np.ndarray]:
        """Get latest frame (RGB)"""
        with self.lock:
            return self.frame.copy() if self.frame is not None else None
    
    def stop(self):
        """Stop camera capture"""
        self.running = False
        if self.cap:
            self.cap.release()
        print("[Camera] Stopped")


class SerialInterface:
    """Serial interface for motor control"""
    
    def __init__(self, port: str = '/dev/ttyUSB0', baud_rate: int = 115200, simulation: bool = True):
        self.port = port
        self.baud_rate = baud_rate
        self.simulation = simulation
        self.connected = False
        self.last_command = np.zeros(2)
        self.ser = None

        self.current_pos = np.zeros(2)      # 记录当前累积位置 (单位：步数或编码器值)
        self.max_limit = np.array([500, 500])   # 正向最大行程限制
        self.min_limit = np.array([-500, -500]) # 反向最大行程限制
        
        if not simulation:
            self._connect()
        else:
            print("[Serial] Running in SIMULATION mode (no real robot)")
            self.connected = True
    
    def _connect(self):
        """Connect to serial port"""
        try:
            import serial
            import serial.tools.list_ports
            
            ports = list(serial.tools.list_ports.comports())
            print(f"[Serial] Available ports: {[p.device for p in ports]}")
            
            self.ser = serial.Serial(self.port, self.baud_rate, timeout=0.1)
            self.connected = True
            print(f"[Serial] Connected to {self.port}")
        except Exception as e:
            print(f"[Serial] Failed to connect: {e}")
            self.connected = False
    
    def send_command(self, action: np.ndarray, max_speed: float = 1.0):
        """Send motor command"""
        self.last_command = action.copy()
        dt = 1.0 / 30 # 30hz
        delta = np.zeros(2)
        # max_speed = 100.0
        
        
        if self.simulation:
            return True
        
        if not self.connected or self.ser is None:
            return False
        
        try:
            m1 = int(action[0] * 30)
            m2 = int(action[1] * 30)
            m1 = max(-1000, min(1000, m1))
            m2 = max(-1000, min(1000, m2))

            delta[0] = m1 * dt
            delta[1] = m2 * dt
            predicted_pos = self.current_pos + delta
            # 电机运动到边界停止滑动
            safe_predicted_pos = np.clip(predicted_pos, self.min_limit, self.max_limit)
            actual_delta = safe_predicted_pos - self.current_pos
            # print(actual_delta)
            # m1 *= -2.5
            # m2 *= -7

            print(f"[Serial] Command: ({m1:.1f}, {m2:.1f})")

            cmd_str = f"{m1:>6d}{m2:>6d}{0:>6d}{0:>6d}"
            self.ser.write(cmd_str.encode('ascii'))

            self.current_pos += actual_delta
            
            return True
        except Exception as e:
            print(f"[Serial] Send error: {e}")
            return False
    
    def send_stop(self):
        """Send stop command"""
        return self.send_command(np.zeros(2))
    
    def close(self):
        """Close connection"""
        if not self.simulation and self.connected and self.ser is not None:
            self.send_stop()
            self.ser.close()
        print("[Serial] Closed")


class IntegratedController:
    """Main controller integrating YOLO-E perception, control, and safety"""
    
    def __init__(self, args):
        self.args = args
        self.config = get_config_from_args(args)
        
        # State
        self.running = True
        self.tracking_active = False
        self.paused = False
        self.show_heatmap = args.show_heatmap
        self.recording = False
        
        # Components
        self.perception = None
        self.control = None       # bc_model (or mock)
        self.control_p = None     # ProportionalController (used when control-mode != bc_model)
        self.safety = None
        self.recorder = None
        
        # Flags
        self.use_mock_perception = args.mock_all or args.mock_perception
        self.use_mock_control = args.mock_all or args.mock_control
        
        # Target name
        self.target_name = args.target
        
        # Initialize components
        self._init_camera()
        self._init_serial()
        self._init_display()
        self._init_safety()
        self._init_recorder()
        
        # Timing
        self.fps = 0.0
        self.frame_times = []
        self.motor_state = np.zeros(4)
        self.start_time = time.time()
        self.action_p = None   # Proportional controller output (comparison mode)
    
    def _init_camera(self):
        self.camera = CameraCapture(
            camera_id=self.args.camera_id,
            width=self.config.camera.width,
            height=self.config.camera.height,
            fps=self.config.camera.fps,
        )
        if not self.camera.start():
            print("[Error] Failed to start camera")
            sys.exit(1)
        time.sleep(0.5)
    
    def _init_serial(self):
        simulation = (self.args.mode == 'simulation')
        self.serial = SerialInterface(
            port=self.config.robot.motor_port,
            baud_rate=self.config.robot.baud_rate,
            simulation=simulation,
        )
    
    def _init_display(self):
        if self.args.no_display:
            self.screen = None
            self.clock = None
            return
        
        if PYGAME_AVAILABLE:
            pygame.init()
            pygame.display.set_caption("YOLO-E Visual Servoing")
            self.screen = pygame.display.set_mode(
                (self.config.visualization.ui_width, self.config.visualization.ui_height)
            )
            self.clock = pygame.time.Clock()
            self.fonts = (
                pygame.font.SysFont('monospace', 20, bold=True),
                pygame.font.SysFont('monospace', 16, bold=True),
                pygame.font.SysFont('monospace', 14),
            )
        else:
            self.screen = None
            self.clock = None
    
    def _init_safety(self):
        self.safety = SafetyManager(self.config)
    
    def _init_recorder(self):
        if not self.args.no_logging:
            self.recorder = DataRecorder(self.config)
        else:
            self.recorder = None
    
    def _init_models(self):
        """Initialize YOLO-E perception and control models"""
        if self.use_mock_perception:
            self.perception = MockYOLOEPerceptionInterface(target_classes=self.target_name)
        else:
            self.perception = YOLOEPerceptionInterface(
                model_path=self.args.yoloe_model,
                target_classes=self.target_name,
                confidence_threshold=self.args.confidence_threshold,
                device=self.config.device,
            )
        
        self.control = create_control_interface(self.config, use_mock=self.use_mock_control)
        
        # Initialise proportional controller when needed for comparison or as primary.
        # Uses the full camera resolution since the crop is resized back before inference.
        if self.args.control_mode in ('proportional', 'both'):
            self.control_p = ProportionalController(
                kp_x=self.args.p_gain_x,
                kp_y=self.args.p_gain_y,
                img_width=self.config.camera.width,
                img_height=self.config.camera.height,
            )
            print(f"[Control] ProportionalController initialised "
                  f"(kp_x={self.args.p_gain_x}, kp_y={self.args.p_gain_y})")
        
        print(f"[Control] Models initialized")
        print(f"[Control] Target: '{self.target_name}'")
    
    def _run_pipeline(self, frame: np.ndarray) -> Tuple[DetectionResult, ControlAction, SafetyOutput]:
        """Run the full perception -> control -> safety pipeline.

        Depending on --control-mode:
          - 'bc_model'      : only bc_model runs; action_p is None.
          - 'proportional'  : only P-controller runs; stored in self.control.
          - 'both'          : both run; bc_model command is sent, P-controller
                              command is stored in self.action_p for display.
        """
        timestamp = time.time() - self.start_time

        detection = self.perception.detect(frame, return_heatmap=self.show_heatmap)

        if self.paused:
            action = ControlAction(action=np.zeros(2), action_mean=np.zeros(2), valid=False)
            self.action_p = ControlAction(action=np.zeros(2), action_mean=np.zeros(2), valid=False)
        else:
            ctrl_mode = self.args.control_mode

            # --- bc_model action ---
            if ctrl_mode in ('bc_model', 'both'):
                action = self.control.compute_action(
                    frame=frame, detection_result=detection,
                    timestamp=timestamp, deterministic=True,
                )
            
            # --- proportional action ---
            if ctrl_mode in ('proportional', 'both'):
                action_p = self.control_p.compute_action(
                    frame=frame, detection_result=detection,
                    timestamp=timestamp,
                )
                self.action_p = action_p
            else:
                self.action_p = None

            # When 'proportional' is primary, promote it to 'action'
            if ctrl_mode == 'proportional':
                action = action_p

        # The action that is actually executed goes through the safety filter.
        # In 'both' mode this is always bc_model.
        flipped = action.action * np.array([-1.0, -1.0])

        safety_output = self.safety.process(
            raw_action=flipped,
            detection_valid=not detection.no_detection,
            confidence=detection.confidence,
            timestamp=timestamp,
        )

        return detection, action, safety_output
    
    def _send_command(self, safety_output: SafetyOutput):
        if self.args.mode == 'simulation':
            return
        if safety_output.is_safe and not self.paused:
            self.serial.send_command(safety_output.action, self.config.robot.max_motor_speed)
        else:
            self.serial.send_stop()
    
    def _handle_events(self):
        if self.screen is not None:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.running = False
                elif event.type == pygame.KEYDOWN:
                    self._handle_key(event.key)
        else:
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                self.running = False
            elif key == ord('h'):
                self.show_heatmap = not self.show_heatmap
            elif key == ord(' '):
                self.paused = not self.paused
            elif key == ord('s'):
                self._toggle_recording()
            elif key == ord('t'):
                self._change_target()
    
    def _handle_key(self, key):
        if key == pygame.K_q or key == pygame.K_ESCAPE:
            self.running = False
        elif key == pygame.K_h:
            self.show_heatmap = not self.show_heatmap
            print(f"[Control] Heatmap: {'ON' if self.show_heatmap else 'OFF'}")
        elif key == pygame.K_SPACE:
            self.paused = not self.paused
            if self.paused:
                self.serial.send_stop()
            print(f"[Control] {'PAUSED' if self.paused else 'RESUMED'}")
        elif key == pygame.K_s:
            self._toggle_recording()
        elif key == pygame.K_t:
            self._change_target()
    
    def _toggle_recording(self):
        if self.recorder is None:
            print("[Control] Recording disabled")
            return
        if self.recording:
            self.recorder.stop_session()
            self.recording = False
            print("[Control] Recording stopped")
        else:
            self.recorder.start_session(f"yoloe_{self.target_name}")
            self.recording = True
            print("[Control] Recording started")
    
    def _change_target(self):
        print("\n[Control] Enter new target name (or press Enter to cancel):")
        try:
            new_target = input("> ").strip()
            if new_target:
                self.target_name = new_target
                if self.perception is not None:
                    self.perception.set_target_classes(new_target)
                print(f"[Control] Target changed to: '{new_target}'")
            else:
                print("[Control] Target change cancelled")
        except:
            print("[Control] Target change cancelled")
    
    def _update_display_opencv(self, frame, detection, action, safety):
        if self.args.control_mode == 'proportional' and self.action_p is not None:
            action = self.action_p
        vis = VisualizationRecorder.draw_visualization(frame, detection, action, safety, self.show_heatmap)
        cv2.putText(vis, f"Target: {self.target_name}", (10, 160), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        cv2.putText(vis, f"FPS: {self.fps:.1f}", (10, 180), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        ctrl_mode = self.args.control_mode
        if ctrl_mode == 'proportional':
            label = "P-Ctrl"
        elif ctrl_mode == 'bc_model':
            label = "bc_model"
        else:
            label = "bc_model(sent)"
        cv2.putText(vis, f"{label}: ({action.action[0]:.2f}, {action.action[1]:.2f})",
                    (10, 200), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 220, 255), 2)

        if ctrl_mode == 'both' and self.action_p is not None:
            cv2.putText(vis,
                        f"P-Ctrl : ({self.action_p.action[0]:.2f}, {self.action_p.action[1]:.2f})",
                        (10, 222), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 180, 0), 2)
            delta = action.action - self.action_p.action
            d_color = (100, 255, 100) if np.linalg.norm(delta) < 0.15 else (100, 100, 255)
            cv2.putText(vis,
                        f"Delta  : ({delta[0]:+.2f}, {delta[1]:+.2f})",
                        (10, 244), cv2.FONT_HERSHEY_SIMPLEX, 0.55, d_color, 2)

        vis_bgr = cv2.cvtColor(vis, cv2.COLOR_RGB2BGR)
        cv2.imshow('YOLO-E Visual Servoing', vis_bgr)
    
    def _update_display_pygame(self, frame, detection, action, safety):
        self.screen.fill((30, 30, 40))
        f_large, f_mid, f_small = self.fonts
        vx, vy = self.config.visualization.video_x, self.config.visualization.video_y
        
        if self.args.control_mode == 'proportional' and self.action_p is not None:
            action = self.action_p
        vis = VisualizationRecorder.draw_visualization(frame, detection, action, safety, self.show_heatmap)
        surf = pygame.surfarray.make_surface(vis.swapaxes(0, 1))
        self.screen.blit(surf, (vx, vy))
        
        y = 20
        mode_str = 'ROBOT' if self.args.mode == 'robot' else 'SIMULATION'
        mode_color = (0, 255, 0) if self.args.mode == 'robot' else (255, 200, 0)
        self.screen.blit(f_mid.render(f"Mode: {mode_str}", True, mode_color), (20, y)); y += 25
        self.screen.blit(f_mid.render(f"Target: {self.target_name}", True, (0, 255, 255)), (20, y)); y += 25
        
        status_color = {'normal': (0, 255, 0), 'low_conf': (255, 255, 0), 'holding': (255, 200, 0),
                        'searching': (100, 200, 255), 'stopped': (255, 0, 0), 'error': (255, 0, 0)}.get(safety.state.value, (200, 200, 200))
        self.screen.blit(f_mid.render(f"State: {safety.state.value}", True, status_color), (20, y)); y += 25
        self.screen.blit(f_small.render(f"FPS: {self.fps:.1f}", True, (200, 200, 200)), (20, y)); y += 20
        self.screen.blit(f_small.render(f"Confidence: {detection.confidence:.2f}", True, (200, 200, 200)), (20, y)); y += 20
        
        error = detection.get_pixel_error()
        self.screen.blit(f_small.render(f"Error: ({error[0]:.0f}, {error[1]:.0f}) px", True, (200, 200, 200)), (20, y)); y += 20

        # Command display — label depends on control mode
        ctrl_mode = self.args.control_mode
        if ctrl_mode == 'proportional':
            cmd_label = "P-Ctrl"
        elif ctrl_mode == 'bc_model':
            cmd_label = "bc_model"
        else:
            cmd_label = "bc_model (sent)"
        self.screen.blit(f_small.render(
            f"{cmd_label}: ({action.action[0]:.2f}, {action.action[1]:.2f})",
            True, (0, 220, 255)), (20, y)); y += 20

        # Second row: proportional command in 'both' mode
        if ctrl_mode == 'both' and self.action_p is not None:
            p_color = (255, 180, 0)
            self.screen.blit(f_small.render(
                f"P-Ctrl  : ({self.action_p.action[0]:.2f}, {self.action_p.action[1]:.2f})",
                True, p_color), (20, y)); y += 20

            # Delta between the two for quick inspection
            delta = action.action - self.action_p.action
            d_color = (180, 255, 180) if np.linalg.norm(delta) < 0.15 else (255, 100, 100)
            self.screen.blit(f_small.render(
                f"Δ cmd   : ({delta[0]:+.2f}, {delta[1]:+.2f})",
                True, d_color), (20, y)); y += 20
        
        if self.recording:
            self.screen.blit(f_small.render("● RECORDING", True, (255, 0, 0)), (20, y)); y += 20
        if self.paused:
            self.screen.blit(f_mid.render("⏸ PAUSED", True, (255, 200, 0)), (20, y)); y += 25
        
        instructions = ["'t' Change target | 'h' Heatmap", "'Space' Pause | 's' Record", "'q' Quit"]
        y = self.config.visualization.ui_height - 80
        for text in instructions:
            self.screen.blit(f_small.render(text, True, (200, 200, 200)), (20, y)); y += 18
        
        pygame.display.flip()
    
    def run(self):
        """Main control loop"""
        print("\n" + "=" * 60)
        print("YOLO-E Visual Servoing Controller")
        print("=" * 60)
        print(f"Mode: {self.args.mode}")
        print(f"Target: '{self.target_name}'")
        print(f"YOLO-E Model: {self.args.yoloe_model}")
        print(f"Mock Perception: {self.use_mock_perception}")
        print(f"Mock Control: {self.use_mock_control}")
        print(f"Control Mode: {self.args.control_mode}")
        print("=" * 60)
        print("Controls: 't' change target, 'h' heatmap, 'Space' pause, 's' record, 'q' quit")
        print("=" * 60 + "\n")
        
        self._init_models()
        self.tracking_active = True
        self.start_time = time.time()
        
        dummy_detection = DetectionResult(bbox=None, center=None, confidence=0.0, detection_score=0.0,
                                          no_detection=True, status='no_detection', lost_count=0, inference_time=0.0, fps=0.0)
        dummy_action = ControlAction(action=np.zeros(2), action_mean=np.zeros(2), valid=False)
        dummy_safety = SafetyOutput(action=np.zeros(2), state=SafetyState.HOLDING, is_safe=False, applied_gain=0.0, reason="Waiting")
        
        try:
            while self.running:
                frame_start = time.time()
                self._handle_events()
                
                frame = self.camera.get_frame()
                if frame is None:
                    continue
                
                if self.tracking_active and self.perception is not None:
                    detection, action, safety = self._run_pipeline(frame)
                    self._send_command(safety)
                    
                    if self.recording and self.recorder:
                        self.recorder.record(frame=frame, detection_result=detection,
                                            control_action=action, safety_output=safety, motor_state=self.motor_state)
                else:
                    detection, action, safety = dummy_detection, dummy_action, dummy_safety
                    if self.args.mode == 'robot':
                        self.serial.send_stop()
                
                if self.screen is not None:
                    self._update_display_pygame(frame, detection, action, safety)
                    self.clock.tick(60)
                elif not self.args.no_display:
                    self._update_display_opencv(frame, detection, action, safety)
                
                frame_time = time.time() - frame_start
                self.frame_times.append(frame_time)
                if len(self.frame_times) > 30:
                    self.frame_times.pop(0)
                self.fps = 1.0 / (sum(self.frame_times) / len(self.frame_times))
        
        except KeyboardInterrupt:
            print("\n[Control] Interrupted by user")
        finally:
            self._cleanup()
    
    def _cleanup(self):
        print("\n[Control] Shutting down...")
        self.serial.send_stop()
        time.sleep(0.1)
        self.serial.close()
        self.camera.stop()
        if self.recording and self.recorder:
            self.recorder.stop_session()
        if self.screen is not None:
            pygame.quit()
        else:
            cv2.destroyAllWindows()
        print("[Control] Shutdown complete")


def main():
    args = parse_args()
    
    if not args.mock_all and not args.mock_control:
        if not args.control_checkpoint:
            print("[Warning] No control checkpoint specified, using mock model")
            args.mock_control = True
        elif not os.path.exists(args.control_checkpoint):
            print(f"[Warning] Control checkpoint not found: {args.control_checkpoint}")
            print("[Warning] Using mock control model")
            args.mock_control = True
    
    controller = IntegratedController(args)
    controller.run()


if __name__ == '__main__':
    main()