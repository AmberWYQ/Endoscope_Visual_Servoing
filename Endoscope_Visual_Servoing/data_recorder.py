"""
Data Recorder for Visual Servoing Pipeline

Handles:
- Frame recording (video)
- Detection logging (bounding boxes, confidence)
- Action logging (motor commands)
- Timestamp synchronization
- CSV export for training data

Designed for debugging and future training data collection.
"""

import os
import time
import json
import csv
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, Any, List
from dataclasses import dataclass, asdict
import numpy as np
import cv2


@dataclass
class RecordEntry:
    """Single timestep record"""
    timestamp: float
    frame_idx: int
    
    # Detection
    detection_valid: bool
    bbox_x: float
    bbox_y: float
    bbox_w: float
    bbox_h: float
    center_x: float
    center_y: float
    confidence: float
    detection_status: str
    
    # Error
    error_x: float
    error_y: float
    
    # Action
    action_m1: float
    action_m2: float
    action_valid: bool
    
    # Safety
    safety_state: str
    applied_gain: float
    
    # Motor state (if available)
    motor_pos_m1: float = 0.0
    motor_pos_m2: float = 0.0
    motor_spd_m1: float = 0.0
    motor_spd_m2: float = 0.0
    
    # Performance
    detection_fps: float = 0.0
    control_fps: float = 0.0


class DataRecorder:
    """
    Records data from the visual servoing pipeline for debugging and training.
    
    Usage:
        recorder = DataRecorder(config.logging)
        recorder.start_session()
        
        # In control loop:
        recorder.record(
            frame=frame,
            detection=detection_result,
            action=control_output,
            safety=safety_output,
            timestamp=timestamp,
        )
        
        # When done:
        recorder.stop_session()
    """
    
    def __init__(self, config):
        """
        Initialize data recorder.
        
        Args:
            config: LoggingConfig or full IntegratedConfig
        """
        if hasattr(config, 'logging'):
            self.config = config.logging
        else:
            self.config = config
        
        # State
        self.session_active = False
        self.session_dir = None
        self.video_writer = None
        self.csv_file = None
        self.csv_writer = None
        
        # Counters
        self.frame_count = 0
        self.start_time = 0.0
        
        # Data buffer
        self.records: List[RecordEntry] = []
        
        # Detection buffer for separate logging
        self.detections: List[Dict] = []
        self.actions: List[Dict] = []
    
    def start_session(self, session_name: Optional[str] = None):
        """
        Start a new recording session.
        
        Args:
            session_name: Optional name for the session
        """
        if self.session_active:
            print("[Recorder] Warning: Session already active, stopping previous")
            self.stop_session()
        
        # Create session directory
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        if session_name:
            dir_name = f"{session_name}_{timestamp}"
        else:
            dir_name = f"session_{timestamp}"
        
        self.session_dir = Path(self.config.log_dir) / dir_name
        self.session_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize video writer
        if self.config.save_video:
            self._init_video_writer()
        
        # Initialize CSV writer
        if self.config.save_csv:
            self._init_csv_writer()
        
        # Reset counters
        self.frame_count = 0
        self.start_time = time.time()
        self.records = []
        self.detections = []
        self.actions = []
        
        self.session_active = True
        print(f"[Recorder] Started session: {self.session_dir}")
    
    def _init_video_writer(self):
        """Initialize video writer"""
        video_path = self.session_dir / "video.mp4"
        fourcc = cv2.VideoWriter_fourcc(*self.config.video_codec)
        self.video_writer = cv2.VideoWriter(
            str(video_path),
            fourcc,
            self.config.video_fps,
            (640, 480)
        )
    
    def _init_csv_writer(self):
        """Initialize CSV writer"""
        csv_path = self.session_dir / "data.csv"
        self.csv_file = open(csv_path, 'w', newline='')
        
        # Get field names from RecordEntry
        fields = list(RecordEntry.__dataclass_fields__.keys())
        self.csv_writer = csv.DictWriter(self.csv_file, fieldnames=fields)
        self.csv_writer.writeheader()
    
    def record(
        self,
        frame: np.ndarray,
        detection_result,
        control_action,
        safety_output,
        timestamp: Optional[float] = None,
        motor_state: Optional[np.ndarray] = None,
    ):
        """
        Record a single timestep.
        
        Args:
            frame: Camera frame (RGB or BGR)
            detection_result: DetectionResult from perception
            control_action: ControlAction from control
            safety_output: SafetyOutput from safety manager
            timestamp: Optional timestamp (uses time.time() if not provided)
            motor_state: Optional motor state [m1_pos, m2_pos, m1_spd, m2_spd]
        """
        if not self.session_active:
            return
        
        if timestamp is None:
            timestamp = time.time() - self.start_time
        
        # Write video frame
        if self.config.save_video and self.video_writer is not None:
            # Convert RGB to BGR for OpenCV
            frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            self.video_writer.write(frame_bgr)
        
        # Build record entry
        bbox = detection_result.bbox if detection_result.bbox is not None else np.zeros(4)
        center = detection_result.center if detection_result.center is not None else np.zeros(2)
        error = detection_result.get_pixel_error() if detection_result.center is not None else np.zeros(2)
        
        entry = RecordEntry(
            timestamp=timestamp,
            frame_idx=self.frame_count,
            
            # Detection
            detection_valid=not detection_result.no_detection,
            bbox_x=float(bbox[0]),
            bbox_y=float(bbox[1]),
            bbox_w=float(bbox[2]),
            bbox_h=float(bbox[3]),
            center_x=float(center[0]),
            center_y=float(center[1]),
            confidence=float(detection_result.confidence),
            detection_status=detection_result.status,
            
            # Error
            error_x=float(error[0]),
            error_y=float(error[1]),
            
            # Action
            action_m1=float(control_action.action[0]),
            action_m2=float(control_action.action[1]),
            action_valid=control_action.valid,
            
            # Safety
            safety_state=safety_output.state.value,
            applied_gain=float(safety_output.applied_gain),
            
            # Performance
            detection_fps=float(detection_result.fps),
            control_fps=1.0 / control_action.inference_time if control_action.inference_time > 0 else 0.0,
        )
        
        # Add motor state if available
        if motor_state is not None and len(motor_state) >= 4:
            entry.motor_pos_m1 = float(motor_state[0])
            entry.motor_pos_m2 = float(motor_state[1])
            entry.motor_spd_m1 = float(motor_state[2])
            entry.motor_spd_m2 = float(motor_state[3])
        
        # Write to CSV
        if self.config.save_csv and self.csv_writer is not None:
            self.csv_writer.writerow(asdict(entry))
        
        # Store in buffer
        self.records.append(entry)
        
        # Store separate detection and action logs
        if self.config.save_detections:
            self.detections.append({
                'timestamp': timestamp,
                'frame_idx': self.frame_count,
                'bbox': bbox.tolist() if isinstance(bbox, np.ndarray) else list(bbox),
                'confidence': detection_result.confidence,
                'status': detection_result.status,
            })
        
        if self.config.save_actions:
            self.actions.append({
                'timestamp': timestamp,
                'frame_idx': self.frame_count,
                'action': control_action.action.tolist(),
                'valid': control_action.valid,
                'safety_state': safety_output.state.value,
            })
        
        self.frame_count += 1
    
    def stop_session(self):
        """Stop the recording session and save all data"""
        if not self.session_active:
            return
        
        # Close video writer
        if self.video_writer is not None:
            self.video_writer.release()
            self.video_writer = None
        
        # Close CSV file
        if self.csv_file is not None:
            self.csv_file.close()
            self.csv_file = None
            self.csv_writer = None
        
        # Save separate detection log
        if self.config.save_detections and self.detections:
            det_path = self.session_dir / "detections.json"
            with open(det_path, 'w') as f:
                json.dump(self.detections, f, indent=2)
        
        # Save separate action log
        if self.config.save_actions and self.actions:
            act_path = self.session_dir / "actions.json"
            with open(act_path, 'w') as f:
                json.dump(self.actions, f, indent=2)
        
        # Save metadata
        duration = time.time() - self.start_time
        metadata = {
            'session_dir': str(self.session_dir),
            'frame_count': self.frame_count,
            'duration_sec': duration,
            'avg_fps': self.frame_count / duration if duration > 0 else 0,
            'start_time': datetime.fromtimestamp(self.start_time).isoformat(),
            'end_time': datetime.now().isoformat(),
        }
        
        meta_path = self.session_dir / "metadata.json"
        with open(meta_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        print(f"[Recorder] Session stopped: {self.frame_count} frames, {duration:.1f}s")
        print(f"[Recorder] Data saved to: {self.session_dir}")
        
        self.session_active = False
    
    def get_session_stats(self) -> Dict[str, Any]:
        """Get current session statistics"""
        if not self.session_active:
            return {'active': False}
        
        duration = time.time() - self.start_time
        return {
            'active': True,
            'session_dir': str(self.session_dir),
            'frame_count': self.frame_count,
            'duration_sec': duration,
            'avg_fps': self.frame_count / duration if duration > 0 else 0,
        }
    
    def __del__(self):
        """Cleanup on deletion"""
        if self.session_active:
            self.stop_session()


class VisualizationRecorder:
    """
    Records visualization frames with overlays for debugging.
    
    Saves annotated frames showing:
    - Bounding boxes
    - Error arrows
    - Control commands
    - Status information
    """
    
    def __init__(self, output_dir: str, fps: float = 20.0):
        """
        Initialize visualization recorder.
        
        Args:
            output_dir: Output directory
            fps: Video frame rate
        """
        self.output_dir = Path(output_dir)
        self.fps = fps
        self.video_writer = None
        self.frame_count = 0
    
    def start(self, filename: str = "visualization.mp4"):
        """Start recording"""
        self.output_dir.mkdir(parents=True, exist_ok=True)
        video_path = self.output_dir / filename
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        self.video_writer = cv2.VideoWriter(
            str(video_path),
            fourcc,
            self.fps,
            (640, 480)
        )
        self.frame_count = 0
    
    def write_frame(self, frame: np.ndarray):
        """Write a single frame"""
        if self.video_writer is None:
            return
        
        # Ensure correct format
        if frame.shape[-1] == 3 and frame.dtype == np.uint8:
            # Assume RGB, convert to BGR
            frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        else:
            frame_bgr = frame
        
        self.video_writer.write(frame_bgr)
        self.frame_count += 1
    
    def stop(self):
        """Stop recording"""
        if self.video_writer is not None:
            self.video_writer.release()
            self.video_writer = None
            print(f"[VisRecorder] Saved {self.frame_count} frames")
    
    @staticmethod
    def draw_visualization(
        frame: np.ndarray,
        detection_result,
        control_action,
        safety_output,
        show_heatmap: bool = False,
    ) -> np.ndarray:
        """
        Draw visualization overlay on frame.
        
        Args:
            frame: Base frame (RGB)
            detection_result: DetectionResult
            control_action: ControlAction
            safety_output: SafetyOutput
            show_heatmap: Whether to overlay heatmap
        
        Returns:
            Annotated frame (RGB)
        """
        vis = frame.copy()
        h, w = vis.shape[:2]
        
        # Draw heatmap if available and requested
        if show_heatmap and detection_result.heatmap is not None:
            heatmap = detection_result.heatmap
            heatmap = cv2.resize(heatmap, (w, h))
            heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min() + 1e-6)
            heatmap_colored = cv2.applyColorMap(
                (heatmap * 255).astype(np.uint8),
                cv2.COLORMAP_JET
            )
            heatmap_colored = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)
            vis = cv2.addWeighted(vis, 0.6, heatmap_colored, 0.4, 0)
        
        # Image center
        cx, cy = w // 2, h // 2
        
        # Draw center crosshair
        cv2.line(vis, (cx - 20, cy), (cx + 20, cy), (0, 255, 0), 2)
        cv2.line(vis, (cx, cy - 20), (cx, cy + 20), (0, 255, 0), 2)
        
        # Draw bounding box
        if detection_result.bbox is not None:
            x, y, bw, bh = detection_result.bbox.astype(int)
            
            # Color based on status
            if detection_result.status == 'tracking':
                color = (0, 200, 255)  # Orange
            elif detection_result.status == 'lost_recovering':
                color = (255, 255, 0)  # Yellow
            else:
                color = (255, 0, 0)  # Red
            
            cv2.rectangle(vis, (x, y), (x + bw, y + bh), color, 2)
            
            # Draw center point
            bcx, bcy = int(x + bw / 2), int(y + bh / 2)
            cv2.circle(vis, (bcx, bcy), 5, color, -1)
            
            # Draw error arrow (from bbox center to image center)
            cv2.arrowedLine(vis, (bcx, bcy), (cx, cy), (255, 0, 255), 2)
        
        # Draw command arrow
        if control_action.valid:
            # Scale action for visualization
            scale = 50
            cmd_x = int(cx - control_action.action[0] * scale)
            cmd_y = int(cy - control_action.action[1] * scale)  # Flip y for display
            cv2.arrowedLine(vis, (cx, cy), (cmd_x, cmd_y), (0, 255, 0), 3)
        
        # Draw status text
        status_color = (255, 255, 255)
        if safety_output.state.value == 'error':
            status_color = (255, 0, 0)
        elif safety_output.state.value == 'holding':
            status_color = (255, 255, 0)
        
        y_offset = 30
        cv2.putText(vis, f"State: {safety_output.state.value}",
                   (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, status_color, 2)
        
        y_offset += 25
        cv2.putText(vis, f"Conf: {detection_result.confidence:.2f}",
                   (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        y_offset += 25
        error = detection_result.get_pixel_error() if detection_result.center is not None else np.zeros(2)
        cv2.putText(vis, f"Error: ({error[0]:.0f}, {error[1]:.0f})",
                   (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        y_offset += 25
        cv2.putText(vis, f"Cmd: ({control_action.action[0]:.2f}, {control_action.action[1]:.2f})",
                   (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        y_offset += 25
        cv2.putText(vis, f"FPS: {detection_result.fps:.1f}",
                   (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        return vis


if __name__ == '__main__':
    # Test data recorder
    print("Testing DataRecorder...")
    
    from dataclasses import dataclass
    
    @dataclass
    class TestLoggingConfig:
        enabled: bool = True
        log_dir: str = '/tmp/test_recording'
        save_video: bool = True
        save_csv: bool = True
        save_detections: bool = True
        save_actions: bool = True
        video_fps: float = 20.0
        video_codec: str = 'mp4v'
    
    # Create mock detection and action results
    @dataclass
    class MockDetection:
        bbox: np.ndarray = None
        center: np.ndarray = None
        confidence: float = 0.8
        status: str = 'tracking'
        no_detection: bool = False
        fps: float = 30.0
        heatmap: np.ndarray = None
        
        def get_pixel_error(self):
            if self.center is None:
                return np.zeros(2)
            return np.array([320 - self.center[0], 240 - self.center[1]])
    
    @dataclass
    class MockControlAction:
        action: np.ndarray = None
        valid: bool = True
        inference_time: float = 0.01
    
    from safety_manager import SafetyOutput, SafetyState
    
    config = TestLoggingConfig()
    recorder = DataRecorder(config)
    
    # Start session
    recorder.start_session("test")
    
    # Record some frames
    for i in range(10):
        frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        
        detection = MockDetection(
            bbox=np.array([200 + i, 150 + i, 80, 60]),
            center=np.array([240 + i, 180 + i]),
        )
        
        action = MockControlAction(
            action=np.array([0.1 * i, -0.05 * i]),
        )
        
        safety = SafetyOutput(
            action=action.action,
            state=SafetyState.NORMAL,
            is_safe=True,
            applied_gain=1.0,
            reason="Normal",
        )
        
        recorder.record(frame, detection, action, safety, timestamp=i * 0.05)
    
    # Stop session
    recorder.stop_session()
    
    print("\nDataRecorder test complete!")
