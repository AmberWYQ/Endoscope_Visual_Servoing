"""
YOLO-E Perception Interface for Visual Servoing Pipeline

Provides a standardized interface for instruction-based object detection using YOLO-E.
Replaces the reference-based detection with text prompt-based detection.

Key Features:
- Text/instruction-based target detection (e.g., "polyp", "lesion", "coin")
- Compatible with pretrained YOLO-World or finetuned YOLO-E models
- Same output interface as the original PerceptionInterface (DetectionResult)
- Supports both single and multiple class prompts

Output Format:
    DetectionResult with:
    - bbox: [x, y, w, h] in 640x480 coordinates, or None if no detection
    - center: [cx, cy] bbox center, or None
    - confidence: float 0-1
    - no_detection: bool
    - status: 'tracking', 'lost_recovering', 'no_detection'
"""

import time
from dataclasses import dataclass
from typing import Optional, List, Dict, Any, Union
import numpy as np
import torch
import cv2

# Try to import ultralytics
try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except ImportError:
    YOLO_AVAILABLE = False
    print("[YOLOEPerception] Warning: ultralytics not installed. Run: pip install ultralytics")


@dataclass
class DetectionResult:
    """Standardized detection result (same as original perception_interface.py)"""
    # Bounding box [x, y, w, h] in original image coordinates
    bbox: Optional[np.ndarray]
    
    # Bbox center [cx, cy]
    center: Optional[np.ndarray]
    
    # Detection confidence (0-1)
    confidence: float
    
    # Detection score (raw model output)
    detection_score: float
    
    # No detection flag
    no_detection: bool
    
    # Status string
    status: str  # 'tracking', 'lost_recovering', 'no_detection'
    
    # Frames since last detection
    lost_count: int
    
    # Processing time
    inference_time: float
    fps: float
    
    # Optional heatmap (not used for YOLO-E, kept for compatibility)
    heatmap: Optional[np.ndarray] = None
    
    # Detected class name
    class_name: Optional[str] = None
    
    def get_normalized_center(self, img_width: int = 640, img_height: int = 480) -> Optional[np.ndarray]:
        """Get center normalized to [-1, 1] range"""
        if self.center is None:
            return None
        cx = (self.center[0] - img_width / 2) / (img_width / 2)
        cy = (self.center[1] - img_height / 2) / (img_height / 2)
        return np.array([cx, cy])
    
    def get_pixel_error(self, img_width: int = 640, img_height: int = 480) -> np.ndarray:
        """Get pixel error from image center (positive = target is to the left/above center)"""
        if self.center is None:
            return np.zeros(2)
        ex = img_width / 2 - self.center[0]
        ey = img_height / 2 - self.center[1]
        return np.array([ex, ey])


class YOLOEPerceptionInterface:
    """
    Standardized interface for YOLO-E instruction-based detection.
    
    Uses YOLO-World/YOLO-E models for open-vocabulary detection based on text prompts.
    """
    
    def __init__(
        self,
        model_path: str = "yolov8x-worldv2.pt",
        target_classes: Union[str, List[str]] = "object",
        confidence_threshold: float = 0.25,
        iou_threshold: float = 0.45,
        max_lost_frames: int = 10,
        device: Optional[str] = None,
    ):
        """
        Initialize the YOLO-E perception interface.
        
        Args:
            model_path: Path to YOLO model checkpoint (pretrained or finetuned)
            target_classes: Target class name(s) for detection (text prompt)
            confidence_threshold: Minimum confidence for detection
            iou_threshold: IoU threshold for NMS
            max_lost_frames: Frames to wait before declaring target lost
            device: Device for inference ('cuda', 'cpu', or None for auto)
        """
        if not YOLO_AVAILABLE:
            raise ImportError("ultralytics not installed. Run: pip install ultralytics")
        
        self.confidence_threshold = confidence_threshold
        self.iou_threshold = iou_threshold
        self.max_lost_frames = max_lost_frames
        
        # Image dimensions (fixed for endoscopy)
        self.img_width = 640
        self.img_height = 480
        
        # Set device
        if device is None:
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        else:
            self.device = device
        
        # Load model
        print(f"[YOLOEPerception] Loading model: {model_path}")
        self.model = YOLO(model_path)

        # Detect whether the loaded model supports open-vocabulary text prompts
        # (YOLO-World / YOLO-E) or is a standard fine-tuned DetectionModel whose
        # classes are fixed at training time.
        self._is_world_model = hasattr(self.model.model, "set_classes")
        if self._is_world_model:
            print("[YOLOEPerception] Model type: YOLO-World / open-vocabulary")
        else:
            # Standard fine-tuned model: read the trained class names from the
            # checkpoint and use them as-is.
            trained_names = getattr(self.model.model, "names", {})
            self._trained_classes = list(trained_names.values()) if trained_names else ["object"]
            print(f"[YOLOEPerception] Model type: standard DetectionModel "
                  f"(fixed classes: {self._trained_classes})")

        # Set target classes
        self.set_target_classes(target_classes)
        
        # Tracking state
        self.last_bbox = None
        self.lost_count = 0
        
        # Statistics
        self.frame_count = 0
        self.total_time = 0.0
        
        print(f"[YOLOEPerception] Initialized on device: {self.device}")
        print(f"[YOLOEPerception] Target classes: {self.target_classes}")
    
    def set_target_classes(self, target_classes: Union[str, List[str]]):
        """
        Set the target classes for detection.

        For YOLO-World / open-vocabulary models, this calls model.set_classes()
        to update the text prompt.  For standard fine-tuned DetectionModels the
        classes are fixed at training time, so we just record the requested name
        for display purposes and use the model's own class list for inference.

        Args:
            target_classes: Single class name or list of class names
        """
        if isinstance(target_classes, str):
            self.target_classes = [target_classes]
        else:
            self.target_classes = list(target_classes)

        if self._is_world_model:
            # YOLO-World: dynamically update text prompt
            self.model.set_classes(self.target_classes)
            print(f"[YOLOEPerception] Target classes set to: {self.target_classes}")
        else:
            # Standard fine-tuned model: classes are baked in — nothing to set.
            # We still keep self.target_classes for the HUD / CSV label, but the
            # model will detect whatever it was trained on.
            print(f"[YOLOEPerception] Fine-tuned model — ignoring set_classes call. "
                  f"Model will detect: {self._trained_classes}  "
                  f"(requested label kept for display: {self.target_classes})")
    
    def detect(self, frame: np.ndarray, return_heatmap: bool = False) -> DetectionResult:
        """
        Detect target in current frame using YOLO-E.
        
        Args:
            frame: Current frame (RGB, any size - will be resized to 640x480)
            return_heatmap: Whether to return attention heatmap (not supported for YOLO-E)
        
        Returns:
            DetectionResult with detection information
        """
        start_time = time.time()
        
        # Ensure correct size
        if frame.shape[:2] != (self.img_height, self.img_width):
            frame = cv2.resize(frame, (self.img_width, self.img_height))
        
        # Convert RGB to BGR for YOLO (YOLO expects BGR)
        frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        
        # Run inference
        results = self.model.predict(
            frame_bgr,
            conf=self.confidence_threshold,
            iou=self.iou_threshold,
            device=self.device,
            verbose=False,
        )
        
        # Process results
        bbox = None
        center = None
        confidence = 0.0
        detection_score = 0.0
        status = 'no_detection'
        class_name = None
        
        if len(results) > 0 and results[0].boxes is not None and len(results[0].boxes) > 0:
            # Get the detection with highest confidence
            boxes = results[0].boxes
            confidences = boxes.conf.cpu().numpy()
            
            if len(confidences) > 0:
                best_idx = np.argmax(confidences)
                best_conf = confidences[best_idx]
                
                if best_conf >= self.confidence_threshold:
                    # Get bbox in xyxy format
                    xyxy = boxes.xyxy[best_idx].cpu().numpy()
                    x1, y1, x2, y2 = xyxy
                    
                    # Convert to [x, y, w, h] format
                    x = float(x1)
                    y = float(y1)
                    w = float(x2 - x1)
                    h = float(y2 - y1)
                    
                    # Clamp to image boundaries
                    x = max(0, min(x, self.img_width - 1))
                    y = max(0, min(y, self.img_height - 1))
                    w = min(w, self.img_width - x)
                    h = min(h, self.img_height - y)
                    
                    bbox = np.array([x, y, w, h])
                    center = np.array([x + w / 2, y + h / 2])
                    confidence = float(best_conf)
                    detection_score = confidence
                    
                    # Get class name
                    if hasattr(boxes, 'cls') and len(boxes.cls) > best_idx:
                        cls_idx = int(boxes.cls[best_idx].cpu().numpy())
                        # For fine-tuned models use the model's own name list;
                        # for YOLO-World use the user-supplied target_classes.
                        name_list = (self.target_classes if self._is_world_model
                                     else self._trained_classes)
                        if cls_idx < len(name_list):
                            class_name = name_list[cls_idx]
                    
                    self.last_bbox = bbox.copy()
                    self.lost_count = 0
                    status = 'tracking'
        
        # Handle no detection
        if bbox is None:
            self.lost_count += 1
            
            if self.lost_count <= self.max_lost_frames and self.last_bbox is not None:
                # Use last known position during recovery period
                bbox = self.last_bbox.copy()
                center = np.array([
                    bbox[0] + bbox[2] / 2,
                    bbox[1] + bbox[3] / 2
                ])
                status = 'lost_recovering'
            else:
                # Fully lost
                self.last_bbox = None
                status = 'no_detection'
        
        # Timing
        elapsed = time.time() - start_time
        self.frame_count += 1
        self.total_time += elapsed
        fps = 1.0 / max(elapsed, 0.001)
        
        return DetectionResult(
            bbox=bbox,
            center=center,
            confidence=confidence,
            detection_score=detection_score,
            no_detection=(bbox is None or status == 'no_detection'),
            status=status,
            lost_count=self.lost_count,
            inference_time=elapsed,
            fps=fps,
            heatmap=None,  # YOLO-E doesn't provide attention heatmaps
            class_name=class_name,
        )
    
    def reset(self):
        """Reset tracking state"""
        self.last_bbox = None
        self.lost_count = 0
        self.frame_count = 0
        self.total_time = 0.0
    
    def get_stats(self) -> Dict[str, Any]:
        """Get detection statistics"""
        return {
            'frame_count': self.frame_count,
            'total_time': self.total_time,
            'avg_fps': self.frame_count / self.total_time if self.total_time > 0 else 0,
            'lost_count': self.lost_count,
            'target_classes': self.target_classes,
        }


class MockYOLOEPerceptionInterface:
    """
    Mock YOLO-E perception interface for testing without the actual model.
    
    Simulates detection by generating a moving target.
    """
    
    def __init__(
        self,
        target_classes: Union[str, List[str]] = "object",
        img_width: int = 640,
        img_height: int = 480,
        **kwargs  # Ignore other arguments
    ):
        """Initialize mock perception"""
        print("[MockYOLOEPerception] Initializing mock YOLO-E perception interface")
        
        self.img_width = img_width
        self.img_height = img_height
        
        if isinstance(target_classes, str):
            self.target_classes = [target_classes]
        else:
            self.target_classes = list(target_classes)
        
        # Simulated target state
        self.bbox_size = (80, 60)  # Fixed size for mock
        self.sim_x = img_width // 2 - self.bbox_size[0] // 2
        self.sim_y = img_height // 2 - self.bbox_size[1] // 2
        self.sim_vx = np.random.uniform(-2, 2)
        self.sim_vy = np.random.uniform(-2, 2)
        
        # Statistics
        self.frame_count = 0
        self.total_time = 0.0
        self.lost_count = 0
    
    def set_target_classes(self, target_classes: Union[str, List[str]]):
        """Set target classes"""
        if isinstance(target_classes, str):
            self.target_classes = [target_classes]
        else:
            self.target_classes = list(target_classes)
        self.lost_count = 0
    
    def detect(self, frame: np.ndarray, return_heatmap: bool = False) -> DetectionResult:
        """Generate mock detection"""
        start_time = time.time()
        
        # Update simulated position with random walk
        self.sim_x += self.sim_vx + np.random.uniform(-1, 1)
        self.sim_y += self.sim_vy + np.random.uniform(-1, 1)
        
        # Velocity variation
        self.sim_vx += np.random.uniform(-0.5, 0.5)
        self.sim_vy += np.random.uniform(-0.5, 0.5)
        self.sim_vx = np.clip(self.sim_vx, -3, 3)
        self.sim_vy = np.clip(self.sim_vy, -3, 3)
        
        # Bounce off edges
        margin = 50
        if self.sim_x < margin:
            self.sim_x = margin
            self.sim_vx = abs(self.sim_vx)
        if self.sim_x + self.bbox_size[0] > self.img_width - margin:
            self.sim_x = self.img_width - margin - self.bbox_size[0]
            self.sim_vx = -abs(self.sim_vx)
        if self.sim_y < margin:
            self.sim_y = margin
            self.sim_vy = abs(self.sim_vy)
        if self.sim_y + self.bbox_size[1] > self.img_height - margin:
            self.sim_y = self.img_height - margin - self.bbox_size[1]
            self.sim_vy = -abs(self.sim_vy)
        
        # Simulate occasional detection failures (5% chance)
        no_detection = np.random.random() < 0.05
        
        if no_detection:
            self.lost_count += 1
            bbox = None
            center = None
            status = 'no_detection'
        else:
            self.lost_count = 0
            bbox = np.array([self.sim_x, self.sim_y, self.bbox_size[0], self.bbox_size[1]])
            center = np.array([self.sim_x + self.bbox_size[0] / 2, self.sim_y + self.bbox_size[1] / 2])
            status = 'tracking'
        
        # Mock confidence
        confidence = 0.8 + np.random.uniform(-0.1, 0.1) if not no_detection else 0.2
        
        # Timing
        elapsed = time.time() - start_time
        self.frame_count += 1
        self.total_time += elapsed
        fps = 1.0 / max(elapsed, 0.001)
        
        return DetectionResult(
            bbox=bbox,
            center=center,
            confidence=confidence,
            detection_score=confidence * 0.9,
            no_detection=no_detection,
            status=status,
            lost_count=self.lost_count,
            inference_time=elapsed,
            fps=fps,
            class_name=self.target_classes[0] if self.target_classes else None,
        )
    
    def reset(self):
        """Reset tracking state"""
        self.sim_x = self.img_width // 2 - self.bbox_size[0] // 2
        self.sim_y = self.img_height // 2 - self.bbox_size[1] // 2
        self.lost_count = 0
        self.frame_count = 0
        self.total_time = 0.0
    
    def get_stats(self) -> Dict[str, Any]:
        return {
            'frame_count': self.frame_count,
            'total_time': self.total_time,
            'avg_fps': self.frame_count / self.total_time if self.total_time > 0 else 0,
            'lost_count': self.lost_count,
            'target_classes': self.target_classes,
        }


def create_yoloe_perception_interface(
    config,
    target_classes: Union[str, List[str]],
    use_mock: bool = False,
) -> YOLOEPerceptionInterface:
    """
    Factory function to create YOLO-E perception interface.
    
    Args:
        config: IntegratedConfig or YOLOEPerceptionConfig
        target_classes: Target class name(s) for detection
        use_mock: Whether to use mock interface for testing
    
    Returns:
        YOLOEPerceptionInterface or MockYOLOEPerceptionInterface
    """
    if use_mock:
        return MockYOLOEPerceptionInterface(
            target_classes=target_classes,
            img_width=640,
            img_height=480,
        )
    
    # Get config parameters
    if hasattr(config, 'yoloe_perception'):
        perc_config = config.yoloe_perception
        model_path = perc_config.model_path
        confidence_threshold = perc_config.confidence_threshold
        iou_threshold = perc_config.iou_threshold
        max_lost_frames = perc_config.max_lost_frames
        device = config.device if hasattr(config, 'device') else None
    elif hasattr(config, 'perception'):
        # Fallback to original perception config
        perc_config = config.perception
        model_path = getattr(perc_config, 'yoloe_model_path', 'yolov8x-worldv2.pt')
        confidence_threshold = perc_config.confidence_threshold
        iou_threshold = getattr(perc_config, 'iou_threshold', 0.45)
        max_lost_frames = perc_config.max_lost_frames
        device = config.device if hasattr(config, 'device') else None
    else:
        # Use defaults
        model_path = 'yolov8x-worldv2.pt'
        confidence_threshold = 0.25
        iou_threshold = 0.45
        max_lost_frames = 10
        device = None
    
    return YOLOEPerceptionInterface(
        model_path=model_path,
        target_classes=target_classes,
        confidence_threshold=confidence_threshold,
        iou_threshold=iou_threshold,
        max_lost_frames=max_lost_frames,
        device=device,
    )


if __name__ == '__main__':
    # Test YOLO-E perception
    print("Testing YOLO-E Perception Interface...")
    
    # Test mock interface
    print("\n1. Testing MockYOLOEPerceptionInterface:")
    mock = MockYOLOEPerceptionInterface(target_classes="polyp")
    
    frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    
    for i in range(10):
        result = mock.detect(frame)
        print(f"Frame {i}: center={result.center}, conf={result.confidence:.2f}, "
              f"status={result.status}, class={result.class_name}")
    
    # Test real interface if ultralytics is available
    if YOLO_AVAILABLE:
        print("\n2. Testing YOLOEPerceptionInterface:")
        try:
            yoloe = YOLOEPerceptionInterface(
                model_path="yolov8x-worldv2.pt",
                target_classes=["polyp", "lesion"],
                confidence_threshold=0.25,
            )
            
            result = yoloe.detect(frame)
            print(f"Detection: center={result.center}, conf={result.confidence:.2f}, "
                  f"status={result.status}, class={result.class_name}")
        except Exception as e:
            print(f"Could not test real interface: {e}")
    
    print("\nYOLO-E perception test complete!")
