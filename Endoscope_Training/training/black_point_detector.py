"""
Black Point Detection for Robotic Endoscope using YOLOv8-World

This module provides real-time detection of black point markers
in endoscopic images using open-vocabulary detection.

Requirements:
    pip install ultralytics opencv-python --break-system-packages
"""

import os
# Fix Qt platform plugin error for headless Linux servers
os.environ['QT_QPA_PLATFORM'] = 'offscreen'

import cv2
import numpy as np
from typing import Dict, Optional, Tuple, List
from dataclasses import dataclass
import time

try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except ImportError:
    print("Warning: ultralytics not installed.")
    print("Install with: pip install ultralytics --break-system-packages")
    YOLO_AVAILABLE = False


@dataclass
class DetectionResult:
    """Detection result container"""
    detected: bool
    bbox: Optional[List[float]]  # [x1, y1, x2, y2]
    center: Optional[Tuple[float, float]]  # (cx, cy)
    confidence: float
    image_error: np.ndarray  # Normalized error from image center
    processing_time: float  # Detection time in ms


class YOLOWorldDetector:
    """
    YOLOv8-World based detector for open-vocabulary detection
    Specifically configured for detecting black points/markers in endoscopic images
    """
    
    def __init__(self, 
                 model_path: str = "best_blackpoint_base.pt",
                 device: str = "cuda",
                 prompts: List[str] = None):
        """
        Initialize detector

        Args:
            model_path: Path to YOLO model weights.
                        Accepts either a fine-tuned standard YOLO model
                        (e.g. best_blackpoint_base.pt) or a YOLOv8-World
                        model (yolov8x-worldv2.pt).
            device: 'cuda' or 'cpu'
            prompts: Text prompts — only used for YOLOv8-World models.
                     Ignored for standard fine-tuned YOLO models.
        """
        self.device = device
        self.model = None
        self.is_world_model = False

        # Prompts only matter for YOLOv8-World open-vocabulary models
        self.prompts = prompts or [
            "black point",
            "black dot",
            "black marker",
            "dark spot",
            "target marker"
        ]

        if YOLO_AVAILABLE:
            self._load_model(model_path)
        else:
            print("YOLO not available, using fallback detection")
    
    def _load_model(self, model_path: str):
        """Load YOLO model — supports both fine-tuned and YOLOv8-World."""
        try:
            print(f"Loading YOLO model: {model_path}")
            self.model = YOLO(model_path)

            # YOLOv8-World models expose set_classes() for open-vocabulary
            # detection. Standard fine-tuned models do not — detect by checking
            # whether the method exists and the model name contains 'world'.
            model_name = str(model_path).lower()
            self.is_world_model = (
                hasattr(self.model, 'set_classes') and 'world' in model_name
            )

            if self.is_world_model:
                self.model.set_classes(self.prompts)
                print(f"World model — prompts set: {self.prompts}")
            else:
                print("Fine-tuned model — using trained class heads (no prompts needed)")

            # Warm up
            dummy = np.zeros((480, 640, 3), dtype=np.uint8)
            self.model.predict(dummy, verbose=False, device=self.device)

            print(f"Model loaded successfully: {model_path}")
        except Exception as e:
            print(f"Failed to load model: {e}")
            self.model = None
    
    def detect(self, 
               image: np.ndarray,
               conf_threshold: float = 0.25,
               iou_threshold: float = 0.45) -> DetectionResult:
        """
        Detect black point in image
        
        Args:
            image: BGR image (H, W, 3)
            conf_threshold: Confidence threshold
            iou_threshold: IoU threshold for NMS
            
        Returns:
            DetectionResult with detection info
        """
        start_time = time.time()
        
        h, w = image.shape[:2]
        image_center = np.array([w / 2, h / 2])
        
        result = DetectionResult(
            detected=False,
            bbox=None,
            center=None,
            confidence=0.0,
            image_error=np.zeros(2),
            processing_time=0.0
        )
        
        if self.model is not None:
            result = self._yolo_detect(image, image_center, conf_threshold, iou_threshold)
        else:
            result = self._fallback_detect(image, image_center)
        
        result.processing_time = (time.time() - start_time) * 1000  # ms
        return result
    
    def _yolo_detect(self, 
                     image: np.ndarray,
                     image_center: np.ndarray,
                     conf_threshold: float,
                     iou_threshold: float) -> DetectionResult:
        """YOLOv8-World detection"""
        h, w = image.shape[:2]
        
        # Run inference
        detections = self.model.predict(
            image,
            conf=conf_threshold,
            iou=iou_threshold,
            verbose=False,
            device=self.device
        )
        
        if len(detections) > 0 and len(detections[0].boxes) > 0:
            boxes = detections[0].boxes
            
            # Get highest confidence detection
            best_idx = boxes.conf.argmax().item()
            bbox = boxes.xyxy[best_idx].cpu().numpy()
            conf = boxes.conf[best_idx].item()
            
            # Calculate center
            cx = (bbox[0] + bbox[2]) / 2
            cy = (bbox[1] + bbox[3]) / 2
            center = (cx, cy)
            
            # Calculate normalized error from image center
            image_error = (np.array(center) - image_center) / np.array([w/2, h/2])
            
            return DetectionResult(
                detected=True,
                bbox=bbox.tolist(),
                center=center,
                confidence=conf,
                image_error=image_error,
                processing_time=0.0
            )
        
        return DetectionResult(
            detected=False,
            bbox=None,
            center=None,
            confidence=0.0,
            image_error=np.zeros(2),
            processing_time=0.0
        )
    
    def _fallback_detect(self, 
                         image: np.ndarray,
                         image_center: np.ndarray) -> DetectionResult:
        """
        Fallback detection using traditional CV
        Uses color thresholding for black points
        """
        h, w = image.shape[:2]
        
        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Apply Gaussian blur to reduce noise
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        
        # Threshold for dark regions (black points)
        _, binary = cv2.threshold(blurred, 60, 255, cv2.THRESH_BINARY_INV)
        
        # Morphological operations to clean up
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
        
        # Find contours
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            return DetectionResult(
                detected=False,
                bbox=None,
                center=None,
                confidence=0.0,
                image_error=np.zeros(2),
                processing_time=0.0
            )
        
        # Filter contours by area and circularity
        valid_contours = []
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < 20 or area > 15000:
                continue
            
            # Check circularity
            perimeter = cv2.arcLength(cnt, True)
            if perimeter > 0:
                circularity = 4 * np.pi * area / (perimeter ** 2)
                if circularity > 0.3:  # Reasonably circular
                    valid_contours.append((cnt, area, circularity))
        
        if not valid_contours:
            return DetectionResult(
                detected=False,
                bbox=None,
                center=None,
                confidence=0.0,
                image_error=np.zeros(2),
                processing_time=0.0
            )
        
        # Select best contour (largest area among circular ones)
        best_cnt, best_area, best_circ = max(valid_contours, key=lambda x: x[1])
        
        # Calculate moments and center
        M = cv2.moments(best_cnt)
        if M["m00"] > 0:
            cx = M["m10"] / M["m00"]
            cy = M["m01"] / M["m00"]
            center = (cx, cy)
            
            # Bounding box
            x, y, bw, bh = cv2.boundingRect(best_cnt)
            bbox = [float(x), float(y), float(x + bw), float(y + bh)]
            
            # Normalized error
            image_error = (np.array(center) - image_center) / np.array([w/2, h/2])
            
            # Confidence based on circularity and area
            conf = min(0.5 + best_circ * 0.3 + min(best_area / 5000, 0.2), 1.0)
            
            return DetectionResult(
                detected=True,
                bbox=bbox,
                center=center,
                confidence=conf,
                image_error=image_error,
                processing_time=0.0
            )
        
        return DetectionResult(
            detected=False,
            bbox=None,
            center=None,
            confidence=0.0,
            image_error=np.zeros(2),
            processing_time=0.0
        )
    
    def visualize(self, 
                  image: np.ndarray, 
                  result: DetectionResult,
                  draw_center_cross: bool = True) -> np.ndarray:
        """
        Visualize detection result on image
        
        Args:
            image: Original BGR image
            result: DetectionResult from detect()
            draw_center_cross: Draw crosshair at image center
            
        Returns:
            Annotated image
        """
        vis = image.copy()
        h, w = vis.shape[:2]
        
        # Draw image center crosshair
        if draw_center_cross:
            cx, cy = w // 2, h // 2
            cv2.line(vis, (cx - 20, cy), (cx + 20, cy), (0, 255, 255), 2)
            cv2.line(vis, (cx, cy - 20), (cx, cy + 20), (0, 255, 255), 2)
        
        if result.detected:
            # Draw bounding box
            if result.bbox:
                x1, y1, x2, y2 = [int(v) for v in result.bbox]
                cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
            # Draw center point
            if result.center:
                cx, cy = int(result.center[0]), int(result.center[1])
                cv2.circle(vis, (cx, cy), 5, (0, 0, 255), -1)
                
                # Draw line from image center to detection
                img_cx, img_cy = w // 2, h // 2
                cv2.line(vis, (img_cx, img_cy), (cx, cy), (255, 0, 255), 2)
            
            # Add text info
            info_text = f"Conf: {result.confidence:.2f}"
            cv2.putText(vis, info_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 
                       0.7, (0, 255, 0), 2)
            
            error_text = f"Error: ({result.image_error[0]:.3f}, {result.image_error[1]:.3f})"
            cv2.putText(vis, error_text, (10, 60), cv2.FONT_HERSHEY_SIMPLEX,
                       0.7, (0, 255, 0), 2)
        else:
            cv2.putText(vis, "No detection", (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                       0.7, (0, 0, 255), 2)
        
        # Processing time
        time_text = f"Time: {result.processing_time:.1f}ms"
        cv2.putText(vis, time_text, (10, h - 20), cv2.FONT_HERSHEY_SIMPLEX,
                   0.6, (255, 255, 255), 2)
        
        return vis


class DetectorBenchmark:
    """Benchmark detector performance"""
    
    def __init__(self, detector: YOLOWorldDetector):
        self.detector = detector
        self.times = []
    
    def run(self, image: np.ndarray, num_iterations: int = 100) -> Dict:
        """Run benchmark"""
        self.times = []
        
        # Warmup
        for _ in range(10):
            self.detector.detect(image)
        
        # Benchmark
        for _ in range(num_iterations):
            result = self.detector.detect(image)
            self.times.append(result.processing_time)
        
        return {
            'mean_time_ms': np.mean(self.times),
            'std_time_ms': np.std(self.times),
            'min_time_ms': np.min(self.times),
            'max_time_ms': np.max(self.times),
            'fps': 1000 / np.mean(self.times)
        }


def test_detector():
    """Test detector with camera or test image"""
    print("Testing Black Point Detector...")
    
    # Initialize detector
    detector = YOLOWorldDetector(
        model_path="yolov8x-worldv2.pt",
        device="cuda" if cv2.cuda.getCudaEnabledDeviceCount() > 0 else "cpu"
    )
    
    # Try to open camera
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    
    if not cap.isOpened():
        print("No camera available, using synthetic test image")
        # Create synthetic test image
        image = np.ones((480, 640, 3), dtype=np.uint8) * 200
        # Draw black circle as target
        cv2.circle(image, (400, 300), 20, (0, 0, 0), -1)
        
        result = detector.detect(image)
        print(f"Detection result: {result}")
        
        vis = detector.visualize(image, result)
        cv2.imwrite("test_detection.jpg", vis)
        print("Saved test_detection.jpg")
        return
    
    print("Press 'q' to quit, 's' to save frame")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Detect
        result = detector.detect(frame)
        
        # Visualize
        vis = detector.visualize(frame, result)
        
        # Note: imshow may not work on headless server
        try:
            cv2.imshow("Detection", vis)
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord('q'):
                break
            elif key == ord('s'):
                cv2.imwrite("captured_frame.jpg", vis)
                print("Saved captured_frame.jpg")
        except:
            # Headless mode - just print results
            print(f"Detected: {result.detected}, Error: {result.image_error}, Time: {result.processing_time:.1f}ms")
            time.sleep(0.1)
    
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    test_detector()
