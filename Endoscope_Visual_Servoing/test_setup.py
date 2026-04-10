#!/usr/bin/env python
"""
Test Script for YOLO-E Visual Servoing System

Verifies that all components are properly installed and working.
Run this script to check your setup before running the main control system.

Usage:
    python test_setup.py
"""

import sys

def test_imports():
    """Test all required imports"""
    print("=" * 50)
    print("Testing imports...")
    print("=" * 50)
    
    errors = []
    
    # Core dependencies
    try:
        import numpy as np
        print(f"✓ numpy: {np.__version__}")
    except ImportError as e:
        errors.append(f"✗ numpy: {e}")
    
    try:
        import cv2
        print(f"✓ opencv: {cv2.__version__}")
    except ImportError as e:
        errors.append(f"✗ opencv: {e}")
    
    try:
        import torch
        cuda_status = "CUDA available" if torch.cuda.is_available() else "CPU only"
        print(f"✓ torch: {torch.__version__} ({cuda_status})")
    except ImportError as e:
        errors.append(f"✗ torch: {e}")
    
    # YOLO-E
    try:
        from ultralytics import YOLO
        print(f"✓ ultralytics (YOLO-E)")
    except ImportError as e:
        errors.append(f"✗ ultralytics: {e}")
    
    # Pygame (optional)
    try:
        import pygame
        print(f"✓ pygame: {pygame.version.ver}")
    except ImportError:
        print("○ pygame: Not installed (optional, will use OpenCV display)")
    
    # Pyserial (optional)
    try:
        import serial
        print(f"✓ pyserial: {serial.VERSION}")
    except ImportError:
        print("○ pyserial: Not installed (only needed for real robot)")
    
    print()
    return errors


def test_local_modules():
    """Test local module imports"""
    print("=" * 50)
    print("Testing local modules...")
    print("=" * 50)
    
    errors = []
    
    modules = [
        ('yoloe_perception_interface', 'YOLOEPerceptionInterface'),
        ('yoloe_combined_config', 'YOLOEIntegratedConfig'),
        ('control_interface', 'ControlInterface'),
        ('safety_manager', 'SafetyManager'),
        ('data_recorder', 'DataRecorder'),
    ]
    
    for module_name, class_name in modules:
        try:
            module = __import__(module_name)
            cls = getattr(module, class_name)
            print(f"✓ {module_name}.{class_name}")
        except ImportError as e:
            errors.append(f"✗ {module_name}: {e}")
        except AttributeError as e:
            errors.append(f"✗ {module_name}.{class_name}: {e}")
    
    print()
    return errors


def test_mock_perception():
    """Test mock perception interface"""
    print("=" * 50)
    print("Testing mock perception...")
    print("=" * 50)
    
    try:
        import numpy as np
        from yoloe_perception_interface import MockYOLOEPerceptionInterface
        
        mock = MockYOLOEPerceptionInterface(target_classes="test")
        frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        
        result = mock.detect(frame)
        print(f"✓ Mock detection: center={result.center}, conf={result.confidence:.2f}")
        return []
    except Exception as e:
        return [f"✗ Mock perception test failed: {e}"]


def test_yolo_model():
    """Test YOLO model loading"""
    print("=" * 50)
    print("Testing YOLO model...")
    print("=" * 50)
    
    try:
        from ultralytics import YOLO
        import numpy as np
        
        # Try to load pretrained model
        print("Loading yolov8s-worldv2.pt (small model for testing)...")
        model = YOLO("yolov8s-worldv2.pt")
        model.set_classes(["test object"])
        
        # Test inference on dummy image
        frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        results = model.predict(frame, conf=0.25, verbose=False)
        
        print(f"✓ YOLO-World model loaded and working")
        print(f"  Model: yolov8s-worldv2.pt")
        print(f"  Inference successful")
        return []
    except Exception as e:
        return [f"✗ YOLO model test failed: {e}"]


def test_control_interface():
    """Test control interface"""
    print("=" * 50)
    print("Testing control interface...")
    print("=" * 50)
    
    try:
        import numpy as np
        from control_interface import MockControlInterface
        from yoloe_perception_interface import DetectionResult
        
        mock_ctrl = MockControlInterface()
        
        # Create dummy detection result
        detection = DetectionResult(
            bbox=np.array([100, 100, 80, 60]),
            center=np.array([140, 130]),
            confidence=0.8,
            detection_score=0.75,
            no_detection=False,
            status='tracking',
            lost_count=0,
            inference_time=0.01,
            fps=100.0,
        )
        
        frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        action = mock_ctrl.compute_action(frame, detection, timestamp=0.0)
        
        print(f"✓ Mock control: action={action.action}")
        return []
    except Exception as e:
        return [f"✗ Control interface test failed: {e}"]


def main():
    """Run all tests"""
    print("\n")
    print("=" * 50)
    print("YOLO-E Visual Servoing Setup Test")
    print("=" * 50)
    print()
    
    all_errors = []
    
    all_errors.extend(test_imports())
    all_errors.extend(test_local_modules())
    all_errors.extend(test_mock_perception())
    all_errors.extend(test_control_interface())
    
    # Optional: test YOLO model
    print("=" * 50)
    print("Optional: Test YOLO model loading?")
    print("(This will download ~50MB model file)")
    print("=" * 50)
    
    try:
        user_input = input("Test YOLO model? [y/N]: ").strip().lower()
        if user_input == 'y':
            all_errors.extend(test_yolo_model())
        else:
            print("Skipping YOLO model test")
    except:
        print("Skipping YOLO model test (non-interactive mode)")
    
    # Summary
    print("\n")
    print("=" * 50)
    print("Summary")
    print("=" * 50)
    
    if all_errors:
        print(f"\n❌ {len(all_errors)} error(s) found:\n")
        for error in all_errors:
            print(f"  {error}")
        print("\nPlease fix these errors before running the control system.")
        return 1
    else:
        print("\n✅ All tests passed!")
        print("\nYou can now run the control system:")
        print("  python yoloe_control_main.py --mode simulation --mock-all --target polyp")
        return 0


if __name__ == '__main__':
    sys.exit(main())
