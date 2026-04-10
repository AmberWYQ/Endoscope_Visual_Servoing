"""
YOLO-E Visual Servoing Package

Instruction-based visual servoing using YOLO-World for detection
and pretrained neural network controller for endoscope control.
"""

from .yoloe_perception_interface import (
    YOLOEPerceptionInterface,
    MockYOLOEPerceptionInterface,
    DetectionResult,
    create_yoloe_perception_interface,
)

from .yoloe_combined_config import (
    IntegratedConfig,
    get_config,
    get_config_from_args,
)

__version__ = "1.0.0"
__author__ = "NCK"
