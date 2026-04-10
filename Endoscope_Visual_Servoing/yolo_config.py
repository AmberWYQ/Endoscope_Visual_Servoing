"""
Multi-Dataset Configuration for YOLO Training Pipeline
=======================================================

Centralized configuration file for managing multiple endoscopy datasets.
Each dataset has its own JSON label file and image directory.

Supports new label_template.json format with:
- roi_name: the class/prompt name for YOLO (e.g., "coin", "fish bone")
- annotations: list of {image_name, detection_result} where detection_result
  is either "no_detection" or [x, y, w, h]
- image_size: [width, height] array (e.g., [640, 480])
- coordinate_format: "[x, y, w, h]"
- coordinate_origin: "upper-left"

Author: NCK
Date: 2025
"""

import os
import json
from pathlib import Path
from typing import List, Dict, Optional, Tuple, Set
from dataclasses import dataclass, field
from datetime import datetime


# =============================================================================
# Dataset Configuration
# =============================================================================

@dataclass
class DatasetConfig:
    """Configuration for a single dataset."""
    name: str                    # Dataset identifier (roi_name from JSON, e.g., "coin")
    labels_json: str             # Path to labels JSON file
    images_dir: str              # Path to images directory
    enabled: bool = True         # Whether to include in training/evaluation
    
    def __post_init__(self):
        """Validate paths exist."""
        if self.enabled:
            if not os.path.exists(self.labels_json):
                print(f"⚠️  Warning: Labels file not found: {self.labels_json}")
            if not os.path.exists(self.images_dir):
                print(f"⚠️  Warning: Images directory not found: {self.images_dir}")


# =============================================================================
# DATASETS - Add your datasets here
# =============================================================================

DATASETS: List[DatasetConfig] = [
    # Example datasets - modify these paths to match your setup
    DatasetConfig(
        name="white oval suspicious region",
        labels_json="/media/kit/endovla+/dataset/data_processing/endovla/phantom/biopsy/white_oval_suspicious_region/json/0001.json",
        images_dir="/media/kit/endovla+/dataset/data_processing/endovla/phantom/biopsy/white_oval_suspicious_region/0001",
        enabled=True
    ),
    DatasetConfig(
        name="white oval suspicious region",
        labels_json="/media/kit/endovla+/dataset/data_processing/endovla/phantom/biopsy/white_oval_suspicious_region/json/0002.json",
        images_dir="/media/kit/endovla+/dataset/data_processing/endovla/phantom/biopsy/white_oval_suspicious_region/0002",
        enabled=True
    ),
    DatasetConfig(
        name="white oval suspicious region",
        labels_json="/media/kit/endovla+/dataset/data_processing/endovla/phantom/biopsy/white_oval_suspicious_region/json/0003.json",
        images_dir="/media/kit/endovla+/dataset/data_processing/endovla/phantom/biopsy/white_oval_suspicious_region/0003",
        enabled=True
    ),
    DatasetConfig(
        name="white oval suspicious region",
        labels_json="/media/kit/endovla+/dataset/data_processing/endovla/phantom/biopsy/white_oval_suspicious_region/json/0004.json",
        images_dir="/media/kit/endovla+/dataset/data_processing/endovla/phantom/biopsy/white_oval_suspicious_region/0004",
        enabled=True
    ),
    DatasetConfig(
        name="polyp",
        labels_json="/media/kit/endovla+/dataset/data_processing/endovla/phantom/biopsy/polyp/json/0001.json",
        images_dir="/media/kit/endovla+/dataset/data_processing/endovla/phantom/biopsy/polyp/0001",
        enabled=True
    ), 
    # DatasetConfig(
    #     name="polyp",
    #     labels_json="/media/kit/endovla+/dataset/data_processing/endovla/phantom/biopsy/polyp/json/0002_correct.json",
    #     images_dir="/media/kit/endovla+/dataset/data_processing/endovla/phantom/biopsy/polyp/0002",
    #     enabled=True
    # ), 
    DatasetConfig(
        name="polyp",
        labels_json="/media/kit/endovla+/dataset/data_processing/endovla/phantom/biopsy/polyp/json/0003.json",
        images_dir="/media/kit/endovla+/dataset/data_processing/endovla/phantom/biopsy/polyp/0003",
        enabled=True
    ),
    DatasetConfig(
        name="polyp",
        labels_json="/media/kit/endovla+/dataset/data_processing/endovla/phantom/biopsy/polyp/json/0004.json",
        images_dir="/media/kit/endovla+/dataset/data_processing/endovla/phantom/biopsy/polyp/0004",
        enabled=True
    ),
    DatasetConfig(
        name="orange protruding lesion",
        labels_json="/media/kit/endovla+/dataset/data_processing/endovla/phantom/biopsy/orange_protruding_lesion/json/0001.json",
        images_dir="/media/kit/endovla+/dataset/data_processing/endovla/phantom/biopsy/orange_protruding_lesion/0001",
        enabled=True
    ),
]


# =============================================================================
# Model Configuration
# =============================================================================

@dataclass
class ModelConfig:
    """Model training and inference configuration."""
    # Base model
    base_model: str = "yolov8x-worldv2.pt"
    
    # Training parameters
    epochs: int = 100
    batch_size: int = 16
    img_size: int = 640
    lr0: float = 0.01
    lrf: float = 0.01
    momentum: float = 0.937
    weight_decay: float = 0.0005
    warmup_epochs: float = 3.0
    patience: int = 50
    
    # Dataset split
    train_split: float = 0.8
    val_split: float = 0.15
    test_split: float = 0.05
    
    # Inference parameters
    conf_threshold: float = 0.25
    iou_threshold: float = 0.45
    
    # Device
    device: Optional[str] = None  # None for auto-detect
    
    # Output directories
    output_dir: str = "./yolo_dataset"
    project: str = "runs/train"
    run_name: Optional[str] = None  # Auto-generated if None


# Default model configuration
MODEL_CONFIG = ModelConfig()


# =============================================================================
# Color Palette for Visualization
# =============================================================================

COLORS = [
    (6, 182, 212),    # Cyan
    (139, 92, 246),   # Purple
    (245, 158, 11),   # Amber
    (16, 185, 129),   # Emerald
    (239, 68, 68),    # Red
    (236, 72, 153),   # Pink
    (59, 130, 246),   # Blue
    (100, 116, 139),  # Slate
    (132, 204, 22),   # Lime
    (249, 115, 22),   # Orange
    (20, 184, 166),   # Teal
    (168, 85, 247),   # Violet
    (34, 197, 94),    # Green
    (251, 146, 60),   # Orange-400
    (56, 189, 248),   # Sky
]


# =============================================================================
# Label JSON Parser (New Format)
# =============================================================================

def parse_label_json(json_path: str) -> Dict:
    """
    Parse the new label_template.json format.
    
    Expected format:
    {
        "roi_name": "coin",
        "image_size": [640, 480],
        "coordinate_format": "[x, y, w, h]",
        "coordinate_origin": "upper-left",
        "ref_images": [...],  # ignored
        "annotations": [
            {"image_name": "0001.jpg", "detection_result": "no_detection"},
            {"image_name": "0002.jpg", "detection_result": [x, y, w, h]},
            ...
        ]
    }
    
    Returns:
        Dict with keys: roi_name, image_size, labels, negatives
    """
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    roi_name = data.get('roi_name', 'unknown')
    
    # Parse image_size - can be [w, h] array or {"width": w, "height": h}
    image_size_raw = data.get('image_size', [640, 480])
    if isinstance(image_size_raw, list):
        image_size = {'width': image_size_raw[0], 'height': image_size_raw[1]}
    else:
        image_size = image_size_raw
    
    # Parse annotations into labels and negatives
    labels = {}  # image_name -> [box_dict]
    negatives = []
    
    annotations = data.get('annotations', [])
    for ann in annotations:
        img_name = ann.get('image_name', '')
        result = ann.get('detection_result', 'no_detection')
        
        if result == 'no_detection':
            negatives.append(img_name)
        elif isinstance(result, list) and len(result) >= 4:
            # [x, y, w, h] format
            x, y, w, h = result[0], result[1], result[2], result[3]
            box = {
                'x': x,
                'y': y,
                'w': w,
                'h': h,
                'promptIndex': 0,  # Single class per dataset
                'prompt': roi_name
            }
            labels[img_name] = [box]
    
    return {
        'roi_name': roi_name,
        'image_size': image_size,
        'labels': labels,
        'negatives': negatives,
        'total_images': data.get('total_images', len(annotations)),
        'labeled_count': data.get('labeled_count', len(labels)),
        'no_detection_count': data.get('no_detection_count', len(negatives))
    }


# =============================================================================
# Multi-Dataset Manager
# =============================================================================

class MultiDatasetManager:
    """
    Manages multiple datasets, collecting prompts and validating labels.
    
    Each dataset has a single class (roi_name), and we build a unified
    class list across all datasets.
    """
    
    def __init__(self, datasets: Optional[List[DatasetConfig]] = None):
        """
        Initialize with list of dataset configurations.
        
        Args:
            datasets: List of DatasetConfig objects. Uses DATASETS if None.
        """
        self.datasets = datasets or DATASETS
        self.enabled_datasets = [d for d in self.datasets if d.enabled]
        
        # Collected data
        self._prompt_names: List[str] = []
        self._prompt_to_index: Dict[str, int] = {}
        self._dataset_data: Dict[str, Dict] = {}
        
        # Load and validate all datasets
        self._load_datasets()
    
    def _load_datasets(self):
        """Load all enabled datasets and collect prompts."""
        print("\n" + "="*60)
        print("Loading Datasets")
        print("="*60)
        
        all_prompt_names: Set[str] = set()
        
        for ds in self.enabled_datasets:
            print(f"\n📁 Loading: {ds.name}")
            print(f"   Labels: {ds.labels_json}")
            print(f"   Images: {ds.images_dir}")
            
            if not os.path.exists(ds.labels_json):
                print(f"   ❌ Labels file not found, skipping")
                continue
            
            # Parse the new JSON format
            parsed = parse_label_json(ds.labels_json)
            
            self._dataset_data[ds.name] = {
                'config': ds,
                'data': parsed
            }
            
            # The roi_name is the class for this dataset
            roi_name = parsed['roi_name']
            all_prompt_names.add(roi_name)
            
            # Print stats
            labels = parsed.get('labels', {})
            negatives = parsed.get('negatives', [])
            total_boxes = sum(len(boxes) for boxes in labels.values())
            
            print(f"   ✓ ROI Name (class): {roi_name}")
            print(f"   ✓ Labeled images: {len(labels)}")
            print(f"   ✓ No detection images: {len(negatives)}")
            print(f"   ✓ Total boxes: {total_boxes}")
        
        # Build unified prompt list (sorted for consistency)
        self._prompt_names = sorted(list(all_prompt_names))
        self._prompt_to_index = {name: i for i, name in enumerate(self._prompt_names)}
        
        print(f"\n{'='*60}")
        print(f"📊 Combined Statistics")
        print(f"{'='*60}")
        print(f"  Total datasets: {len(self.enabled_datasets)}")
        print(f"  Total unique classes: {len(self._prompt_names)}")
        print(f"  Classes: {self._prompt_names}")
    
    @property
    def prompts(self) -> List[str]:
        """Get list of all unique prompts/classes."""
        return self._prompt_names
    
    @property
    def num_classes(self) -> int:
        """Get total number of classes."""
        return len(self._prompt_names)
    
    def get_unified_prompt_index(self, dataset_name: str, local_index: int = 0) -> int:
        """
        Convert a dataset's local prompt index to unified index.
        
        For the new format, each dataset has only one class (roi_name),
        so local_index is typically 0.
        """
        if dataset_name not in self._dataset_data:
            raise ValueError(f"Unknown dataset: {dataset_name}")
        
        roi_name = self._dataset_data[dataset_name]['data']['roi_name']
        return self._prompt_to_index.get(roi_name, 0)
    
    def get_dataset_labels(self, dataset_name: str) -> Tuple[Dict, Set, Dict]:
        """
        Get labels, negatives, and image size for a specific dataset.
        
        Args:
            dataset_name: Name of the dataset
            
        Returns:
            Tuple of (labels dict, negatives set, image_size dict)
        """
        if dataset_name not in self._dataset_data:
            raise ValueError(f"Unknown dataset: {dataset_name}")
        
        data = self._dataset_data[dataset_name]['data']
        labels = data.get('labels', {})
        negatives = set(data.get('negatives', []))
        image_size = data.get('image_size', {'width': 640, 'height': 480})
        
        return labels, negatives, image_size
    
    def iter_all_labeled_images(self):
        """
        Iterate over all labeled images across all datasets.
        
        Yields:
            Tuple of (dataset_name, image_name, boxes, is_negative, images_dir)
        """
        for ds in self.enabled_datasets:
            if ds.name not in self._dataset_data:
                continue
            
            data = self._dataset_data[ds.name]['data']
            labels = data.get('labels', {})
            negatives = set(data.get('negatives', []))
            
            # Images with boxes
            for img_name, boxes in labels.items():
                if boxes:  # Non-empty boxes
                    # Convert prompt indices to unified indices
                    unified_boxes = []
                    for box in boxes:
                        unified_box = box.copy()
                        unified_box['promptIndex'] = self.get_unified_prompt_index(ds.name, 0)
                        unified_boxes.append(unified_box)
                    
                    yield ds.name, img_name, unified_boxes, False, ds.images_dir
            
            # Negative samples (no_detection)
            for img_name in negatives:
                yield ds.name, img_name, [], True, ds.images_dir
    
    def get_all_images_count(self) -> Dict[str, int]:
        """Get count of images per dataset."""
        counts = {}
        for ds in self.enabled_datasets:
            if ds.name not in self._dataset_data:
                counts[ds.name] = 0
                continue
            
            data = self._dataset_data[ds.name]['data']
            labels = data.get('labels', {})
            negatives = data.get('negatives', [])
            
            labeled_count = len([k for k, v in labels.items() if len(v) > 0])
            counts[ds.name] = labeled_count + len(negatives)
        
        return counts
    
    def get_total_labeled_images(self) -> int:
        """Get total number of labeled images across all datasets."""
        return sum(self.get_all_images_count().values())


# =============================================================================
# Utility Functions
# =============================================================================

def get_all_images(images_dir: Path, extensions: tuple = ('.jpg', '.jpeg', '.png')) -> List[str]:
    """
    Recursively get all images from a directory, supporting both flat and nested structures.
    
    Args:
        images_dir: Root directory containing images
        extensions: Tuple of valid image extensions
        
    Returns:
        List of relative image paths (e.g., 'image.jpg' or 'subdir/image.jpg')
    """
    images_dir = Path(images_dir)
    all_images = []
    
    for f in images_dir.rglob("*"):
        if f.is_file() and f.suffix.lower() in extensions:
            rel_path = str(f.relative_to(images_dir))
            rel_path = rel_path.replace("\\", "/")
            all_images.append(rel_path)
    
    return sorted(all_images)


def flatten_path(path: str) -> str:
    """
    Flatten a path with subdirectories to a safe filename.
    
    Args:
        path: Path like 'subdir/image.jpg'
        
    Returns:
        Flattened name like 'subdir_image.jpg'
    """
    return path.replace("/", "_").replace("\\", "_")


def generate_run_name(prefix: str = "endovlayolo") -> str:
    """Generate a timestamped run name."""
    timestamp = datetime.now().strftime("%Y%m%d")
    return f"{prefix}_{timestamp}"


def load_prompts_from_json(json_path: str) -> List[str]:
    """
    Load prompts from a JSON file.
    
    Supports both old format (prompts list) and new format (roi_name).
    
    Args:
        json_path: Path to JSON file
        
    Returns:
        List of prompt names
    """
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    # New format - single roi_name
    if 'roi_name' in data:
        return [data['roi_name']]
    
    # Old format - prompts list
    if isinstance(data, list):
        return data
    elif 'prompts' in data:
        return [p['name'] if isinstance(p, dict) else p for p in data['prompts']]
    else:
        return []


def print_config_summary(manager: MultiDatasetManager, model_config: ModelConfig):
    """Print configuration summary."""
    print("\n" + "="*60)
    print("Configuration Summary")
    print("="*60)
    
    print("\n📊 Datasets:")
    for ds in manager.enabled_datasets:
        print(f"  - {ds.name}")
    
    print(f"\n📋 Classes ({manager.num_classes}):")
    for i, name in enumerate(manager.prompts):
        print(f"  {i}: {name}")
    
    print(f"\n⚙️  Model Configuration:")
    print(f"  Base model: {model_config.base_model}")
    print(f"  Epochs: {model_config.epochs}")
    print(f"  Batch size: {model_config.batch_size}")
    print(f"  Image size: {model_config.img_size}")
    print(f"  Learning rate: {model_config.lr0}")
    print(f"  Train/Val/Test split: {model_config.train_split}/{model_config.val_split}/{model_config.test_split}")


# =============================================================================
# Quick Setup Functions
# =============================================================================

def add_dataset(name: str, labels_json: str, images_dir: str, enabled: bool = True):
    """
    Add a dataset to the global DATASETS list.
    
    Args:
        name: Dataset identifier (roi_name)
        labels_json: Path to labels JSON file
        images_dir: Path to images directory
        enabled: Whether to include in training/evaluation
    """
    DATASETS.append(DatasetConfig(
        name=name,
        labels_json=labels_json,
        images_dir=images_dir,
        enabled=enabled
    ))


def setup_datasets(dataset_configs: List[Dict]):
    """
    Setup multiple datasets from a list of dictionaries.
    
    Args:
        dataset_configs: List of dicts with keys: name, labels_json, images_dir, enabled
    """
    global DATASETS
    DATASETS = []
    
    for config in dataset_configs:
        add_dataset(
            name=config['name'],
            labels_json=config['labels_json'],
            images_dir=config['images_dir'],
            enabled=config.get('enabled', True)
        )


# =============================================================================
# Exports
# =============================================================================

__all__ = [
    # Classes
    'DatasetConfig',
    'ModelConfig',
    'MultiDatasetManager',
    
    # Global configs
    'DATASETS',
    'MODEL_CONFIG',
    'COLORS',
    
    # Functions
    'parse_label_json',
    'get_all_images',
    'flatten_path',
    'generate_run_name',
    'load_prompts_from_json',
    'print_config_summary',
    'add_dataset',
    'setup_datasets',
]
