"""
YOLO-World Fine-tuning for Endoscopic Images (Multi-Dataset)
=============================================================

This script fine-tunes YOLO-World model on multiple custom endoscopic image 
datasets with open-vocabulary detection capabilities.

Supports new label_template.json format:
- roi_name: class name (e.g., "coin")
- annotations: [{image_name, detection_result}] where detection_result
  is "no_detection" or [x, y, w, h]
- image_size: [640, 480]

Best checkpoint is saved as endovlayolo_YYYYMMDD.pt

Author: NCK
Date: 2025
"""

import os
import sys
import json
import yaml
import shutil
import argparse
import random
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from datetime import datetime

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from yolo_config import (
    DatasetConfig,
    ModelConfig,
    MultiDatasetManager,
    DATASETS,
    MODEL_CONFIG,
    get_all_images,
    flatten_path,
    generate_run_name,
    print_config_summary,
    add_dataset,
    setup_datasets,
)

try:
    import cv2
except ImportError:
    print("Installing opencv-python...")
    os.system("pip install opencv-python --break-system-packages")
    import cv2

try:
    from ultralytics import YOLO
except ImportError:
    print("Installing ultralytics...")
    os.system("pip install ultralytics --break-system-packages")
    from ultralytics import YOLO

try:
    import torch
except ImportError:
    print("Installing torch...")
    os.system("pip install torch torchvision --break-system-packages")
    import torch


class MultiDatasetConverter:
    """
    Converts multiple labeled endoscopy datasets to YOLO training format.
    
    Handles conversion from new label_template.json format to YOLO txt format.
    Supports partial labeling mode and nested directory structures.
    """
    
    def __init__(
        self,
        manager: MultiDatasetManager,
        output_dir: str,
        train_split: float = 0.8,
        val_split: float = 0.15,
        test_split: float = 0.05,
        use_only_labeled: bool = True
    ):
        """
        Initialize the dataset converter.
        
        Args:
            manager: MultiDatasetManager instance with loaded datasets
            output_dir: Output directory for YOLO dataset
            train_split: Fraction for training set
            val_split: Fraction for validation set
            test_split: Fraction for test set
            use_only_labeled: If True, only use labeled images for training
        """
        self.manager = manager
        self.output_dir = Path(output_dir)
        self.train_split = train_split
        self.val_split = val_split
        self.test_split = test_split
        self.use_only_labeled = use_only_labeled
        
        # Collect all labeled images
        self.all_labeled = []
        for ds_name, img_name, boxes, is_negative, images_dir in manager.iter_all_labeled_images():
            self.all_labeled.append({
                'dataset': ds_name,
                'image': img_name,
                'boxes': boxes,
                'is_negative': is_negative,
                'images_dir': images_dir
            })

    from pathlib import Path
    from typing import Optional

    def _resolve_image_path(self, images_dir: Path, img_name: str) -> Optional[Path]:
        """
        Resolve image path even if extension in label differs from disk.
        Tries exact name first, then swaps common extensions by stem.
        """
        p = images_dir / img_name
        if p.exists():
            return p

        stem = Path(img_name).stem
        for ext in (".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG"):
            cand = images_dir / f"{stem}{ext}"
            if cand.exists():
                return cand

        # last resort: any file with same stem
        hits = list(images_dir.glob(f"{stem}.*"))
        return hits[0] if hits else None

    def convert(self) -> str:
        """
        Convert all datasets to YOLO format.
        
        Returns:
            Path to the generated dataset YAML file
        """
        print(f"\n{'='*60}")
        print("Converting Multi-Dataset to YOLO Format")
        print(f"{'='*60}")
        
        print(f"\n📊 Dataset Statistics:")
        print(f"  - Number of datasets: {len(self.manager.enabled_datasets)}")
        print(f"  - Number of classes: {self.manager.num_classes}")
        print(f"  - Total labeled images: {len(self.all_labeled)}")
        
        # Count per dataset
        per_dataset = {}
        for item in self.all_labeled:
            ds = item['dataset']
            per_dataset[ds] = per_dataset.get(ds, 0) + 1
        
        print(f"\n  Per-dataset breakdown:")
        for ds, count in per_dataset.items():
            print(f"    - {ds}: {count} images")
        
        # Create directory structure
        self._create_directories()
        
        # Shuffle and split
        random.shuffle(self.all_labeled)
        n_total = len(self.all_labeled)
        n_train = int(n_total * self.train_split)
        n_val = int(n_total * self.val_split)
        
        train_items = self.all_labeled[:n_train]
        val_items = self.all_labeled[n_train:n_train + n_val]
        test_items = self.all_labeled[n_train + n_val:]
        
        print(f"\n📁 Dataset split:")
        print(f"  - Train: {len(train_items)}")
        print(f"  - Val: {len(val_items)}")
        print(f"  - Test: {len(test_items)}")
        
        # Process each split
        self._process_split(train_items, "train")
        self._process_split(val_items, "val")
        self._process_split(test_items, "test")
        
        # Create dataset YAML
        yaml_path = self._create_yaml()
        
        # Save dataset info
        self._save_dataset_info()
        
        print(f"\n✅ Dataset created at: {self.output_dir}")
        print(f"   YAML config: {yaml_path}")
        
        return str(yaml_path)
    
    def _create_directories(self):
        """Create YOLO dataset directory structure."""
        for split in ["train", "val", "test"]:
            (self.output_dir / split / "images").mkdir(parents=True, exist_ok=True)
            (self.output_dir / split / "labels").mkdir(parents=True, exist_ok=True)
    
    def _process_split(self, items: List[Dict], split: str):
        """Process images for a specific split."""
        images_out = self.output_dir / split / "images"
        labels_out = self.output_dir / split / "labels"
        
        for item in items:
            ds_name = item['dataset']
            img_name = item['image']
            boxes = item['boxes']
            is_negative = item['is_negative']
            images_dir = Path(item['images_dir'])
            
            # Source image path (robust resolution)
            src_image = self._resolve_image_path(images_dir, img_name)
            if src_image is None:
                print(f"  Warning: Image not found (any ext): {images_dir / img_name}")
                continue

            # Create unique output name (keep REAL filename & extension)
            flat_name = f"{ds_name.replace(' ', '_')}_{flatten_path(src_image.name)}"

            # Copy image
            dst_image = images_out / flat_name
            shutil.copy2(src_image, dst_image)

            # Read dims from resolved path
            img = cv2.imread(str(src_image))
            if img is None:
                print(f"  Warning: cv2 failed to read: {src_image}")
                continue
            h, w = img.shape[:2]
            
            # Create label file
            label_name = Path(flat_name).stem + ".txt"
            label_path = labels_out / label_name
            
            if boxes:
                # Get image size from dataset config
                labels_data, _, image_size = self.manager.get_dataset_labels(ds_name)
                img_w = image_size.get('width', 640)
                img_h = image_size.get('height', 480)
                
                with open(label_path, "w") as f:
                    for box in boxes:
                        # Convert from [x, y, w, h] format to YOLO format
                        x = box["x"]
                        y = box["y"]
                        bw = box["w"]
                        bh = box["h"]
                        
                        # Scale to actual image coordinates, then normalize
                        scale_x = w / img_w
                        scale_y = h / img_h
                        
                        x_scaled = x * scale_x
                        y_scaled = y * scale_y
                        w_scaled = bw * scale_x
                        h_scaled = bh * scale_y
                        
                        # Convert to YOLO format (normalized x_center, y_center, width, height)
                        x_center = (x_scaled + w_scaled / 2) / w
                        y_center = (y_scaled + h_scaled / 2) / h
                        norm_w = w_scaled / w
                        norm_h = h_scaled / h
                        
                        # Use unified prompt index
                        class_id = box.get("promptIndex", 0)
                        
                        f.write(f"{class_id} {x_center:.6f} {y_center:.6f} {norm_w:.6f} {norm_h:.6f}\n")
            
            elif is_negative:
                # Create empty label file for negative samples (no_detection)
                with open(label_path, "w") as f:
                    pass
    
    def _create_yaml(self) -> Path:
        """Create YOLO dataset configuration YAML."""
        config = {
            "path": str(self.output_dir.absolute()),
            "train": "train/images",
            "val": "val/images",
            "test": "test/images",
            "nc": self.manager.num_classes,
            "names": self.manager.prompts
        }
        
        yaml_path = self.output_dir / "dataset.yaml"
        with open(yaml_path, "w") as f:
            yaml.dump(config, f, default_flow_style=False, sort_keys=False)
        
        return yaml_path
    
    def _save_dataset_info(self):
        """Save dataset information for reference."""
        info = {
            "created": datetime.now().isoformat(),
            "datasets": [ds.name for ds in self.manager.enabled_datasets],
            "classes": self.manager.prompts,
            "num_classes": self.manager.num_classes,
            "splits": {
                "train": self.train_split,
                "val": self.val_split,
                "test": self.test_split
            },
            "total_images": len(self.all_labeled),
            "per_dataset": {
                ds.name: sum(1 for item in self.all_labeled if item['dataset'] == ds.name)
                for ds in self.manager.enabled_datasets
            }
        }
        
        info_path = self.output_dir / "dataset_info.json"
        with open(info_path, "w") as f:
            json.dump(info, f, indent=2)


class YOLOWorldFineTuner:
    """
    Fine-tuning class for YOLO-World on endoscopic images.
    
    Saves best checkpoint as endovlayolo_YYYYMMDD.pt
    """
    
    def __init__(
        self,
        base_model: str = "yolov8x-worldv2.pt",
        device: Optional[str] = None
    ):
        """
        Initialize the fine-tuner.
        
        Args:
            base_model: Base YOLO-World model to fine-tune
            device: Device to train on ('cuda', 'cpu', or None for auto)
        """
        self.base_model = base_model
        
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
        
        print(f"\n🔧 Fine-tuner initialized")
        print(f"   Base model: {base_model}")
        print(f"   Device: {self.device}")
        
        # Check GPU info
        if torch.cuda.is_available():
            print(f"   GPU: {torch.cuda.get_device_name(0)}")
            print(f"   GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    
    def train(
        self,
        dataset_yaml: str,
        epochs: int = 100,
        batch_size: int = 16,
        img_size: int = 640,
        lr0: float = 0.01,
        lrf: float = 0.01,
        momentum: float = 0.937,
        weight_decay: float = 0.0005,
        warmup_epochs: float = 3.0,
        patience: int = 50,
        project: str = "runs/train",
        name: Optional[str] = None,
        resume: bool = False,
        freeze_layers: int = 0,
        augment: bool = True,
        cache: bool = False,
        workers: int = 8,
        save_period: int = -1,
        verbose: bool = True
    ) -> str:
        """
        Fine-tune the YOLO-World model.
        
        Returns:
            Path to the best model weights (named endovlayolo_YYYYMMDD.pt)
        """
        print("\n" + "="*60)
        print("YOLO-World Fine-tuning")
        print("="*60)
        
        # Generate run name with date format if not provided
        if name is None:
            name = generate_run_name("endovlayolo")
        
        # Load model
        print(f"\nLoading base model: {self.base_model}")
        model = YOLO(self.base_model)
        
        # Training arguments
        train_args = {
            "data": dataset_yaml,
            "epochs": epochs,
            "batch": batch_size,
            "imgsz": img_size,
            "lr0": lr0,
            "lrf": lrf,
            "momentum": momentum,
            "weight_decay": weight_decay,
            "warmup_epochs": warmup_epochs,
            "patience": patience,
            "project": project,
            "name": name,
            "resume": resume,
            "augment": augment,
            "cache": cache,
            "workers": workers,
            "save_period": save_period,
            "verbose": verbose,
            "device": self.device,
            "exist_ok": True,
            "plots": True,
            "save": True,
        }
        
        # Freeze layers if specified
        if freeze_layers > 0:
            train_args["freeze"] = freeze_layers
            print(f"Freezing first {freeze_layers} layers")
        
        print(f"\nTraining configuration:")
        for key, value in train_args.items():
            print(f"  - {key}: {value}")
        
        print("\n" + "-"*60)
        print("Starting training...")
        print("-"*60 + "\n")
        
        # Train
        results = model.train(**train_args)
        
        # Get best model path
        run_dir = Path(project) / name
        original_best = run_dir / "weights" / "best.pt"
        
        # Rename best.pt to endovlayolo_YYYYMMDD.pt
        date_str = datetime.now().strftime("%Y%m%d")
        final_model_name = f"endovlayolo_{date_str}.pt"
        final_model_path = run_dir / "weights" / final_model_name
        
        if original_best.exists():
            shutil.copy2(original_best, final_model_path)
            print(f"\n✅ Best model saved as: {final_model_path}")
        
        print("\n" + "="*60)
        print("✅ Training Complete!")
        print("="*60)
        print(f"\nBest model saved at: {final_model_path}")
        
        # Print final metrics
        if hasattr(results, "results_dict"):
            print("\nFinal Metrics:")
            for key, value in results.results_dict.items():
                if isinstance(value, float):
                    print(f"  - {key}: {value:.4f}")
        
        return str(final_model_path)


def prepare_and_train(
    manager: Optional[MultiDatasetManager] = None,
    output_dir: str = "./yolo_dataset",
    base_model: str = "yolov8x-worldv2.pt",
    epochs: int = 100,
    batch_size: int = 16,
    img_size: int = 640,
    train_split: float = 0.8,
    val_split: float = 0.15,
    device: Optional[str] = None,
    project: str = "runs/train",
    name: Optional[str] = None
) -> str:
    """
    Complete pipeline: convert multi-dataset and train model.
    
    Args:
        manager: MultiDatasetManager instance. If None, creates from DATASETS.
        output_dir: Output directory for converted dataset
        base_model: Base YOLO-World model
        epochs: Training epochs
        batch_size: Batch size
        img_size: Image size
        train_split: Training split ratio
        val_split: Validation split ratio
        device: Training device
        project: Project directory for training
        name: Run name (auto-generated as endovlayolo_YYYYMMDD if None)
        
    Returns:
        Path to the trained model weights
    """
    # Create manager if not provided
    if manager is None:
        manager = MultiDatasetManager()
    
    if manager.num_classes == 0:
        raise ValueError("No classes found in datasets. Please configure DATASETS in config.py")
    
    # Step 1: Convert dataset
    print("\n" + "="*60)
    print("Step 1: Converting Multi-Dataset")
    print("="*60)
    
    converter = MultiDatasetConverter(
        manager=manager,
        output_dir=output_dir,
        train_split=train_split,
        val_split=val_split,
        test_split=1 - train_split - val_split
    )
    
    yaml_path = converter.convert()
    
    # Step 2: Train model
    print("\n" + "="*60)
    print("Step 2: Training Model")
    print("="*60)
    
    trainer = YOLOWorldFineTuner(
        base_model=base_model,
        device=device
    )
    
    if name is None:
        name = generate_run_name("endovlayolo")
    
    best_model = trainer.train(
        dataset_yaml=yaml_path,
        epochs=epochs,
        batch_size=batch_size,
        img_size=img_size,
        project=project,
        name=name
    )
    
    return best_model


def main():
    """Main function for command-line usage."""
    parser = argparse.ArgumentParser(
        description="YOLO-World Fine-tuning for Multiple Endoscopic Datasets"
    )
    
    subparsers = parser.add_subparsers(dest="command", help="Commands")
    
    # Convert command
    convert_parser = subparsers.add_parser("convert", help="Convert datasets to YOLO format")
    convert_parser.add_argument(
        "--output", "-o",
        type=str,
        default="./yolo_dataset",
        help="Output directory"
    )
    convert_parser.add_argument(
        "--train-split",
        type=float,
        default=0.8,
        help="Training set ratio"
    )
    convert_parser.add_argument(
        "--val-split",
        type=float,
        default=0.15,
        help="Validation set ratio"
    )
    
    # Train command
    train_parser = subparsers.add_parser("train", help="Train the model")
    train_parser.add_argument(
        "--dataset", "-d",
        type=str,
        required=True,
        help="Path to dataset YAML file"
    )
    train_parser.add_argument(
        "--model", "-m",
        type=str,
        default="yolov8x-worldv2.pt",
        help="Base model to fine-tune"
    )
    train_parser.add_argument(
        "--epochs", "-e",
        type=int,
        default=100,
        help="Number of training epochs"
    )
    train_parser.add_argument(
        "--batch-size", "-b",
        type=int,
        default=16,
        help="Batch size"
    )
    train_parser.add_argument(
        "--img-size",
        type=int,
        default=640,
        help="Input image size"
    )
    train_parser.add_argument(
        "--lr",
        type=float,
        default=0.01,
        help="Initial learning rate"
    )
    train_parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device (cuda/cpu)"
    )
    train_parser.add_argument(
        "--project",
        type=str,
        default="runs/train",
        help="Project directory"
    )
    train_parser.add_argument(
        "--name",
        type=str,
        default=None,
        help="Run name (auto-generated as endovlayolo_YYYYMMDD if not specified)"
    )
    train_parser.add_argument(
        "--freeze",
        type=int,
        default=0,
        help="Number of layers to freeze"
    )
    train_parser.add_argument(
        "--patience",
        type=int,
        default=50,
        help="Early stopping patience"
    )
    train_parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from last checkpoint"
    )
    
    # Full pipeline command
    full_parser = subparsers.add_parser("full", help="Run full pipeline (convert + train)")
    full_parser.add_argument(
        "--output", "-o",
        type=str,
        default="./yolo_dataset",
        help="Output directory for dataset"
    )
    full_parser.add_argument(
        "--model", "-m",
        type=str,
        default="yolov8x-worldv2.pt",
        help="Base model"
    )
    full_parser.add_argument(
        "--epochs", "-e",
        type=int,
        default=100,
        help="Training epochs"
    )
    full_parser.add_argument(
        "--batch-size", "-b",
        type=int,
        default=16,
        help="Batch size"
    )
    full_parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device (cuda/cpu)"
    )
    full_parser.add_argument(
        "--project",
        type=str,
        default="runs/train",
        help="Project directory"
    )
    full_parser.add_argument(
        "--name",
        type=str,
        default=None,
        help="Run name (auto-generated as endovlayolo_YYYYMMDD if not specified)"
    )
    full_parser.add_argument(
        "--train-split",
        type=float,
        default=0.8,
        help="Training set ratio"
    )
    full_parser.add_argument(
        "--val-split",
        type=float,
        default=0.15,
        help="Validation set ratio"
    )
    
    # Add dataset command
    add_parser = subparsers.add_parser("add-dataset", help="Add a dataset to config")
    add_parser.add_argument(
        "--name", "-n",
        type=str,
        required=True,
        help="Dataset name (roi_name)"
    )
    add_parser.add_argument(
        "--labels", "-l",
        type=str,
        required=True,
        help="Path to labels JSON"
    )
    add_parser.add_argument(
        "--images", "-i",
        type=str,
        required=True,
        help="Path to images directory"
    )
    
    # List datasets command
    list_parser = subparsers.add_parser("list", help="List configured datasets")
    
    args = parser.parse_args()
    
    if args.command == "convert":
        manager = MultiDatasetManager()
        converter = MultiDatasetConverter(
            manager=manager,
            output_dir=args.output,
            train_split=args.train_split,
            val_split=args.val_split,
            test_split=1 - args.train_split - args.val_split
        )
        converter.convert()
    
    elif args.command == "train":
        trainer = YOLOWorldFineTuner(
            base_model=args.model,
            device=args.device
        )
        trainer.train(
            dataset_yaml=args.dataset,
            epochs=args.epochs,
            batch_size=args.batch_size,
            img_size=args.img_size,
            lr0=args.lr,
            project=args.project,
            name=args.name,
            freeze_layers=args.freeze,
            patience=args.patience,
            resume=args.resume
        )
    
    elif args.command == "full":
        manager = MultiDatasetManager()
        prepare_and_train(
            manager=manager,
            output_dir=args.output,
            base_model=args.model,
            epochs=args.epochs,
            batch_size=args.batch_size,
            train_split=args.train_split,
            val_split=args.val_split,
            device=args.device,
            project=args.project,
            name=args.name
        )
    
    elif args.command == "add-dataset":
        add_dataset(args.name, args.labels, args.images)
        print(f"✅ Added dataset: {args.name}")
        print(f"   Labels: {args.labels}")
        print(f"   Images: {args.images}")
        print("\nNote: This only adds to the current session.")
        print("To persist, add to config.py DATASETS list.")
    
    elif args.command == "list":
        print("\n📁 Configured Datasets:")
        if not DATASETS:
            print("   No datasets configured. Add datasets to config.py DATASETS list.")
        else:
            for ds in DATASETS:
                status = "✓" if ds.enabled else "✗"
                print(f"   {status} {ds.name}")
                print(f"      Labels: {ds.labels_json}")
                print(f"      Images: {ds.images_dir}")
    
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
