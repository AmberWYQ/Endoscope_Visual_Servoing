"""
Utility functions for Endoscope Visual Servoing System
"""

import os
import json
import numpy as np
import cv2
import torch
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
import matplotlib.pyplot as plt


def set_seed(seed: int = 42):
    """Set random seeds for reproducibility"""
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def setup_environment():
    """Setup environment variables for headless operation"""
    os.environ['QT_QPA_PLATFORM'] = 'offscreen'
    os.environ['MPLBACKEND'] = 'Agg'


def normalize_quaternion(q: np.ndarray) -> np.ndarray:
    """Normalize quaternion to unit length"""
    norm = np.linalg.norm(q)
    if norm > 0:
        return q / norm
    return np.array([1.0, 0.0, 0.0, 0.0])


def quaternion_to_rotation_matrix(q: np.ndarray) -> np.ndarray:
    """Convert quaternion [qw, qx, qy, qz] to 3x3 rotation matrix"""
    qw, qx, qy, qz = q
    
    R = np.array([
        [1 - 2*(qy**2 + qz**2), 2*(qx*qy - qz*qw), 2*(qx*qz + qy*qw)],
        [2*(qx*qy + qz*qw), 1 - 2*(qx**2 + qz**2), 2*(qy*qz - qx*qw)],
        [2*(qx*qz - qy*qw), 2*(qy*qz + qx*qw), 1 - 2*(qx**2 + qy**2)]
    ])
    
    return R


def rotation_matrix_to_quaternion(R: np.ndarray) -> np.ndarray:
    """Convert 3x3 rotation matrix to quaternion [qw, qx, qy, qz]"""
    trace = np.trace(R)
    
    if trace > 0:
        s = 0.5 / np.sqrt(trace + 1.0)
        qw = 0.25 / s
        qx = (R[2, 1] - R[1, 2]) * s
        qy = (R[0, 2] - R[2, 0]) * s
        qz = (R[1, 0] - R[0, 1]) * s
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = 2.0 * np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2])
        qw = (R[2, 1] - R[1, 2]) / s
        qx = 0.25 * s
        qy = (R[0, 1] + R[1, 0]) / s
        qz = (R[0, 2] + R[2, 0]) / s
    elif R[1, 1] > R[2, 2]:
        s = 2.0 * np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2])
        qw = (R[0, 2] - R[2, 0]) / s
        qx = (R[0, 1] + R[1, 0]) / s
        qy = 0.25 * s
        qz = (R[1, 2] + R[2, 1]) / s
    else:
        s = 2.0 * np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1])
        qw = (R[1, 0] - R[0, 1]) / s
        qx = (R[0, 2] + R[2, 0]) / s
        qy = (R[1, 2] + R[2, 1]) / s
        qz = 0.25 * s
        
    return normalize_quaternion(np.array([qw, qx, qy, qz]))


def compute_tracking_error(target: np.ndarray, center: np.ndarray) -> float:
    """Compute Euclidean tracking error in pixels"""
    return np.linalg.norm(target - center)


def exponential_moving_average(data: np.ndarray, alpha: float = 0.1) -> np.ndarray:
    """Apply exponential moving average smoothing"""
    smoothed = np.zeros_like(data)
    smoothed[0] = data[0]
    for i in range(1, len(data)):
        smoothed[i] = alpha * data[i] + (1 - alpha) * smoothed[i-1]
    return smoothed


class TrainingLogger:
    """Logger for training metrics"""
    
    def __init__(self, log_dir: str, name: str = "training"):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.name = name
        
        self.history = {
            "epoch": [],
            "train_loss": [],
            "val_loss": [],
            "metrics": []
        }
        
    def log(self, epoch: int, train_loss: float, val_loss: float = None, 
            metrics: Dict = None):
        """Log training metrics"""
        self.history["epoch"].append(epoch)
        self.history["train_loss"].append(train_loss)
        self.history["val_loss"].append(val_loss)
        self.history["metrics"].append(metrics or {})
        
        # Print
        msg = f"Epoch {epoch}: train_loss={train_loss:.6f}"
        if val_loss is not None:
            msg += f", val_loss={val_loss:.6f}"
        print(msg)
        
    def save(self):
        """Save history to JSON"""
        filepath = self.log_dir / f"{self.name}_history.json"
        
        # Convert numpy arrays to lists
        history_serializable = {}
        for key, values in self.history.items():
            if isinstance(values, list):
                history_serializable[key] = [
                    float(v) if isinstance(v, (np.floating, float)) else v 
                    for v in values
                ]
            else:
                history_serializable[key] = values
                
        with open(filepath, "w") as f:
            json.dump(history_serializable, f, indent=2)
            
        print(f"Saved training history to {filepath}")
        
    def plot(self, save_path: str = None):
        """Plot training curves"""
        fig, ax = plt.subplots(figsize=(10, 6))
        
        epochs = self.history["epoch"]
        ax.plot(epochs, self.history["train_loss"], label="Train Loss", color="blue")
        
        if self.history["val_loss"][0] is not None:
            ax.plot(epochs, self.history["val_loss"], label="Val Loss", color="orange")
            
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss")
        ax.set_title(f"{self.name} Training Curves")
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Saved training plot to {save_path}")
        else:
            save_path = self.log_dir / f"{self.name}_curves.png"
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            
        plt.close()


class VideoWriter:
    """Wrapper for OpenCV VideoWriter with error handling"""
    
    def __init__(self, filepath: str, fps: int = 20, 
                 frame_size: Tuple[int, int] = (640, 480)):
        self.filepath = filepath
        self.fps = fps
        self.frame_size = frame_size
        
        # Create directory if needed
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        
        # Try different codecs
        codecs = ['mp4v', 'avc1', 'XVID', 'MJPG']
        self.writer = None
        
        for codec in codecs:
            try:
                fourcc = cv2.VideoWriter_fourcc(*codec)
                self.writer = cv2.VideoWriter(filepath, fourcc, fps, frame_size)
                if self.writer.isOpened():
                    print(f"[VideoWriter] Using codec: {codec}")
                    break
            except:
                continue
                
        if self.writer is None or not self.writer.isOpened():
            print("[VideoWriter] WARNING: Could not create video writer")
            self.writer = None
            
    def write(self, frame: np.ndarray):
        """Write frame"""
        if self.writer is not None:
            # Ensure correct size
            if frame.shape[:2][::-1] != self.frame_size:
                frame = cv2.resize(frame, self.frame_size)
            self.writer.write(frame)
            
    def release(self):
        """Release writer"""
        if self.writer is not None:
            self.writer.release()
            print(f"[VideoWriter] Saved video to {self.filepath}")


def visualize_jacobian(jacobian: np.ndarray, ax=None) -> plt.Axes:
    """Visualize 2x2 Jacobian matrix"""
    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 5))
        
    im = ax.imshow(jacobian, cmap='coolwarm', vmin=-1, vmax=1)
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(['Motor 1', 'Motor 2'])
    ax.set_yticklabels(['Error X', 'Error Y'])
    ax.set_title('Jacobian Matrix')
    
    # Add text annotations
    for i in range(2):
        for j in range(2):
            ax.text(j, i, f'{jacobian[i, j]:.3f}', 
                   ha='center', va='center', color='black')
            
    plt.colorbar(im, ax=ax)
    return ax


def visualize_attention(attention_map: np.ndarray, image: np.ndarray = None,
                       ax=None) -> plt.Axes:
    """Visualize attention map overlay on image"""
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 6))
        
    # Resize attention to image size
    if image is not None:
        h, w = image.shape[:2]
        attention_resized = cv2.resize(attention_map, (w, h))
        
        # Overlay
        ax.imshow(image)
        ax.imshow(attention_resized, alpha=0.5, cmap='jet')
    else:
        ax.imshow(attention_map, cmap='jet')
        
    ax.set_title('Attention Map')
    ax.axis('off')
    return ax


def create_comparison_figure(
    images: List[np.ndarray],
    titles: List[str],
    save_path: str = None
) -> plt.Figure:
    """Create side-by-side comparison figure"""
    n = len(images)
    fig, axes = plt.subplots(1, n, figsize=(4*n, 4))
    
    if n == 1:
        axes = [axes]
        
    for ax, img, title in zip(axes, images, titles):
        if len(img.shape) == 3:
            # Convert BGR to RGB
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        ax.imshow(img)
        ax.set_title(title)
        ax.axis('off')
        
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        
    return fig


def compute_metrics(
    predictions: np.ndarray,
    targets: np.ndarray
) -> Dict[str, float]:
    """Compute evaluation metrics"""
    mse = np.mean((predictions - targets) ** 2)
    mae = np.mean(np.abs(predictions - targets))
    rmse = np.sqrt(mse)
    
    # Per-dimension metrics
    mse_per_dim = np.mean((predictions - targets) ** 2, axis=0)
    
    return {
        "mse": float(mse),
        "mae": float(mae),
        "rmse": float(rmse),
        "mse_per_dim": mse_per_dim.tolist()
    }


def load_checkpoint(
    filepath: str,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer = None,
    device: str = "cpu"
) -> Dict[str, Any]:
    """Load model checkpoint"""
    checkpoint = torch.load(filepath, map_location=device)
    
    # Load model state
    if "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        model.load_state_dict(checkpoint)
        
    # Load optimizer state if provided
    if optimizer is not None and "optimizer_state_dict" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        
    print(f"Loaded checkpoint from {filepath}")
    
    return checkpoint


def save_checkpoint(
    filepath: str,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer = None,
    epoch: int = 0,
    loss: float = 0.0,
    **kwargs
):
    """Save model checkpoint"""
    Path(filepath).parent.mkdir(parents=True, exist_ok=True)
    
    checkpoint = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "loss": loss
    }
    
    if optimizer is not None:
        checkpoint["optimizer_state_dict"] = optimizer.state_dict()
        
    checkpoint.update(kwargs)
    
    torch.save(checkpoint, filepath)
    print(f"Saved checkpoint to {filepath}")


def count_parameters(model: torch.nn.Module) -> int:
    """Count trainable parameters in model"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def print_model_summary(model: torch.nn.Module):
    """Print model summary with parameter counts"""
    print("\n" + "="*60)
    print("Model Summary")
    print("="*60)
    
    total_params = 0
    for name, module in model.named_modules():
        if len(list(module.children())) == 0:  # Leaf modules only
            params = sum(p.numel() for p in module.parameters())
            if params > 0:
                print(f"{name:40s} {params:>12,d}")
                total_params += params
                
    print("="*60)
    print(f"{'Total Parameters':40s} {total_params:>12,d}")
    print(f"{'Trainable Parameters':40s} {count_parameters(model):>12,d}")
    print("="*60 + "\n")
