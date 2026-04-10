# Endoscope Visual Servoing System

An instruction-based visual servoing system for robotic endoscope control using a fine-tuned vision detector for text-prompt object detection combined with a pretrained low-level neural network controller.

### Architecture

```
Camera Frame (400x400) 
    ↓
Target Detection by fine-tuned detector
    ↓
Bounding Box [x, y, w, h] + Confidence
    ↓
Low-Level Neural Network Controller
    ↓
2-DoF Bending Commands [m1, m2]
    ↓
Safety Filter
    ↓
Motor Commands
```

## Installation

### 1. Create Virtual Environment (Recommended)

```bash
python3 -m venv venv
source venv/bin/activate
```

### 2. Install Dependencies

```bash
pip install torch torchvision
pip install ultralytics
pip install opencv-python
pip install numpy
pip install pygame  # Optional, for better visualization
pip install pyserial  # For real robot control
```

Or install all at once:

```bash
pip install torch torchvision ultralytics opencv-python numpy pygame pyserial
```

### 3. Fine-tuned vision detector

Should be saved at: ./checkpoints/best_blackpoint_base.pt

### 4. Pre-trained controllers

./checkpoints/bc_model.pt
./checkpoints/bc+iql.pt
./checkpoints/bc+td3bc.pt

### 5. Test video

./videos/video1.mp4
./videos/video2.mp4



## Usage

### Video Simulation

Evaluate the pipeline offline on pre-recorded video files. No robot, serial port, or live camera required. Annotated output videos are saved automatically.

**Single video:**

```bash
python video_sim.py \
    --input ./test_videos/video1.mp4 \
    --yoloe-model ./checkpoints/best_blackpoint_base.pt \
    --control-checkpoint ./checkpoints/bc_model.pt \
```

**Multiple videos in one run:**

```bash
python video_sim.py \
    --input ./videos/video1.mp4 ./videos/video2.mp4 \
    --yoloe-model ./checkpoints/best_blackpoint_base.pt \
    --control-checkpoint ./checkpoints/bc_model.pt \
```

**Side-by-side comparison: learned controller vs proportional controller:**

```bash
python video_sim.py \
    --input ./videos/video1.mp4 \
    --yoloe-model ./checkpoints/best_blackpoint_base.pt \
    --control-checkpoint ./checkpoints/bc_model.pt \
    --control-mode both \
    --p-gain-x 2 --p-gain-y 2
