# Endoscope Visual Servoing System Training Pipeline

A two-stage neural network training system for robotic endoscope visual servoing with target tracking.

## Overview

This system enables a continuum robotic endoscope to automatically track a dynamic black point target to the center of the camera view. It uses a unified neural network architecture trained through three progressive stages:

1. **Behavior Cloning**: Adapt to real-world challenges from expert demonstrations  
2. **Offline Reinforcement Learning**: Optimize beyond human performance

## Features

- **Unified Architecture**: Same neural network structure across all training stages
- **Modular Jacobian Learning**: Learn local image-to-actuation mapping for better control
- **Multi-Modal Fusion**: Combines visual, temporal, and proprioceptive information
- **Temporal Dynamics**: Trajectory encoding for target motion prediction
- **Physics Simulator**: Custom gymnasium environment for PINN training
- **Hardware Integration**: Real-time control with EM tracker and motor interfaces


## Installation

```bash
# Clone or copy the project
cd endoscope_tracking

# Create conda environment
conda create -n endoscope python=3.10
conda activate endoscope

# Install dependencies
pip install torch torchvision
pip install gymnasium numpy opencv-python pandas
pip install matplotlib tensorboard tqdm
pip install ultralytics

# Optional: For Xbox controller support
pip install pygame

# Optional: For hardware interfaces
pip install pyserial  # Serial communication
```

## Quick Start

### 1. Expert Demonstration Dataset

Unpublished. Email the author to get the dataset.
Email: weiyuqing0805@gmail.com

### 3. Stage 1: Behavior Cloning

Train from expert demonstrations (your collected data):

```bash
# Using your CSV data
python training/train_bc.py \
    --data path_to_your_data.csv \
    --epochs 200
```

### 4. Stage 2: Offline Reinforcement Learning

Fine-tune with RL for optimal performance: iql or td3bc, choose at '--algo'

```bash
python /content/project/training/train_offline_rl.py \
    --data     path_to_your_data.csv \
    --bc_ckpt  ./checkpoints/bc/best.pt \
    --algo     iql \
    --epochs   200 \
    --batch    256 \
    --out_dir  ./checkpoints/offline_rl
```
