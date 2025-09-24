# Depth Anything V2 Real-time Camera Demo

A high-performance, real-time depth estimation demo using **Depth Anything V2** model. This optimized implementation provides smooth depth visualization with multiple model sizes and robust loading methods.

![Demo Preview](https://img.shields.io/badge/Demo-Real--time%20Depth%20Estimation-blue)
![Python](https://img.shields.io/badge/Python-3.8%2B-green)
![PyTorch](https://img.shields.io/badge/PyTorch-1.9%2B-red)
![OpenCV](https://img.shields.io/badge/OpenCV-4.0%2B-orange)

![Depth Output Example](/examples/depth_output.png)

## Features

- **Real-time performance** (8-15+ FPS on GPU)
- **Multiple model sizes** (small, base, large)
- **Live camera feed** with side-by-side depth visualization
- **Robust model loading** (transformers + torch hub fallbacks)
- **Optimized preprocessing** for maximum speed
- **Video recording** and screenshot capabilities
- **GPU acceleration** with CPU fallback
- **Real-time FPS monitoring**

## Installation

### Prerequisites

- Python 3.8+
- Webcam/laptop camera
- CUDA-capable GPU (optional, but recommended)

### Install Dependencies

```bash
# Clone the repository
git clone https://github.com/Abadli-Badro/Depth-Estimation-Using-The-Depth-Anything-V2-Model.git
cd Depth-Estimation-Using-The-Depth-Anything-V2-Model

# Install required packages
pip install torch torchvision torchaudio
pip install transformers accelerate
pip install opencv-python
pip install Pillow
pip install numpy

# For CUDA support (if you have NVIDIA GPU)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

## Quick Start

### Basic Usage

```bash
# Run with default settings (small model, camera 0)
python depth_demo.py

# Specify camera and model size
python depth_demo.py --camera 0 --model-size small

# Save output video
python depth_demo.py --save --output my_depth_demo.mp4
```

### Model Options

- `--model-size small`: Fastest performance (~10-15 FPS on GPU)
- `--model-size base`: Balanced quality/speed (~5-10 FPS on GPU)
- `--model-size large`: Best quality (~2-5 FPS on GPU)

### Controls

- **Press 'q'**: Quit the application
- **Press 's'**: Save screenshot
- **ESC**: Also quits the application

## ⚙️ Configuration

### Camera Settings

The script automatically configures your camera for optimal performance:

- Resolution: 640x480 (configurable)
- FPS: 30 (configurable)
- Format: MJPG (for better performance)
- Buffer: Minimal for low latency

### Performance Optimization

- **Input resolution**: Automatically scaled (256px for small, 384px for larger models)
- **Frame skipping**: Smart frame processing to maintain smooth display
- **GPU acceleration**: Automatic GPU detection and usage
- **Memory management**: Optimized tensor operations and caching

### Performance Tips

- Close other applications to free up GPU/CPU resources
- Ensure good lighting for better depth estimation
- Use small model for real-time applications
- Consider reducing camera resolution for very old hardware

## References

- [Depth Anything V2 Paper](https://arxiv.org/abs/2406.09414)
- [Original GitHub Repository](https://github.com/LiheYoung/Depth-Anything-V2)
- [Hugging Face Model Hub](https://huggingface.co/depth-anything)

---
