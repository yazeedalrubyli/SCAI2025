# Triton Inference Server for Soccer Analysis

This directory contains models and scripts for deploying YOLO-based object detection models with NVIDIA Triton Inference Server for soccer video analysis.

## Overview

The system uses Triton Inference Server to efficiently deploy and serve two key models:
- Ball detector - For tracking the soccer ball in video frames
- Player detector - For identifying and locating players on the field

These models are optimized for real-time inference using NVIDIA TensorRT acceleration.

## Directory Structure

```
triton_inference_server/
├── models/             # Pre-trained model weights
│   ├── ball/           # Ball detection model
│   └── player/         # Player detection model
└── scripts/            # Deployment and export scripts
    └── export.py       # Script to export models to Triton format
```

## Requirements

- Docker with NVIDIA Container Toolkit
- NVIDIA GPU with CUDA support
- Python 3.8+
- tritonclient package
- ultralytics package

## Usage

### Exporting Models

The `export.py` script converts YOLOv8 models to ONNX format and sets up the Triton model repository:

```bash
cd triton_inference_server
python scripts/export.py
```

This script:
1. Converts the YOLO models to ONNX format
2. Creates Triton model repositories for both ball and player detection models
3. Starts Docker containers running Triton Inference Server
4. Exposes the models on different ports:
   - Ball detector: http://localhost:8000
   - Player detector: http://localhost:8001

### Testing Inference

You can test the deployed models using sample Python code:

```python
import numpy as np
import tritonclient.http as httpclient
import cv2

# Prepare your image
image = cv2.imread("sample_frame.jpg")
image = cv2.resize(image, (640, 640))
img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
img_norm = img_rgb.astype(np.float32) / 255.0
img_chw = np.transpose(img_norm, (2, 0, 1))
img_batch = np.expand_dims(img_chw, 0)

# Connect to Triton server
client = httpclient.InferenceServerClient('localhost:8000')

# Prepare inputs
inputs = [httpclient.InferInput('images', img_batch.shape, 'FP32')]
inputs[0].set_data_from_numpy(img_batch)

# Request outputs
outputs = [httpclient.InferRequestedOutput('output0')]

# Run inference
result = client.infer('player_detector', inputs, outputs=outputs)

# Process results
detections = result.as_numpy('output0')
```

### Performance Optimization

The models are configured to use TensorRT acceleration with FP16 precision, with engine caching enabled for faster subsequent loads.

## Stopping the Servers

To stop the Triton servers:

```bash
# Get container IDs
docker ps

# Stop the containers
docker stop <container_id1> <container_id2>
```

## Model Information

- Both models are trained using YOLOv12 architecture 
- Input size: 640x640 pixels
- Input format: RGB images normalized to [0,1]
- Output format: Standard YOLO detection format [batch, dimensions, detections]