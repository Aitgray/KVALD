# KVALD Architecture

A high-level overview of the KVALD processing pipeline, from raw video input to output mask generation.

## 1. Input & Abstraction Layer
- **Device Discovery**: V4L2 (Linux), DirectShow (Windows), GStreamer
- **VideoCapture Module**: OpenCV wrapper for multiple backends
- **Preprocessing**:
  - Color conversion (RGB → YUV or grayscale)
  - Optional resolution scaling/pyramid creation

## 2. Mask Generation Network
- **Model**: MobileNetV2-based U-Net (keras-like pseudocode)
- **Input**: single frame (HxWx3)
- **Output**: probability mask (HxWx1)
- **Loss Components**:
  1. **Segmentation Loss**: BCE/Dice vs. synthetic hotspot mask
  2. **Brightness Preservation**: MSE between target mean brightness and \
     post-mask brightness
  3. **Smoothness**: L1 gradient regularizer on mask

## 3. Differentiable Kalman Filter Module
- **Predict**: state & covariance propagation via learned dynamics
- **Update**: measurement fusion using neural mask as observation
- **Learnable Parameters**: process & measurement noise covariances
- **Backprop**: gradients flow through both predict and update steps

## 4. Spatial & Temporal Smoothing
- **Spatial**: 3×3 bilateral or Gaussian blur
- **Temporal**: exponential moving average:
  \[ M_t = (1 - \lambda)\hat M_t + \lambda M_{t-1} \]
- **Tunable**: \(\lambda \in [0.8, 0.95]\)

## 5. Feedback Verification
- **Re-apply**: smoothed mask + frame → original network or verifier net
- **Consistency Check**: ensure glare suppression and brightness target

## 6. Output & Integration
- **Overlay Renderer**: RGBA composite for debugging
- **Control Interface**:
  - Shared memory / UDP packet exporter
  - API → local-dimming hardware zones

## 7. Performance & Scaling
- **Inference Path**:
  - Python prototype (PyTorch)
  - Export to ONNX → load via TensorRT (GPU) or ONNX Runtime (CPU)
  - libtorch C++ fallback for embedded
- **Benchmarking**:
  - Latency (ms/frame) in `/benchmarks`
  - Accuracy & stability metrics (post-mask brightness error, mask jitter)
- **CI/CD**: GitHub Actions build & test on Linux/macOS/Windows