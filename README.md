# KVALD - Kalman Vision-based Automated Local Dimming
I'll use python to prototype the code and C++ for the final production version.

I want to write a general overview of my current plans for the project.

## Overview
KVALD is a real time system for detecting and dimming high-intensity reigons in a video stream (e.g., bright headlights at night). It uses a lightweight neural network to generate a dynamic brightness mask, applies a differentiable Kalman-based smoothing filter, and outputs a control mask for localized dimming hardware.

## Current Features:
None

## Planned Features:
- **Dynamic Input**: Supports USB cameras, network streams, and video files via OpenCV/GStreamer.
- **Neural Mask Generation**: MobileNet-backed U-Net produces a per-pixel [0,1] dimming mask.
- **Differentiable Kalman Filter**: Embedded as a trainable module for temporal consistency.
- **Spatial & Temporal Smoothing**: Bilateral/Gaussian blur plus EMA to reduce flicker.
- **Feedback Verification**: Optional light-weight verifier network to ensure consistency post-smoothing.
- **Efficient Inference**: Exportable to ONNX/TensorRT or libtorch C++ for sub-33ms latency.
- **Benchmarking Suite**: Automated latency and accuracy tests in `/benchmarks`.

## Repository Structure:
/
├─ src/                   # C++ inference & I/O code
├─ include/               # Public headers
├─ models/                # Trained network weights (.pt, .onnx)
├─ data/                  # Sample videos & test inputs
├─ benchmarks/            # Performance test harnesses
├─ docs/
│   ├─ ARCHITECTURE.md    # System design and data flow
│   └─ ...                # Additional docs
├─ scripts/               # Data generation & utility scripts
├─ tests/                 # Unit & integration tests
├─ CMakeLists.txt         # Build configuration
├─ Dockerfile             # Reproducible dev environment <- do i need this?
├─ README.md              # This file
├─ LICENSE                # None yet, may add in the future
└─ .github/               # Again, none yet, may add in the future