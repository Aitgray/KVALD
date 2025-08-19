# KVALD: Kalman Filter for Video Glare Reduction

KVALD is a real-time video processing application designed to reduce glare and stabilize brightness in video streams. It uses a combination of a deep learning-based mask generator and a differentiable Kalman filter to achieve this.

## Table of Contents

- [Project Status](#project-status)
- [Architecture](#architecture)
- [Installation](#installation)
- [Running the Application](#running-the-application)
- [Next Steps](#next-steps)

## Project Status

This project is currently in the process of being converted from a Python prototype to a production-ready C++ implementation for improved performance.

## Architecture

The core of KVALD consists of the following components:

1.  **Mask Generation Network**: A U-Net based model that generates a probability mask to identify glare hotspots.
2.  **Differentiable Kalman Filter**: A Kalman filter that uses the generated mask to predict and update the state of the video stream, smoothing out brightness variations.
3.  **Spatial and Temporal Smoothing**: Additional smoothing algorithms to further improve the quality of the output video.

For a more detailed overview of the architecture, please see the [Architecture.md](docs/Architecture.md) document.

## Installation

To build and run this project, you will need the following dependencies installed:

*   **C++ Compiler**: A modern C++ compiler that supports C++17 (e.g., GCC, Clang, MSVC).
*   **CMake**: Version 3.15 or higher.
*   **OpenCV**: Required for video capture and processing.
*   **LibTorch**: The C++ distribution of PyTorch.
*   **Eigen**: A C++ template library for linear algebra.
*   **cxxopts**: A lightweight C++ option parser library.

It is recommended to use a package manager like [vcpkg](https://vcpkg.io/) to install these dependencies.

Once the dependencies are installed, you can build the project using the following commands:

```bash
mkdir build
cd build
cmake ..
cmake --build .
```

## Running the Application

After building the project, you can run the main application from the `build` directory:

```bash
./kvald_app
```

### Troubleshooting

When running the application, you may encounter errors related to missing DLLs, such as:

*   `c10.dll was not found`
*   `torch_cuda.dll was not found`
*   `torch_cpu.dll was not found`

These files are part of the LibTorch library. To resolve this, you need to ensure that the directory containing these DLLs is in your system's `PATH` environment variable. For example, if you installed LibTorch to `C:\libtorch`, you would add `C:\libtorch\lib` to your `PATH`.

## Next Steps

The following are the immediate next steps for this project:

1. **Executable Error**: Running the executable currently returns the error: "Could not open config.json at prototypes/config.json.
2.  **Fix Eigen Include Path**: Resolve the Eigen include path issue in the `CMakeLists.txt` to allow the Kalman Filter to compile and its tests to run.
3.  **Complete Kalman Filter Tests**: Finish and verify the unit tests for the C++ Kalman Filter.
4.  **Translate `model.py`**: Convert the `model.py` to C++, which orchestrates the Kalman Filter.
6.  **Implement Pipeline Verification Test**: Create a test to ensure the C++ implementation is a faithful translation of the Python prototype.