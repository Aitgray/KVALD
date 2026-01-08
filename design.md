# Design Decisions and Technologies

This document outlines the design decisions and technologies used in the KVALD project.

## Core Technologies

*   **Python 3**: The primary programming language.
*   **PyTorch**: The core deep learning framework for model creation, training, and inference.
*   **OpenCV (cv2)**: Used for video and image processing tasks like reading/writing videos, color space conversions, and applying Gaussian blur.
*   **NumPy**: For numerical operations, especially on image and mask arrays.
*   **Matplotlib**: For plotting training metrics.
*   **JSON**: For configuration files.

## File-by-File Breakdown

### `check_cuda.py`

*   **Purpose**: A simple utility to verify the CUDA and PyTorch installation. It prints out versions and device information.
*   **Design**: This is a standalone script that imports `torch` and `sys` to provide a quick check of the environment. It's not part of the main application logic but is a useful debugging tool.

### `dataset_factory.py`

*   **Purpose**: To provide a single point of entry for creating datasets for training and evaluation.
*   **Design**: It implements the factory design pattern. The `get_dataset` function reads a `config.json` file to decide which dataset to instantiate. It can return a synthetic dataset (`GlareDataset`) or a real-world dataset (`RealGlareDataset` or a fallback). This makes it easy to switch between different data sources without changing the training script.

### `kalman_filter.py`

*   **Purpose**: To smooth the generated glare masks over time.
*   **Design**: Implements a simple pixel-wise Kalman filter. Each pixel in the mask is treated as a separate state to be updated. This is a lightweight approach to introduce temporal consistency to the glare detection.

### `model.py`

*   **Purpose**: Defines the neural network architecture for glare detection.
*   **Design**: A standard U-Net architecture is implemented. The U-Net is well-suited for image segmentation tasks, which is analogous to generating a glare mask. The model consists of an encoder path, a bottleneck, and a decoder path with skip connections.

### `proof_of_concept.py`

*   **Purpose**: The main application logic for processing a video to detect and visualize glare.
*   **Design**: This script orchestrates the entire process. It uses multithreading with a queue to read frames from the video in a separate thread, which can improve performance by overlapping I/O and processing. It uses the `UNet` model for mask generation, the `MaskKalmanFilter` for smoothing, and OpenCV for video I/O and visualization. It's designed to be run from the command line and takes a video path as input.

### `real_data.py`

*   **Purpose**: To load and preprocess real-world image data.
*   **Design**: It defines a `RealImageDataset` class that loads images from a manifest file (in JSON format). The manifest can contain paths to images and associated metadata. The dataset can be configured to resize and crop images. It also includes helper functions for splitting the data into training, validation, and test sets and for creating `DataLoader`s.

### `synthetic_data.py`

*   **Purpose**: To generate synthetic data for training and testing.
*   **Design**: This script can generate videos with different types of glare (static, pulsing, traveling) overlaid on a noisy background. This is useful for initial training and for creating a controlled environment to test the model's performance. The `GlareDataset` class provides an interface to use this synthetic data in a PyTorch `DataLoader`.

### `train.py`
*   **Purpose**: To train the U-Net model on glare detection and dimming tasks using both supervised and unsupervised objectives. The script manages dataset loading, model training, evaluation, and performance logging in a reproducible and hardware-optimized manner.
*   **Design Overview**:
This file implements a robust PyTorch training pipeline with a strong emphasis on performance, stability, and reproducibility. It supports GPU acceleration, dynamic loss balancing, and automatic metric tracking for the IoU and Dice scores.

**Key Features and Design Choices:**
*   **Reproducibility and Performance**:
    *   Uses a seed_everything() utility to initialize RNGs across Python, NumPy, and CUDA, while keeping torch.backends.cudnn.benchmark = True for faster convolutional performance on consistent input shapes.
    *   Mixed-precision training via torch.cuda.amp significantly improves speed and memory efficiency on NVIDIA GPUs without compromising accuracy.
*   **Data Handling**:
    *   Retrieves datasets dynamically from the modular dataset_factory using get_dataset().
    *   Automatically splits the dataset into training and validation subsets using PyTorch’s random_split, ensuring reproducible partitions.
    *   Employs multi-threaded DataLoaders (num_workers=6, pin_memory=True, persistent_workers=True) to maximize I/O throughput on large datasets.
*   **Model and Optimization**:
    *   Uses a standard U-Net architecture (UNet(in_channels=1, out_channels=1)).
    *   The main supervised objective is binary cross-entropy with logits (BCEWithLogitsLoss), chosen for stable binary segmentation convergence.
    *   An unsupervised auxiliary loss (unsupervised_evaluate) penalizes excessive brightness reduction and contrast loss in the masked image, encouraging visually consistent dimming without degrading the frame.
    *   Optimized with Adam and adaptive learning rate scheduling (ReduceLROnPlateau), which halves the LR when validation loss stagnates.
*   **Metrics and Evaluation**:
    *   Computes Intersection-over-Union (IoU) and Dice coefficients each epoch using logits-safe helper functions to measure segmentation accuracy.
    *   Implements early stopping through a custom EarlyStopper class to terminate training when validation loss fails to improve, reducing overfitting and training time.
*   **Checkpointing and Logging**:
    *   Saves both the best (kvald_unet_best.pth) and final (kvald_unet_last.pth) model weights.
    *   Logs metrics for each epoch into training_metrics.csv for reproducibility and post-analysis.
    *   Automatically generates a three-panel plot (training_metrics_plot.png) showing Loss, IoU, and Dice progression over epochs.
*   **Configuration and Execution**:
    *   Reads hyperparameters from an external config.json, including epochs, batch size, learning rate, patience, threshold, and unsupervised loss weight.
    *   Supports both CPU and CUDA execution, auto-selecting the device at runtime.
*   **Rationale**: The design focuses on repeatable, efficient, and interpretable training. Using both supervised and unsupervised objectives allows the model to generalize to real-world glare beyond synthetic examples. Mixed-precision, early stopping, and checkpointing minimize compute cost and prevent wasted epochs. All training statistics are automatically saved for transparency and later validation.

### `verify_performance.py`
*   **Purpose**: To evaluate the model's performance on synthetic data.
*   **Design**: This script generates synthetic videos with known ground-truth glare masks. It then runs the `video_processing` pipeline on these videos and compares the generated masks with the ground truth using IoU and Dice scores. This provides a quantitative measure of the model's accuracy.

### `tests/test_proof_of_concept.py`

*   **Purpose**: To provide unit tests for the `proof_of_concept.py` script.
*   **Design**: It uses Python's `unittest` framework. It includes a `MockUNet` to avoid the need for a real trained model during testing. It tests the video processing pipeline and the frame reader thread.
