# KVALD: Kalman Vision-based Automated Local Dimming

KVALD is a real-time video processing application designed to reduce glare and stabilize brightness in video streams. It uses a combination of a deep learning-based mask generator and a differentiable Kalman filter to achieve this.

## Table of Contents

- [Architecture](#architecture)
- [Running the Application](#running-the-application)

## Architecture

The core of KVALD consists of the following components:

1.  **Mask Generation Network**: A U-Net based model that generates a probability mask to identify glare hotspots.
2.  **Differentiable Kalman Filter**: A Kalman filter that uses the generated mask to predict and update the state of the video stream, smoothing out brightness variations.
3.  **Spatial and Temporal Smoothing**: Additional smoothing algorithms to further improve the quality of the output video.

**Note on Architecture:** This project has been refactored to be a pure Python implementation, removing the C++ components for simplicity and faster prototyping.

## Running the Application

To run this project, you will need Python and the dependencies listed in `requirements.txt`. You can install them using pip:

```bash
pip install -r requirements.txt
```

You can then run the main application:

```bash
python proof_of_concept.py
```
