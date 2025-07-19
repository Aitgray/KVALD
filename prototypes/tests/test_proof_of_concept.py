import unittest
import cv2
import numpy as np
import os
import torch
import torch.nn as nn
import json
import threading
from queue import Queue

# Add the parent directory to the path to allow imports
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from proof_of_concept import (
    generate_mask,
    smooth_output,
    verify_feedback,
    finalize_output,
    create_video_writer,
    MaskVerificationError,
    video_processing,
    frame_reader
)

# Mock U-Net model for testing
class MockUNet(nn.Module):
    def __init__(self):
        super(MockUNet, self).__init__()
        self.conv = nn.Conv2d(1, 1, kernel_size=3, padding=1)

    def forward(self, x):
        return torch.sigmoid(self.conv(x))

class TestVideoProcessing(unittest.TestCase):

    def setUp(self):
        """Set up a mock model, test data, and a dummy video file."""
        self.mock_model = MockUNet()
        self.device = 'cpu'
        self.mock_model.to(self.device)
        self.test_frame = np.random.rand(128, 128).astype(np.float32)
        self.test_bgr_frame = (np.random.rand(128, 128, 3) * 255).astype(np.uint8)
        self.video_path = "test_video.mp4"
        self.create_dummy_video(self.video_path, 128, 128, 5)
        self.config = self.create_dummy_config()

    def tearDown(self):
        """Clean up created files."""
        if os.path.exists(self.video_path):
            os.remove(self.video_path)
        if os.path.exists("test_output_mask.mp4"):
            os.remove("test_output_mask.mp4")
        if os.path.exists("test_output_final.mp4"):
            os.remove("test_output_final.mp4")
        if os.path.exists("config.json"):
            os.remove("config.json")

    def create_dummy_video(self, filename, width, height, num_frames):
        """Creates a dummy video file for testing."""
        writer = create_video_writer(filename, 30, width, height, is_color=True)
        for _ in range(num_frames):
            frame = np.random.randint(0, 256, (height, width, 3), dtype=np.uint8)
            writer.write(frame)
        writer.release()

    def create_dummy_config(self):
        """Creates a dummy config file for testing."""
        config = {
            "smoothing": {"kernel_size": [5, 5], "sigma": 1.5},
            "train": {"device": "cpu"},
            "kalman_filter": {"Q": 0.0001, "R": 0.01}
        }
        with open("config.json", "w") as f:
            json.dump(config, f)
        return config

    def test_frame_reader(self):
        """Test the frame reader thread."""
        frame_queue = Queue()
        stop_event = threading.Event()
        reader_thread = threading.Thread(target=frame_reader, args=(self.video_path, frame_queue, stop_event))
        reader_thread.start()
        reader_thread.join(timeout=2)
        stop_event.set()
        self.assertFalse(frame_queue.empty())

    def test_video_processing_multithreaded(self):
        """Test the full multithreaded video processing pipeline."""
        video_processing(self.video_path, "test_output", self.mock_model, self.config)
        self.assertTrue(os.path.exists("test_output_mask.mp4"))
        self.assertTrue(os.path.exists("test_output_final.mp4"))

if __name__ == '__main__':
    unittest.main()