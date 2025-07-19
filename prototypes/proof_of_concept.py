# Full proof of concept script
# Imports
import cv2
import numpy as np
import json
from kalman_filter import MaskKalmanFilter
import argparse
import torch
from model import UNet
import torch.nn as nn
import threading
from queue import Queue

# --- Custom Exceptions ---
class VideoProcessingError(Exception):
    """Base exception for errors in this script."""
    pass

class VideoCaptureError(VideoProcessingError):
    """Raised when the video file cannot be opened."""
    pass

class FrameProcessingError(VideoProcessingError):
    """Raised for errors during the frame processing loop."""
    pass

class MaskVerificationError(FrameProcessingError):
    """Raised when a generated mask fails the verification step."""
    pass

# --- Threading Functions ---
def frame_reader(video_path: str, frame_queue: Queue, stop_event: threading.Event):
    """
    Reads frames from a video file and puts them into a queue.

    Args:
        video_path: Path to the input video file.
        frame_queue: The queue to store frames.
        stop_event: An event to signal when to stop reading frames.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise VideoCaptureError(f"Cannot open video file: {video_path}")

    while not stop_event.is_set():
        ret, frame = cap.read()
        if not ret:
            break
        frame_queue.put(frame)

    cap.release()
    frame_queue.put(None)  # Sentinel to indicate the end of the stream

def video_processing(video_path: str, output_id: str, model: nn.Module, config: dict) -> None:
    """
    Processes a video to detect and visualize glare, frame by frame, using multithreading.

    Args:
        video_path: Path to the input video file.
        output_id: A unique identifier used for naming the output files.
        model: The pre-trained PyTorch model to use for mask generation.
        config: A dictionary containing configuration parameters.
    """
    # Load configuration
    smoothing_ksize = tuple(config['smoothing']['kernel_size'])
    smoothing_sigma = config['smoothing']['sigma']
    device = config['train']['device']

    model.to(device)
    model.eval()

    # Get video properties for output
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise VideoCaptureError(f"Cannot open video file: {video_path}")
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    cap.release()

    # Initialize video writers
    mask_writer = create_video_writer(f"{output_id}_mask.mp4", fps, width, height, is_color=False)
    final_writer = create_video_writer(f"{output_id}_final.mp4", fps, width, height, is_color=True)

    # Initialize Kalman Filter
    kf = MaskKalmanFilter(
        shape=(height, width),
        q_val=config['kalman_filter']['Q'],
        r_val=config['kalman_filter']['R']
    )

    # Frame queue and threading events
    frame_queue = Queue(maxsize=int(fps))
    stop_event = threading.Event()

    # Start the frame reader thread
    reader_thread = threading.Thread(target=frame_reader, args=(video_path, frame_queue, stop_event))
    reader_thread.start()

    frame_count = 0
    try:
        while True:
            frame = frame_queue.get()
            if frame is None:  # End of stream
                break

            frame_count += 1
            gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0

            mask = generate_mask(model, gray_frame, device)
            mask = smooth_output(mask, smoothing_ksize, smoothing_sigma)
            filtered_mask = kf.update(mask)

            try:
                verify_feedback(filtered_mask)
            except MaskVerificationError as e:
                print(f"Warning: Frame {frame_count} failed verification: {e}")
                continue

            mask_output = (np.clip(filtered_mask, 0.0, 1.0) * 255).astype(np.uint8)
            final_output = finalize_output(frame, filtered_mask)

            mask_writer.write(mask_output)
            final_writer.write(final_output)

    finally:
        # Clean up
        stop_event.set()
        reader_thread.join()
        mask_writer.release()
        final_writer.release()

    if frame_count == 0:
        raise FrameProcessingError("No frames were read from the video.")

    print(f"Processing complete. Videos saved as {output_id}_mask.mp4 and {output_id}_final.mp4")


def generate_mask(model: nn.Module, frame: np.ndarray, device: str) -> np.ndarray:
    """
    Generates a glare mask for a single frame using the U-Net model.

    Args:
        model: The pre-trained PyTorch model.
        frame: The input frame (H, W), normalized to [0, 1].
        device: The device to run the model on ('cuda' or 'cpu').

    Returns:
        The generated mask as a numpy array (H, W).
    """
    # Add batch and channel dimensions for the model
    inp = torch.from_numpy(frame).unsqueeze(0).unsqueeze(0).to(device)
    with torch.no_grad():
        # Squeeze batch and channel dimensions from the output
        out = model(inp).squeeze(0).squeeze(0).cpu().numpy()
    return out


def smooth_output(mask: np.ndarray, ksize: tuple[int, int], sigma: float) -> np.ndarray:
    """
    Applies a Gaussian blur to smooth the mask.

    Args:
        mask: The input mask (H, W).
        ksize: The kernel size for the Gaussian blur.
        sigma: The sigma value for the Gaussian blur.

    Returns:
        The smoothed mask.
    """
    return cv2.GaussianBlur(mask, ksize, sigma)


def verify_feedback(mask: np.ndarray, min_threshold: float = 0.01, max_threshold: float = 0.5):
    """
    Verifies the generated mask based on simple heuristics.

    Checks if the proportion of the activated mask area is within a reasonable range.

    Args:
        mask: The mask to verify.
        min_threshold: The minimum acceptable proportion of the mask being active.
        max_threshold: The maximum acceptable proportion of the mask being active.

    Raises:
        MaskVerificationError: If the mask's active area is outside the defined bounds.
    """
    mask_proportion = np.mean(mask)
    if not (min_threshold < mask_proportion < max_threshold):
        raise MaskVerificationError(
            f"Mask covers {mask_proportion:.2%} of the frame, "
            f"which is outside the acceptable range of ({min_threshold:.2%}, {max_threshold:.2%})."
        )


def finalize_output(frame: np.ndarray, mask: np.ndarray, color: tuple[int, int, int] = (0, 0, 255), alpha: float = 0.4) -> np.ndarray:
    """
    Overlays the mask onto the original frame as a colored highlight.

    Args:
        frame: The original BGR video frame.
        mask: The glare mask (H, W), values in [0, 1].
        color: The BGR color of the overlay.
        alpha: The transparency of the overlay.

    Returns:
        The frame with the mask overlaid.
    """
    overlay = frame.copy()
    colored_mask = (np.stack([mask] * 3, axis=-1) * color).astype(np.uint8)
    cv2.addWeighted(colored_mask, alpha, overlay, 1 - alpha, 0, overlay)
    return overlay


def create_video_writer(filename: str, fps: float, width: int, height: int, is_color: bool) -> cv2.VideoWriter:
    """
    Creates and returns a cv2.VideoWriter object.

    Args:
        filename: The name of the output video file.
        fps: Frames per second for the output video.
        width: Width of the video frames.
        height: Height of the video frames.
        is_color: Flag indicating if the video is color.

    Returns:
        An initialized cv2.VideoWriter object.

    Raises:
        IOError: If the VideoWriter cannot be opened.
    """
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(filename, fourcc, fps, (width, height), isColor=is_color)
    if not writer.isOpened():
        raise IOError(f"Failed to open VideoWriter for {filename}")
    return writer


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Process a video file to detect and highlight glare.",
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument("video_path", type=str, help="Path to the input video file.")
    parser.add_argument(
        "-o", "--output_id", type=str,
        help="A unique ID for the output files. Defaults to the video's filename."
    )
    parser.add_argument("--config", type=str, default="config.json", help="Path to the config file.")
    args = parser.parse_args()

    video_path = args.video_path
    output_id = args.output_id or video_path.split('/')[-1].split('.')[0]

    try:
        with open(args.config, 'r') as f:
            config = json.load(f)
        
        model = UNet(in_channels=1, out_channels=1)
        model.load_state_dict(torch.load("kvald_unet.pth", map_location=config['train']['device']))

        video_processing(video_path, output_id, model, config)
    except (VideoProcessingError, FileNotFoundError) as e:
        print(f"An error occurred: {e}")
