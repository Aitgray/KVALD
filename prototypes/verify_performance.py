import os
import json
import numpy as np
import torch
from tqdm import tqdm
import cv2

from synthetic_data import generate_sample_videos
from model import UNet
from proof_of_concept import video_processing

def iou_score(pred, target):
    intersection = (pred * target).sum()
    union = pred.sum() + target.sum() - intersection
    return (intersection + 1e-6) / (union + 1e-6)

def dice_score(pred, target):
    intersection = (pred * target).sum()
    return (2. * intersection + 1e-6) / (pred.sum() + target.sum() + 1e-6)

def verify_performance(config_path: str = 'config.json'):
    """
    Verifies the performance of the glare detection model by comparing its output
    on synthetic data against ground-truth masks.

    Args:
        config_path: Path to the JSON configuration file.
    """
    # Load configuration
    current_dir = os.path.dirname(__file__)
    config_full_path = os.path.join(current_dir, config_path)
    with open(config_full_path, 'r') as f:
        config = json.load(f)

    device = config['train']['device']
    model_path = "kvald_unet.pth"
    n_videos = 10  # Number of videos to test on

    # Load model
    model = UNet(in_channels=1, out_channels=1)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()

    total_iou = total_dice = 0.0

    for i in tqdm(range(n_videos), desc="Verifying Performance"):
        # Generate synthetic data
        video_dict, mask_dict = generate_sample_videos(seed=i)
        style = next(iter(video_dict))
        video = video_dict[style]
        true_mask = mask_dict[style]

        # Save the video to a temporary file
        video_path = f"temp_video_{i}.mp4"
        save_video(video, video_path)

        # Run the video processing pipeline
        video_processing(video_path, f"temp_output_{i}", model, config)

        # Load the generated mask
        pred_mask = load_video(f"temp_output_{i}_mask.mp4")

        # Calculate metrics
        pred_mask_tensor = torch.from_numpy(pred_mask).to(device)
        true_mask_tensor = torch.from_numpy(true_mask).to(device)

        total_iou += iou_score(pred_mask_tensor, true_mask_tensor).item()
        total_dice += dice_score(pred_mask_tensor, true_mask_tensor).item()

        # Clean up temporary files
        os.remove(video_path)
        os.remove(f"temp_output_{i}_mask.mp4")
        os.remove(f"temp_output_{i}_final.mp4")

    avg_iou = total_iou / n_videos
    avg_dice = total_dice / n_videos

    print(f"Average IoU: {avg_iou:.4f}")
    print(f"Average Dice: {avg_dice:.4f}")

    iou_threshold = config['verification']['iou_threshold']
    dice_threshold = config['verification']['dice_threshold']

    if avg_iou >= iou_threshold and avg_dice >= dice_threshold:
        print("Performance verification passed!")
    else:
        print("Performance verification failed.")

def save_video(video: np.ndarray, path: str, fps: int = 30):
    """
    Saves a numpy array as a video file.

    Args:
        video: The video to save (T, H, W).
        path: The path to save the video to.
        fps: The frames per second of the video.
    """
    T, H, W = video.shape
    writer = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (W, H), isColor=False)
    for t in range(T):
        frame = (video[t] * 255).astype(np.uint8)
        writer.write(frame)
    writer.release()

def load_video(path: str) -> np.ndarray:
    """
    Loads a video file into a numpy array.

    Args:
        path: The path to the video file.

    Returns:
        The video as a numpy array (T, H, W).
    """
    cap = cv2.VideoCapture(path)
    frames = []
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
        frames.append(gray_frame)
    cap.release()
    return np.stack(frames, axis=0)

if __name__ == "__main__":
    verify_performance()