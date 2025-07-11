# Full proof of concept script
# Imports
import cv2
import numpy as np
import json
from kalman_filter import MaskKalmanFilter
import argparse
import torch

# This will eventually need to be replaced, streaming the video frames directly into the neural network will be more efficient
# and less memory intensive.
def extract_frames(path):
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise IOError("Cannot open video file") # Temp, I'll figure out a more accurate error later on.

    frames = []
    while True: # Better way to do this?
        ret, frame = cap.read() # pull frames one by one until 
        if not ret: # If no successful read
            break

        # Conversion to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = gray.astype(np.float32) / 255.0 # would maintaining the type be better for better accuracy or is compressing it better for feeding into the nn?
        frames.append(gray)

    cap.release()

    if not frames:
        raise ValueError("No frames read")
    
    # stack into shape
    video_tensor = np.stack(frames, axis=0)
    return video_tensor

# Initial function
# I'll add multithreading later on, so that we can extract the frames and process them concurrently.
def video_processing(path: str, id: str) -> None:
    # For now this will just have a path to some kind of video file

    '''
    Processing steps:
        Take in a video and convert it to a single-key dict {style: (T,H,W) float32 in [0,1]}
        Not sure how converting it to grayscale will work but that's fine for now.
    '''
    # Extract the config parameters
    with open('config.json', 'r') as f:
        config = json.load(f)

    ksize = tuple(config['smoothing']['kernel_size'])
    sigma = config['smoothing']['sigma']

    # Extract the frames and generate the first mask for the kf
    video_tensor = extract_frames(path)
    first_mask = generate_mask(video_tensor[0])
    kf = MaskKalmanFilter(shape=first_mask.shape)

    mask_frames = []
    final_frames = []

    for frame in video_tensor:
        mask = generate_mask(frame)
        mask = smooth_output(mask, ksize, sigma)

        filtered = kf.update(mask)

        """
        # Do our verification step here, this is a temporary placeholder
        how_good = verify_feedback(filtered)
        
        if how_good is not good:
            raise ValueError("Generated mask is flawed or not suitable for processing")
        """

        # Now send the filtered mask out to our output function, along with the original frame for combination.
        mask_frames.append(filtered)
        final_frames.append(finalize_output(frame, filtered))

    mask_array = np.stack(mask_frames, axis=0)
    final_array = np.stack(final_frames, axis=0)
    
    frames_to_vid(mask_array, id + "_mask.mp4")
    frames_to_vid(final_array, id + "_final.mp4")
    return None

# Mask generation function
def generate_mask(frame: np.ndarray) -> np.ndarray:
    inp = torch.from_numpy(frame[None, ...]).to(device='cuda')  # Assuming the model is on GPU
    with torch.no_grad():
        out = model(inp).cpu().squeeze().numpy()  # Get the output from the model
    return out

# Smoothing function, makes the code more readable
def smooth_output(mask, ksize, sigma):    
    return cv2.GaussianBlur(mask, ksize, sigma)

# Feedback verification function
def verify_feedback(smoothed_output):
    # This function will verify the feedback from the smoothed output
    # This is kinda just a sanity check, is the processed mask that we've output good according to our heuristics?
    # Not sure how to achieve this for now

    pass

# Finalized output function
def finalize_output(frame, mask):
    # This function will layer the mask onto the frame
    return "finalized_output"

# Converts our frames back into a video
def frames_to_vid(frames, id) -> None:
    # recombine all of the frames 
    T, H, W = frames.shape

    fps = 30.0 # can change
    codec: str = 'mp4v'

    fourcc = cv2.VideoWriter_fourcc(*codec)
    
    writer = cv2.VideoWriter(id, fourcc, fps, (W, H), isColor=False)
    if not writer.isOpened():
        raise IOError
    
    for t in range(T):
        img = (np.clip(frames[t], 0.0, 1.0) *255).astype(np.uint8)
        writer.write(img)

    writer.release()
    return

# Main function to run the code
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process a video file")
    parser.add_argument("video_path", type=str, help="Path to the input video file")
    args = parser.parse_args()

    video_path = args.video_path
    id = video_path.split('/')[-1].split('.')[0]  # Extracting the video ID from the path

    video_processing(video_path, id)
