'''
A few key notes:
    I can use this as initial training data to start the model when I write it in C++, but I'm going to need some real data to actually train the model.
    I'll compile a text document full of driving footage, then I'll use a Python script to get the videos and plug them into the model.
    Initially I'll be using supervised learning, but this secondary step will be unsupervised so I'll need to figure out proper heuristics for the model.
'''

import numpy as np
from typing import Dict, Tuple
from torch.utils.data import Dataset
import torch


class GlareDataset(Dataset):
    def __init__(self, n_videos, transform=None):
        self.n_videos = n_videos
        self.transform = transform

    def __len__(self): return self.n_videos * 150 # Assuming 150 frames per video
    def __getitem__(self, idx):
        video_idx = idx // 150  # Assuming 150 frames per video
        frame_idx = idx % 150

        video_dict, mask_dict = generate_sample_videos(seed=video_idx)
        style = next(iter(video_dict))
        video = video_dict[style]  # (T,H,W)
        mask  = mask_dict[style]

        frame = video[frame_idx]
        mask  = mask[frame_idx]

        if self.transform:
            frame = self.transform(frame)
            mask  = self.transform(mask)
            
        return torch.from_numpy(frame)[None,...].float(), torch.from_numpy(mask)[None,...].float()  # shape (1,H,W), (1,H,W)

def create_glare(
        H: int, 
        W: int, 
        center: Tuple[int, int], 
        sigma: float
) -> np.ndarray:
    """2D Gaussian spot of shape (H, W), normalized to [0,1]."""
    y = np.arange(H)[:, None]
    x = np.arange(W)[None, :]
    cy, cx = center
    g = np.exp(-(((y - cy) ** 2 + (x - cx) ** 2) / (2 * sigma ** 2)))
    return g / g.max()

# Apply a static Gaussian glare at a random position.
def static_glare(
    background: np.ndarray,
    rng: np.random.RandomState,
    intensity: float,
    sigma: int
) -> Tuple[np.ndarray, np.ndarray]:
    
    T, H, W = background.shape
    cy = rng.randint(sigma, H - sigma)
    cx = rng.randint(sigma, W - sigma)
    spot = create_glare(H, W, (cy, cx), sigma)
    mask = np.tile(spot, (T, 1, 1)).astype(np.float32)
    video = np.clip(background + intensity * mask, 0, 1)
    return video, mask

# Apply a pulsing Gaussian glare at a random position with random period.
def pulsing_glare(
    background: np.ndarray,
    rng: np.random.RandomState,
    intensity: float,
    sigma: int
) -> Tuple[np.ndarray, np.ndarray]:
    
    T, H, W = background.shape
    cy = rng.randint(sigma, H - sigma)
    cx = rng.randint(sigma, W - sigma)
    spot = create_glare(H, W, (cy, cx), sigma)
    period = rng.randint(20, 60)
    mask = np.zeros_like(background)
    for t in range(T):
        alpha = 0.5 * (1 + np.sin(2 * np.pi * t / period))
        mask[t] = spot * alpha
    video = np.clip(background + intensity * mask, 0, 1)
    return video, mask.astype(np.float32)

# Apply a traveling Gaussian glare moving linearly between random start/end points.
def traveling_glare(
    background: np.ndarray,
    rng: np.random.RandomState,
    intensity: float,
    sigma: int
) -> Tuple[np.ndarray, np.ndarray]:

    T, H, W = background.shape
    start = (rng.randint(sigma, H - sigma), rng.randint(sigma, W - sigma))
    end = (rng.randint(sigma, H - sigma), rng.randint(sigma, W - sigma))
    mask = np.zeros_like(background)
    video = np.zeros_like(background)
    for t in range(T):
        frac = t / float(T - 1)
        cy = int(start[0] + frac * (end[0] - start[0]))
        cx = int(start[1] + frac * (end[1] - start[1]))
        spot = create_glare(H, W, (cy, cx), sigma)
        mask[t] = spot
        video[t] = np.clip(background[t] + intensity * spot, 0, 1)
    return video, mask.astype(np.float32)

# Generate one random glare style over a Gaussian background.
def generate_sample_videos(
    seed: int,
    T: int = 150,
    H: int = 128,
    W: int = 128,
    mean: float = 0.5,
    std: float = 0.1
) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray]]:
    """
    Returns:
      video: single-key dict {style: (T,H,W) float32 in [0,1]}
      mask:  single-key dict {style: (T,H,W) float32 ground-truth}
    """
    rng = np.random.RandomState(seed)
    background = rng.normal(loc=mean, scale=std, size=(T, H, W)).astype(np.float32)
    background = np.clip(background, 0.0, 1.0)

    glare_funcs = { # can add more if necessary
        'static': static_glare,
        'pulsing': pulsing_glare,
        'traveling': traveling_glare,
    }
    style = "pulsing" # For consistent testing
    intensity = float(rng.uniform(0.5, 1.0))
    sigma = int(rng.uniform(1, 5))

    video, mask = glare_funcs[style](background, rng, intensity, sigma)
    return {style: video}, {style: mask}

if __name__ == '__main__':
    import json
    import os

    # Create the data directory if it doesn't exist
    data_dir = os.path.join(os.path.dirname(__file__), '..', 'data')
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)

    video_dict, mask_dict = generate_sample_videos(seed=0, T=10, H=16, W=16)
    style = next(iter(video_dict))
    video = video_dict[style]
    mask = mask_dict[style]

    data_to_serialize = {
        'style': style,
        'video': video.tolist(),
        'mask': mask.tolist()
    }

    with open(os.path.join(data_dir, 'synthetic_data.json'), 'w') as f:
        json.dump(data_to_serialize, f, indent=4)