import os
import json
import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image
from typing import Tuple, Dict, List
from scipy.ndimage import convolve

def srgb_to_linear(img):
    # Approximate sRGB to linear conversion
    return np.power(img, 2.2)

def linear_to_srgb(img):
    # Approximate linear to sRGB conversion
    return np.power(img, 1/2.2)

def create_glare_kernel(H: int, W: int, center: Tuple[int, int], sigma: float) -> np.ndarray:
    """Creates a 2D Gaussian spot of shape (H, W), normalized to [0,1]."""
    y = np.arange(H)[:, None]
    x = np.arange(W)[None, :]
    cy, cx = center
    g = np.exp(-(((y - cy) ** 2 + (x - cx) ** 2) / (2 * sigma ** 2)))
    return g / g.max()

def hotspot_glare(H: int, W: int, rng: np.random.RandomState) -> Tuple[np.ndarray, np.ndarray]:
    """Generates a single hotspot glare source and its mask."""
    sigma = rng.uniform(10, 30)
    cy = rng.randint(sigma, H - sigma)
    cx = rng.randint(sigma, W - sigma)
    
    glare_kernel = create_glare_kernel(H, W, (cy, cx), sigma)
    
    intensity = rng.uniform(0.5, 1.0)
    glare = intensity * glare_kernel
    
    mask = (glare_kernel > 0.1).astype(np.float32)
    
    return glare, mask

def multi_hotspot_glare(H: int, W: int, rng: np.random.RandomState) -> Tuple[np.ndarray, np.ndarray]:
    """Generates multiple hotspot glare sources and their combined mask."""
    num_hotspots = rng.randint(2, 4)
    
    total_glare = np.zeros((H, W), dtype=np.float32)
    total_mask = np.zeros((H, W), dtype=np.float32)
    
    for _ in range(num_hotspots):
        glare, mask = hotspot_glare(H, W, rng)
        total_glare += glare
        total_mask = np.maximum(total_mask, mask)
        
    total_glare = np.clip(total_glare, 0, 1)
    
    return total_glare, total_mask

def traveling_glare(H: int, W: int, rng: np.random.RandomState, sequence_len: int) -> Tuple[np.ndarray, np.ndarray]:
    """Generates a traveling hotspot glare source and its mask across a sequence."""
    sigma = rng.uniform(10, 30)
    start_center = (rng.randint(sigma, H - sigma), rng.randint(sigma, W - sigma))
    end_center = (rng.randint(sigma, H - sigma), rng.randint(sigma, W - sigma))
    
    total_glare = np.zeros((sequence_len, H, W), dtype=np.float32)
    total_mask = np.zeros((sequence_len, H, W), dtype=np.float32)
    
    for t in range(sequence_len):
        # Linear interpolation with jitter
        interp_factor = t / (sequence_len - 1) if sequence_len > 1 else 0
        center_y = start_center[0] + (end_center[0] - start_center[0]) * interp_factor + rng.normal(0, 2)
        center_x = start_center[1] + (end_center[1] - start_center[1]) * interp_factor + rng.normal(0, 2)
        
        center = (np.clip(center_y, sigma, H-sigma), np.clip(center_x, sigma, W-sigma))

        glare_kernel = create_glare_kernel(H, W, center, sigma)
        intensity = rng.uniform(0.5, 1.0)
        glare = intensity * glare_kernel
        mask = (glare_kernel > 0.1).astype(np.float32)
        
        total_glare[t] = glare
        total_mask[t] = mask
        
    return total_glare, total_mask

def streak_glare(H: int, W: int, rng: np.random.RandomState) -> Tuple[np.ndarray, np.ndarray]:
    """Generates an elongated streak glare source and its mask."""
    center_y = rng.randint(0, H)
    center_x = rng.randint(0, W)
    
    length = rng.uniform(50, 150)
    width = rng.uniform(2, 5)
    angle = rng.uniform(-10, 10)
    
    y, x = np.ogrid[-center_y:H-center_y, -center_x:W-center_x]
    
    angle_rad = np.deg2rad(angle)
    rot_x = x * np.cos(angle_rad) + y * np.sin(angle_rad)
    rot_y = -x * np.sin(angle_rad) + y * np.cos(angle_rad)
    
    glare_kernel = np.exp(-( (rot_x**2 / (length**2)) + (rot_y**2 / (width**2)) ))
    glare_kernel /= glare_kernel.max()
    
    intensity = rng.uniform(0.5, 1.0)
    glare = intensity * glare_kernel
    mask = (glare_kernel > 0.2).astype(np.float32)
    
    return glare, mask

def bloom_glare(H: int, W: int, rng: np.random.RandomState) -> Tuple[np.ndarray, np.ndarray]:
    """Generates a bloom glare effect."""
    hotspot, mask = hotspot_glare(H, W, rng)
    
    # Convolve with a wider kernel to create a halo
    bloom_kernel_sigma = rng.uniform(30, 60)
    bloom_kernel = create_glare_kernel(H, W, (H//2, W//2), bloom_kernel_sigma)
    
    bloom_glare = convolve(hotspot, bloom_kernel)
    bloom_glare /= bloom_glare.max()
    
    intensity = rng.uniform(0.2, 0.5)
    glare = intensity * bloom_glare
    
    # The mask is the union of the hotspot and the bloom
    mask = np.maximum(mask, (bloom_glare > 0.1).astype(np.float32))
    
    return glare, mask

def flicker_glare(H: int, W: int, rng: np.random.RandomState, sequence_len: int) -> Tuple[np.ndarray, np.ndarray]:
    """Generates a global flicker effect."""
    glare = np.zeros((sequence_len, H, W), dtype=np.float32)
    mask = np.zeros((sequence_len, H, W), dtype=np.float32)
    
    flicker_frames = rng.choice(sequence_len, size=rng.randint(1, 5), replace=False)
    
    for t in flicker_frames:
        gain = rng.uniform(0.1, 0.3)
        glare[t] = gain
        mask[t] = 1.0
        
    return glare, mask

class RealGlareDataset(Dataset):
    def __init__(self, manifest_path: str, glare_config: Dict, transform=None, H=128, W=128, sequence_len=1):
        with open(manifest_path, 'r') as f:
            self.image_paths = json.load(f)
            
        self.glare_config = glare_config
        self.transform = transform
        self.H = H
        self.W = W
        self.sequence_len = sequence_len
        self.rng = np.random.RandomState(42)

    def __len__(self):
        return len(self.image_paths) - self.sequence_len + 1

    def __getitem__(self, idx):
        
        frames = []
        for i in range(self.sequence_len):
            img_path = self.image_paths[idx + i]
            try:
                with Image.open(img_path).convert('L') as img:
                    img = img.resize((self.W, self.H))
                    img = np.array(img, dtype=np.float32) / 255.0
            except FileNotFoundError:
                img = self.rng.rand(self.H, self.W).astype(np.float32)
            frames.append(img)
        
        frames = np.stack(frames)

        # Convert to linear space for glare addition
        frames_linear = srgb_to_linear(frames)

        # Generate glare
        glare_mode = self.rng.choice(list(self.glare_config['mode_weights'].keys()), p=list(self.glare_config['mode_weights'].values()))
        
        if glare_mode == 'hotspot':
            glare, mask = hotspot_glare(self.H, self.W, self.rng)
            glare = np.tile(glare, (self.sequence_len, 1, 1))
            mask = np.tile(mask, (self.sequence_len, 1, 1))
        elif glare_mode == 'multi-hotspot':
            glare, mask = multi_hotspot_glare(self.H, self.W, self.rng)
            glare = np.tile(glare, (self.sequence_len, 1, 1))
            mask = np.tile(mask, (self.sequence_len, 1, 1))
        elif glare_mode == 'traveling':
            glare, mask = traveling_glare(self.H, self.W, self.rng, self.sequence_len)
        elif glare_mode == 'streak':
            glare, mask = streak_glare(self.H, self.W, self.rng)
            glare = np.tile(glare, (self.sequence_len, 1, 1))
            mask = np.tile(mask, (self.sequence_len, 1, 1))
        elif glare_mode == 'bloom':
            glare, mask = bloom_glare(self.H, self.W, self.rng)
            glare = np.tile(glare, (self.sequence_len, 1, 1))
            mask = np.tile(mask, (self.sequence_len, 1, 1))
        elif glare_mode == 'flicker':
            glare, mask = flicker_glare(self.H, self.W, self.rng, self.sequence_len)
        else: # 'none'
            glare, mask = np.zeros_like(frames_linear), np.zeros_like(frames_linear)

        # Scene-aware intensity scaling
        base_luminance = frames_linear.mean()
        k_base = self.rng.uniform(self.glare_config['k_range'][0], self.glare_config['k_range'][1])
        k = k_base * (1 - base_luminance)**2 # Glare is stronger in darker scenes

        # Composite glare
        composited_img_linear = frames_linear + k * glare

        # Exposure adaptation and noise
        gain = self.rng.normal(1.0, 0.1)
        bias = self.rng.normal(0.0, 0.05)
        noise = self.rng.normal(0.0, 0.02, size=composited_img_linear.shape)
        
        composited_img_linear = (composited_img_linear * gain + bias) + noise
        composited_img_linear = np.clip(composited_img_linear, 0, 1)

        # For now, let's assume the model works in linear space, as it's better for the task
        final_frames = composited_img_linear

        if self.transform:
            # This part needs to be adapted for sequences if you have a transform that works on sequences
            pass
        
        # Return a random frame from the sequence
        frame_idx_to_return = self.rng.randint(self.sequence_len)
        
        return torch.from_numpy(final_frames[frame_idx_to_return])[None,...].float(), torch.from_numpy(mask[frame_idx_to_return])[None,...].float()