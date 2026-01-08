# verify_performance.py
import os
import json
import math
import csv
from typing import Tuple
import cv2

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
import matplotlib.pyplot as plt

from model import UNet
from dataset_factory import get_dataset

# -----------------------------
# Metrics
# -----------------------------
@torch.no_grad()
def _to_probs(logits: torch.Tensor) -> torch.Tensor:
    # Handles either logits or already-in-[0,1] tensors
    if logits.min() < 0 or logits.max() > 1:
        return torch.sigmoid(logits)
    return logits

@torch.no_grad()
def _binarize(probs: torch.Tensor, thr: float = 0.5) -> torch.Tensor:
    return (probs > thr).to(torch.float32)

@torch.no_grad()
def iou_score(pred_logits: torch.Tensor, target: torch.Tensor, thr: float = 0.5, eps: float = 1e-6, fp_bias: float = 1.0, fn_bias: float = 1.0) -> float:
    probs = _to_probs(pred_logits)
    pred  = _binarize(probs, thr)
    tgt   = (target > 0.5).to(torch.float32)

    inter = (pred * tgt).sum().item()
    fp    = (pred * (1.0 - tgt)).sum().item()
    fn    = ((1.0 - pred) * tgt).sum().item()
    return (inter + eps) / (inter + fp_bias * fp + fn_bias * fn + eps)

@torch.no_grad()
def dice_score(pred_logits: torch.Tensor, target: torch.Tensor, thr: float = 0.5, eps: float = 1e-6) -> float:
    probs = _to_probs(pred_logits)
    pred  = _binarize(probs, thr)
    tgt   = (target > 0.5).to(torch.float32)

    inter = (pred * tgt).sum().item()
    fp    = (pred * (1.0 - tgt)).sum().item()
    fn    = ((1.0 - pred) * tgt).sum().item()
    return (2.0 * inter + eps) / (2.0 * inter + fp + fn + eps)

# -----------------------------
# Visualization helpers
# -----------------------------
def _to_numpy_img(t: torch.Tensor) -> np.ndarray:
    """
    t: [C,H,W] or [H,W] in [0,1] -> np.uint8 [H,W,C] or [H,W]
    """
    if t.dim() == 3:
        if t.size(0) == 1:
            t = t[0] # Grayscale, remove channel dim
        else:
            t = t.permute(1, 2, 0) # [C,H,W] -> [H,W,C]
    t = t.clamp(0,1).detach().cpu().numpy()
    return (t * 255.0).astype(np.uint8)

def _make_overlay(gt_mask: np.ndarray, pred_mask: np.ndarray) -> np.ndarray:
    """
    gt_mask, pred_mask: uint8 [H,W] (0..255)
    Returns color overlay [H,W,3]:
      - GT positive (green), Pred positive (red), overlap (yellow)
    """
    gt = (gt_mask > 127).astype(np.uint8)
    pr = (pred_mask > 127).astype(np.uint8)

    h, w = gt.shape
    overlay = np.zeros((h, w, 3), dtype=np.uint8)
    # red for pred
    overlay[..., 2] = (pr * 255)
    # green for gt
    overlay[..., 1] = (gt * 255)
    # yellow where both
    both = (gt & pr).astype(bool)
    overlay[both] = np.array([255, 255, 0], dtype=np.uint8)
    return overlay

def _masked_frame(frame_u8: np.ndarray, mask_u8: np.ndarray) -> np.ndarray:
    """
    Simple dimming: frame * (1 - mask)
    Expect frame_u8: [H,W] 0..255, mask_u8: [H,W] 0..255
    """
    frame = frame_u8.astype(np.float32) / 255.0
    mask  = (mask_u8.astype(np.float32) / 255.0)
    if frame.ndim == 3 and mask.ndim == 2:
        mask = np.expand_dims(mask, axis=2) # so it broadcasts over color channels
    out = frame * (1.0 - mask)
    return (out * 255.0).clip(0,255).astype(np.uint8)

def _save_panel(
    out_dir: str,
    idx: int,
    clean_frame: torch.Tensor, # [C,H,W] in [0,1]
    frame: torch.Tensor,      # [C,H,W] in [0,1]
    gt_mask: torch.Tensor,    # [1,H,W] in {0,1} (float)
    pred_logits: torch.Tensor # [1,H,W]
):
    os.makedirs(out_dir, exist_ok=True)
    clean_frame_u8 = _to_numpy_img(clean_frame)
    frame_u8 = _to_numpy_img(frame)
    gt_u8    = _to_numpy_img(gt_mask)
    pred_u8  = _to_numpy_img(_binarize(_to_probs(pred_logits)))

    overlay = _make_overlay(gt_u8, pred_u8)
    pred_masked = _masked_frame(frame_u8, pred_u8)

    # 2x3 panel
    fig, axes = plt.subplots(2, 3, figsize=(12, 8))
    axes[0,0].imshow(clean_frame_u8); axes[0,0].set_title("Input (clean)"); axes[0,0].axis('off')
    axes[0,1].imshow(frame_u8); axes[0,1].set_title("Input (glare)"); axes[0,1].axis('off')
    axes[0,2].imshow(pred_masked); axes[0,2].set_title("Pred-Masked"); axes[0,2].axis('off')
    axes[1,0].imshow(gt_u8, cmap='gray', vmin=0, vmax=255);    axes[1,0].set_title("GT Mask");       axes[1,0].axis('off')
    axes[1,1].imshow(pred_u8, cmap='gray', vmin=0, vmax=255);  axes[1,1].set_title("Pred Mask");     axes[1,1].axis('off')
    axes[1,2].imshow(overlay);                                 axes[1,2].set_title("Overlay (GT vs Pred)"); axes[1,2].axis('off')

    fig.tight_layout()
    path = os.path.join(out_dir, f"sample_{idx:04d}.png")
    fig.savefig(path, dpi=120)
    plt.close(fig)

# -----------------------------
# Main verification
# -----------------------------
@torch.no_grad()
def verify_performance(config_path: str = "config.json",
                       model_path: str = "kvald_unet_best.pth",
                       out_dir: str = "verify_out",
                       num_samples: int = 64,
                       batch_size: int = 8,
                       force_split: str = None):
    """
    Evaluate the trained model on online-synthesized samples and save visual panels.

    - Uses the dataset_factory's glare_synth_online (or whatever your config selects).
    - Computes per-sample IoU/Dice against the synthetic ground truth masks.
    - Saves panels: input, GT mask, Pred mask, overlay, Pred-masked, GT-masked.

    Args:
        config_path: path to your config.json
        model_path:  path to model weights (default: kvald_unet_best.pth from train.py)
        out_dir:     where to save panels + CSV
        num_samples: number of samples to evaluate/visualize
        batch_size:  dataloader batch size
        force_split: override config.data.split (e.g., "val"), or None to use config
    """
    # Load config
    current_dir = os.path.dirname(__file__)
    cfg_full = os.path.join(current_dir, config_path)
    with open(cfg_full, "r") as f:
        cfg = json.load(f)

    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("CUDA available:", torch.cuda.is_available(), "| Using:", device)

    # If requested, override split to 'val' for verification
    if force_split is not None:
        cfg = json.loads(json.dumps(cfg))  # deep copy via JSON
        cfg.setdefault("data", {})["split"] = force_split

        # write a temporary config to keep get_dataset signature simple
        tmp_cfg = os.path.join(current_dir, "_tmp_verify_config.json")
        with open(tmp_cfg, "w") as f:
            json.dump(cfg, f, indent=2)
        cfg_path_used = tmp_cfg
    else:
        cfg_path_used = cfg_full

    # Dataset + loader
    ds = get_dataset(cfg_path_used)
    if len(ds) > num_samples:
        # Take a deterministic subset (first N indices)
        sel_indices = list(range(num_samples))
    else:
        sel_indices = list(range(len(ds)))

    # Build a small sampler via Subset without importing torch.utils.data.Subset explicitly
    from torch.utils.data import Subset
    sub_ds = Subset(ds, sel_indices)
    dl = DataLoader(sub_ds, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=False)

    # Model
    model = UNet(in_channels=3, out_channels=1).to(device)
    ckpt_path = os.path.join(current_dir, model_path)
    if not os.path.isfile(ckpt_path):
        raise FileNotFoundError(f"Model weights not found: {ckpt_path}")
    state = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(state)
    model.eval()

    # Threshold from training config (default 0.5)
    thr = float(cfg.get("train", {}).get("threshold", 0.5))

    # IoU biases
    iou_fp_bias = float(cfg.get("verification", {}).get("iou_fp_bias", 1.0))
    iou_fn_bias = float(cfg.get("verification", {}).get("iou_fn_bias", 1.0))

    # Output dirs
    os.makedirs(out_dir, exist_ok=True)
    panels_dir = os.path.join(out_dir, "panels")
    os.makedirs(panels_dir, exist_ok=True)

    # CSV
    csv_path = os.path.join(out_dir, "metrics.csv")
    csv_f = open(csv_path, "w", newline="")
    writer = csv.writer(csv_f)
    writer.writerow(["index", "iou", "dice"])

    total_iou = 0.0
    total_dice = 0.0
    n = 0
    idx_global = 0

    for clean_frames, frames, gt_masks in tqdm(dl, desc="Verifying"):
        frames   = frames.to(device, non_blocking=True)  # [B,1,H,W], [0,1]
        gt_masks = gt_masks.to(device, non_blocking=True)  # [B,1,H,W], {0,1]
        clean_frames = clean_frames.to(device, non_blocking=True)

        logits = model(frames)  # [B,1,H,W] logits
        probs = torch.sigmoid(logits)

        # Per-sample metrics + visualizations
        B = frames.size(0)
        for b in range(B):
            p = probs[b,0].detach().cpu().numpy()

            thr_clean = 0.6 # May need to tune
            binary = (p>thr_clean).astype(np.uint8)

            kernel_open = np.ones((3,3), np.uint8)
            kernel_close = np.ones((5,5), np.uint8)

            # Remove tiny speckles
            binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel_open)

            # Fill gaps, smooth shape
            binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel_close)

            # Widen prediction slightly
            binary = cv2.dilate(binary, kernel_open, iterations=1)

            # Back to torch
            pred_clean = torch.from_numpy(binary).to(device).float().unsqueeze(0)

            # Metrics
            iou  = iou_score(pred_clean, gt_masks[b], thr=0.5,
                         fp_bias=iou_fp_bias, fn_bias=iou_fn_bias)
            dice = dice_score(pred_clean, gt_masks[b], thr=0.5)

            total_iou += iou
            total_dice += dice
            n += 1
            writer.writerow([idx_global, f"{iou:.6f}", f"{dice:.6f}"])

            # Save panel
            _save_panel(
                panels_dir,
                idx_global,
                clean_frame=clean_frames[b].detach(),
                frame=frames[b].detach(),
                gt_mask=gt_masks[b].detach(),
                pred_logits=logits[b].detach(),
            )
            idx_global += 1

    csv_f.close()

    avg_iou = total_iou / max(1, n)
    avg_dice = total_dice / max(1, n)
    print(f"Average IoU:  {avg_iou:.4f}")
    print(f"Average Dice: {avg_dice:.4f}")

    # Also write a quick README with thresholds
    iou_thr  = float(cfg.get("verification", {}).get("iou_threshold", 0.5))
    dice_thr = float(cfg.get("verification", {}).get("dice_threshold", 0.7))
    passed = (avg_iou >= iou_thr) and (avg_dice >= dice_thr)

    with open(os.path.join(out_dir, "summary.txt"), "w") as f:
        f.write(f"Model: {model_path}\n")
        f.write(f"Samples: {n}\n")
        f.write(f"Average IoU:  {avg_iou:.4f}\n")
        f.write(f"Average Dice: {avg_dice:.4f}\n")
        f.write(f"Thresholds (IoU/Dice): {iou_thr}/{dice_thr}\n")
        f.write(f"Result: {'PASSED' if passed else 'FAILED'}\n")

    print("Verification", "PASSED" if passed else "FAILED")
    print(f"- Panels: {panels_dir}")
    print(f"- CSV:    {csv_path}")

if __name__ == "__main__":
    # By default, try to evaluate the validation split if available.
    verify_performance(force_split="val")
