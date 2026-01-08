# train.py
import os, csv, json, math, time, random
from typing import Tuple

import sys

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torch import amp
from tqdm import tqdm
import matplotlib.pyplot as plt

from torchvision.transforms.functional import gaussian_blur
from model import UNet
from dataset_factory import get_dataset

# -----------------------
# Repro & cuDNN settings
# -----------------------
def seed_everything(seed: int = 42):
    random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True  # speed

seed_everything(42)

# -----------------------
# Metrics (fixed)
# -----------------------
@torch.no_grad()
def _to_probs(x: torch.Tensor) -> torch.Tensor:
    """
    Convert logits to probabilities in [0,1]. If input is already in [0,1],
    this is a no-op (safe).
    """
    # Heuristic: if any value is outside [0,1], assume logits.
    if x.min() < 0 or x.max() > 1:
        return torch.sigmoid(x)
    return x

@torch.no_grad()
def _binarize(pred_like: torch.Tensor, thr: float = 0.5) -> torch.Tensor:
    # expects probabilities in [0,1]
    return (pred_like > thr).to(torch.float32)

@torch.no_grad()
def iou_score(pred_logits: torch.Tensor,
              target: torch.Tensor,
              threshold: float = 0.5,
              eps: float = 1e-6) -> torch.Tensor:
    """
    IoU = TP / (TP + FP + FN)
    """
    probs = _to_probs(pred_logits)               # (B,1,H,W) in [0,1]
    pred  = _binarize(probs, threshold)
    tgt   = (target > 0.5).to(torch.float32)     # ensure binary

    # reduce over spatial dims; keep batch
    inter = (pred * tgt).sum(dim=(1,2,3))
    fp    = (pred * (1.0 - tgt)).sum(dim=(1,2,3))
    fn    = ((1.0 - pred) * tgt).sum(dim=(1,2,3))
    iou   = (inter + eps) / (inter + fp + fn + eps)

    return iou.mean()

@torch.no_grad()
def dice_score(pred_logits: torch.Tensor,
               target: torch.Tensor,
               threshold: float = 0.5,
               eps: float = 1e-6) -> torch.Tensor:
    """
    Dice = 2*TP / (2*TP + FP + FN)
    (Note: NOT the same denominator as IoU.)
    """
    probs = _to_probs(pred_logits)               # (B,1,H,W) in [0,1]
    pred  = _binarize(probs, threshold)
    tgt   = (target > 0.5).to(torch.float32)     # ensure binary

    inter = (pred * tgt).sum(dim=(1,2,3))
    fp    = (pred * (1.0 - tgt)).sum(dim=(1,2,3))
    fn    = ((1.0 - pred) * tgt).sum(dim=(1,2,3))
    dice  = (2.0 * inter + eps) / (2.0 * inter + fp + fn + eps)

    return dice.mean()

# -----------------------
# Your unsupervised term
# -----------------------
def unsupervised_evaluate(pred_mask_probs: torch.Tensor, frames: torch.Tensor) -> torch.Tensor:
    """
    Penalize deviation in avg brightness and reduction in std-dev of the masked video.
    Inputs expected as probabilities in [0,1].
    """
    avg_before, std_before = frames.mean(), frames.std()
    masked_video = frames * (1 - pred_mask_probs)
    avg_after, std_after = masked_video.mean(), masked_video.std()
    loss_avg = (avg_after - avg_before).pow(2)
    loss_std = torch.relu(std_after - std_before)
    return loss_avg + loss_std

# -----------------------
# Early stopping helper
# -----------------------
class EarlyStopper:
    def __init__(self, patience: int = 5, min_delta: float = 0.0):
        self.patience = patience
        self.min_delta = min_delta
        self.best = math.inf
        self.count = 0

    def step(self, val_loss: float) -> bool:
        if val_loss < self.best - self.min_delta:
            self.best = val_loss
            self.count = 0
            return False
        self.count += 1
        return self.count > self.patience

# -----------------------
# Training loop
# -----------------------
def train_model(
    epochs: int,
    batch_size: int,
    lr: float,
    device: torch.device,
    loss_weight: float,
    patience: int = 4,
    threshold: float = 0.5,
    blur_kernel_size: Tuple[int, int] = (5, 5),
    blur_sigma: float = 1.0,
    config_path: str = "config.json",
    resume_from: str | None = None,
):
    # Dataset
    full_ds = get_dataset(config_path)  # returns the whole set once

    n = len(full_ds)
    n_test = int(0.10 * n)
    n_val  = int(0.15 * n)
    n_train = n - n_val - n_test

    g = torch.Generator().manual_seed(42)  # deterministic split
    train_ds, val_ds, test_ds = torch.utils.data.random_split(full_ds, [n_train, n_val, n_test], generator=g)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                          num_workers=0, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False,
                          num_workers=0, pin_memory=True)

    # Model / loss / opt
    model = UNet(in_channels=3, out_channels=1).to(device)

    # Optional: load existing weights for fine-tuning
    if resume_from is not None and os.path.isfile(resume_from):
        print(f"Loading weights from {resume_from} for fine-tuning...")
        state = torch.load(resume_from, map_location=device)
        model.load_state_dict(state)
    elif resume_from:
        print(f"[Warning] resume_from='{resume_from}' not found, starting from scratch.")


    # We assume model emits probabilities; keep BCELoss.
    # If you switch UNet to output logits, change to BCEWithLogitsLoss and keep metrics as-is (they auto-sigmoid).
    criterion = nn.BCEWithLogitsLoss()

    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=2)
    scaler = amp.GradScaler('cuda', enabled=(device.type == "cuda"))

    # Logs
    train_losses, train_ious, train_dices = [], [], []
    val_losses,   val_ious,   val_dices   = [], [], []

    best_val_loss = math.inf
    early = EarlyStopper(patience=patience, min_delta=0.0)

    for epoch in range(1, epochs + 1):
        # -------- train --------
        model.train()
        t_loss = t_iou = t_dice = 0.0
        
        data_time = 0.0
        compute_time = 0.0
        end_time = time.time()
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{epochs} [Train]")
        
        for _, frames, masks in pbar:
            data_time += time.time() - end_time
            compute_start = time.time()
            
            frames = frames.to(device, non_blocking=True)      # (B,3,H,W), normalized
            masks  = masks.to(device, non_blocking=True).float()  # (B,1,H,W), {0,1}

            optimizer.zero_grad(set_to_none=True)
            with amp.autocast('cuda', enabled=(device.type == "cuda")):
                preds = model(frames)                     # logits
                loss_sup = criterion(preds, masks)
                pred_probs = torch.sigmoid(preds)

                if blur_sigma > 0:
                    pred_probs = gaussian_blur(pred_probs, kernel_size=blur_kernel_size, sigma=blur_sigma)

                loss_uns  = unsupervised_evaluate(pred_probs, frames)
                loss = loss_sup + loss_weight * loss_uns

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            compute_time += time.time() - compute_start

            # Metrics (use logits/probs consistently)
            with torch.no_grad():
                t_loss += loss.item() * frames.size(0)
                t_iou  += iou_score(preds, masks, threshold=threshold).item() * frames.size(0)
                t_dice += dice_score(preds, masks, threshold=threshold).item() * frames.size(0)
            
            end_time = time.time()

        avg_t_loss = t_loss / n_train
        avg_t_iou  = t_iou  / n_train
        avg_t_dice = t_dice / n_train
        train_losses.append(avg_t_loss); train_ious.append(avg_t_iou); train_dices.append(avg_t_dice)
        print(f"Train -> Loss: {avg_t_loss:.4f}, IoU: {avg_t_iou:.4f}, Dice: {avg_t_dice:.4f}")
        
        total_time = data_time + compute_time
        if total_time > 0:
            print(f"  [Timing] Data: {data_time:.2f}s ({data_time/total_time*100:.1f}%), Compute: {compute_time:.2f}s ({compute_time/total_time*100:.1f}%)")

        # -------- val --------
        model.eval()
        v_loss = v_iou = v_dice = 0.0
        with torch.no_grad():
            pbar = tqdm(val_loader, desc=f"Epoch {epoch}/{epochs} [Val]")
            for _, frames, masks in pbar:
                frames = frames.to(device, non_blocking=True)
                masks  = masks.to(device, non_blocking=True).float()

                with amp.autocast('cuda', enabled=(device.type == "cuda")):
                    preds = model(frames)                 # logits
                    loss_sup = criterion(preds, masks)

                v_loss += loss_sup.item() * frames.size(0)
                v_iou  += iou_score(preds, masks, threshold=threshold).item() * frames.size(0)
                v_dice += dice_score(preds, masks, threshold=threshold).item() * frames.size(0)

        avg_v_loss = v_loss / n_val
        avg_v_iou  = v_iou  / n_val
        avg_v_dice = v_dice / n_val
        val_losses.append(avg_v_loss); val_ious.append(avg_v_iou); val_dices.append(avg_v_dice)

        print(f" Val -> Loss: {avg_v_loss:.4f}, IoU: {avg_v_iou:.4f}, Dice: {avg_v_dice:.4f}")

        # LR schedule + early stop + best ckpt
        old_lr = optimizer.param_groups[0]['lr']
        scheduler.step(avg_v_loss)
        new_lr = optimizer.param_groups[0]['lr']
        if new_lr != old_lr:
            print(f"Learning rate reduced to {new_lr:.2e}")


        if avg_v_loss < best_val_loss:
            best_val_loss = avg_v_loss
            torch.save(model.state_dict(), "kvald_unet_best.pth")
            print("Saved best model -> kvald_unet_best.pth")

        if device.type == "cuda":
            torch.cuda.empty_cache()

        if early.step(avg_v_loss):
            print(f"Early stopping at epoch {epoch} (best val loss {best_val_loss:.6f})")
            break

    # Save last ckpt too
    torch.save(model.state_dict(), "kvald_unet_last.pth")
    print("Saved last model -> kvald_unet_last.pth")

    # -------- CSV + Plot --------
    out_dir = os.path.dirname(__file__)
    csv_path = os.path.join(out_dir, "training_metrics.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Epoch","Train Loss","Train IoU","Train Dice","Val Loss","Val IoU","Val Dice"])
        for i in range(len(train_losses)):
            writer.writerow([i+1, train_losses[i], train_ious[i], train_dices[i],
                             val_losses[i], val_ious[i], val_dices[i]])
    print(f"Training metrics saved to {csv_path}")

    plt.figure(figsize=(12,5))
    epochs_range = range(1, len(train_losses)+1)

    plt.subplot(1,3,1)
    plt.plot(epochs_range, train_losses, label="Train Loss")
    plt.plot(epochs_range, val_losses, label="Val Loss")
    plt.title("Loss over Epochs"); plt.xlabel("Epoch"); plt.ylabel("Loss"); plt.legend()

    plt.subplot(1,3,2)
    plt.plot(epochs_range, train_ious, label="Train IoU")
    plt.plot(epochs_range, val_ious, label="Val IoU")
    plt.title("IoU over Epochs"); plt.xlabel("Epoch"); plt.ylabel("IoU"); plt.legend()

    plt.subplot(1,3,3)
    plt.plot(epochs_range, train_dices, label="Train Dice")
    plt.plot(epochs_range, val_dices, label="Val Dice")
    plt.title("Dice over Epochs"); plt.xlabel("Epoch"); plt.ylabel("Dice"); plt.legend()

    plt.tight_layout()
    plot_path = os.path.join(out_dir, "training_metrics_plot.png")
    plt.savefig(plot_path)
    print(f"Training metrics plot saved to {plot_path}")

# -----------------------
# Main
# -----------------------
if __name__ == "__main__":
    current_dir = os.path.dirname(__file__)

    # Allow alternate config via CLI: python train.py config_flare7k_ft.json
    cfg_name = "config.json"
    if len(sys.argv) > 1:
        cfg_name = sys.argv[1]

    config_path = os.path.join(current_dir, cfg_name)
    with open(config_path, "r") as f:
        config = json.load(f)

    epochs     = int(config["train"]["num_epochs"])
    batch_size = int(config["train"]["batch_size"])
    lr         = float(config["train"]["learning_rate"])
    loss_weight = float(config["unsupervised"]["loss_weight"])
    patience   = int(config["train"].get("patience", 4))
    threshold  = float(config["train"].get("threshold", 0.5))
    smoothing_cfg = config.get("smoothing", {})
    blur_kernel_size = tuple(smoothing_cfg.get("kernel_size", [0, 0]))
    blur_sigma = float(smoothing_cfg.get("sigma", 0.0))

    # Optional resume_from for fine-tuning
    resume_from = config["train"].get("resume_from", None)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Is CUDA available? ", torch.cuda.is_available())
    print("Using device:", device)

    train_model(
        epochs=epochs,
        batch_size=batch_size,
        lr=lr,
        device=device,
        loss_weight=loss_weight,
        patience=patience,
        threshold=threshold,
        blur_kernel_size=blur_kernel_size,
        blur_sigma=blur_sigma,
        config_path=config_path,
        resume_from=resume_from,
    )

