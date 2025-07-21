import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm


from synthetic_data import GlareDataset

import os
import os
import json

from model import UNet             # your CNN definition
# from data import VideoFrameMaskDataset  # replace with your real dataset

def supervised_evaluate(pred_mask: torch.Tensor, true_mask: torch.Tensor) -> torch.Tensor:
    """
    Compute supervised MSE between predicted and true masks.
    """
    return torch.mean((pred_mask - true_mask) ** 2)

def unsupervised_evaluate(pred_mask: torch.Tensor, video: torch.Tensor) -> torch.Tensor:
    """
    Example unsupervised loss: penalize deviation in average brightness
    and reduction in std‐dev of the masked video.
    """
    avg_before, std_before = video.mean(), video.std()
    masked_video = video * (1 - pred_mask)
    avg_after, std_after   = masked_video.mean(), masked_video.std()

    loss_avg = (avg_after - avg_before).pow(2)
    loss_std = torch.relu(std_after - std_before)
    return loss_avg + loss_std

def iou_score(pred, target):
    intersection = (pred * target).sum()
    union = pred.sum() + target.sum() - intersection
    return (intersection + 1e-6) / (union + 1e-6)

def dice_score(pred, target):
    intersection = (pred * target).sum()
    return (2. * intersection + 1e-6) / (pred.sum() + target.sum() + 1e-6)

def create_dataset(n_videos: int):
    return GlareDataset(n_videos=n_videos, transform=None)  # replace with your dataset class

def train_model(
    dataset,
    epochs: int,
    batch_size: int,
    lr: float,
    device: str,
    loss_weight: float,
    val_split: float = 0.2,
):
    # Split into train/val
    n_val   = int(len(dataset) * val_split)
    n_train = len(dataset) - n_val
    train_ds, val_ds = random_split(dataset, [n_train, n_val])

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,  num_workers=4)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False, num_workers=4)

    # Model, loss, optimizer
    model     = UNet(in_channels=1, out_channels=1).to(device)
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    for epoch in range(1, epochs + 1):
        # — Training —
        model.train()
        total_loss = total_iou = total_dice = 0.0

        loop = tqdm(train_loader, desc=f"Epoch {epoch}/{epochs} [Train]")
        for frames, masks in loop:
            frames = frames.to(device)           # shape (B,1,H,W)
            masks  = masks.to(device).float()    # shape (B,1,H,W)

            optimizer.zero_grad()
            preds = model(frames)                # (B,1,H,W) with sigmoid
            loss_sup = criterion(preds, masks)
            loss_unsup = unsupervised_evaluate(preds, frames)
            loss = loss_sup + loss_weight * loss_unsup

            loss.backward()
            optimizer.step()

            total_loss += loss.item() * frames.size(0)
            total_iou  += iou_score(preds, masks).item() * frames.size(0)
            total_dice += dice_score(preds, masks).item() * frames.size(0)

        n_train = len(train_loader.dataset)
        print(f"Train -> Loss: {total_loss/n_train:.4f}, "
              f"IoU: {total_iou/n_train:.4f}, Dice: {total_dice/n_train:.4f}")

        # — Validation —
        model.eval()
        val_loss = val_iou = val_dice = 0.0

        with torch.no_grad():
            vloop = tqdm(val_loader, desc=f"Epoch {epoch}/{epochs} [Val]")
            for frames, masks in vloop:
                frames = frames.to(device)
                masks  = masks.to(device).float()

                preds = model(frames)
                loss_sup = criterion(preds, masks)

                val_loss += loss_sup.item() * frames.size(0)
                val_iou  += iou_score((preds > 0.5).int(), masks.int()).item() * frames.size(0)
                val_dice += dice_score((preds > 0.5).int(), masks.int()).item() * frames.size(0)

        n_val = len(val_loader.dataset)
        print(f" Val -> Loss: {val_loss/n_val:.4f}, ")
        print(f" Val -> Loss: {val_loss/n_val:.4f}, "
              f"IoU: {val_iou/n_val:.4f}, Dice: {val_dice/n_val:.4f}")
    # Save checkpoint
    torch.save(model.state_dict(), "kvald_unet.pth")
    print("Training complete. Model saved to kvald_unet.pth")

if __name__ == "__main__":
    # Get config parameters
    current_dir = os.path.dirname(__file__)
    config_full_path = os.path.join(current_dir, 'config.json')
    with open(config_full_path, 'r') as f:
        config = json.load(f)
    
    n_videos   = config['train']['n_videos']
    epochs     = config['train']['num_epochs']
    batch_size = config['train']['batch_size']
    lr         = config['train']['learning_rate']
    device     = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    loss_weight = config['unsupervised']['loss_weight']

    dataset = create_dataset(n_videos)
    train_model(dataset, epochs, batch_size, lr, device, loss_weight)
