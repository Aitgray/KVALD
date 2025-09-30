import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm
import csv
import matplotlib.pyplot as plt
import os
import json
from model import UNet

from dataset_factory import get_dataset

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


def train_model(
    epochs: int,
    batch_size: int,
    lr: float,
    device: str,
    loss_weight: float,
    val_split: float = 0.2,
):
    # Create datasets
    full_dataset = get_dataset()
    n_total = len(full_dataset)
    n_val = int(n_total * val_split)
    n_train = n_total - n_val
    train_ds, val_ds = random_split(full_dataset, [n_train, n_val])

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,  num_workers=6, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False, num_workers=6, pin_memory=True)

    # Model, loss, optimizer
    model     = UNet(in_channels=1, out_channels=1).to(device)
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    # Lists to store metrics for plotting
    train_losses = []
    train_ious = []
    train_dices = []
    val_losses = []
    val_ious = []
    val_dices = []

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

        avg_train_loss = total_loss / n_train
        avg_train_iou = total_iou / n_train
        avg_train_dice = total_dice / n_train
        train_losses.append(avg_train_loss)
        train_ious.append(avg_train_iou)
        train_dices.append(avg_train_dice)

        print(f"Train -> Loss: {avg_train_loss:.4f}, "
              f"IoU: {avg_train_iou:.4f}, Dice: {avg_train_dice:.4f}")

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

        avg_val_loss = val_loss / n_val
        avg_val_iou = val_iou / n_val
        avg_val_dice = val_dice / n_val
        val_losses.append(avg_val_loss)
        val_ious.append(avg_val_iou)
        val_dices.append(avg_val_dice)

        print(f" Val -> Loss: {avg_val_loss:.4f}, "
              f"IoU: {avg_val_iou:.4f}, Dice: {avg_val_dice:.4f}")
        if device == "cuda":
            torch.cuda.empty_cache()
    
    # Save metrics to CSV
    metrics_path = os.path.join(os.path.dirname(__file__), 'training_metrics.csv')
    with open(metrics_path, 'w', newline='') as csvfile:
        fieldnames = ['Epoch', 'Train Loss', 'Train IoU', 'Train Dice', 'Val Loss', 'Val IoU', 'Val Dice']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

        writer.writeheader()
        for i in range(epochs):
            writer.writerow({
                'Epoch': i + 1,
                'Train Loss': train_losses[i],
                'Train IoU': train_ious[i],
                'Train Dice': train_dices[i],
                'Val Loss': val_losses[i],
                'Val IoU': val_ious[i],
                'Val Dice': val_dices[i]
            })
    print(f"Training metrics saved to {metrics_path}")

    # Plotting
    epochs_range = range(1, epochs + 1)

    plt.figure(figsize=(12, 5))

    plt.subplot(1, 3, 1)
    plt.plot(epochs_range, train_losses, label='Train Loss')
    plt.plot(epochs_range, val_losses, label='Val Loss')
    plt.title('Loss over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    plt.subplot(1, 3, 2)
    plt.plot(epochs_range, train_ious, label='Train IoU')
    plt.plot(epochs_range, val_ious, label='Val IoU')
    plt.title('IoU over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('IoU')
    plt.legend()

    plt.subplot(1, 3, 3)
    plt.plot(epochs_range, train_dices, label='Train Dice')
    plt.plot(epochs_range, val_dices, label='Val Dice')
    plt.title('Dice Score over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Dice Score')
    plt.legend()

    plt.tight_layout()
    plot_path = os.path.join(os.path.dirname(__file__), 'training_metrics_plot.png')
    plt.savefig(plot_path)
    print(f"Training metrics plot saved to {plot_path}")

    # Save checkpoint
    torch.save(model.state_dict(), "kvald_unet.pth")
    print("Training complete. Model saved to kvald_unet.pth")

if __name__ == "__main__":
    # Get config parameters
    current_dir = os.path.dirname(__file__)
    config_full_path = os.path.join(current_dir, 'config.json')
    with open(config_full_path, 'r') as f:
        config = json.load(f)
    
    epochs     = config['train']['num_epochs']
    batch_size = config['train']['batch_size']
    lr         = config['train']['learning_rate']

    print(f"Is cuda available? {torch.cuda.is_available()}")

    device     = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    loss_weight = config['unsupervised']['loss_weight']

    train_model(epochs, batch_size, lr, device, loss_weight)
