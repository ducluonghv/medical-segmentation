"""
Training script - consistent with TransAttUnet paper settings:
  Optimizer  : SGD, momentum=0.9, weight_decay=1e-4
  LR schedule: step-decay by ×0.1 every 40 epochs (MultiStepLR)
  Initial LR : 0.0001  (paper setting, p.6 — matches TransAttUnet exactly)
  Epochs     : 100
  Batch size : 4
  Loss       : 0.5 * BCE + 0.5 * Dice  (Eq. 9 of the paper)
  Threshold  : 0.5 (binary output)

Usage:
  python train.py --dataset isic --data_root /path/to/ISIC
  python train.py --dataset glas --data_root /path/to/GlaS --model proposed
  python train.py --dataset covid --data_root /path/to/COVID --model baseline
"""
import argparse
import os
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from datasets        import DATASET_REGISTRY, DATASET_DEFAULTS
from models          import TransAttUnet, ProposedModel, UNet, AttUNet
from utils.metrics   import MetricAccumulator
from utils.visualize import plot_training_curves


# ──────────────────────────────────────────────────────────────────────
# Loss
# ──────────────────────────────────────────────────────────────────────

class CombinedLoss(nn.Module):
    """α * BCE + β * Dice  (α=β=0.5, ε=1e-6, same as paper Eq. 9)."""

    def __init__(self, alpha: float = 0.5, beta: float = 0.5, eps: float = 1e-6):
        super().__init__()
        self.alpha = alpha
        self.beta  = beta
        self.eps   = eps
        self.bce   = nn.BCEWithLogitsLoss()

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        bce_loss = self.bce(logits, targets)

        probs = torch.sigmoid(logits)
        inter = (probs * targets).sum(dim=(1, 2, 3))
        union = probs.sum(dim=(1, 2, 3)) + targets.sum(dim=(1, 2, 3))
        dice_loss = 1.0 - ((2 * inter + self.eps) / (union + self.eps)).mean()

        return self.alpha * bce_loss + self.beta * dice_loss


# ──────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────

def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark     = False


def build_model(model_name: str, n_channels: int, n_classes: int):
    if model_name == 'baseline':
        return TransAttUnet(n_channels=n_channels, n_classes=n_classes)
    if model_name == 'proposed':
        return ProposedModel(n_channels=n_channels, n_classes=n_classes, use_adar=True, use_csr=True)
    if model_name == 'adar_only':
        return ProposedModel(n_channels=n_channels, n_classes=n_classes, use_adar=True, use_csr=False)
    if model_name == 'csr_only':
        return ProposedModel(n_channels=n_channels, n_classes=n_classes, use_adar=False, use_csr=True)
    if model_name == 'unet':
        return UNet(n_channels=n_channels, n_classes=n_classes)
    if model_name == 'att_unet':
        return AttUNet(n_channels=n_channels, n_classes=n_classes)
    raise ValueError(f"Unknown model: {model_name}")


def get_logits(output):
    """Handle models that return (logits, routing_info) or just logits."""
    if isinstance(output, tuple):
        return output[0]
    return output


# ──────────────────────────────────────────────────────────────────────
# Train / validate one epoch
# ──────────────────────────────────────────────────────────────────────

def train_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0.0
    for imgs, masks, _ in loader:
        imgs, masks = imgs.to(device), masks.to(device)
        optimizer.zero_grad()
        logits = get_logits(model(imgs))
        loss   = criterion(logits, masks)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * imgs.size(0)
    return total_loss / len(loader.dataset)


@torch.no_grad()
def validate_epoch(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    acc = MetricAccumulator()
    for imgs, masks, _ in loader:
        imgs, masks = imgs.to(device), masks.to(device)
        logits = get_logits(model(imgs))
        loss   = criterion(logits, masks)
        total_loss += loss.item() * imgs.size(0)
        probs = torch.sigmoid(logits)
        for p, m in zip(probs, masks):
            acc.update(p.cpu(), m.cpu())
    return total_loss / len(loader.dataset), acc.summary()


# ──────────────────────────────────────────────────────────────────────
# Main training loop
# ──────────────────────────────────────────────────────────────────────

def train(args):
    set_seed(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # macOS + CPU: num_workers > 0 gây deadlock do fork; tự động về 0
    import platform
    nw = args.num_workers
    if nw > 0 and platform.system() == 'Darwin' and not torch.cuda.is_available():
        nw = 0
    pin = device.type == 'cuda'

    print(f"Device: {device} | Model: {args.model} | Dataset: {args.dataset} | workers: {nw}")

    defaults = DATASET_DEFAULTS[args.dataset]
    DatasetCls = DATASET_REGISTRY[args.dataset]

    train_ds = DatasetCls(args.data_root, split='train', img_size=defaults['img_size'], augment=True)
    val_ds   = DatasetCls(args.data_root, split='val',   img_size=defaults['img_size'], augment=False)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=nw, pin_memory=pin)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch_size, shuffle=False,
                              num_workers=nw, pin_memory=pin)

    model = build_model(args.model, n_channels=defaults['n_channels'], n_classes=1).to(device)
    criterion = CombinedLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr,
                                momentum=0.9, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer, milestones=[40, 80], gamma=0.1
    )

    save_dir = Path(args.save_dir) / args.dataset / args.model
    save_dir.mkdir(parents=True, exist_ok=True)

    history       = {'train_loss': [], 'val_loss': [], 'val_dice': []}
    best_dice     = 0.0
    no_improve    = 0
    patience      = args.patience

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        train_loss             = train_epoch(model, train_loader, optimizer, criterion, device)
        val_loss, val_summary  = validate_epoch(model, val_loader, criterion, device)
        scheduler.step()

        val_dice = val_summary['dice']['mean']
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['val_dice'].append(val_dice)

        elapsed = time.time() - t0
        print(
            f"Epoch [{epoch:3d}/{args.epochs}]  "
            f"train_loss={train_loss:.4f}  val_loss={val_loss:.4f}  "
            f"val_dice={val_dice*100:.2f}%  lr={scheduler.get_last_lr()[0]:.2e}  "
            f"({elapsed:.1f}s)"
        )

        if val_dice > best_dice:
            best_dice  = val_dice
            no_improve = 0
            torch.save({
                'epoch': epoch,
                'model_state': model.state_dict(),
                'optimizer_state': optimizer.state_dict(),
                'val_dice': val_dice,
            }, save_dir / 'best_model.pth')
            print(f"  ✓ Saved best model  (val_dice={best_dice*100:.2f}%)")
        else:
            no_improve += 1
            if patience > 0 and no_improve >= patience:
                print(f"\nEarly stopping at epoch {epoch} (no improvement for {patience} epochs).")
                break

    # Save final model and training curves
    torch.save(model.state_dict(), save_dir / 'last_model.pth')
    plot_training_curves(history, save_path=str(save_dir / 'training_curves.png'))
    print(f"\nTraining done. Best val Dice: {best_dice*100:.2f}%")
    print(f"Checkpoints saved to: {save_dir}")


# ──────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description='Train segmentation model')
    p.add_argument('--dataset',     type=str, required=True,
                   choices=['isic', 'glas', 'covid', 'lung', 'dsb2018'])
    p.add_argument('--data_root',   type=str, required=True, help='Path to dataset root')
    p.add_argument('--model',       type=str, default='proposed',
                   choices=['baseline', 'proposed', 'adar_only', 'csr_only',
                            'unet', 'att_unet'])
    p.add_argument('--epochs',      type=int, default=100)
    p.add_argument('--batch_size',  type=int, default=4)
    p.add_argument('--lr',          type=float, default=0.0001)
    p.add_argument('--num_workers', type=int, default=4)
    p.add_argument('--patience',    type=int, default=20,
                   help='Early-stop after N epochs without val-dice improvement (0=disable)')
    p.add_argument('--save_dir',    type=str, default='checkpoints')
    p.add_argument('--seed',        type=int, default=42)
    return p.parse_args()


if __name__ == '__main__':
    train(parse_args())
