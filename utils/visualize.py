"""
Visualization utilities for thesis experiments.

Four visualization types:
  1. plot_segmentation      - qualitative comparison grid (thesis Fig. 4/5 style)
  2. plot_adar_weights      - ADAR routing weight distributions across test samples
  3. plot_csr_weights       - CSR scale weights per decoder stage (bar chart)
  4. plot_feature_maps      - activation heatmap of bottleneck features

All functions save to disk and optionally display inline.
"""
import os
from pathlib import Path
from typing import Optional

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import torch
import torch.nn.functional as F


# ──────────────────────────────────────────────────────────────────────
# 1. Segmentation comparison grid
# ──────────────────────────────────────────────────────────────────────

def plot_segmentation(
    images:    list,          # list of (3,H,W) tensors or (H,W,3) numpy arrays
    gt_masks:  list,          # list of (1,H,W) or (H,W) binary arrays
    pred_masks: list,         # list of (1,H,W) probability arrays (after sigmoid)
    model_names: list[str],   # e.g. ['TransAttUnet', 'Proposed']
    all_preds:  Optional[list] = None,  # list of lists if comparing multiple models
    save_path:  Optional[str]  = None,
    show:       bool           = False,
    n_samples:  int            = 4,
    threshold:  float          = 0.5,
):
    """Segmentation comparison grid in the style of TransAttUnet Fig. 4.

    Columns: Input | GT | model_1 pred | model_2 pred | ...
    Rows   : n_samples randomly selected images
    """
    n = min(n_samples, len(images))
    # Determine number of model columns
    if all_preds is None:
        all_preds = [pred_masks]
    n_models = len(all_preds)
    n_cols   = 2 + n_models           # Input + GT + each model

    fig, axes = plt.subplots(n, n_cols, figsize=(3 * n_cols, 3 * n))
    if n == 1:
        axes = axes[np.newaxis, :]

    col_titles = ['Input', 'Ground Truth'] + model_names
    for j, title in enumerate(col_titles):
        axes[0, j].set_title(title, fontsize=10, fontweight='bold')

    for i in range(n):
        # Input image (denormalize roughly)
        img = _to_numpy_img(images[i])
        axes[i, 0].imshow(img)
        axes[i, 0].axis('off')

        # GT
        gt = np.squeeze(gt_masks[i]) if hasattr(gt_masks[i], '__len__') else gt_masks[i]
        if isinstance(gt, torch.Tensor):
            gt = gt.cpu().numpy()
        axes[i, 1].imshow(gt, cmap='gray')
        axes[i, 1].axis('off')

        # Each model prediction
        for j, preds in enumerate(all_preds):
            pred = np.squeeze(preds[i])
            if isinstance(pred, torch.Tensor):
                pred = pred.cpu().numpy()
            binary_pred = (pred > threshold).astype(np.float32)
            axes[i, 2 + j].imshow(binary_pred, cmap='gray')
            axes[i, 2 + j].axis('off')

    plt.tight_layout()
    _save_or_show(fig, save_path, show)


# ──────────────────────────────────────────────────────────────────────
# 2. ADAR routing weight distributions
# ──────────────────────────────────────────────────────────────────────

def plot_adar_weights(
    weights_list: list,       # list of (3,) numpy arrays collected over test set
    save_path: Optional[str] = None,
    show: bool = False,
):
    """Box plots showing distribution of [w_tsa, w_gsa, w_orig] across test images.

    This visualizes that ADAR genuinely adapts routing rather than converging
    to fixed weights - key evidence for the thesis contribution.
    """
    weights_arr = np.stack(weights_list)          # (N, 3)
    labels = ['w_tsa\n(channel att.)', 'w_gsa\n(spatial att.)', 'w_orig\n(residual)']

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    # Box plot - distribution per weight type
    axes[0].boxplot(
        [weights_arr[:, 0], weights_arr[:, 1], weights_arr[:, 2]],
        labels=labels, patch_artist=True,
        boxprops=dict(facecolor='steelblue', alpha=0.6),
        medianprops=dict(color='red', linewidth=2),
    )
    axes[0].set_ylabel('Routing weight')
    axes[0].set_title('ADAR routing weight distribution (test set)')
    axes[0].set_ylim(0, 1)
    axes[0].axhline(1/3, color='gray', linestyle='--', linewidth=1, label='Uniform (1/3)')
    axes[0].legend(fontsize=8)

    # Mean bar chart with std error bars
    means = weights_arr.mean(axis=0)
    stds  = weights_arr.std(axis=0)
    x = np.arange(3)
    bars = axes[1].bar(x, means, yerr=stds, capsize=5,
                       color=['#4C72B0', '#DD8452', '#55A868'], alpha=0.8)
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(labels)
    axes[1].set_ylabel('Mean routing weight')
    axes[1].set_title('ADAR mean routing weights ± std')
    axes[1].set_ylim(0, 1)
    axes[1].axhline(1/3, color='gray', linestyle='--', linewidth=1, label='Uniform (1/3)')
    axes[1].legend(fontsize=8)
    for bar, mean in zip(bars, means):
        axes[1].text(bar.get_x() + bar.get_width()/2, mean + 0.02,
                     f'{mean:.3f}', ha='center', va='bottom', fontsize=9)

    plt.tight_layout()
    _save_or_show(fig, save_path, show)


# ──────────────────────────────────────────────────────────────────────
# 3. CSR scale weights across decoder stages
# ──────────────────────────────────────────────────────────────────────

def plot_csr_weights(
    csr_weights: dict,        # {'csr1': (N,2), 'csr2': (N,2), 'csr3': (N,2), 'csr4': (N,2)}
    save_path: Optional[str] = None,
    show: bool = False,
):
    """Stacked bar chart showing mean [w_prev, w_cur] per decoder stage.

    Visualizes that shallow decoder stages rely more on current skip features
    (boundary info) while deep stages rely more on previous semantic features
    - the key mechanistic insight of CSR.
    """
    stage_labels  = ['Stage 1\n(deep)', 'Stage 2', 'Stage 3', 'Stage 4\n(shallow)']
    source_labels = ['Previous scale\n(semantic)', 'Current stage\n(local)']
    colors = ['#4C72B0', '#DD8452']

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    means_prev, means_cur = [], []
    stds_prev,  stds_cur  = [], []
    for key in ['csr1', 'csr2', 'csr3', 'csr4']:
        arr = np.stack(csr_weights[key])       # (N, 2)
        means_prev.append(arr[:, 0].mean())
        means_cur.append(arr[:, 1].mean())
        stds_prev.append(arr[:, 0].std())
        stds_cur.append(arr[:, 1].std())

    x = np.arange(4)
    w = 0.35

    # Grouped bar chart
    b1 = axes[0].bar(x - w/2, means_prev, w, yerr=stds_prev, capsize=4,
                     label=source_labels[0], color=colors[0], alpha=0.8)
    b2 = axes[0].bar(x + w/2, means_cur,  w, yerr=stds_cur,  capsize=4,
                     label=source_labels[1], color=colors[1], alpha=0.8)
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(stage_labels)
    axes[0].set_ylabel('Mean routing weight')
    axes[0].set_title('CSR routing weights per decoder stage')
    axes[0].set_ylim(0, 1)
    axes[0].axhline(0.5, color='gray', linestyle='--', linewidth=1, label='Uniform (0.5)')
    axes[0].legend(fontsize=9)

    # Stacked 100% bar to show relative preference
    axes[1].bar(x, means_prev, color=colors[0], alpha=0.8, label=source_labels[0])
    bottom = np.array(means_prev)
    axes[1].bar(x, means_cur,  bottom=bottom, color=colors[1], alpha=0.8, label=source_labels[1])
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(stage_labels)
    axes[1].set_ylabel('Weight proportion')
    axes[1].set_title('CSR routing - relative scale preference')
    axes[1].legend(fontsize=9)

    plt.tight_layout()
    _save_or_show(fig, save_path, show)


# ──────────────────────────────────────────────────────────────────────
# 4. Bottleneck feature activation heatmap
# ──────────────────────────────────────────────────────────────────────

def plot_feature_maps(
    model,
    image: torch.Tensor,       # (1,3,H,W) input tensor
    device: str = 'cpu',
    layer_name: str = 'bottleneck',
    save_path: Optional[str] = None,
    show: bool = False,
):
    """Plot channel-mean activation heatmap at the bottleneck, overlaid on image.

    Useful for qualitatively showing that ADAR produces more discriminative
    bottleneck representations compared to fixed SAA.
    """
    activations = {}

    def hook_fn(module, inp, out):
        # out may be (tensor, weights) tuple from ADAR
        feat = out[0] if isinstance(out, tuple) else out
        activations['feat'] = feat.detach().cpu()

    # Register hook on bottleneck
    target = getattr(model, layer_name, None) or getattr(model, 'saa', None)
    if target is None:
        raise AttributeError(f"Cannot find layer '{layer_name}' or 'saa' on model")
    handle = target.register_forward_hook(hook_fn)

    model.eval()
    with torch.no_grad():
        model.to(device)
        out = model(image.to(device))

    handle.remove()

    feat = activations['feat'][0]              # (C, H, W)
    heatmap = feat.mean(dim=0).numpy()         # (H, W) channel-mean activation
    heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min() + 1e-6)

    img_np = _to_numpy_img(image[0])

    fig, axes = plt.subplots(1, 3, figsize=(10, 3.5))
    axes[0].imshow(img_np)
    axes[0].set_title('Input image')
    axes[0].axis('off')

    im = axes[1].imshow(heatmap, cmap='jet')
    axes[1].set_title(f'Bottleneck activation\n({layer_name})')
    axes[1].axis('off')
    plt.colorbar(im, ax=axes[1], fraction=0.046, pad=0.04)

    # Overlay: resize heatmap to image size
    heatmap_resized = F.interpolate(
        torch.tensor(heatmap).unsqueeze(0).unsqueeze(0),
        size=img_np.shape[:2], mode='bilinear', align_corners=True
    )[0, 0].numpy()
    overlay = 0.5 * img_np / 255.0 + 0.5 * plt.cm.jet(heatmap_resized)[:, :, :3]
    overlay = np.clip(overlay, 0, 1)
    axes[2].imshow(overlay)
    axes[2].set_title('Activation overlay')
    axes[2].axis('off')

    plt.tight_layout()
    _save_or_show(fig, save_path, show)


# ──────────────────────────────────────────────────────────────────────
# 5. Training curves
# ──────────────────────────────────────────────────────────────────────

def plot_training_curves(
    history:    dict,         # {'train_loss': [...], 'val_loss': [...], 'val_dice': [...]}
    save_path:  Optional[str] = None,
    show:       bool = False,
):
    """Plot loss and Dice curves over epochs."""
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    epochs = range(1, len(history['train_loss']) + 1)

    axes[0].plot(epochs, history['train_loss'], label='Train loss', color='#4C72B0')
    if 'val_loss' in history:
        axes[0].plot(epochs, history['val_loss'],  label='Val loss',   color='#DD8452')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].set_title('Training & Validation Loss')
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    if 'val_dice' in history:
        axes[1].plot(epochs, history['val_dice'], label='Val Dice', color='#55A868')
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('Dice coefficient')
        axes[1].set_title('Validation Dice')
        axes[1].legend()
        axes[1].grid(alpha=0.3)

    plt.tight_layout()
    _save_or_show(fig, save_path, show)


# ──────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────

def _to_numpy_img(img) -> np.ndarray:
    """Convert (3,H,W) tensor or (H,W,3) ndarray to uint8 (H,W,3)."""
    if isinstance(img, torch.Tensor):
        img = img.detach().cpu()
        if img.ndim == 3 and img.shape[0] in (1, 3):
            img = img.permute(1, 2, 0)
        img = img.numpy()
    img = np.clip(img, 0, None)
    if img.max() <= 1.0:
        img = (img * 255).astype(np.uint8)
    return img.astype(np.uint8)


def _save_or_show(fig, save_path, show):
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
    if show:
        plt.show()
    plt.close(fig)
