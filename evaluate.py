"""
Evaluation script - computes all metrics on the test split and generates
qualitative visualizations (segmentation grid, routing weight plots).

Usage:
  python evaluate.py --dataset isic --data_root /path/to/ISIC \\
                     --model proposed --checkpoint checkpoints/isic/proposed/best_model.pth

Outputs (under results/<dataset>/<model>/):
  metrics.csv              - per-sample metrics table
  summary.txt              - mean ± std for each metric
  segmentation_grid.png    - qualitative comparison (first N test samples)
  adar_weights.png         - ADAR routing weight distribution
  csr_weights.png          - CSR scale weights per decoder stage
  feature_maps/            - bottleneck activation overlays
"""
import argparse
import csv
import os
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from datasets        import DATASET_REGISTRY, DATASET_DEFAULTS
from models          import TransAttUnet, ProposedModel
from utils.metrics   import MetricAccumulator, compute_all
from utils.visualize import (
    plot_segmentation,
    plot_adar_weights,
    plot_csr_weights,
    plot_feature_maps,
)
from train import build_model, set_seed


# ──────────────────────────────────────────────────────────────────────
# Routing info collection
# ──────────────────────────────────────────────────────────────────────

def _extract_routing(output):
    """Return (logits, routing_info_or_None)."""
    if isinstance(output, tuple):
        return output[0], output[1]
    return output, None


def _accumulate_routing(routing_acc, routing_info):
    if routing_info is None:
        return
    for key, val in routing_info.items():
        if val is not None:
            routing_acc.setdefault(key, []).append(val.detach().cpu().numpy())


# ──────────────────────────────────────────────────────────────────────
# Main evaluation
# ──────────────────────────────────────────────────────────────────────

@torch.no_grad()
def evaluate(args):
    set_seed(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    defaults   = DATASET_DEFAULTS[args.dataset]
    DatasetCls = DATASET_REGISTRY[args.dataset]
    test_ds    = DatasetCls(args.data_root, split='test',
                            img_size=defaults['img_size'], augment=False)
    test_loader = DataLoader(test_ds, batch_size=1, shuffle=False, num_workers=2)

    model = build_model(args.model, n_channels=defaults['n_channels'], n_classes=1)
    ckpt  = torch.load(args.checkpoint, map_location='cpu')
    state = ckpt.get('model_state', ckpt)     # handle both raw state_dict and wrapped ckpt
    model.load_state_dict(state)
    model.to(device).eval()

    out_dir = Path(args.output_dir) / args.dataset / args.model
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── Per-sample evaluation ─────────────────────────────────────────
    accumulator  = MetricAccumulator()
    routing_acc  = {}        # key => list of (B,K) arrays
    per_sample   = []        # for CSV export
    viz_images, viz_gts, viz_preds = [], [], []

    for i, (imgs, masks, paths) in enumerate(test_loader):
        imgs  = imgs.to(device)
        output = model(imgs)
        logits, routing_info = _extract_routing(output)
        probs = torch.sigmoid(logits)

        for b in range(imgs.size(0)):
            m = compute_all(probs[b].cpu(), masks[b].cpu())
            accumulator.update(probs[b].cpu(), masks[b].cpu())
            per_sample.append({
                'sample': str(paths[b]),
                **{k: f"{v:.4f}" for k, v in m.items()},
            })

        _accumulate_routing(routing_acc, routing_info)

        # Collect first N samples for visualization
        if i < args.n_viz:
            viz_images.append(imgs[0].cpu())
            viz_gts.append(masks[0].cpu())
            viz_preds.append(probs[0].cpu())

    # ── Print and save summary ─────────────────────────────────────────
    summary = accumulator.print_summary(title=f"{args.model.upper()} on {args.dataset.upper()}")
    with open(out_dir / 'summary.txt', 'w') as f:
        f.write(f"Model   : {args.model}\n")
        f.write(f"Dataset : {args.dataset}\n")
        f.write(f"Samples : {len(per_sample)}\n\n")
        f.write(f"{'Metric':>8}  {'Mean (%)':>10}  {'Std (%)':>10}\n")
        f.write('-' * 34 + '\n')
        for k, v in summary.items():
            f.write(f"{k.upper():>8}  {v['mean']*100:10.2f}  {v['std']*100:10.2f}\n")

    # ── Per-sample CSV ─────────────────────────────────────────────────
    csv_path = out_dir / 'metrics.csv'
    with open(csv_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=per_sample[0].keys())
        writer.writeheader()
        writer.writerows(per_sample)
    print(f"Per-sample metrics saved to {csv_path}")

    # ── Visualizations ─────────────────────────────────────────────────
    if viz_images:
        plot_segmentation(
            images=viz_images, gt_masks=viz_gts, pred_masks=viz_preds,
            model_names=[args.model],
            save_path=str(out_dir / 'segmentation_grid.png'),
        )
        print(f"Segmentation grid saved.")

    # ADAR weight distribution
    if 'adar' in routing_acc and routing_acc['adar']:
        all_w = np.concatenate(routing_acc['adar'], axis=0)   # (N, 3)
        plot_adar_weights(
            list(all_w),
            save_path=str(out_dir / 'adar_weights.png'),
        )
        print(f"ADAR weight plot saved.")

    # CSR weight per stage
    csr_keys = [k for k in routing_acc if k.startswith('csr')]
    if len(csr_keys) == 4:
        csr_weights = {
            k: list(np.concatenate(routing_acc[k], axis=0))
            for k in ['csr1', 'csr2', 'csr3', 'csr4']
        }
        plot_csr_weights(csr_weights, save_path=str(out_dir / 'csr_weights.png'))
        print(f"CSR weight plot saved.")

    # Feature activation maps (first sample)
    if viz_images:
        fm_dir = out_dir / 'feature_maps'
        fm_dir.mkdir(exist_ok=True)
        try:
            layer = 'bottleneck' if args.model != 'baseline' else 'saa'
            plot_feature_maps(
                model, viz_images[0].unsqueeze(0), device=str(device),
                layer_name=layer,
                save_path=str(fm_dir / 'sample_0.png'),
            )
            print(f"Feature map saved.")
        except Exception as e:
            print(f"Feature map skipped: {e}")

    print(f"\nAll outputs written to: {out_dir}")
    return summary


# ──────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description='Evaluate segmentation model')
    p.add_argument('--dataset',    type=str, required=True,
                   choices=['isic', 'glas', 'covid', 'lung', 'dsb2018'])
    p.add_argument('--data_root',  type=str, required=True)
    p.add_argument('--model',      type=str, default='proposed',
                   choices=['baseline', 'proposed', 'adar_only', 'csr_only'])
    p.add_argument('--checkpoint', type=str, required=True)
    p.add_argument('--output_dir', type=str, default='results')
    p.add_argument('--n_viz',      type=int, default=4, help='Samples for segmentation grid')
    p.add_argument('--seed',       type=int, default=42)
    return p.parse_args()


if __name__ == '__main__':
    evaluate(parse_args())
