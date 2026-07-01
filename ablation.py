"""
Ablation study runner - executes all 4 experiment configurations sequentially
and prints a consolidated comparison table at the end.

Ablation design (Section IV of thesis):
  Exp 1: baseline     - TransAttUnet_R (fixed SAA, fixed cat)
  Exp 2: adar_only    - fixed cat, ADAR bottleneck
  Exp 3: csr_only     - fixed SAA, CSR decoder
  Exp 4: proposed     - ADAR + CSR (full proposed model)

For each configuration the script:
  1. Trains the model (calls train.train())
  2. Evaluates on the test split (calls evaluate.evaluate())
  3. Collects mean Dice, IoU, ACC, REC, PRE, HD95

Usage:
  python ablation.py --dataset glas --data_root /path/to/GlaS
  python ablation.py --dataset isic --data_root /path/to/ISIC --epochs 100

Optional --skip_train to run evaluation only (requires pre-trained checkpoints).
"""
import argparse
import os
from pathlib import Path

import numpy as np

from train    import train,    parse_args as train_parse
from evaluate import evaluate, parse_args as eval_parse


ABLATION_CONFIGS = [
    {'name': 'baseline',  'use_adar': False, 'use_csr': False,
     'label': 'TransAttUnet_R (baseline)'},
    {'name': 'adar_only', 'use_adar': True,  'use_csr': False,
     'label': '+ ADAR (Ours)'},
    {'name': 'csr_only',  'use_adar': False, 'use_csr': True,
     'label': '+ CSR (Ours)'},
    {'name': 'proposed',  'use_adar': True,  'use_csr': True,
     'label': '+ ADAR + CSR (Full Proposed)'},
]

METRIC_KEYS = ['dice', 'iou', 'acc', 'rec', 'pre', 'hd95']


# ──────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────

def run_ablation(args):
    results = {}

    for cfg in ABLATION_CONFIGS:
        model_name = cfg['name']
        print(f"\n{'='*60}")
        print(f"  Running: {cfg['label']}")
        print(f"{'='*60}")

        ckpt_dir  = Path(args.ckpt_dir) / args.dataset / model_name
        ckpt_path = ckpt_dir / 'best_model.pth'

        # ── Train ─────────────────────────────────────────────────────
        if not args.skip_train:
            train_args = argparse.Namespace(
                dataset     = args.dataset,
                data_root   = args.data_root,
                model       = model_name,
                epochs      = args.epochs,
                batch_size  = args.batch_size,
                lr          = args.lr,
                num_workers = args.num_workers,
                save_dir    = args.ckpt_dir,
                seed        = args.seed,
            )
            train(train_args)
        else:
            if not ckpt_path.exists():
                print(f"  ⚠ Checkpoint not found: {ckpt_path}  - skipping.")
                continue

        # ── Evaluate ──────────────────────────────────────────────────
        eval_args = argparse.Namespace(
            dataset    = args.dataset,
            data_root  = args.data_root,
            model      = model_name,
            checkpoint = str(ckpt_path),
            output_dir = args.result_dir,
            n_viz      = 4,
            seed       = args.seed,
        )
        summary = evaluate(eval_args)
        results[model_name] = {'label': cfg['label'], 'summary': summary}

    # ── Print consolidated table ───────────────────────────────────────
    _print_ablation_table(results, args.dataset)


def _print_ablation_table(results: dict, dataset: str):
    if not results:
        print("No results to display.")
        return

    header_metrics = ['DICE', 'IoU', 'ACC', 'REC', 'PRE', 'HD95']
    col_w = 10
    name_w = 32

    sep = '─' * (name_w + col_w * len(header_metrics) + 2)
    print(f"\n{'='*len(sep)}")
    print(f"  ABLATION STUDY - {dataset.upper()}")
    print(f"{'='*len(sep)}")
    print(f"{'Model':<{name_w}}" + "".join(f"{h:>{col_w}}" for h in header_metrics))
    print(sep)

    best = {k: -np.inf if k != 'hd95' else np.inf for k in METRIC_KEYS}
    for cfg in ABLATION_CONFIGS:
        name = cfg['name']
        if name not in results:
            continue
        s = results[name]['summary']
        for k in METRIC_KEYS:
            v = s[k]['mean']
            if k == 'hd95':
                best[k] = min(best[k], v)
            else:
                best[k] = max(best[k], v)

    for cfg in ABLATION_CONFIGS:
        name = cfg['name']
        if name not in results:
            print(f"{'  ' + cfg['label']:<{name_w}}" + f"{'(skipped)':>{col_w}}")
            continue
        s     = results[name]['summary']
        label = '  ' + cfg['label']
        row   = f"{label:<{name_w}}"
        for k in METRIC_KEYS:
            v    = s[k]['mean'] * (1 if k == 'hd95' else 100)
            mark = '*' if abs(v - best[k] * (1 if k == 'hd95' else 100)) < 0.005 else ' '
            row += f"{v:>{col_w-1}.2f}{mark}"
        print(row)

    print(sep)
    print("  * = best result in column")
    print()

    # Save table to text file
    result_dir = Path('results') / dataset / 'ablation'
    result_dir.mkdir(parents=True, exist_ok=True)
    with open(result_dir / 'ablation_table.txt', 'w') as f:
        f.write(f"ABLATION STUDY - {dataset.upper()}\n")
        f.write(f"{'Model':<{name_w}}" + "".join(f"{h:>{col_w}}" for h in header_metrics) + '\n')
        f.write(sep + '\n')
        for cfg in ABLATION_CONFIGS:
            name = cfg['name']
            if name not in results:
                continue
            s     = results[name]['summary']
            label = '  ' + cfg['label']
            row   = f"{label:<{name_w}}"
            for k in METRIC_KEYS:
                v = s[k]['mean'] * (1 if k == 'hd95' else 100)
                row += f"{v:>{col_w}.2f}"
            f.write(row + '\n')
    print(f"Ablation table saved to {result_dir / 'ablation_table.txt'}")


# ──────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description='Ablation study runner')
    p.add_argument('--dataset',     type=str, required=True,
                   choices=['isic', 'glas', 'covid', 'lung', 'dsb2018'])
    p.add_argument('--data_root',   type=str, required=True)
    p.add_argument('--epochs',      type=int, default=100)
    p.add_argument('--batch_size',  type=int, default=4)
    p.add_argument('--lr',          type=float, default=0.01)
    p.add_argument('--num_workers', type=int, default=4)
    p.add_argument('--ckpt_dir',    type=str, default='checkpoints')
    p.add_argument('--result_dir',  type=str, default='results')
    p.add_argument('--seed',        type=int, default=42)
    p.add_argument('--skip_train',  action='store_true',
                   help='Skip training and run evaluation only (checkpoints must exist)')
    return p.parse_args()


if __name__ == '__main__':
    run_ablation(parse_args())
