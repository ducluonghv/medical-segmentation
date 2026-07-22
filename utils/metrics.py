"""
Segmentation evaluation metrics.

All functions accept numpy boolean/binary arrays OR torch tensors.
Inputs are binarised at threshold=0.5 if they contain probabilities.

Metrics (consistent with TransAttUnet paper, Eq. 10):
    Dice  = 2*TP / (2*TP + FP + FN)
    IoU   = TP / (TP + FP + FN)
    ACC   = (TP + TN) / (TP + TN + FP + FN)
    REC   = TP / (TP + FN)          Recall / Sensitivity
    PRE   = TP / (TP + FP)          Precision

Additional (not in paper, added for thesis):
    HD95  = 95th-percentile bidirectional Hausdorff distance (pixel units)
            Measures boundary quality - key evidence for CSR improvement.
"""
import numpy as np
import torch
from scipy.ndimage import binary_erosion, distance_transform_edt


EPS = 1e-6   # smoothing to avoid division by zero


# ──────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────

def _to_numpy_binary(x, threshold: float = 0.5) -> np.ndarray:
    """Convert tensor or ndarray to a 2-D boolean numpy array."""
    if isinstance(x, torch.Tensor):
        x = x.detach().cpu().numpy()
    x = np.squeeze(x)          # remove batch / channel dims of size 1
    return x > threshold


def _confusion(pred: np.ndarray, gt: np.ndarray):
    tp = np.logical_and(pred,  gt).sum()
    tn = np.logical_and(~pred, ~gt).sum()
    fp = np.logical_and(pred,  ~gt).sum()
    fn = np.logical_and(~pred, gt).sum()
    return int(tp), int(tn), int(fp), int(fn)


# ──────────────────────────────────────────────────────────────────────
# Individual metrics
# ──────────────────────────────────────────────────────────────────────

def dice_score(pred, gt, threshold: float = 0.5) -> float:
    p = _to_numpy_binary(pred, threshold)
    g = _to_numpy_binary(gt,   threshold)
    tp, _, fp, fn = _confusion(p, g)
    return (2 * tp) / (2 * tp + fp + fn + EPS)


def iou_score(pred, gt, threshold: float = 0.5) -> float:
    p = _to_numpy_binary(pred, threshold)
    g = _to_numpy_binary(gt,   threshold)
    tp, _, fp, fn = _confusion(p, g)
    return tp / (tp + fp + fn + EPS)


def accuracy(pred, gt, threshold: float = 0.5) -> float:
    p = _to_numpy_binary(pred, threshold)
    g = _to_numpy_binary(gt,   threshold)
    tp, tn, fp, fn = _confusion(p, g)
    return (tp + tn) / (tp + tn + fp + fn + EPS)


def recall(pred, gt, threshold: float = 0.5) -> float:
    p = _to_numpy_binary(pred, threshold)
    g = _to_numpy_binary(gt,   threshold)
    tp, _, _, fn = _confusion(p, g)
    return tp / (tp + fn + EPS)


def precision(pred, gt, threshold: float = 0.5) -> float:
    p = _to_numpy_binary(pred, threshold)
    g = _to_numpy_binary(gt,   threshold)
    tp, _, fp, _ = _confusion(p, g)
    return tp / (tp + fp + EPS)


def hd95(pred, gt, threshold: float = 0.5) -> float:
    """95th-percentile bidirectional Hausdorff distance (pixel units).

    Returns 0.0 when either surface is empty (edge case for nearly empty masks).
    """
    p = _to_numpy_binary(pred, threshold)
    g = _to_numpy_binary(gt,   threshold)

    # Extract surface pixels via erosion
    p_surface = p ^ binary_erosion(p)
    g_surface = g ^ binary_erosion(g)

    if not p_surface.any() and not g_surface.any():
        return 0.0   # both empty (no lesion, nothing predicted) → perfect
    if not p_surface.any() or not g_surface.any():
        # one side empty → maximal distance (image diagonal)
        return float(np.sqrt(p.shape[0] ** 2 + p.shape[1] ** 2))

    # distance_transform_edt(~surface) = distance to nearest surface pixel
    dist_p = distance_transform_edt(~p_surface)   # dist from any pixel to pred surface
    dist_g = distance_transform_edt(~g_surface)   # dist from any pixel to gt surface

    d_pred_to_gt = dist_g[p_surface]              # for each pred surface px => nearest gt surface
    d_gt_to_pred = dist_p[g_surface]              # for each gt surface px   => nearest pred surface

    all_d = np.concatenate([d_pred_to_gt, d_gt_to_pred])
    return float(np.percentile(all_d, 95))


# ──────────────────────────────────────────────────────────────────────
# Batch evaluation
# ──────────────────────────────────────────────────────────────────────

def compute_all(pred, gt, threshold: float = 0.5) -> dict:
    """Compute all metrics for a single prediction/GT pair.

    Args:
        pred : (1,1,H,W) tensor or (H,W) ndarray - raw logits or probabilities
        gt   : (1,1,H,W) tensor or (H,W) ndarray - ground truth binary mask

    Returns:
        dict with keys: dice, iou, acc, rec, pre, hd95
    """
    # Apply sigmoid if logits (values outside [0,1])
    if isinstance(pred, torch.Tensor):
        if pred.min() < 0 or pred.max() > 1:
            pred = torch.sigmoid(pred)

    return {
        'dice': dice_score(pred, gt, threshold),
        'iou':  iou_score(pred,  gt, threshold),
        'acc':  accuracy(pred,   gt, threshold),
        'rec':  recall(pred,     gt, threshold),
        'pre':  precision(pred,  gt, threshold),
        'hd95': hd95(pred,       gt, threshold),
    }


class MetricAccumulator:
    """Accumulate per-sample metrics and report mean ± std over a dataset split."""

    def __init__(self):
        self._records: list[dict] = []

    def update(self, pred, gt, threshold: float = 0.5):
        self._records.append(compute_all(pred, gt, threshold))

    def summary(self) -> dict:
        """Returns {metric: {'mean': float, 'std': float}} for all metrics."""
        if not self._records:
            return {}
        keys = list(self._records[0].keys())
        result = {}
        for k in keys:
            vals = np.array([r[k] for r in self._records])
            result[k] = {'mean': float(vals.mean()), 'std': float(vals.std())}
        return result

    def print_summary(self, title: str = ''):
        s = self.summary()
        if title:
            print(f'\n{"─"*50}')
            print(f'  {title}')
            print(f'{"─"*50}')
        header = f"{'Metric':>8}  {'Mean':>8}  {'Std':>8}"
        print(header)
        print('─' * len(header))
        for k, v in s.items():
            if k == 'hd95':
                print(f"{k.upper():>8}  {v['mean']:8.2f}  {v['std']:8.2f}  px")
            else:
                print(f"{k.upper():>8}  {v['mean']*100:8.2f}  {v['std']*100:8.2f}  %")
        print()
        return s

    def reset(self):
        self._records.clear()
