"""
Cross-scale Semantic Routing (CSR)

Motivation (from thesis Outline.pdf):
  TransAttUnet uses fixed concatenation at each decoder stage:
      x_cat = cat(x_prev_scale, x_current)
  giving equal weight to all scale sources regardless of content.

  CSR replaces this with a learned gating mechanism that predicts
  per-sample importance weights for each scale source before concatenation:
      [w_prev, w_cur] = Softmax(Gate(cat(x_prev_scale, x_current)))
      x_cat = cat(w_prev * x_prev_scale, w_cur * x_current)

  The output shape is identical to the original concatenation, so all
  downstream Up modules require no modification.

Design:
  - Gate: GlobalAvgPool => Linear(total_ch, total_ch//8) => ReLU =>
          Linear(total_ch//8, 2) => Softmax
  - Weights are returned for visualization of per-stage routing behavior.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class CSR(nn.Module):
    """Cross-scale Semantic Routing for one decoder fusion point.

    Args:
        prev_channels: channels of the upsampled previous-stage features
        cur_channels:  channels of the current decoder-stage features

    Returns:
        out     : (B, prev_ch + cur_ch, H, W)  - weighted concatenation
        weights : (B, 2)                        - [w_prev, w_cur] for viz
    """

    def __init__(self, prev_channels: int, cur_channels: int):
        super().__init__()
        total = prev_channels + cur_channels
        reduced = max(total // 8, 8)

        self.gate = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(total, reduced),
            nn.ReLU(inplace=True),
            nn.Linear(reduced, 2),
            nn.Softmax(dim=-1),
        )

    def forward(self, x_prev: torch.Tensor, x_cur: torch.Tensor) -> tuple:
        # Align spatial resolution if they differ (guard for edge cases)
        if x_prev.shape[2:] != x_cur.shape[2:]:
            x_prev = F.interpolate(x_prev, size=x_cur.shape[2:], mode='bilinear', align_corners=True)

        x_cat = torch.cat([x_prev, x_cur], dim=1)    # (B, prev+cur, H, W)
        weights = self.gate(x_cat)                    # (B, 2)

        w_prev = weights[:, 0].view(-1, 1, 1, 1)
        w_cur  = weights[:, 1].view(-1, 1, 1, 1)

        out = torch.cat([w_prev * x_prev, w_cur * x_cur], dim=1)
        return out, weights
