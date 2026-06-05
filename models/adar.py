"""
Adaptive Dual-path Attention Routing (ADAR)

Motivation (from thesis Outline.pdf):
  TransAttUnet fuses TSA + GSA with a fixed sum (F_SAA = F_tsa + F_gsa).
  ADAR replaces this with a content-adaptive gate that learns per-image
  routing weights, so the model can decide how much long-range channel
  context (TSA) vs spatial context (GSA) vs original features to use.

Architecture:
  F_ADAR = w_tsa * F_tsa + w_gsa * F_gsa + w_orig * F_en
  where [w_tsa, w_gsa, w_orig] = Softmax(Gate(F_en)), sum-to-1.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class PositionEmbeddingLearned(nn.Module):
    """Learned 2-D positional encoding (same as TransAttUnet)."""

    def __init__(self, num_pos_feats=256, max_len=32):
        super().__init__()
        self.row_embed = nn.Embedding(max_len, num_pos_feats)
        self.col_embed = nn.Embedding(max_len, num_pos_feats)
        nn.init.uniform_(self.row_embed.weight)
        nn.init.uniform_(self.col_embed.weight)

    def forward(self, x):
        h, w = x.shape[-2:]
        i = torch.arange(w, device=x.device)
        j = torch.arange(h, device=x.device)
        x_emb = self.col_embed(i)                          # (W, C/2)
        y_emb = self.row_embed(j)                          # (H, C/2)
        pos = torch.cat([
            x_emb.unsqueeze(0).repeat(h, 1, 1),            # (H, W, C/2)
            y_emb.unsqueeze(1).repeat(1, w, 1),            # (H, W, C/2)
        ], dim=-1).permute(2, 0, 1).unsqueeze(0)           # (1, C, H, W)
        return pos.repeat(x.shape[0], 1, 1, 1)            # (B, C, H, W)


class SpatialAttention(nn.Module):
    """Global Spatial Attention (GSA / PAM) - same as TransAttUnet."""

    def __init__(self, in_channels):
        super().__init__()
        self.query_conv = nn.Conv2d(in_channels, in_channels // 8, kernel_size=1)
        self.key_conv   = nn.Conv2d(in_channels, in_channels // 8, kernel_size=1)
        self.value_conv = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        B, C, H, W = x.size()
        q = self.query_conv(x).view(B, -1, H * W).permute(0, 2, 1)  # (B, HW, C')
        k = self.key_conv(x).view(B, -1, H * W)                      # (B, C', HW)
        attn = torch.softmax(torch.bmm(q, k), dim=-1)                # (B, HW, HW)
        v = self.value_conv(x).view(B, -1, H * W)                    # (B, C, HW)
        out = torch.bmm(v, attn.permute(0, 2, 1)).view(B, C, H, W)
        return self.gamma * out + x


class ChannelAttention(nn.Module):
    """Transformer Self Attention (TSA) over channel dimension - same as TransAttUnet."""

    def __init__(self, in_channels, dropout=0.1):
        super().__init__()
        self.scale = in_channels ** 0.5
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B, C, H, W = x.size()
        q = x.view(B, C, -1)                          # (B, C, HW)
        k = x.view(B, C, -1).permute(0, 2, 1)        # (B, HW, C)
        v = x.view(B, C, -1)                          # (B, C, HW)
        attn = self.dropout(torch.softmax(torch.matmul(q / self.scale, k), dim=-1))  # (B, C, C)
        out = torch.matmul(attn, v).view(B, C, H, W)
        return out


class FixedSAA(nn.Module):
    """Original fixed Self-Aware Attention from TransAttUnet.

    F_SAA = F_tsa + F_gsa   (no learnable fusion weights)
    Used as the baseline when use_adar=False in ProposedModel.
    """

    def __init__(self, in_channels):
        super().__init__()
        self.pos   = PositionEmbeddingLearned(in_channels // 2)
        self.tsa   = ChannelAttention(in_channels)
        self.gsa   = SpatialAttention(in_channels)

    def forward(self, x):
        f_gsa = self.gsa(x)
        f_tsa = self.tsa(x + self.pos(x))
        return f_tsa + f_gsa


class ADAR(nn.Module):
    """Adaptive Dual-path Attention Routing.

    Replaces the fixed SAA fusion with a content-driven gate:
        weights = Softmax(Gate(x))          # (B, 3)
        out = w_tsa*F_tsa + w_gsa*F_gsa + w_orig*x

    The gate is a lightweight channel-squeeze MLP applied after global
    average pooling, adding negligible parameters and no spatial overhead.

    Returns:
        out     : (B, C, H, W) - enriched bottleneck features
        weights : (B, 3)       - [w_tsa, w_gsa, w_orig] for visualization
    """

    def __init__(self, in_channels, dropout=0.1):
        super().__init__()
        self.pos = PositionEmbeddingLearned(in_channels // 2)
        self.tsa = ChannelAttention(in_channels, dropout)
        self.gsa = SpatialAttention(in_channels)

        # Routing gate: squeeze => bottleneck => 3-way softmax
        reduced = max(in_channels // 8, 16)
        self.gate = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(in_channels, reduced),
            nn.ReLU(inplace=True),
            nn.Linear(reduced, 3),
            nn.Softmax(dim=-1),
        )

    def forward(self, x):
        f_tsa = self.tsa(x + self.pos(x))   # channel-attention path
        f_gsa = self.gsa(x)                  # spatial-attention path

        weights = self.gate(x)               # (B, 3)
        w_tsa  = weights[:, 0].view(-1, 1, 1, 1)
        w_gsa  = weights[:, 1].view(-1, 1, 1, 1)
        w_orig = weights[:, 2].view(-1, 1, 1, 1)

        out = w_tsa * f_tsa + w_gsa * f_gsa + w_orig * x
        return out, weights
