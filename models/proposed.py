"""
Proposed model: TransAttUnet + ADAR + CSR

Two new components replace fixed counterparts in TransAttUnet:
  - ADAR at the bottleneck: adaptive routing between TSA, GSA, and original features
  - CSR at each decoder fusion: adaptive weighting of multi-scale sources

Ablation support via boolean flags:
  use_adar=False, use_csr=False  =>  equivalent to TransAttUnet_R (baseline)
  use_adar=True,  use_csr=False  =>  +ADAR only
  use_adar=False, use_csr=True   =>  +CSR only
  use_adar=True,  use_csr=True   =>  full proposed model

Forward returns:
  logits        : (B, n_classes, H, W)
  routing_info  : dict with keys 'adar', 'csr1'..'csr4' (all None when disabled)
                  Used for visualization; ignored during normal training.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from .backbone import DoubleConv, Down, Up, OutConv
from .adar import ADAR, FixedSAA
from .csr  import CSR


class ProposedModel(nn.Module):
    """
    Args:
        n_channels : input image channels (3 for RGB, 1 for grayscale)
        n_classes  : output segmentation classes
        bilinear   : use bilinear upsampling (True) or transposed conv (False)
        use_adar   : replace fixed SAA with adaptive ADAR at the bottleneck
        use_csr    : replace fixed cat with adaptive CSR at each decoder stage
    """

    def __init__(
        self,
        n_channels: int = 3,
        n_classes:  int = 1,
        bilinear:   bool = True,
        use_adar:   bool = True,
        use_csr:    bool = True,
    ):
        super().__init__()
        self.n_channels = n_channels
        self.n_classes  = n_classes
        self.use_adar   = use_adar
        self.use_csr    = use_csr
        factor = 2 if bilinear else 1

        # ── Encoder (identical to TransAttUnet) ──────────────────────────
        self.inc   = DoubleConv(n_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        self.down4 = Down(512, 1024 // factor)              # => 512 ch

        # ── Bottleneck ────────────────────────────────────────────────────
        # ADAR returns (out, weights); FixedSAA returns out only.
        self.bottleneck = ADAR(512) if use_adar else FixedSAA(512)

        # ── Decoder (same Up modules as TransAttUnet) ─────────────────────
        # Channel sizes at each Up input = (prev_scale_ch + cur_decoder_ch) + skip_ch
        #   up1: x6_cat(768) + x4(512)? No - Up takes (x_input, x_skip) and cats them.
        #   up1(x5=512, x4=512):       512+512=1024 => 256
        #   up2(x6_cat=768, x3=256):   768+256=1024 => 128
        #   up3(x7_cat=384, x2=128):   384+128=512  => 64
        #   up4(x8_cat=192, x1=64):    192+64=256   => 64
        self.up1  = Up(1024, 512 // factor, bilinear)
        self.up2  = Up(1024, 256 // factor, bilinear)
        self.up3  = Up(512,  128 // factor, bilinear)
        self.up4  = Up(256,  64,            bilinear)
        self.outc = OutConv(128, n_classes)                  # 64+64=128 => n_classes

        # ── CSR at each multi-scale fusion point ──────────────────────────
        # prev_ch = channels of the upsampled previous feature
        # cur_ch  = channels of the current decoder output
        if use_csr:
            self.csr1 = CSR(prev_channels=512, cur_channels=256)   # x5_scale + x6
            self.csr2 = CSR(prev_channels=256, cur_channels=128)   # x6_scale + x7
            self.csr3 = CSR(prev_channels=128, cur_channels=64)    # x7_scale + x8
            self.csr4 = CSR(prev_channels=64,  cur_channels=64)    # x8_scale + x9

    # ------------------------------------------------------------------
    def _fuse(self, x_prev: torch.Tensor, x_cur: torch.Tensor, stage: int):
        """Fuse previous-scale and current features, return (tensor, weights|None)."""
        if self.use_csr:
            csr = getattr(self, f'csr{stage}')
            return csr(x_prev, x_cur)
        # Fixed concatenation - same as TransAttUnet
        x_prev_aligned = F.interpolate(x_prev, size=x_cur.shape[2:], mode='bilinear', align_corners=True)
        return torch.cat([x_prev_aligned, x_cur], dim=1), None

    # ------------------------------------------------------------------
    def forward(self, x: torch.Tensor):
        # ── Encoder ──────────────────────────────────────────────────────
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)

        # ── Bottleneck ────────────────────────────────────────────────────
        if self.use_adar:
            x5, adar_w = self.bottleneck(x5)
        else:
            x5      = self.bottleneck(x5)
            adar_w  = None

        # ── Decoder with adaptive multi-scale routing ─────────────────────
        x6 = self.up1(x5, x4)
        x5_scale = F.interpolate(x5, size=x6.shape[2:], mode='bilinear', align_corners=True)
        x6_cat, csr1_w = self._fuse(x5_scale, x6, stage=1)    # 512+256=768

        x7 = self.up2(x6_cat, x3)
        x6_scale = F.interpolate(x6, size=x7.shape[2:], mode='bilinear', align_corners=True)
        x7_cat, csr2_w = self._fuse(x6_scale, x7, stage=2)    # 256+128=384

        x8 = self.up3(x7_cat, x2)
        x7_scale = F.interpolate(x7, size=x8.shape[2:], mode='bilinear', align_corners=True)
        x8_cat, csr3_w = self._fuse(x7_scale, x8, stage=3)    # 128+64=192

        x9 = self.up4(x8_cat, x1)
        x8_scale = F.interpolate(x8, size=x9.shape[2:], mode='bilinear', align_corners=True)
        x9_final, csr4_w = self._fuse(x8_scale, x9, stage=4)  # 64+64=128

        logits = self.outc(x9_final)

        routing_info = {
            'adar': adar_w,   # (B,3) or None
            'csr1': csr1_w,   # (B,2) or None
            'csr2': csr2_w,
            'csr3': csr3_w,
            'csr4': csr4_w,
        }
        return logits, routing_info
