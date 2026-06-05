"""
TransAttUnet baseline - clean rewrite of the original TransAttUnet/model/TransAttUnet.py.

This is the TransAttUnet_R variant (residual multi-scale skip connections),
which is the best-performing variant reported in the paper.

Channel flow (bilinear=True):
  Encoder : x1=64, x2=128, x3=256, x4=512, x5=512
  Bottleneck (SAA): x5 => 512
  Decoder :
    up1(x5,x4)   => x6=256;  cat(x5_scale=512, x6=256) => x6_cat=768
    up2(x6_cat,x3) => x7=128; cat(x6_scale=256, x7=128) => x7_cat=384
    up3(x7_cat,x2) => x8=64;  cat(x7_scale=128, x8=64)  => x8_cat=192
    up4(x8_cat,x1) => x9=64;  cat(x8_scale=64, x9=64)   => x9_final=128
  Output: OutConv(128 => n_classes)
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from .backbone import DoubleConv, Down, Up, OutConv
from .adar import FixedSAA


class TransAttUnet(nn.Module):
    """TransAttUnet_R: residual multi-scale skip connections + fixed SAA."""

    def __init__(self, n_channels: int = 3, n_classes: int = 1, bilinear: bool = True):
        super().__init__()
        self.n_channels = n_channels
        self.n_classes  = n_classes
        self.bilinear   = bilinear
        factor = 2 if bilinear else 1

        # Encoder
        self.inc   = DoubleConv(n_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        self.down4 = Down(512, 1024 // factor)   # => 512

        # Bottleneck: fixed SAA (TSA + GSA, sum fusion)
        self.saa = FixedSAA(512)

        # Decoder  (in_channels accounts for residual cat input)
        self.up1 = Up(1024, 512 // factor, bilinear)   # 512+512=1024 => 256
        self.up2 = Up(1024, 256 // factor, bilinear)   # 768+256=1024 => 128
        self.up3 = Up(512,  128 // factor, bilinear)   # 384+128=512  => 64
        self.up4 = Up(256,  64,            bilinear)   # 192+64=256   => 64
        self.outc = OutConv(128, n_classes)             # 64+64=128

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Encoder
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)

        # Fixed SAA bottleneck
        x5 = self.saa(x5)

        # Decoder with residual multi-scale skip connections
        x6 = self.up1(x5, x4)
        x5_scale = F.interpolate(x5, size=x6.shape[2:], mode='bilinear', align_corners=True)
        x6_cat = torch.cat([x5_scale, x6], dim=1)          # 512+256=768

        x7 = self.up2(x6_cat, x3)
        x6_scale = F.interpolate(x6, size=x7.shape[2:], mode='bilinear', align_corners=True)
        x7_cat = torch.cat([x6_scale, x7], dim=1)          # 256+128=384

        x8 = self.up3(x7_cat, x2)
        x7_scale = F.interpolate(x7, size=x8.shape[2:], mode='bilinear', align_corners=True)
        x8_cat = torch.cat([x7_scale, x8], dim=1)          # 128+64=192

        x9 = self.up4(x8_cat, x1)
        x8_scale = F.interpolate(x8, size=x9.shape[2:], mode='bilinear', align_corners=True)
        x9_final = torch.cat([x8_scale, x9], dim=1)        # 64+64=128

        return self.outc(x9_final)
