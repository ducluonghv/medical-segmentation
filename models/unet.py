"""
Standard U-Net (Ronneberger et al., MICCAI 2015).

Architecture:
    Encoder : 64, 128, 256, 512, 512  (double-conv + maxpool)
    Decoder : symmetric up-path with skip connections (cat)
    Output  : 1×1 conv => n_classes

No attention, no positional encoding, no residual cross-scale connections.
Used as the simplest baseline in the ablation comparison table.
"""
import torch
import torch.nn as nn

from .backbone import DoubleConv, Down, Up, OutConv


class UNet(nn.Module):
    """Standard U-Net baseline.

    Args:
        n_channels : input image channels
        n_classes  : number of output classes
        bilinear   : use bilinear upsampling (True) or transposed convolution (False)
    """

    def __init__(self, n_channels: int = 3, n_classes: int = 1, bilinear: bool = True):
        super().__init__()
        self.n_channels = n_channels
        self.n_classes  = n_classes
        factor = 2 if bilinear else 1

        self.inc   = DoubleConv(n_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        self.down4 = Down(512, 1024 // factor)

        self.up1  = Up(1024, 512 // factor, bilinear)
        self.up2  = Up(512,  256 // factor, bilinear)
        self.up3  = Up(256,  128 // factor, bilinear)
        self.up4  = Up(128,  64,            bilinear)
        self.outc = OutConv(64, n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)

        x = self.up1(x5, x4)
        x = self.up2(x,  x3)
        x = self.up3(x,  x2)
        x = self.up4(x,  x1)
        return self.outc(x)
