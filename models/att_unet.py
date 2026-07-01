"""
Attention U-Net (Oktay et al., MIDL 2018).
https://arxiv.org/abs/1804.03999

Adds soft attention gates to each skip connection in the standard U-Net decoder.
The gate suppresses irrelevant background activations in the skip features,
focusing the decoder on salient gland/lesion regions.

Gate mechanism (Fig. 1 in paper):
    g   : gating signal from the coarser decoder level (1×1 conv)
    x   : skip connection features (1×1 conv)
    ψ   : Sigmoid(ReLU(g + x)) => 1-channel soft mask
    out : ψ * x  (element-wise selection)

This is identical to Table V's "Attention U-Net [10]" row.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from .backbone import DoubleConv, Down, OutConv


class AttentionGate(nn.Module):
    """Soft attention gate for one decoder skip connection.

    Args:
        g_channels : channels of the gating signal (from coarser decoder level)
        x_channels : channels of the skip connection feature
        inter_channels : bottleneck channels inside the gate (default: x_channels // 2)
    """

    def __init__(self, g_channels: int, x_channels: int, inter_channels: int = None):
        super().__init__()
        if inter_channels is None:
            inter_channels = max(x_channels // 2, 1)

        self.Wg = nn.Sequential(
            nn.Conv2d(g_channels, inter_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(inter_channels),
        )
        self.Wx = nn.Sequential(
            nn.Conv2d(x_channels, inter_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(inter_channels),
        )
        self.psi = nn.Sequential(
            nn.Conv2d(inter_channels, 1, kernel_size=1, bias=False),
            nn.BatchNorm2d(1),
            nn.Sigmoid(),
        )

    def forward(self, g: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        """
        g : gating signal (B, g_ch, H', W')  — coarser spatial resolution
        x : skip connection (B, x_ch, H, W)  — finer spatial resolution
        """
        # Upsample g to match x spatially
        g_up = F.interpolate(self.Wg(g), size=x.shape[2:], mode='bilinear', align_corners=True)
        att  = F.relu(g_up + self.Wx(x), inplace=True)
        att  = self.psi(att)                          # (B, 1, H, W) in [0,1]
        return att * x                                 # attended skip features


class AttentionUpBlock(nn.Module):
    """Decoder up-block with attention gate on the skip connection."""

    def __init__(self, in_channels: int, skip_channels: int, out_channels: int,
                 bilinear: bool = True):
        super().__init__()
        self.up  = (nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
                    if bilinear else
                    nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2))
        g_ch = in_channels if bilinear else in_channels // 2
        self.attn = AttentionGate(g_channels=g_ch, x_channels=skip_channels)
        self.conv = DoubleConv(g_ch + skip_channels, out_channels)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x    = self.up(x)
        skip = self.attn(x, skip)                     # gate the skip connection
        if x.shape[2:] != skip.shape[2:]:
            diff_h = skip.shape[2] - x.shape[2]
            diff_w = skip.shape[3] - x.shape[3]
            x = F.pad(x, [diff_w // 2, diff_w - diff_w // 2,
                           diff_h // 2, diff_h - diff_h // 2])
        return self.conv(torch.cat([skip, x], dim=1))


class AttUNet(nn.Module):
    """Attention U-Net (Oktay et al., 2018).

    Identical to standard U-Net but with soft attention gates gating each
    skip connection before concatenation in the decoder.

    Args:
        n_channels : input image channels
        n_classes  : number of output classes
        bilinear   : use bilinear upsampling (True) or transposed conv (False)
    """

    def __init__(self, n_channels: int = 3, n_classes: int = 1, bilinear: bool = True):
        super().__init__()
        self.n_channels = n_channels
        self.n_classes  = n_classes
        factor = 2 if bilinear else 1

        # Encoder (same as U-Net)
        self.inc   = DoubleConv(n_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        self.down4 = Down(512, 1024 // factor)

        # Decoder with attention gates
        self.up1  = AttentionUpBlock(1024 // factor, 512, 512 // factor, bilinear)
        self.up2  = AttentionUpBlock(512 // factor,  256, 256 // factor, bilinear)
        self.up3  = AttentionUpBlock(256 // factor,  128, 128 // factor, bilinear)
        self.up4  = AttentionUpBlock(128 // factor,   64,  64,           bilinear)
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
