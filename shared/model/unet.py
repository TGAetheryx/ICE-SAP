"""
U-Net with MobileNetV2 encoder for 6-channel Sentinel-2 glacier segmentation.

Architecture:
  - Input: (B, 6, H, W)  — RGB + SWIR (B11, B12) + Thermal (B10)
  - Encoder: MobileNetV2 backbone (first conv adapted to 6 channels)
  - Decoder: standard U-Net skip connections + upsampling
  - Output: (B, 1, H, W)  — binary glacier/non-glacier probability

Used in all three papers (ICE-SAP / TGlacierEdge / ASPT).

References:
  Ronneberger et al., "U-Net", MICCAI 2015.
  Howard et al., "MobileNetV2", CVPR 2018.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple


# ---------------------------------------------------------------------------
# Basic building blocks
# ---------------------------------------------------------------------------

class ConvBnRelu(nn.Module):
    """Conv2d → BatchNorm2d → ReLU6."""

    def __init__(self, in_ch: int, out_ch: int, kernel: int = 3,
                 stride: int = 1, padding: int = 1, groups: int = 1):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel, stride=stride,
                      padding=padding, groups=groups, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU6(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class InvertedResidual(nn.Module):
    """MobileNetV2 inverted residual block."""

    def __init__(self, in_ch: int, out_ch: int, stride: int = 1,
                 expand_ratio: int = 6):
        super().__init__()
        self.stride = stride
        assert stride in (1, 2)
        hidden = int(in_ch * expand_ratio)
        self.use_res = (stride == 1 and in_ch == out_ch)

        layers: List[nn.Module] = []
        if expand_ratio != 1:
            layers.append(ConvBnRelu(in_ch, hidden, kernel=1, padding=0))
        layers += [
            ConvBnRelu(hidden, hidden, stride=stride, groups=hidden),
            nn.Conv2d(hidden, out_ch, 1, bias=False),
            nn.BatchNorm2d(out_ch),
        ]
        self.conv = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.use_res:
            return x + self.conv(x)
        return self.conv(x)


# ---------------------------------------------------------------------------
# MobileNetV2 encoder (6-channel first conv)
# ---------------------------------------------------------------------------

class MobileNetV2Encoder(nn.Module):
    """
    MobileNetV2 backbone adapted to 6-channel Sentinel-2 input.

    Returns five feature maps at strides 1, 2, 4, 8, 16 for skip connections.
    """

    # (t=expand_ratio, c=out_channels, n=repeats, s=stride)
    _SETTINGS = [
        (1,  16, 1, 1),
        (6,  24, 2, 2),
        (6,  32, 3, 2),
        (6,  64, 4, 2),
        (6,  96, 3, 1),
        (6, 160, 3, 2),
        (6, 320, 1, 1),
    ]

    def __init__(self, in_channels: int = 6):
        super().__init__()
        # Stem: 6-ch → 32, stride 2
        self.stem = ConvBnRelu(in_channels, 32, stride=2)

        # Build inverted-residual stages
        self.stages = nn.ModuleList()
        in_ch = 32
        for t, c, n, s in self._SETTINGS:
            blocks: List[nn.Module] = []
            for i in range(n):
                stride = s if i == 0 else 1
                blocks.append(InvertedResidual(in_ch, c, stride, t))
                in_ch = c
            self.stages.append(nn.Sequential(*blocks))

        # Feature-map channels at each skip level
        # skip0: stem output  → 32   (stride 2)
        # skip1: stage 0      → 16   (stride 2, same spatial as stem)
        # skip2: stage 1      → 24   (stride 4)
        # skip3: stage 2      → 32   (stride 8)
        # skip4: stage 4      → 96   (stride 16)  [bottleneck]
        self.out_channels = [32, 16, 24, 32, 96]

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        """Return list of feature maps [skip0 .. skip4]."""
        s0 = self.stem(x)               # stride 2,  32 ch
        s1 = self.stages[0](s0)         # stride 2,  16 ch  (no stride inside)
        s2 = self.stages[1](s1)         # stride 4,  24 ch
        s3 = self.stages[2](s2)         # stride 8,  32 ch
        _  = self.stages[3](s3)         # stride 16, 64 ch  (internal)
        s4 = self.stages[4](_)          # stride 16, 96 ch  (bottleneck)
        return [s0, s1, s2, s3, s4]


# ---------------------------------------------------------------------------
# Decoder block
# ---------------------------------------------------------------------------

class DecoderBlock(nn.Module):
    """
    Upsample × 2 → concat skip → Conv → Conv.
    """

    def __init__(self, in_ch: int, skip_ch: int, out_ch: int):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode='bilinear',
                              align_corners=False)
        self.conv = nn.Sequential(
            ConvBnRelu(in_ch + skip_ch, out_ch),
            ConvBnRelu(out_ch, out_ch),
        )

    def forward(self, x: torch.Tensor,
                skip: torch.Tensor) -> torch.Tensor:
        x = self.up(x)
        # Pad if spatial dims differ by 1 pixel (odd input sizes)
        if x.shape[-2:] != skip.shape[-2:]:
            x = F.interpolate(x, size=skip.shape[-2:],
                              mode='bilinear', align_corners=False)
        x = torch.cat([x, skip], dim=1)
        return self.conv(x)


# ---------------------------------------------------------------------------
# Full U-Net
# ---------------------------------------------------------------------------

class UNet(nn.Module):
    """
    U-Net with MobileNetV2 encoder for 6-channel glacier segmentation.

    Args:
        in_channels:  number of input spectral bands (default 6).
        num_classes:  output channels (1 for binary segmentation).
        input_size:   spatial size (H, W); used only for shape checks.
    """

    def __init__(self, in_channels: int = 6, num_classes: int = 1,
                 input_size: Tuple[int, int] = (256, 256)):
        super().__init__()
        self.encoder = MobileNetV2Encoder(in_channels)
        enc_ch = self.encoder.out_channels  # [32, 16, 24, 32, 96]

        # Bottleneck: enc_ch[-1] → 256
        self.bottleneck = nn.Sequential(
            ConvBnRelu(enc_ch[-1], 256),
            ConvBnRelu(256, 256),
        )

        # Decoder (bottom-up)
        self.dec4 = DecoderBlock(256,    enc_ch[3], 128)   # 96→96
        self.dec3 = DecoderBlock(128,    enc_ch[2], 64)
        self.dec2 = DecoderBlock(64,     enc_ch[1], 32)
        self.dec1 = DecoderBlock(32,     enc_ch[0], 16)

        # Final upsampling to original resolution (×2 from stem)
        self.final_up = nn.Upsample(scale_factor=2, mode='bilinear',
                                    align_corners=False)
        self.head = nn.Sequential(
            ConvBnRelu(16, 16),
            nn.Conv2d(16, num_classes, kernel_size=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, 6, H, W) Sentinel-2 patch

        Returns:
            logits: (B, 1, H, W) — apply sigmoid for probabilities
        """
        skips = self.encoder(x)        # [s0, s1, s2, s3, s4]
        b = self.bottleneck(skips[-1]) # (B, 256, H/16, W/16)

        d = self.dec4(b,    skips[3])
        d = self.dec3(d,    skips[2])
        d = self.dec2(d,    skips[1])
        d = self.dec1(d,    skips[0])
        d = self.final_up(d)
        return self.head(d)

    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        """Return sigmoid-activated probability map."""
        return torch.sigmoid(self.forward(x))

    @property
    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())


# ---------------------------------------------------------------------------
# Tiny variant (used as fair SWIR-preserving baseline in ablation)
# ---------------------------------------------------------------------------

class UNetTiny(UNet):
    """
    Lightweight U-Net (float32, no SAP/SAAQ) — SWIR-preserving baseline.
    Matches the 'U-Net-Tiny (SWIR)‡' row in Table I of TGlacierEdge paper.
    26.7 MB, 1.3 FPS, 3120 mW on RPi 4.
    """

    def __init__(self, in_channels: int = 6, num_classes: int = 1):
        super().__init__(in_channels=in_channels, num_classes=num_classes)
        # Same architecture; "tiny" refers to training without compression.
        # The full compression pipeline (SAP+SAAQ) reduces it to 4.8 MB.


if __name__ == "__main__":
    model = UNet(in_channels=6)
    x = torch.randn(2, 6, 256, 256)
    out = model(x)
    print(f"UNet output shape: {out.shape}")
    print(f"Parameters: {model.num_parameters:,}")
