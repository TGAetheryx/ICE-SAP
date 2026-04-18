"""
Meta-Net: Lightweight Physical Parameter Mapping.

Maps per-patch geophysical observables to the boundary-bandwidth parameter
σ_meta ∈ [4, 10] px (ICE-SAP / EIA Nordic paper) or spatially varying
σ̂_meta(x) (ASPT / Computers & Geosciences paper).

Two variants are implemented:

  MetaNetMLP   — 2-layer MLP (1.2 KB, <2 mW on RPi 4).
                 Inputs: scalar patch-level statistics.
                 Output: single scalar σ_meta ∈ [4, 10].
                 Used in: ICE-SAP (EIA Nordic) + TGlacierEdge (DCOSS-IoT).

  MetaNetConv  — 3-block convolutional network (64 ch, 3×3 kernels).
                 Inputs: spatial feature maps.
                 Output: spatially varying σ̂_meta(x) map ∈ [4, 10].
                 Used in: ASPT (Computers & Geosciences).

Physical interpretation of σ_meta:
  σ_meta ≈ 4–5 px  →  high albedo, low temperature → stable glacier edge
  σ_meta ≈ 6–10 px →  low albedo, surface meltwater → active ablation/calving

References:
  Tang (2026). ICE-SAP. EIA Nordic 2026.
  Tang (2026). ASPT. Computers & Geosciences.
  Tang (2026). TGlacierEdge. DCOSS-IoT 2026.
"""

import torch
import torch.nn as nn
from typing import Tuple


# ---------------------------------------------------------------------------
# Scalar MLP variant (used in ICE-SAP and TGlacierEdge)
# ---------------------------------------------------------------------------

class MetaNetMLP(nn.Module):
    """
    2-layer MLP (hidden=16) mapping patch-level statistics to σ_meta.

    Input features (4 scalars per patch):
        ndsi       : Normalised Difference Snow Index  (albedo proxy)
                     NDSI = (B3 − B11) / (B3 + B11)
        thermal    : Thermal mean (Band B10 surface temperature proxy, Ts)
        glcm_entr  : GLCM Spatial Entropy (roughness ξ)
        delta_ndsi : ΔNDSI (temporal ablation dynamics)

    Output:
        sigma_meta : scalar ∈ [sigma_min, sigma_max]

    Architecture: Linear(4→16) → ReLU → Linear(16→1) → Sigmoid → scale

    Overhead: ~180 parameters ≈ 1.2 KB fp32 weights, <2 mW on RPi 4.

    Generalisation bound (Theorem 2, ICE-SAP):
        Constraining output to [4, 10] reduces Rademacher complexity by 36×
        vs. unconstrained regressor.
    """

    SIGMA_MIN: float = 4.0    # px — stable glacier margin at 10 m/px = 40 m
    SIGMA_MAX: float = 10.0   # px — active ablation front at 10 m/px = 100 m

    def __init__(self, in_features: int = 4, hidden: int = 16,
                 sigma_min: float = 4.0, sigma_max: float = 10.0):
        super().__init__()
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max
        self.sigma_range = sigma_max - sigma_min

        self.net = nn.Sequential(
            nn.Linear(in_features, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, 1),
            nn.Sigmoid(),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """
        Args:
            features: (B, 4) — [ndsi, thermal_mean, glcm_entropy, delta_ndsi]

        Returns:
            sigma_meta: (B, 1) — predicted bandwidth in [sigma_min, sigma_max]
        """
        raw = self.net(features)                            # (B, 1) ∈ [0, 1]
        sigma = self.sigma_min + raw * self.sigma_range     # scale to [4, 10]
        return sigma

    def predict_numpy(self, ndsi: float, thermal: float,
                      glcm_entropy: float, delta_ndsi: float) -> float:
        """Convenience wrapper for a single patch (numpy inputs)."""
        import numpy as np
        feat = torch.tensor([[ndsi, thermal, glcm_entropy, delta_ndsi]],
                            dtype=torch.float32)
        with torch.no_grad():
            return self.forward(feat).item()

    @property
    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())

    @property
    def size_kb(self) -> float:
        """Approximate weight size in KB (float32)."""
        return self.num_parameters * 4 / 1024


# ---------------------------------------------------------------------------
# Spatial convolutional variant (used in ASPT / Computers & Geosciences)
# ---------------------------------------------------------------------------

class _ConvBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class MetaNetConv(nn.Module):
    """
    Spatially varying Meta-Net: outputs σ̂_meta(x) per pixel.

    Three convolutional blocks (64 channels each) → 1×1 conv → Sigmoid → scale.

    Input feature maps (per pixel):
        Channel 0 : local entropy of NDSI
        Channel 1 : thermal infrared variance (B10 resampled to 10 m)
        Channel 2 : GLCM contrast texture
        Channel 3 : GLCM correlation texture

    Output:
        sigma_map: (B, 1, H, W) ∈ [sigma_min, sigma_max]

    Used in ASPT paper for the spatially adaptive boundary-weighted loss:
        w(x) = exp(−d(x,∂Ω)² / (2·σ̂_meta(x)²))
    """

    def __init__(self, in_channels: int = 4, hidden_ch: int = 64,
                 sigma_min: float = 4.0, sigma_max: float = 10.0):
        super().__init__()
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max

        self.encoder = nn.Sequential(
            _ConvBlock(in_channels, hidden_ch),
            _ConvBlock(hidden_ch, hidden_ch),
            _ConvBlock(hidden_ch, hidden_ch),
        )
        self.head = nn.Sequential(
            nn.Conv2d(hidden_ch, 1, kernel_size=1),
            nn.Sigmoid(),
        )

    def forward(self, feature_map: torch.Tensor) -> torch.Tensor:
        """
        Args:
            feature_map: (B, 4, H, W) — spatial feature channels

        Returns:
            sigma_map: (B, 1, H, W) ∈ [sigma_min, sigma_max]
        """
        x = self.encoder(feature_map)
        raw = self.head(x)
        sigma = self.sigma_min + raw * (self.sigma_max - self.sigma_min)
        return sigma

    @property
    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())


# ---------------------------------------------------------------------------
# Factory function
# ---------------------------------------------------------------------------

def build_meta_net(variant: str = "mlp", **kwargs) -> nn.Module:
    """
    Build a Meta-Net.

    Args:
        variant: "mlp"  → MetaNetMLP  (ICE-SAP / TGlacierEdge)
                 "conv" → MetaNetConv (ASPT)
    """
    if variant == "mlp":
        return MetaNetMLP(**kwargs)
    elif variant == "conv":
        return MetaNetConv(**kwargs)
    else:
        raise ValueError(f"Unknown variant: {variant!r}. Choose 'mlp' or 'conv'.")


if __name__ == "__main__":
    # MLP variant
    mlp = MetaNetMLP()
    feat = torch.randn(8, 4)
    sigma = mlp(feat)
    print(f"MetaNetMLP output: {sigma.squeeze().tolist()}")
    print(f"  Parameters: {mlp.num_parameters}  ({mlp.size_kb:.2f} KB)")

    # Conv variant
    conv = MetaNetConv()
    fmap = torch.randn(2, 4, 64, 64)
    smap = conv(fmap)
    print(f"MetaNetConv output shape: {smap.shape}")
    print(f"  Parameters: {conv.num_parameters:,}")
