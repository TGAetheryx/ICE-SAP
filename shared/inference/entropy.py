"""
Entropy computations for the ICE-SAP / ASPT pipeline.

Two entropy types are used:

1. GLCM Spatial Entropy (roughness ξ)
   Used as input feature for Meta-Net.
   Measures local texture heterogeneity in a 7×7 neighbourhood.

2. Boundary Entropy Field H_b(x)
   H_b(x) = −P(x) log P(x) − (1−P(x)) log(1−P(x))
   where P(x) is the per-pixel segmentation probability map.
   Used to compute Δ̂_spec = Var[H_b] (spectral gap surrogate).

Reference:
  ASPT paper §4.1 (Computers & Geosciences 2026).
  ICE-SAP paper §3.2 (EIA Nordic 2026).
"""

import numpy as np
from typing import Optional


# ---------------------------------------------------------------------------
# Boundary entropy field H_b
# ---------------------------------------------------------------------------

def boundary_entropy_field(prob_map: np.ndarray,
                            eps: float = 1e-7) -> np.ndarray:
    """
    Compute per-pixel boundary entropy H_b(x).

    H_b(x) = −P·log(P) − (1−P)·log(1−P)

    On a well-resolved boundary, P(x) is sharply bimodal (≈0 in interior,
    ≈1 at calving front), so H_b is strongly heterogeneous → high Var[H_b].
    Near structural failure, P(x) → Bernoulli(0.5) everywhere → H_b → log 2
    at every pixel → Var[H_b] → 0  (spectral collapse signature).

    Args:
        prob_map: (H, W) or (B, H, W) float array ∈ [0, 1].
        eps:      Numerical floor to prevent log(0).

    Returns:
        H_b: same shape as prob_map, values ∈ [0, log2] ≈ [0, 0.693].
    """
    p = np.clip(prob_map.astype(np.float64), eps, 1.0 - eps)
    H = -p * np.log(p) - (1 - p) * np.log(1 - p)
    return H.astype(np.float32)


def boundary_entropy_torch(prob_map: "torch.Tensor",
                            eps: float = 1e-7) -> "torch.Tensor":
    """
    PyTorch version of boundary entropy field.

    Args:
        prob_map: (...) float tensor ∈ [0, 1].

    Returns:
        H_b: same shape, float tensor.
    """
    import torch
    p = prob_map.clamp(eps, 1.0 - eps)
    return -p * torch.log(p) - (1 - p) * torch.log(1 - p)


# ---------------------------------------------------------------------------
# Spectral gap surrogate Δ̂_spec = Var[H_b]
# ---------------------------------------------------------------------------

def delta_spec(prob_map: np.ndarray,
               mask: Optional[np.ndarray] = None,
               eps: float = 1e-7) -> float:
    """
    Compute the spectral gap surrogate Δ̂_spec = Var_{x∈Ω}[H_b(x,t)].

    This is the computationally tractable proxy for the Laplace-Beltrami
    first spectral gap Δspec = λ₁ − λ₀ of the Perceptual Operator Ω̂.

    Geometric motivation (ASPT §4.1):
      On a well-resolved boundary: P bimodal → H_b heterogeneous → high Var.
      At spectral collapse (Δspec → 0): P flat → H_b uniform → Var → 0.
      Δ̂_spec ↓ is therefore a detectable precursor to calving.

    Validation (ASPT Supplementary S2):
      ε_max = 0.02, relative error ≤ 3.5%, PCC = 0.998, SCC = 1.0
      across 128,000 simulated samples.

    Args:
        prob_map: (H, W) segmentation probability map ∈ [0, 1].
        mask:     Optional (H, W) binary mask — compute only within glacier
                  region of interest (ROI). If None, uses all pixels.
        eps:      Numerical floor.

    Returns:
        delta: scalar float — Δ̂_spec value.
    """
    H = boundary_entropy_field(prob_map, eps=eps)

    if mask is not None:
        mask = mask.astype(bool)
        if mask.sum() < 4:          # degenerate case
            return float(np.var(H))
        return float(np.var(H[mask]))

    return float(np.var(H))


def delta_spec_series(
    prob_maps: np.ndarray,
    masks: Optional[np.ndarray] = None,
    eps: float = 1e-7,
) -> np.ndarray:
    """
    Compute Δ̂_spec for a time-series of probability maps.

    Args:
        prob_maps: (T, H, W) array of segmentation probabilities.
        masks:     (T, H, W) or (H, W) optional ROI masks.
        eps:       Numerical floor.

    Returns:
        delta_series: (T,) array of Δ̂_spec values.
    """
    T = prob_maps.shape[0]
    series = np.zeros(T, dtype=np.float32)
    for t in range(T):
        mask_t = masks[t] if (masks is not None and masks.ndim == 3) else masks
        series[t] = delta_spec(prob_maps[t], mask=mask_t, eps=eps)
    return series


# ---------------------------------------------------------------------------
# GLCM spatial entropy (Meta-Net input feature ξ)
# ---------------------------------------------------------------------------

def glcm_entropy(
    image: np.ndarray,
    radius: int = 3,
) -> np.ndarray:
    """
    Compute local GLCM entropy (roughness ξ) used as Meta-Net input.

    Uses scikit-image rank entropy over a disk neighbourhood.

    Args:
        image: (H, W) float image, will be normalised to uint8.
        radius: disk radius for local neighbourhood (default 3 → 7×7 px).

    Returns:
        entropy_map: (H, W) float32 entropy values.
    """
    from skimage.filters.rank import entropy as rank_entropy
    from skimage.morphology import disk

    img_u8 = _to_uint8(image)
    ent = rank_entropy(img_u8, disk(radius)).astype(np.float32)
    # Normalise to [0, 1]
    ent_max = np.log2(256)
    return ent / (ent_max + 1e-8)


def glcm_features(
    image: np.ndarray,
    distances: list = [1],
    angles: list = [0, np.pi / 2],
    levels: int = 64,
    patch_size: int = 7,
) -> dict:
    """
    Compute patch-level GLCM contrast and correlation texture features.

    Used as input features for MetaNetConv (ASPT paper §4.2).

    Args:
        image:      (H, W) float image.
        distances:  GLCM offsets (pixels).
        angles:     GLCM angles (radians).
        levels:     Quantisation levels.
        patch_size: Not used directly (uses full-image GLCM for scalar output).

    Returns:
        dict with keys 'contrast' and 'correlation' — scalar values.
    """
    from skimage.feature import graycomatrix, graycoprops

    img_u8 = _to_uint8(image)
    # Quantise to `levels` bins
    img_q = (img_u8.astype(np.float32) * (levels - 1) / 255).astype(np.uint8)
    img_q = np.clip(img_q, 0, levels - 1)

    glcm = graycomatrix(img_q, distances=distances, angles=angles,
                        levels=levels, symmetric=True, normed=True)

    contrast = float(graycoprops(glcm, 'contrast').mean())
    correlation = float(graycoprops(glcm, 'correlation').mean())
    return {"contrast": contrast, "correlation": correlation}


# ---------------------------------------------------------------------------
# NDSI and thermal features (Meta-Net inputs)
# ---------------------------------------------------------------------------

def compute_ndsi(band3: np.ndarray, band11: np.ndarray,
                 eps: float = 1e-8) -> np.ndarray:
    """
    Normalised Difference Snow Index.

    NDSI = (B3 − B11) / (B3 + B11)

    Sentinel-2 Band 3 = Green (560 nm), Band 11 = SWIR (1614 nm).
    High NDSI (≈1) → clean snow/ice. Low NDSI (≈0) → bare rock or water.

    Args:
        band3:  (H, W) Green reflectance ∈ [0, 1].
        band11: (H, W) SWIR reflectance ∈ [0, 1].

    Returns:
        ndsi: (H, W) float32 ∈ [−1, 1].
    """
    b3 = band3.astype(np.float32)
    b11 = band11.astype(np.float32)
    ndsi = (b3 - b11) / (b3 + b11 + eps)
    return np.clip(ndsi, -1.0, 1.0)


def extract_meta_net_features(
    ndsi_map: np.ndarray,
    thermal_band: np.ndarray,
    ndsi_prev: Optional[np.ndarray] = None,
    glcm_radius: int = 3,
) -> np.ndarray:
    """
    Extract the 4 scalar patch-level features used by MetaNetMLP.

    Features:
        [0] NDSI mean          (albedo proxy)
        [1] Thermal mean       (surface temperature Ts proxy)
        [2] GLCM entropy mean  (roughness ξ)
        [3] ΔNDSI              (ablation dynamics; 0 if prev not available)

    Args:
        ndsi_map:     (H, W) NDSI map.
        thermal_band: (H, W) thermal reflectance / radiance.
        ndsi_prev:    (H, W) NDSI from previous scene (for ΔNDSI).
        glcm_radius:  Disk radius for entropy.

    Returns:
        features: (4,) float32 array.
    """
    feat_ndsi = float(np.mean(ndsi_map))
    feat_thermal = float(np.mean(thermal_band))

    ent_map = glcm_entropy(ndsi_map, radius=glcm_radius)
    feat_glcm = float(np.mean(ent_map))

    if ndsi_prev is not None:
        feat_delta = float(np.mean(np.abs(ndsi_map - ndsi_prev)))
    else:
        feat_delta = 0.0

    return np.array([feat_ndsi, feat_thermal, feat_glcm, feat_delta],
                    dtype=np.float32)


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def _to_uint8(image: np.ndarray) -> np.ndarray:
    img = image.astype(np.float32)
    lo, hi = img.min(), img.max()
    if hi - lo < 1e-8:
        return np.zeros_like(img, dtype=np.uint8)
    return ((img - lo) / (hi - lo) * 255).astype(np.uint8)


if __name__ == "__main__":
    np.random.seed(42)
    # Simulate a "good" prediction (bimodal) → high Δ̂_spec
    H, W = 128, 128
    prob_good = np.zeros((H, W), dtype=np.float32)
    prob_good[:, W//2:] = 0.95     # ice
    prob_good[:, :W//2] = 0.05     # water
    prob_good[:, W//2-2:W//2+2] = 0.5   # boundary ambiguity

    # Simulate a "collapsing" prediction (flat ≈ 0.5 everywhere) → low Δ̂_spec
    prob_bad = np.full((H, W), 0.5, dtype=np.float32)
    prob_bad += np.random.randn(H, W) * 0.05

    d_good = delta_spec(prob_good)
    d_bad  = delta_spec(prob_bad)
    print(f"Δ̂_spec (stable boundary) = {d_good:.4f}")
    print(f"Δ̂_spec (collapsing)       = {d_bad:.4f}")
    print(f"Ratio: {d_good / (d_bad + 1e-8):.1f}×")
