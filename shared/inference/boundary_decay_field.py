"""
Boundary Decay Field (Tang Field) W(x).

The central spatial prior shared across all three papers:

    W(x) = exp(−d(x, ∂G) / σ_meta) · Φ(Entropy(x))

where:
  d(x, ∂G)    : Euclidean distance from pixel x to the nearest glacier
                boundary ∂G (in pixels)
  σ_meta      : bandwidth parameter predicted by Meta-Net
                ∈ [4, 10] px (Sentinel-2: 40–100 m)
  Φ(Entropy)  : local entropy modulation factor
                = 1 + α · H(x) / log(2)   (soft up-weighting of ambiguous zones)

Information-theoretic foundation (ICE-SAP, Proposition 1):
  Under the MDL principle with Dirichlet-energy regularisation, W*(x) ∝
  exp(−d(x,∂G)/σ*) is the weight function minimising total description length.
  σ* = β√(λ/ṙ) where ṙ is the empirical boundary uncertainty rate.

The field assigns:
  - HIGH weight near boundaries (small d) → concentrate computation
  - LOW  weight in homogeneous ice interior (large d) → suppress
"""

import numpy as np
from scipy.ndimage import distance_transform_edt
from skimage.filters.rank import entropy as skimage_entropy
from skimage.morphology import disk
from typing import Optional, Union


# ---------------------------------------------------------------------------
# Core BDF computation
# ---------------------------------------------------------------------------

def compute_boundary_decay_field(
    boundary_mask: np.ndarray,
    sigma_meta: Union[float, np.ndarray],
    image: Optional[np.ndarray] = None,
    entropy_alpha: float = 0.3,
    normalise: bool = True,
) -> np.ndarray:
    """
    Compute the Boundary Decay Field W(x) for a single patch.

    Args:
        boundary_mask: (H, W) binary array — 1 on boundary pixels, 0 elsewhere.
                       Typically obtained from Sobel edge detection on the
                       NDSI or GLIMS/RGI boundary raster.
        sigma_meta:    Bandwidth in pixels. Either a scalar (MetaNetMLP output)
                       or a (H, W) spatial map (MetaNetConv output).
        image:         (H, W) greyscale image used for entropy modulation.
                       If None, entropy modulation is disabled (Φ = 1).
        entropy_alpha: Weight for entropy modulation (default 0.3).
        normalise:     If True, scale W(x) to [0, 1].

    Returns:
        W: (H, W) float32 array ∈ (0, 1].

    Notes:
        σ_meta robustness: varying σ_meta from 4–8 px changes IoU by ≤0.2 pp
        and boundary IoU by ≤0.3 pp (validated on Vatnajökull, ICE-SAP §III.B).
    """
    boundary_mask = boundary_mask.astype(bool)

    # --- Distance to nearest boundary ∂G ---
    # distance_transform_edt gives Euclidean distance to the nearest True pixel.
    # We want distance to boundary, so pass the *complement* (non-boundary = True).
    dist = distance_transform_edt(~boundary_mask).astype(np.float32)

    # --- Exponential decay ---
    if isinstance(sigma_meta, np.ndarray):
        # Spatially varying σ_meta(x) from MetaNetConv
        sigma = sigma_meta.astype(np.float32)
        sigma = np.clip(sigma, 1e-3, None)          # prevent division by zero
    else:
        sigma = float(sigma_meta)
        if sigma < 1e-3:
            sigma = 1e-3

    W = np.exp(-dist / sigma)                        # (H, W)

    # --- Entropy modulation Φ(Entropy) ---
    if image is not None and entropy_alpha > 0:
        # Normalise image to [0, 255] uint8 for skimage entropy
        img_u8 = _to_uint8(image)
        H_local = skimage_entropy(img_u8, disk(3)).astype(np.float32)
        # Normalise entropy to [0, 1]  (max entropy = log2(nbins))
        H_norm = H_local / (np.log2(256) + 1e-8)
        phi = 1.0 + entropy_alpha * H_norm
        W = W * phi

    if normalise:
        W_max = W.max()
        if W_max > 0:
            W = W / W_max

    return W.astype(np.float32)


def extract_boundary_from_mask(
    glacier_mask: np.ndarray,
    dilation_px: int = 1,
) -> np.ndarray:
    """
    Extract boundary ∂G from a binary glacier segmentation mask.

    Uses Sobel-based edge detection (approximates the GLIMS/RGI boundary
    extraction used at inference time, §3.2 ICE-SAP).

    Args:
        glacier_mask: (H, W) binary array — 1 = glacier, 0 = non-glacier.
        dilation_px:  Border width in pixels (default 1).

    Returns:
        boundary: (H, W) binary array — 1 on boundary pixels.
    """
    from scipy.ndimage import binary_dilation, binary_erosion
    mask = glacier_mask.astype(bool)
    eroded = binary_erosion(mask, iterations=dilation_px)
    boundary = mask & ~eroded
    return boundary.astype(np.float32)


# ---------------------------------------------------------------------------
# Boundary-weighted loss weight map
# ---------------------------------------------------------------------------

def compute_loss_weight_map(
    glacier_mask: np.ndarray,
    sigma_meta: Union[float, np.ndarray],
    image: Optional[np.ndarray] = None,
    entropy_alpha: float = 0.3,
    min_weight: float = 0.1,
) -> np.ndarray:
    """
    Compute per-pixel loss weight map for the boundary-weighted
    segmentation loss (ICE-SAP §3.2, ASPT §4.2).

        L_geo = −(1/N) Σ_x W(x)·log p(x)
                + 1 − [2 Σ_x W(x)·p(x)·y(x)] / [Σ_x W(x)·p(x) + Σ_x W(x)·y(x)]

    High-weight pixels near boundaries receive 5–10× stronger gradient signal
    than homogeneous interior pixels.

    Args:
        glacier_mask: (H, W) binary glacier mask.
        sigma_meta:   BDF bandwidth (pixels).
        image:        Optional image for entropy modulation.
        entropy_alpha: Entropy modulation weight.
        min_weight:   Floor value to prevent zero gradients in interior.

    Returns:
        weight_map: (H, W) float32.
    """
    boundary = extract_boundary_from_mask(glacier_mask)
    W = compute_boundary_decay_field(
        boundary, sigma_meta, image, entropy_alpha, normalise=True
    )
    # Ensure minimum gradient everywhere
    W = np.clip(W, min_weight, 1.0)
    return W


# ---------------------------------------------------------------------------
# Batch computation (PyTorch tensors)
# ---------------------------------------------------------------------------

def bdf_batch_torch(
    boundary_masks: "torch.Tensor",
    sigma_meta: Union[float, "torch.Tensor"],
) -> "torch.Tensor":
    """
    Compute BDF for a batch of boundary masks using PyTorch operations.

    Args:
        boundary_masks: (B, 1, H, W) binary float tensor.
        sigma_meta:     scalar float or (B, 1, H, W) tensor from MetaNetConv.

    Returns:
        W: (B, 1, H, W) float tensor ∈ (0, 1].

    Note: Uses CPU numpy distance transform per sample (GPU EDT not standard).
    """
    import torch
    B = boundary_masks.shape[0]
    device = boundary_masks.device
    W_list = []

    bm_np = boundary_masks.detach().cpu().numpy()  # (B, 1, H, W)

    for i in range(B):
        bm = bm_np[i, 0] > 0.5                     # (H, W) bool
        if isinstance(sigma_meta, (int, float)):
            sm = float(sigma_meta)
        else:
            sm_np = sigma_meta.detach().cpu().numpy()
            sm = sm_np[i, 0] if sm_np.ndim == 4 else float(sm_np)

        dist = distance_transform_edt(~bm).astype(np.float32)
        W_np = np.exp(-dist / max(sm, 1e-3))
        W_list.append(W_np)

    W_np_batch = np.stack(W_list, axis=0)[:, np.newaxis]   # (B, 1, H, W)
    return torch.from_numpy(W_np_batch).to(device)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _to_uint8(image: np.ndarray) -> np.ndarray:
    """Normalise floating-point image to uint8 [0, 255]."""
    img = image.astype(np.float32)
    lo, hi = img.min(), img.max()
    if hi - lo < 1e-8:
        return np.zeros_like(img, dtype=np.uint8)
    img = (img - lo) / (hi - lo) * 255.0
    return img.astype(np.uint8)


# ---------------------------------------------------------------------------
# Quick sanity check
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # Synthetic boundary mask: thin horizontal line in centre
    H, W = 128, 128
    bm = np.zeros((H, W), dtype=np.float32)
    bm[H // 2, :] = 1.0

    for sigma in [4.0, 6.0, 10.0]:
        W_field = compute_boundary_decay_field(bm, sigma)
        print(f"σ={sigma:.0f}  W(boundary)={W_field[H//2, W//2]:.4f}  "
              f"W(32px away)={W_field[H//2 - 32, W//2]:.4f}")
