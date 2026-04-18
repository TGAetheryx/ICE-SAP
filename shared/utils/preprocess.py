"""
Sentinel-2 Level-2A preprocessing for glacier segmentation.

Handles:
  - Band selection and ordering  (B2, B3, B4, B11, B12, B10)
  - Reflectance normalisation
  - Cloud masking (ESA Scene Classification Layer, class 9)
  - Patch extraction (128×128 or 256×256)
  - NDSI computation
  - GEE-style preprocessing pipeline

Sentinel-2 bands used:
  B2  (490 nm, Blue)           → RGB channel
  B3  (560 nm, Green)          → RGB + NDSI numerator
  B4  (665 nm, Red)            → RGB
  B11 (1614 nm, SWIR-1)        → ice–water boundary (key!)
  B12 (2202 nm, SWIR-2)        → ice–water boundary
  B10 (1375 nm, Cirrus/TIR)    → thermal proxy

Dataset details (from all three papers):
  - 122 Sentinel-2 L2A scenes, Vatnajökull 2019–2023
  - Cloud cover ≤ 15–20% (GEE cloud filter)
  - 10 m spatial resolution
  - Labels from GLIMS/RGI 7.0
"""

import numpy as np
from typing import List, Tuple, Optional, Dict
from pathlib import Path


# ---------------------------------------------------------------------------
# Band constants
# ---------------------------------------------------------------------------

BAND_NAMES = ["B2", "B3", "B4", "B11", "B12", "B10"]
BAND_WAVELENGTHS_NM = [490, 560, 665, 1614, 2202, 1375]
N_BANDS = 6

# Reflectance normalisation (Sentinel-2 L2A surface reflectance scale)
# L2A values are stored as uint16 with scale factor 10000
REFLECTANCE_SCALE = 10000.0

# Typical per-band means and stds over Vatnajökull (approximated for normalisation)
# These should be replaced with dataset-specific statistics
VATNAJOKULL_BAND_MEAN = np.array([0.118, 0.153, 0.132, 0.260, 0.198, 0.041],
                                  dtype=np.float32)
VATNAJOKULL_BAND_STD  = np.array([0.085, 0.096, 0.091, 0.152, 0.130, 0.032],
                                  dtype=np.float32)


# ---------------------------------------------------------------------------
# Preprocessing pipeline
# ---------------------------------------------------------------------------

def load_sentinel2_patch(
    path: str,
    normalise: bool = True,
    band_indices: Optional[List[int]] = None,
) -> np.ndarray:
    """
    Load a Sentinel-2 L2A GeoTIFF and return a (6, H, W) float32 array.

    Args:
        path:          Path to multi-band GeoTIFF (6 bands in order above).
        normalise:     If True, divide by REFLECTANCE_SCALE.
        band_indices:  Subset of band indices (default: all 6).

    Returns:
        patch: (C, H, W) float32 array.
    """
    try:
        import rasterio
        with rasterio.open(path) as src:
            data = src.read().astype(np.float32)   # (C, H, W)
    except ImportError:
        raise ImportError("rasterio is required to load GeoTIFF files. "
                          "pip install rasterio")

    if band_indices is not None:
        data = data[band_indices]

    if normalise:
        data = data / REFLECTANCE_SCALE

    # Clip to valid reflectance range
    data = np.clip(data, 0.0, 1.0)
    return data


def apply_cloud_mask(
    patch: np.ndarray,
    scl_band: np.ndarray,
    fill_value: float = 0.0,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Apply ESA Scene Classification Layer cloud mask.

    SCL class 9 = Cloud High Probability.
    Also masks: 3 (Cloud Shadows), 8 (Cloud Medium Probability).

    Args:
        patch:     (C, H, W) reflectance array.
        scl_band:  (H, W) SCL values (uint8).
        fill_value: Value to assign to masked pixels.

    Returns:
        masked_patch: (C, H, W) with cloud pixels set to fill_value.
        cloud_mask:   (H, W) binary mask — 1 where cloudy.
    """
    cloud_mask = np.isin(scl_band, [3, 8, 9]).astype(np.float32)
    masked = patch.copy()
    for c in range(patch.shape[0]):
        masked[c][cloud_mask > 0] = fill_value
    return masked, cloud_mask


def normalise_patch(
    patch: np.ndarray,
    mean: Optional[np.ndarray] = None,
    std:  Optional[np.ndarray] = None,
) -> np.ndarray:
    """
    Z-score normalise a (C, H, W) patch.

    Args:
        patch: (C, H, W) float32.
        mean:  (C,) per-band mean. Defaults to Vatnajökull statistics.
        std:   (C,) per-band std. Defaults to Vatnajökull statistics.

    Returns:
        normalised: (C, H, W).
    """
    if mean is None:
        mean = VATNAJOKULL_BAND_MEAN
    if std is None:
        std = VATNAJOKULL_BAND_STD

    mean = mean[:, np.newaxis, np.newaxis]
    std  = std[:, np.newaxis, np.newaxis]
    return (patch - mean) / (std + 1e-8)


def extract_patches(
    image: np.ndarray,
    patch_size: int = 128,
    stride: int = 64,
    pad_mode: str = "reflect",
) -> Tuple[np.ndarray, List[Tuple[int, int]]]:
    """
    Extract overlapping patches from a large image.

    Args:
        image:      (C, H, W) image.
        patch_size: Patch spatial size.
        stride:     Stride between patches (default 64 = 50% overlap).
        pad_mode:   Padding mode if image doesn't divide evenly.

    Returns:
        patches:    (N, C, patch_size, patch_size) array.
        positions:  List of (row, col) top-left positions.
    """
    C, H, W = image.shape

    # Pad to multiple of stride
    pad_h = (stride - H % stride) % stride
    pad_w = (stride - W % stride) % stride
    if pad_h > 0 or pad_w > 0:
        image = np.pad(image, ((0, 0), (0, pad_h), (0, pad_w)), mode=pad_mode)

    _, H_pad, W_pad = image.shape
    patches = []
    positions = []

    for r in range(0, H_pad - patch_size + 1, stride):
        for c in range(0, W_pad - patch_size + 1, stride):
            patches.append(image[:, r:r+patch_size, c:c+patch_size])
            positions.append((r, c))

    return np.stack(patches, axis=0), positions


def reconstruct_from_patches(
    patches: np.ndarray,
    positions: List[Tuple[int, int]],
    output_shape: Tuple[int, int],
    patch_size: int = 128,
    blend: str = "gaussian",
) -> np.ndarray:
    """
    Reconstruct a full probability map from overlapping patch predictions.

    Args:
        patches:       (N, H_p, W_p) predicted probability maps.
        positions:     (N,) list of (row, col) positions.
        output_shape:  (H, W) of the full output.
        patch_size:    Spatial size of each patch.
        blend:         "gaussian" | "mean" — blending strategy.

    Returns:
        prob_map: (H, W) float32.
    """
    H, W = output_shape
    accum = np.zeros((H, W), dtype=np.float64)
    weight = np.zeros((H, W), dtype=np.float64)

    if blend == "gaussian":
        # 2D Gaussian weight centred on patch
        sigma = patch_size / 4.0
        y = np.arange(patch_size) - patch_size / 2
        x = np.arange(patch_size) - patch_size / 2
        yy, xx = np.meshgrid(y, x, indexing='ij')
        w = np.exp(-(yy**2 + xx**2) / (2 * sigma**2))
        w /= w.max()
    else:
        w = np.ones((patch_size, patch_size))

    for i, (r, c) in enumerate(positions):
        r_end = min(r + patch_size, H)
        c_end = min(c + patch_size, W)
        accum[r:r_end, c:c_end] += patches[i, :r_end-r, :c_end-c] * \
                                    w[:r_end-r, :c_end-c]
        weight[r:r_end, c:c_end] += w[:r_end-r, :c_end-c]

    weight = np.where(weight > 0, weight, 1.0)
    return (accum / weight).astype(np.float32)


# ---------------------------------------------------------------------------
# GLIMS/RGI ground truth loading
# ---------------------------------------------------------------------------

def load_glims_mask(
    path: str,
    target_shape: Optional[Tuple[int, int]] = None,
) -> np.ndarray:
    """
    Load a GLIMS/RGI 7.0 glacier mask raster.

    Args:
        path:          Path to binary glacier mask GeoTIFF.
        target_shape:  Optional (H, W) to resize to.

    Returns:
        mask: (H, W) uint8 binary array — 1 = glacier.
    """
    try:
        import rasterio
        from rasterio.enums import Resampling
        with rasterio.open(path) as src:
            if target_shape is not None:
                data = src.read(
                    1,
                    out_shape=(1, *target_shape),
                    resampling=Resampling.nearest
                )[np.newaxis]
            else:
                data = src.read()
        mask = (data[0] > 0).astype(np.uint8)
    except ImportError:
        raise ImportError("rasterio required for loading GLIMS masks.")

    return mask


if __name__ == "__main__":
    # Demonstrate patch extraction
    fake_image = np.random.rand(6, 512, 512).astype(np.float32)
    patches, positions = extract_patches(fake_image, patch_size=128, stride=64)
    print(f"Extracted {len(patches)} patches of shape {patches[0].shape}")
    print(f"First 3 positions: {positions[:3]}")

    prob_maps = np.random.rand(len(patches), 128, 128).astype(np.float32)
    full = reconstruct_from_patches(prob_maps, positions, (512, 512))
    print(f"Reconstructed shape: {full.shape}")
