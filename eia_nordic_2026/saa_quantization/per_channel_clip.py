"""
Per-channel asymmetric clip bounds for SAAQ.

Handles the ~10× SWIR reflectance gap between B11/B12 and RGB bands.
Per-channel calibration achieves 4.9 FPS vs. 1.9 FPS for global-scale PTQ.
"""
import numpy as np
from typing import Dict, Tuple


# Spectral band indices in the 6-channel input
BAND_RGB   = [0, 1, 2]   # B2, B3, B4
BAND_SWIR  = [3, 4]      # B11, B12
BAND_TIR   = [5]         # B10

# Typical reflectance ranges per band type (Vatnajökull, normalised)
TYPICAL_RANGES = {
    "rgb":  (0.0, 0.4),
    "swir": (0.0, 0.9),    # ~10× wider than RGB
    "tir":  (0.0, 0.15),
}


class PerChannelClipCalibrator:
    """
    Stores and applies per-channel asymmetric INT8 clip bounds.

    For each Conv2d layer, stores:
        clip_min[c]: lower clip bound for channel c
        clip_max[c]: upper clip bound for channel c

    Quantisation formula:
        scale_c = (clip_max[c] - clip_min[c]) / 255
        zero_point_c = round(-clip_min[c] / scale_c)
        q_c = clamp(round(x_c / scale_c) + zero_point_c, 0, 255)
    """

    def __init__(self):
        self._layer_bounds: Dict[str, Dict] = {}

    def store_bounds(self, layer_name: str, clip_min: np.ndarray,
                     clip_max: np.ndarray) -> None:
        self._layer_bounds[layer_name] = {
            "clip_min": clip_min.astype(np.float32),
            "clip_max": clip_max.astype(np.float32),
            "scale": (clip_max - clip_min) / 255.0,
            "zero_point": np.round(-clip_min / ((clip_max - clip_min) / 255.0 + 1e-8)),
        }

    def quantize_channel(self, activations: np.ndarray, layer_name: str,
                          channel_idx: int) -> np.ndarray:
        """Quantise a single channel to INT8 using stored bounds."""
        bounds = self._layer_bounds[layer_name]
        cmin = bounds["clip_min"][channel_idx]
        cmax = bounds["clip_max"][channel_idx]
        scale = (cmax - cmin) / 255.0 + 1e-8
        zp = int(round(-cmin / scale))

        q = np.clip(np.round(activations / scale) + zp, 0, 255).astype(np.uint8)
        return q

    def dequantize_channel(self, q_acts: np.ndarray, layer_name: str,
                            channel_idx: int) -> np.ndarray:
        """Dequantise INT8 back to float32."""
        bounds = self._layer_bounds[layer_name]
        scale = bounds["scale"][channel_idx]
        zp = bounds["zero_point"][channel_idx]
        return (q_acts.astype(np.float32) - zp) * scale

    def get_compression_stats(self) -> dict:
        total_channels = sum(len(b["clip_min"])
                             for b in self._layer_bounds.values())
        return {
            "n_layers_calibrated": len(self._layer_bounds),
            "total_channels": total_channels,
            "expected_speedup": "4.1×",    # from ICE-SAP §3.3
            "expected_fps": 4.9,
            "model_size_mb": 1.8,          # after WSA pruning + SAAQ (Table 3)
        }


def spectral_aware_calibration_split(
    n_patches: int = 2000,
    optical_frac: float = 0.62,
    swir_frac: float = 0.28,
    thermal_frac: float = 0.10,
    seed: int = 42,
) -> Dict[str, int]:
    """
    Determine calibration patch counts per spectral type.

    ICE-SAP §3.3 calibration: 2,000 patches (optical 62%, SWIR 28%, thermal 10%).
    This spectral-diverse composition handles the ~10× SWIR/RGB reflectance gap.
    """
    assert abs(optical_frac + swir_frac + thermal_frac - 1.0) < 1e-6
    return {
        "optical": int(n_patches * optical_frac),
        "swir": int(n_patches * swir_frac),
        "thermal": n_patches - int(n_patches * optical_frac)
                             - int(n_patches * swir_frac),
    }


if __name__ == "__main__":
    split = spectral_aware_calibration_split(2000)
    print(f"Calibration split: {split}")
    print(f"Total: {sum(split.values())} patches  (paper: 2,000)")

    calibrator = PerChannelClipCalibrator()
    np.random.seed(0)
    C = 32
    clip_min = np.random.uniform(-0.1, 0.0, C).astype(np.float32)
    clip_max = np.random.uniform(0.5, 1.2, C).astype(np.float32)
    calibrator.store_bounds("conv1", clip_min, clip_max)

    acts = np.random.randn(64, 64).astype(np.float32) * 0.5
    q = calibrator.quantize_channel(acts, "conv1", 0)
    deq = calibrator.dequantize_channel(q, "conv1", 0)
    err = float(np.abs(acts - deq).mean())
    print(f"Quantisation MAE: {err:.5f}")
