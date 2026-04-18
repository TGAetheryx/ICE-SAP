"""
L2 Meta Inference — Meta-Net σ_meta generation.

L2: 80 mW, 0.1 Hz, 8% duty cycle.
Runs Meta-Net to predict σ_meta and compute W(x) for current scene.
Triggers L3 when boundary-change metric exceeds θ_up = 0.50.
"""
import numpy as np
from typing import Optional, Tuple
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from shared.inference.entropy import extract_meta_net_features


L2_POWER_MW = 80.0
L2_FREQ_HZ  = 0.1
L2_DUTY     = 0.08


def run_l2_meta_inference(
    ndsi_map: np.ndarray,
    thermal_band: np.ndarray,
    ndsi_prev: Optional[np.ndarray],
    meta_net_fn,
) -> Tuple[float, np.ndarray, float]:
    """
    L2: Run Meta-Net to produce σ_meta and W(x).

    Args:
        ndsi_map:     (H, W) current NDSI.
        thermal_band: (H, W) thermal channel.
        ndsi_prev:    (H, W) previous NDSI (for ΔNDSI).
        meta_net_fn:  Callable (features: np.ndarray) → sigma_meta float.

    Returns:
        sigma_meta:         Predicted bandwidth (px).
        boundary_change:    Scene change metric ∈ [0,1].
        l2_power_contribution: Duty-weighted power (mW).
    """
    features = extract_meta_net_features(ndsi_map, thermal_band, ndsi_prev)
    sigma_meta = float(meta_net_fn(features))

    # Boundary change: ΔNDSI magnitude
    if ndsi_prev is not None:
        boundary_change = float(np.clip(
            np.abs(ndsi_map - ndsi_prev).mean() * 10, 0, 1))
    else:
        boundary_change = 0.3   # neutral default

    return sigma_meta, boundary_change, L2_POWER_MW * L2_DUTY


def l2_power_budget() -> float:
    return L2_POWER_MW * L2_DUTY   # 6.4 mW
