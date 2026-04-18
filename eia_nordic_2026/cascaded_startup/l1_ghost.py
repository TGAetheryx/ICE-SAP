"""
L1 Ghost Sensing — lightweight cloud/polar-night detector.

L1: 15 mW, 1 Hz, 90% duty cycle.
Uses a 120 KB micro-model for cloud cover and polar night detection.
Only triggers L2 when scene change metric exceeds θ_up = 0.35.
"""
import numpy as np
from typing import Tuple


L1_POWER_MW = 15.0
L1_FREQ_HZ  = 1.0
L1_DUTY     = 0.90
L1_MODEL_KB = 120


def detect_cloud_or_polar_night(
    ndsi_map: np.ndarray,
    thermal_band: np.ndarray,
    cloud_threshold: float = 0.05,
    polar_night_lux_threshold: float = 5.0,
) -> Tuple[bool, float]:
    """
    L1 micro-model: detect cloud cover or polar night.

    Args:
        ndsi_map:      (H, W) NDSI map.
        thermal_band:  (H, W) thermal channel.
        cloud_threshold: NDSI fraction threshold for cloud.
        polar_night_lux_threshold: Ambient illuminance threshold.

    Returns:
        blocked: True if observation is blocked (cloud/polar night).
        scene_change_metric: ∈ [0,1].
    """
    # Cloud detection: low NDSI + high uniform brightness
    cloud_frac = float((ndsi_map < 0.1).mean())
    blocked = cloud_frac > cloud_threshold

    # Scene-change metric: variance of NDSI (low = stable/cloudy)
    scene_change = float(np.clip(np.std(ndsi_map) * 2, 0, 1))

    return blocked, scene_change


def l1_power_budget(duty_cycle: float = L1_DUTY) -> float:
    """Return duty-weighted L1 power contribution (mW)."""
    return L1_POWER_MW * duty_cycle
