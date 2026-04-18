"""
L3 Core Execution — SAP-compressed U-Net inference.

L3: 320 mW (356±16 mW at −45°C), trigger-based, 2% duty cycle.
Runs the full SAP-compressed + SAAQ-quantised U-Net via ONNX Runtime.
Inference latency: 201 ± 2 ms on RPi 4.
"""
import numpy as np
from typing import Optional, Tuple
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))


L3_POWER_MW    = 320.0
L3_POWER_MW_45C = 356.0    # at −45°C (Arrhenius-type increase)
L3_DUTY        = 0.02
L3_LATENCY_MS  = 201.0


def run_l3_core_inference(
    patch: np.ndarray,
    model_fn,
    temperature_c: float = 25.0,
) -> Tuple[np.ndarray, float, float]:
    """
    L3: Full SAP+SAAQ U-Net inference.

    Args:
        patch:          (6, H, W) Sentinel-2 patch.
        model_fn:       Callable → probability map.
        temperature_c:  Operating temperature for power adjustment.

    Returns:
        prob_map:    (H, W) segmentation probabilities.
        power_mw:    Instantaneous inference power.
        latency_ms:  Inference latency.
    """
    import time
    t0 = time.perf_counter()
    prob_map = model_fn(patch)
    latency_ms = (time.perf_counter() - t0) * 1000

    # Power scales with temperature (Arrhenius: +11.3% at −45°C)
    if temperature_c <= -45:
        power_mw = L3_POWER_MW_45C
    elif temperature_c < 0:
        # Linear interpolation between 25°C and −45°C
        frac = (25.0 - temperature_c) / (25.0 + 45.0)
        power_mw = L3_POWER_MW + frac * (L3_POWER_MW_45C - L3_POWER_MW)
    else:
        power_mw = L3_POWER_MW

    return prob_map, power_mw, latency_ms


def l3_power_budget(temperature_c: float = 25.0) -> float:
    """Duty-weighted L3 power contribution (mW)."""
    if temperature_c <= -45:
        return L3_POWER_MW_45C * L3_DUTY
    return L3_POWER_MW * L3_DUTY   # 6.4 mW


def average_system_power(temperature_c: float = 25.0) -> float:
    """
    Compute average system power for cascaded startup.
    P_avg = 0.9×15 + 0.08×80 + 0.02×L3_power
    Paper (25°C): 23.9 mW; at −45°C: 27.0 mW.
    """
    from eia_nordic_2026.cascaded_startup.l1_ghost import L1_POWER_MW, L1_DUTY
    from eia_nordic_2026.cascaded_startup.l2_meta  import L2_POWER_MW, L2_DUTY
    l3_pw = L3_POWER_MW_45C if temperature_c <= -45 else L3_POWER_MW
    return L1_POWER_MW * L1_DUTY + L2_POWER_MW * L2_DUTY + l3_pw * L3_DUTY


if __name__ == "__main__":
    p25 = average_system_power(25.0)
    p45 = average_system_power(-45.0)
    print(f"Avg power @+25°C: {p25:.1f} mW  (paper: 23.9 mW)")
    print(f"Avg power @−45°C: {p45:.1f} mW  (paper: 27.0 mW)")
    batt_25 = 4000 * 3.6 / (p25 / 1000) / 86400
    batt_45 = 4000 * 3.6 / (p45 / 1000) / 86400
    print(f"Battery life @+25°C: {batt_25:.1f} days  (paper: 24.6)")
    print(f"Battery life @−45°C: {batt_45:.1f} days  (paper: 22.8)")
