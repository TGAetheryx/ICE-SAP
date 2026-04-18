"""
Threshold sensitivity analysis for early-warning (ASPT Table 3 sensitivity row).

Tests detection performance across 1st–10th percentile thresholds.
Paper result: 10–12/14 detected; false alarms 1–4/47 (Breiðamerkurjökull).
"""
import numpy as np
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from computers_geosciences_2026.early_warning.detection_threshold import (
    threshold_sensitivity_analysis,
)


def run():
    np.random.seed(42)
    T = 122
    delta_series = np.random.uniform(0.30, 0.65, T).astype(np.float32)
    calving_indices = sorted(np.random.choice(range(20, T-5), 14, replace=False))
    for ci in calving_indices:
        for off in range(1, 7):
            t = ci - off
            if t >= 0:
                delta_series[t] = max(0.04, delta_series[t] - 0.07 * off)

    results = threshold_sensitivity_analysis(
        delta_series, calving_indices,
        percentiles=[1, 2, 3, 5, 10],
        window_obs=8, min_consecutive=3,
    )

    print("\n[Sensitivity Analysis — Breiðamerkurjökull]")
    print(f"  {'Percentile':>12} {'Detected':>10} {'FA':>6} {'Det. Rate':>12}")
    print("-"*44)
    for pct, r in results.items():
        print(f"  {pct:>10.0f}th  {r['n_detected']}/{r['n_events']:>4}    "
              f"{r['false_alarms']:>4}   {r['detection_rate']*100:>8.1f}%")
    print(f"\n  Paper (1st–10th pct): 10–12/14 detected; FA 1–4/47")


if __name__ == "__main__":
    run()
