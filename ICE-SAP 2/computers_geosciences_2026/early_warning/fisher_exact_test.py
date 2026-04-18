"""
Fisher exact test for calving detection significance.

ASPT results:
  Breiðamerkurjökull: 11/14 detected, p = 0.003
  Skeiðarárjökull:    10/12 detected, p = 0.004
  Combined:           21/26 detected, p < 0.001

Tests H₀: detection rate is no better than chance (random alerts).
"""
import numpy as np
from scipy.stats import fisher_exact, binom_test
from typing import Tuple


def fisher_exact_detection(
    n_detected: int,
    n_events: int,
    n_false_alarms: int,
    n_non_calving_windows: int,
) -> dict:
    """
    Fisher exact test: detections vs. chance.

    2×2 contingency table:
                      Calving    Non-calving
      Alert issued:   n_det      n_FA
      No alert:       n_miss     n_NC - n_FA

    Args:
        n_detected:              True positives (detected calving events).
        n_events:                Total calving events.
        n_false_alarms:          False alarms in non-calving windows.
        n_non_calving_windows:   Total non-calving windows.

    Returns:
        dict with odds_ratio, p_value, significant.
    """
    n_missed = n_events - n_detected
    n_true_neg = n_non_calving_windows - n_false_alarms

    table = np.array([
        [n_detected, n_false_alarms],
        [n_missed,   n_true_neg],
    ])

    # Guard against negative values
    table = np.clip(table, 0, None)

    odds_ratio, p_value = fisher_exact(table, alternative='greater')

    return {
        "contingency_table": table.tolist(),
        "odds_ratio": float(odds_ratio),
        "p_value": float(p_value),
        "significant_001": p_value < 0.001,
        "significant_005": p_value < 0.005,
        "significant_01":  p_value < 0.01,
    }


def replicate_aspt_results() -> None:
    """Replicate Fisher exact test results from ASPT paper (Table 3)."""
    print("=== ASPT Fisher Exact Test Results ===\n")

    # Breiðamerkurjökull
    r1 = fisher_exact_detection(11, 14, 2, 47)
    print(f"Breiðamerkurjökull: 11/14 detected, 2/47 FA")
    print(f"  p = {r1['p_value']:.4f}  (paper: p = 0.003)")
    print(f"  Odds ratio = {r1['odds_ratio']:.2f}\n")

    # Skeiðarárjökull
    r2 = fisher_exact_detection(10, 12, 1, 41)
    print(f"Skeiðarárjökull: 10/12 detected, 1/41 FA")
    print(f"  p = {r2['p_value']:.4f}  (paper: p = 0.004)")
    print(f"  Odds ratio = {r2['odds_ratio']:.2f}\n")

    # Combined
    r3 = fisher_exact_detection(21, 26, 3, 88)
    print(f"Combined: 21/26 detected, 3/88 FA")
    print(f"  p = {r3['p_value']:.6f}  (paper: p < 0.001)")
    print(f"  Odds ratio = {r3['odds_ratio']:.2f}")


if __name__ == "__main__":
    replicate_aspt_results()
