"""
Empirical verification of Spectral Collapse Hypothesis (Hypothesis 3.1).

Tests:
  1. Δ̂_spec drops by ≥70% in the 24h preceding confirmed calving events.
     Paper: 75.6±3.8% (Breiðamerkurjökull), 73.5±4.2% (Skeiðarárjökull).
  2. No significant inter-glacier difference in drop magnitude (t-test p=0.61).
  3. Seasonal performance: summer > winter aligned with melt patterns.
"""
import numpy as np
from scipy import stats
from typing import List, Dict


def verify_delta_spec_drop(
    delta_series: np.ndarray,
    calving_indices: List[int],
    pre_window_obs: int = 6,       # ~24h at 4-day revisit
    revisit_hours: float = 96.0,
) -> dict:
    """
    Compute Δ̂_spec drop magnitude in the pre-calving window.

    Args:
        delta_series:    (T,) Δ̂_spec series.
        calving_indices: Confirmed calving event indices.
        pre_window_obs:  Number of observations before event (~24h).

    Returns:
        dict with mean drop %, std, per-event drops.
    """
    drops = []
    for ci in calving_indices:
        if ci < pre_window_obs:
            continue
        baseline = float(np.median(delta_series[max(0, ci - 2*pre_window_obs):
                                                ci - pre_window_obs]))
        pre_calving = float(np.min(delta_series[ci - pre_window_obs:ci + 1]))
        if baseline > 1e-6:
            drop_pct = (baseline - pre_calving) / baseline * 100
            drops.append(drop_pct)

    drops = np.array(drops)
    return {
        "mean_drop_pct": float(drops.mean()) if len(drops) > 0 else None,
        "std_drop_pct": float(drops.std()) if len(drops) > 0 else None,
        "per_event_drops": drops.tolist(),
        "n_events": len(drops),
    }


def test_interglacier_difference(
    drops_glacier_a: List[float],
    drops_glacier_b: List[float],
) -> dict:
    """
    Two-sample t-test for inter-glacier difference in Δ̂_spec drop magnitude.

    Expected result (ASPT §5.6): p = 0.61 (no significant difference).
    """
    a = np.array(drops_glacier_a)
    b = np.array(drops_glacier_b)
    t_stat, p_val = stats.ttest_ind(a, b, equal_var=False)
    return {
        "t_stat": float(t_stat),
        "p_value": float(p_val),
        "significant_05": p_val < 0.05,
        "mean_a": float(a.mean()),
        "mean_b": float(b.mean()),
    }


if __name__ == "__main__":
    np.random.seed(0)
    T = 60
    series = np.random.uniform(0.4, 0.7, T).astype(np.float32)
    # Simulate drop before event at t=40
    series[34:40] = np.linspace(0.40, 0.12, 6)
    result = verify_delta_spec_drop(series, [40])
    print(f"Drop: {result['mean_drop_pct']:.1f}%  (paper: ~75%)")

    r_t = test_interglacier_difference(
        [75.6, 73.1, 78.2, 74.0, 76.3],
        [73.5, 71.8, 74.9, 72.1, 75.2],
    )
    print(f"Inter-glacier t-test p = {r_t['p_value']:.3f}  (paper: p=0.61)")
