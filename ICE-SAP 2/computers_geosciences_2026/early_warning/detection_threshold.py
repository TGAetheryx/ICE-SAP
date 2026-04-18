"""
Detection threshold computation for early-warning.

Implements the causal rolling baseline threshold from ASPT §5.4:
  - 5th percentile of 30-day rolling baseline of Δ̂_spec
  - Computed exclusively from [t−30 days, t−1 day]
  - Sensitivity analysis: 1st–10th percentile (Table 3)

Threshold sensitivity results (ASPT Table 3 sensitivity row):
  1st pct:  10–12/14 detected; FA 1–4/47
  5th pct:  11/14 detected; FA 2/47  (primary)
  10th pct: 11–12/14 detected; FA 3–4/47
"""

import numpy as np
from typing import Optional, List, Tuple


def rolling_percentile_threshold(
    series: np.ndarray,
    t: int,
    percentile: float = 5.0,
    window_obs: int = 8,           # 30 days / 4-day revisit ≈ 8 observations
) -> Optional[float]:
    """
    Compute causal rolling baseline threshold at time t.

    Args:
        series:       (T,) Δ̂_spec time series.
        t:            Current time index.
        percentile:   Threshold percentile (default 5).
        window_obs:   Rolling window size in observations.

    Returns:
        threshold: float, or None if insufficient history.
    """
    t_start = max(0, t - window_obs)
    baseline = series[t_start:t]   # [t−window, t−1], strictly causal

    if len(baseline) < 4:
        return None

    return float(np.percentile(baseline, percentile))


def threshold_sensitivity_analysis(
    delta_series: np.ndarray,
    calving_indices: List[int],
    percentiles: List[float] = None,
    window_obs: int = 8,
    min_consecutive: int = 3,
    revisit_hours: float = 96.0,
    min_lead_h: float = 24.0,
) -> dict:
    """
    Evaluate detection performance across a range of percentile thresholds.

    Reproduces ASPT Table 3 sensitivity row.

    Args:
        delta_series:    (T,) Δ̂_spec values.
        calving_indices: Confirmed calving event indices.
        percentiles:     List of threshold percentiles to test.
        window_obs:      Rolling window size (observations).
        min_consecutive: Consecutive below-threshold observations required.
        revisit_hours:   Sentinel-2 revisit cadence.
        min_lead_h:      Minimum lead time for positive detection.

    Returns:
        dict mapping percentile → {n_detected, n_events, false_alarms}.
    """
    if percentiles is None:
        percentiles = [1.0, 2.0, 3.0, 5.0, 10.0]

    T = len(delta_series)
    results = {}

    for pct in percentiles:
        detected = 0
        false_alarms = 0

        for ev_idx in calving_indices:
            consec = 0
            alert_t = None
            for t in range(min_consecutive, ev_idx + 1):
                thresh = rolling_percentile_threshold(delta_series, t, pct,
                                                      window_obs)
                if thresh is None:
                    continue
                if delta_series[t] < thresh:
                    consec += 1
                else:
                    consec = 0
                if consec >= min_consecutive and alert_t is None:
                    alert_t = t - min_consecutive + 1

            if alert_t is not None:
                lead_h = (ev_idx - alert_t) * revisit_hours
                if lead_h >= min_lead_h:
                    detected += 1

        # Count false alarms in non-calving windows
        calving_set = set(calving_indices)
        for window_start in range(window_obs, T - window_obs, window_obs):
            window_end = window_start + window_obs
            window_range = set(range(window_start, window_end))
            if window_range & calving_set:
                continue
            consec = 0
            for t in range(window_start, min(window_end, T)):
                thresh = rolling_percentile_threshold(
                    delta_series, t, pct, window_obs)
                if thresh is None:
                    continue
                if delta_series[t] < thresh:
                    consec += 1
                else:
                    consec = 0
                if consec >= min_consecutive:
                    false_alarms += 1
                    break

        results[pct] = {
            "n_detected": detected,
            "n_events": len(calving_indices),
            "false_alarms": false_alarms,
            "detection_rate": detected / len(calving_indices)
            if calving_indices else 0.0,
        }

    return results


if __name__ == "__main__":
    np.random.seed(0)
    # Simulate series with a collapse at t=30
    T = 50
    series = np.random.uniform(0.3, 0.7, T).astype(np.float32)
    # Insert collapse
    series[28:32] = [0.25, 0.20, 0.15, 0.18]

    thresh = rolling_percentile_threshold(series, t=30, percentile=5.0,
                                          window_obs=8)
    print(f"Threshold at t=30: {thresh:.4f}")
    print(f"Value at t=30:     {series[30]:.4f}")
    print(f"Below threshold:   {series[30] < thresh}")
