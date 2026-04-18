"""
False alarm rate computation.

ASPT results:
  Breiðamerkurjökull: 2/47 = 4.3%
  Skeiðarárjökull:    1/41 = 2.4%
  Combined:           3/88 = 3.4%
"""
import numpy as np
from typing import List, Set


def compute_false_alarm_rate(
    delta_series: np.ndarray,
    calving_indices: List[int],
    window_obs: int = 8,
    percentile: float = 5.0,
    min_consecutive: int = 3,
) -> dict:
    """
    Compute false alarm rate over all non-calving 30-day windows.

    A non-calving window is any window_obs-length segment that does not
    overlap with any confirmed calving event ±1 observation.

    Args:
        delta_series:    (T,) Δ̂_spec series.
        calving_indices: List of confirmed calving event indices.
        window_obs:      Window size in observations (~30 days).
        percentile:      Detection threshold percentile.
        min_consecutive: Min consecutive sub-threshold observations.

    Returns:
        dict: false_alarms, n_windows, false_alarm_rate.
    """
    T = len(delta_series)
    calving_set: Set[int] = set()
    for ci in calving_indices:
        for offset in range(-2, 3):     # ±2 obs buffer around each event
            calving_set.add(ci + offset)

    false_alarms = 0
    n_windows = 0

    step = window_obs
    for ws in range(window_obs, T - window_obs + 1, step):
        we = ws + window_obs
        window_range = set(range(ws, we))
        if window_range & calving_set:
            continue

        n_windows += 1
        consec = 0
        triggered = False
        for t in range(ws, we):
            t_start = max(0, t - window_obs)
            baseline = delta_series[t_start:t]
            if len(baseline) < 4:
                continue
            thresh = float(np.percentile(baseline, percentile))
            if delta_series[t] < thresh:
                consec += 1
            else:
                consec = 0
            if consec >= min_consecutive:
                triggered = True
                break

        if triggered:
            false_alarms += 1

    far = false_alarms / n_windows if n_windows > 0 else 0.0
    return {
        "false_alarms": false_alarms,
        "n_windows": n_windows,
        "false_alarm_rate": far,
    }
