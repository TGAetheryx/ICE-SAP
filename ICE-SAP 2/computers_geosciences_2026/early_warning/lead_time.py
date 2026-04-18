"""
Lead time computation for calving early-warning.

Lead time = time between first Δ̂_spec alert and confirmed calving event.
ASPT results: median 18.3 h (Breiðamerkurjökull), 17.1 h (Skeiðarárjökull).
IQR: 12–24 h combined.
"""
import numpy as np
from typing import List, Optional, Tuple


def compute_lead_time(
    delta_series: np.ndarray,
    calving_idx: int,
    percentile: float = 5.0,
    window_obs: int = 8,
    min_consecutive: int = 3,
    revisit_hours: float = 96.0,
) -> Optional[float]:
    """
    Compute lead time (hours) for a single calving event.

    Args:
        delta_series:    (T,) Δ̂_spec time series.
        calving_idx:     Index of confirmed calving.
        percentile:      Rolling baseline percentile.
        window_obs:      Rolling window observations (30-day / revisit).
        min_consecutive: Min consecutive below-threshold observations.
        revisit_hours:   Sentinel-2 revisit cadence (hours).

    Returns:
        lead_time_h: float hours, or None if not detected.
    """
    alert_idx = None
    consec = 0

    for t in range(min_consecutive, calving_idx + 1):
        t_start = max(0, t - window_obs)
        baseline = delta_series[t_start:t]
        if len(baseline) < 4:
            continue
        thresh = float(np.percentile(baseline, percentile))

        if delta_series[t] < thresh:
            consec += 1
        else:
            consec = 0

        if consec >= min_consecutive and alert_idx is None:
            alert_idx = t - min_consecutive + 1

    if alert_idx is None:
        return None

    return float((calving_idx - alert_idx) * revisit_hours)


def lead_time_statistics(
    lead_times: List[Optional[float]],
) -> dict:
    """Compute median, IQR, min, max from a list of lead times."""
    valid = [lt for lt in lead_times if lt is not None]
    if not valid:
        return {"median": None, "iqr_lo": None, "iqr_hi": None,
                "min": None, "max": None, "n_valid": 0}
    a = np.array(valid)
    return {
        "median": float(np.median(a)),
        "iqr_lo": float(np.percentile(a, 25)),
        "iqr_hi": float(np.percentile(a, 75)),
        "min": float(a.min()),
        "max": float(a.max()),
        "n_valid": len(valid),
    }
