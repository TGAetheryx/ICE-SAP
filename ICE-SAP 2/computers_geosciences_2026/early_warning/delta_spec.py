"""
Spectral gap surrogate Δ̂_spec = Var[H_b(x,t)].

Geometric motivation (ASPT §4.1):
  The Perceptual Operator Ω̂ = −Δ_M + V on the Arithmetic Fisher Manifold has
  a discrete spectrum {λ₀ ≤ λ₁ ≤ …}. As the glacier boundary approaches
  structural failure, the first spectral gap Δspec = λ₁ − λ₀ → 0.

  In the observational representation, spectral collapse ↔ P(x) → uniform
  Bernoulli(0.5) everywhere ↔ H_b → log 2 at every pixel ↔ Var[H_b] → 0.

Synthetic validation (Supplementary S2):
  ε_max = 0.02, relative error ≤ 3.5%, PCC = 0.998, SCC = 1.0
  across 128,000 simulated samples.

This module re-exports from shared/inference/entropy.py and adds
paper-specific wrappers.
"""

import numpy as np
from typing import Optional, List, Tuple
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from shared.inference.entropy import (
    boundary_entropy_field,
    delta_spec,
    delta_spec_series,
)


# ---------------------------------------------------------------------------
# Paper-specific wrappers
# ---------------------------------------------------------------------------

class DeltaSpecMonitor:
    """
    Online Δ̂_spec monitor for a single glacier with rolling baseline.

    Implements the causal detection criterion from ASPT §5.4:
      "Detection was defined as a sustained drop below a percentile threshold
       of the 30-day rolling baseline of Δ̂_spec, maintained for at least
       3 consecutive observations."

    The rolling baseline uses only past observations [t−30 days, t−1 day],
    ensuring real-time deployability.

    Args:
        revisit_hours:     Sentinel-2 revisit cadence (~4 days = 96 h).
        window_days:       Rolling window length (default 30 days).
        percentile:        Detection threshold percentile (default 5).
        min_consecutive:   Minimum consecutive sub-threshold observations.
    """

    def __init__(
        self,
        revisit_hours: float = 96.0,
        window_days: int = 30,
        percentile: float = 5.0,
        min_consecutive: int = 3,
    ):
        self.revisit_hours = revisit_hours
        self.window_days = window_days
        self.percentile = percentile
        self.min_consecutive = min_consecutive

        self._history: List[float] = []
        self._timestamps: List[float] = []    # hours since start
        self._consecutive_below: int = 0
        self._alert_issued: bool = False
        self._alert_time_h: Optional[float] = None

    @property
    def window_obs(self) -> int:
        """Number of observations in the rolling window."""
        return int(self.window_days * 24 / self.revisit_hours)

    def update(self, prob_map: np.ndarray,
               timestamp_h: Optional[float] = None) -> dict:
        """
        Process a new Sentinel-2 scene and update the monitor.

        Args:
            prob_map:    (H, W) segmentation probability map ∈ [0,1].
            timestamp_h: Absolute timestamp (hours since mission start).
                         If None, estimated from observation index.

        Returns:
            status dict with keys:
              'delta_spec'    : float  — current Δ̂_spec value
              'threshold'     : float  — current rolling baseline threshold
              'alert'         : bool   — True if detection criterion met
              'consecutive'   : int    — consecutive below-threshold count
              'lead_time_h'   : float | None — hours since first alert
        """
        ds = delta_spec(prob_map)
        t = timestamp_h if timestamp_h is not None else \
            len(self._history) * self.revisit_hours

        self._history.append(ds)
        self._timestamps.append(t)

        # Causal rolling baseline: [t-window, t-1]
        w = self._history[max(0, len(self._history) - self.window_obs - 1):-1]
        if len(w) >= 4:
            thresh = float(np.percentile(w, self.percentile))
        else:
            thresh = float("inf")    # not enough history yet

        if ds < thresh:
            self._consecutive_below += 1
        else:
            self._consecutive_below = 0

        new_alert = False
        if (self._consecutive_below >= self.min_consecutive
                and not self._alert_issued):
            self._alert_issued = True
            self._alert_time_h = t
            new_alert = True

        lead = (t - self._alert_time_h) if self._alert_issued else None

        return {
            "delta_spec": ds,
            "threshold": thresh,
            "alert": new_alert,
            "alert_active": self._alert_issued,
            "consecutive": self._consecutive_below,
            "lead_time_h": lead,
        }

    def reset(self) -> None:
        """Reset monitor state (new monitoring season)."""
        self._history.clear()
        self._timestamps.clear()
        self._consecutive_below = 0
        self._alert_issued = False
        self._alert_time_h = None

    @property
    def history(self) -> np.ndarray:
        return np.array(self._history, dtype=np.float32)

    @property
    def timestamps(self) -> np.ndarray:
        return np.array(self._timestamps, dtype=np.float64)


# ---------------------------------------------------------------------------
# Batch evaluation over a Sentinel-2 time series
# ---------------------------------------------------------------------------

def evaluate_early_warning(
    prob_series: np.ndarray,
    calving_indices: List[int],
    revisit_hours: float = 96.0,
    window_days: int = 30,
    percentile: float = 5.0,
    min_consecutive: int = 3,
    min_lead_h: float = 24.0,
) -> dict:
    """
    Evaluate early-warning performance over a full Sentinel-2 time series.

    Args:
        prob_series:     (T, H, W) probability maps.
        calving_indices: List of scene indices with confirmed calving events.
        revisit_hours:   Sentinel-2 revisit cadence (hours).
        window_days:     Rolling baseline window.
        percentile:      Detection threshold percentile.
        min_consecutive: Minimum consecutive observations below threshold.
        min_lead_h:      Minimum lead time to count as positive detection.

    Returns:
        dict with:
          'detected':      List[bool] — detection status per event
          'lead_times_h':  List[float | None] — lead times
          'false_alarms':  int
          'detection_rate': float
          'median_lead_h': float
          'false_alarm_rate': float
    """
    T = len(prob_series)
    ds_series = delta_spec_series(prob_series)

    detected = []
    lead_times = []

    for ev_idx in calving_indices:
        monitor = DeltaSpecMonitor(revisit_hours, window_days,
                                   percentile, min_consecutive)
        alert_time = None

        for t in range(min(ev_idx + 1, T)):
            # Reconstruct prob_map from series
            status = monitor.update(prob_series[t],
                                    timestamp_h=t * revisit_hours)
            if status["alert"] and alert_time is None:
                alert_time = t * revisit_hours

        ev_time_h = ev_idx * revisit_hours
        if alert_time is not None:
            lead_h = ev_time_h - alert_time
            detected.append(lead_h >= min_lead_h)
            lead_times.append(lead_h if lead_h >= 0 else None)
        else:
            detected.append(False)
            lead_times.append(None)

    # False alarm rate on non-calving windows
    calving_set = set(calving_indices)
    false_alarms = 0
    n_non_calving_windows = 0

    # Build 30-day non-calving windows
    wobs = int(window_days * 24 / revisit_hours)
    t = wobs
    while t < T:
        # Check if window [t-wobs, t] overlaps any calving event
        window_indices = set(range(max(0, t - wobs), t + 1))
        if not (window_indices & calving_set):
            n_non_calving_windows += 1
            monitor = DeltaSpecMonitor(revisit_hours, window_days,
                                       percentile, min_consecutive)
            for ti in range(max(0, t - wobs), t + 1):
                status = monitor.update(prob_series[ti],
                                        timestamp_h=ti * revisit_hours)
                if status["alert"]:
                    false_alarms += 1
                    break
        t += wobs

    n_det = sum(detected)
    n_ev = len(detected)
    valid_leads = [lt for lt in lead_times if lt is not None and lt >= 0]

    return {
        "detected": detected,
        "lead_times_h": lead_times,
        "false_alarms": false_alarms,
        "n_non_calving_windows": n_non_calving_windows,
        "detection_rate": n_det / n_ev if n_ev > 0 else 0.0,
        "median_lead_h": float(np.median(valid_leads)) if valid_leads else None,
        "iqr_lead_h": (float(np.percentile(valid_leads, 25)),
                       float(np.percentile(valid_leads, 75)))
                       if len(valid_leads) >= 4 else None,
        "false_alarm_rate": false_alarms / n_non_calving_windows
                            if n_non_calving_windows > 0 else 0.0,
    }


if __name__ == "__main__":
    np.random.seed(42)
    T, H, W = 50, 64, 64

    # Simulate: stable → precursory collapse → calving → recovery
    prob_series = np.zeros((T, H, W), dtype=np.float32)
    for t in range(T):
        if t < 30:
            # Stable: bimodal prediction
            p = np.zeros((H, W), dtype=np.float32)
            p[:, W//2:] = 0.92
            p[:, :W//2] = 0.04
        elif t < 40:
            # Collapse: flattening
            frac = (t - 30) / 10
            p = np.full((H, W), 0.5 * frac + (1 - frac) * 0.48, dtype=np.float32)
            p += np.random.randn(H, W).astype(np.float32) * 0.05
        else:
            # Post-calving: new stable boundary
            p = np.zeros((H, W), dtype=np.float32)
            p[:, W//3:] = 0.90
            p[:, :W//3] = 0.05
        prob_series[t] = np.clip(p, 0, 1)

    ds = delta_spec_series(prob_series)
    print("Δ̂_spec series (first 15 values):", np.round(ds[:15], 4))
    print(f"  Stable mean:    {ds[:30].mean():.4f}")
    print(f"  Collapse mean:  {ds[30:40].mean():.4f}")

    result = evaluate_early_warning(prob_series, calving_indices=[39])
    print(f"\nDetection: {result['detected']}")
    print(f"Lead time: {result['lead_times_h']} h")
    print(f"False alarm rate: {result['false_alarm_rate']:.2%}")
