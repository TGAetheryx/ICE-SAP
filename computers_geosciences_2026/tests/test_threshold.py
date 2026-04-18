"""Tests for detection threshold computation."""
import numpy as np
import pytest
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from computers_geosciences_2026.early_warning.detection_threshold import (
    rolling_percentile_threshold,
    threshold_sensitivity_analysis,
)


def test_rolling_threshold_causal():
    """Threshold at t uses only past data."""
    series = np.arange(20, dtype=np.float32) / 20.0
    t = 15
    thresh = rolling_percentile_threshold(series, t=t, percentile=5, window_obs=8)
    # Must use only series[7:15]
    expected = float(np.percentile(series[7:15], 5))
    assert abs(thresh - expected) < 1e-5


def test_rolling_threshold_insufficient_history():
    """Returns None when history < 4 observations."""
    series = np.array([0.5, 0.4, 0.3], dtype=np.float32)
    assert rolling_percentile_threshold(series, t=3, percentile=5) is None


def test_sensitivity_returns_all_percentiles():
    np.random.seed(42)
    series = np.random.uniform(0.2, 0.8, 50).astype(np.float32)
    series[30:35] = 0.05
    result = threshold_sensitivity_analysis(
        series, calving_indices=[35],
        percentiles=[1, 5, 10], window_obs=8,
    )
    assert set(result.keys()) == {1, 5, 10}
    for pct, r in result.items():
        assert "n_detected" in r
        assert "false_alarms" in r
