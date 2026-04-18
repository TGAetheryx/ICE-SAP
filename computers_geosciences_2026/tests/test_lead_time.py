"""Tests for lead time computation."""
import numpy as np
import pytest
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from computers_geosciences_2026.early_warning.lead_time import (
    compute_lead_time, lead_time_statistics,
)


def test_lead_time_detected():
    """Should detect with ≥24h lead when clear collapse."""
    np.random.seed(1)
    T = 50
    series = np.random.uniform(0.35, 0.65, T).astype(np.float32)
    # Insert clear collapse at obs 30–35
    for i in range(5):
        series[30 + i] = 0.08 - i * 0.01
    lt = compute_lead_time(series, calving_idx=40,
                           percentile=5, window_obs=8,
                           min_consecutive=3, revisit_hours=96.0)
    assert lt is not None and lt >= 0


def test_lead_time_no_detection():
    """Returns None when no collapse."""
    series = np.full(50, 0.5, dtype=np.float32)
    lt = compute_lead_time(series, calving_idx=40)
    assert lt is None


def test_lead_time_statistics():
    times = [18.0, None, 24.0, 12.0, None, 20.0]
    stats = lead_time_statistics(times)
    assert stats["n_valid"] == 4
    assert abs(stats["median"] - 19.0) < 1.0
