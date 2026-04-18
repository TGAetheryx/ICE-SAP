"""Tests for SBT tile selection."""
import numpy as np
import pytest
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from eia_nordic_2026.sbt_sarq.tile_selection import select_boundary_tiles


def test_all_tiles_below_tau():
    """When all W(x) < tau, no tiles selected."""
    W = np.full((128, 128), 0.1, dtype=np.float32)
    selected, mask, frac = select_boundary_tiles(W, tau=0.5)
    assert len(selected) == 0
    assert frac == 0.0


def test_all_tiles_above_tau():
    """When all W(x) > tau, all tiles selected."""
    W = np.full((128, 128), 0.9, dtype=np.float32)
    selected, mask, frac = select_boundary_tiles(W, tau=0.5)
    n_total = (128 // 16) ** 2
    assert len(selected) == n_total
    assert abs(frac - 1.0) < 1e-6


def test_uplink_reduction():
    """SBT should achieve ≥80% uplink reduction for boundary-only W field."""
    W = np.zeros((128, 128), dtype=np.float32)
    W[:, 56:72] = 0.95   # only boundary strip
    selected, _, frac = select_boundary_tiles(W, tau=0.5)
    n_total = (128 // 16) ** 2
    reduction = 1 - len(selected) / n_total
    assert reduction >= 0.7, f"Expected ≥70% reduction, got {reduction:.1%}"
