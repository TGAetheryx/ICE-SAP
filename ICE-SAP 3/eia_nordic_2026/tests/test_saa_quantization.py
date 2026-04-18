"""Tests for SAAQ quantization."""
import numpy as np
import pytest
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from eia_nordic_2026.saa_quantization.kl_calibration import (
    kl_divergence, find_optimal_clip_bounds_kl
)


def test_kl_identity():
    """KL(P||P) = 0."""
    p = np.array([0.1, 0.3, 0.4, 0.2], dtype=np.float32)
    assert kl_divergence(p, p) < 1e-6


def test_kl_positive():
    """KL divergence >= 0."""
    p = np.array([0.4, 0.3, 0.2, 0.1], dtype=np.float32)
    q = np.array([0.1, 0.2, 0.3, 0.4], dtype=np.float32)
    assert kl_divergence(p, q) >= 0


def test_clip_bounds_convergence():
    """Calibration should converge to KL < 0.05 on normal distribution."""
    np.random.seed(0)
    acts = np.random.normal(0.5, 0.2, 5000).astype(np.float32)
    cmin, cmax, kl, iters = find_optimal_clip_bounds_kl(
        acts, convergence_threshold=0.05, max_iterations=60)
    assert kl < 0.05, f"KL={kl:.4f} should be < 0.05"
    assert cmin < cmax
