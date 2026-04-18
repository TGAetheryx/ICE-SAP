"""Tests for delta_spec surrogate."""
import numpy as np
import pytest
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from shared.inference.entropy import boundary_entropy_field, delta_spec


def test_boundary_entropy_bimodal():
    """Bimodal prediction → high entropy variance."""
    H, W = 64, 64
    p = np.zeros((H, W), dtype=np.float32)
    p[:, W//2:] = 0.95
    p[:, :W//2] = 0.05
    H_b = boundary_entropy_field(p)
    # Interior entropy should be near 0
    assert H_b[:, W//2+5:].mean() < 0.15, "Interior entropy too high"


def test_boundary_entropy_uniform():
    """Uniform P≈0.5 → entropy near log(2) ≈ 0.693."""
    p = np.full((32, 32), 0.5, dtype=np.float32)
    H_b = boundary_entropy_field(p)
    assert abs(H_b.mean() - np.log(2)) < 0.05


def test_delta_spec_stable_vs_collapsed():
    """Stable boundary → high Δ̂_spec; collapsed → low Δ̂_spec."""
    H, W = 64, 64
    p_stable = np.zeros((H, W), dtype=np.float32)
    p_stable[:, W//2:] = 0.95
    p_stable[:, :W//2] = 0.05

    p_collapsed = np.full((H, W), 0.5, dtype=np.float32)

    ds_stable = delta_spec(p_stable)
    ds_collapsed = delta_spec(p_collapsed)

    assert ds_stable > ds_collapsed * 3, \
        f"Stable should be >> collapsed: {ds_stable:.4f} vs {ds_collapsed:.4f}"


def test_delta_spec_monotone():
    """Δ̂_spec decreases as boundary becomes more diffuse."""
    H, W = 64, 64
    gaps = [1.0, 0.7, 0.4, 0.1]
    ds_vals = []
    for gap in gaps:
        p = np.zeros((H, W), dtype=np.float32)
        p[:, W//2:] = 0.95 * gap + 0.5 * (1 - gap)
        p[:, :W//2] = 0.05 * gap + 0.5 * (1 - gap)
        ds_vals.append(delta_spec(p))

    for i in range(len(ds_vals) - 1):
        assert ds_vals[i] >= ds_vals[i+1] - 1e-4, \
            f"Not monotone at index {i}: {ds_vals[i]:.4f} < {ds_vals[i+1]:.4f}"


def test_delta_spec_values_nonnegative():
    """Δ̂_spec = Var[H_b] must be non-negative."""
    np.random.seed(0)
    for _ in range(20):
        p = np.random.rand(32, 32).astype(np.float32)
        assert delta_spec(p) >= 0
