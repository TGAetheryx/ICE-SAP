"""
Synthetic validation of Δ̂_spec ≈ Var[H_b] as spectral gap surrogate.

ASPT Supplementary S2 results:
  ε_max = 0.02, relative error ≤ 3.5%, PCC = 0.998, SCC = 1.0
  across 128,000 simulated samples spanning multiple manifold dimensions,
  stress levels, and noise conditions.

The validation constructs synthetic Bernoulli probability fields with
known spectral gap properties and verifies that Var[H_b] tracks the
theoretical spectral gap accurately.
"""
import numpy as np
from scipy.stats import pearsonr, spearmanr
from typing import Tuple, List
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from shared.inference.entropy import boundary_entropy_field, delta_spec


def simulate_probability_field(
    H: int,
    W: int,
    spectral_gap: float,
    noise_std: float = 0.02,
    rng: np.random.Generator = None,
) -> np.ndarray:
    """
    Generate a synthetic probability map with a controlled spectral gap.

    When spectral_gap = 1.0: perfectly bimodal (sharp boundary) → high Var[H_b].
    When spectral_gap = 0.0: uniform Bernoulli(0.5) → low Var[H_b].

    Interpolation: P(x) = gap·P_bimodal + (1−gap)·0.5 + noise.

    Args:
        H, W:          Spatial dimensions.
        spectral_gap:  ∈ [0, 1] — 1 = stable, 0 = collapsed.
        noise_std:     Gaussian noise std.
        rng:           NumPy random generator.

    Returns:
        prob_map: (H, W) float32 ∈ [0, 1].
    """
    if rng is None:
        rng = np.random.default_rng(0)

    # Bimodal base: sharp vertical boundary at centre
    P_bimodal = np.zeros((H, W), dtype=np.float32)
    P_bimodal[:, W // 2:] = 0.95
    P_bimodal[:, :W // 2] = 0.05
    # Soft boundary zone ±2 px
    boundary_zone = slice(W // 2 - 2, W // 2 + 2)
    P_bimodal[:, boundary_zone] = 0.5

    # Blend bimodal and uniform
    P_uniform = np.full((H, W), 0.5, dtype=np.float32)
    P = spectral_gap * P_bimodal + (1.0 - spectral_gap) * P_uniform

    # Add noise
    noise = rng.standard_normal((H, W)).astype(np.float32) * noise_std
    P = np.clip(P + noise, 0.01, 0.99)
    return P


def run_synthetic_validation(
    n_samples: int = 128_000,
    H: int = 32,
    W: int = 32,
    noise_levels: List[float] = None,
    gap_levels: int = 50,
    seed: int = 42,
) -> dict:
    """
    Full synthetic validation of Δ̂_spec surrogate.

    Reproduces ASPT Supplementary S2.

    Args:
        n_samples:    Total number of synthetic probability fields.
        H, W:         Spatial dimensions of each field.
        noise_levels: List of noise std values to test.
        gap_levels:   Number of spectral gap values in [0, 1].
        seed:         Random seed.

    Returns:
        dict with pcc, scc, max_abs_error, max_rel_error.
    """
    if noise_levels is None:
        noise_levels = [0.01, 0.02, 0.05, 0.10]

    rng = np.random.default_rng(seed)
    true_gaps = []
    surrogate_vals = []

    samples_per_config = max(1, n_samples // (gap_levels * len(noise_levels)))

    for noise in noise_levels:
        for gap_idx in range(gap_levels):
            gap = float(gap_idx) / (gap_levels - 1)   # 0 → 1
            for _ in range(samples_per_config):
                P = simulate_probability_field(H, W, gap, noise, rng)
                ds = delta_spec(P)
                true_gaps.append(gap)
                surrogate_vals.append(ds)

    true_gaps = np.array(true_gaps, dtype=np.float64)
    surrogate_vals = np.array(surrogate_vals, dtype=np.float64)

    # Normalise both to [0, 1] for comparison
    tg_norm = (true_gaps - true_gaps.min()) / (true_gaps.ptp() + 1e-8)
    sv_norm = (surrogate_vals - surrogate_vals.min()) / \
              (surrogate_vals.ptp() + 1e-8)

    abs_errors = np.abs(tg_norm - sv_norm)
    rel_errors = abs_errors / (tg_norm + 1e-8)

    pcc, pcc_p = pearsonr(tg_norm, sv_norm)
    scc, scc_p = spearmanr(tg_norm, sv_norm)

    results = {
        "n_samples": len(true_gaps),
        "pcc": float(pcc),
        "pcc_p": float(pcc_p),
        "scc": float(scc),
        "scc_p": float(scc_p),
        "max_abs_error": float(abs_errors.max()),
        "mean_abs_error": float(abs_errors.mean()),
        "max_rel_error_pct": float(rel_errors.max() * 100),
        # Paper targets: ε_max=0.02, rel≤3.5%, PCC=0.998, SCC=1.0
        "pass_pcc": pcc >= 0.99,
        "pass_scc": scc >= 0.99,
        "pass_abs": abs_errors.max() <= 0.05,
    }
    return results


if __name__ == "__main__":
    print("Running synthetic validation (may take ~30s)...")
    # Use smaller n for quick test
    r = run_synthetic_validation(n_samples=10_000, H=32, W=32)
    print(f"\n=== Supplementary S2 Validation ===")
    print(f"N samples:        {r['n_samples']:,}")
    print(f"PCC:              {r['pcc']:.4f}  (paper: 0.998)")
    print(f"SCC:              {r['scc']:.4f}  (paper: 1.0)")
    print(f"Max abs error:    {r['max_abs_error']:.4f}  (paper: ε_max=0.02)")
    print(f"Max rel error:    {r['max_rel_error_pct']:.2f}%  (paper: ≤3.5%)")
    print(f"\nPASS PCC≥0.99:   {r['pass_pcc']}")
    print(f"PASS SCC≥0.99:   {r['pass_scc']}")
    print(f"PASS abs≤0.05:   {r['pass_abs']}")
