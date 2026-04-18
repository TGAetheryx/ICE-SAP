"""
Ablation study (ASPT Table 2).

Full ICE-SAP: 91.0 ± 1.2%
w/o Meta-Net (σ fixed=1):       85.1 ± 1.6%  (−6.9 pp)
w/o boundary weighting (BCE):   82.3 ± 2.1%  (−8.7 pp)
Meta-Net: NDSI only:            87.4 ± 1.4%  (−3.6 pp)
Meta-Net: TIR+GLCM only:        88.2 ± 1.3%  (−2.8 pp)
"""
import numpy as np
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))


# Paper results (Table 2, ASPT)
ABLATION_RESULTS = {
    "Full ICE-SAP":                         (91.0, 1.2,  0.0),
    "w/o Meta-Net (σ_meta fixed=1)":        (85.1, 1.6, -6.9),
    "w/o boundary weighting (uniform BCE)": (82.3, 2.1, -8.7),
    "Meta-Net: NDSI entropy only":          (87.4, 1.4, -3.6),
    "Meta-Net: TIR+GLCM only":              (88.2, 1.3, -2.8),
}


def print_ablation_table():
    print("\n" + "="*72)
    print(f"  ASPT Table 2 — Ablation Study (Breiðamerkurjökull, n=30 scenes)")
    print("="*72)
    print(f"  {'Variant':<42} {'B-IoU (%)':>12} {'vs. full':>10}")
    print("-"*72)
    for name, (mean, std, delta) in ABLATION_RESULTS.items():
        delta_str = f"{delta:+.1f} pp" if delta != 0 else "—"
        print(f"  {name:<42} {mean:.1f}±{std:.1f}   {delta_str:>10}")
    print("-"*72)
    print("\n  Key insights:")
    print("  • 8.7 pp gap (uniform→full) isolates boundary-weighting contribution")
    print("  • Additional 1.8 pp from dynamic Meta-Net (adaptive σ̂_meta)")
    print("  • All three Meta-Net input groups contribute meaningfully")


def simulate_ablation_variant(
    variant: str,
    n_scenes: int = 30,
    seed: int = 42,
) -> float:
    """
    Simulate boundary-IoU for an ablation variant.
    Returns a value close to the paper result with synthetic noise.
    """
    np.random.seed(seed)
    mean, std, _ = ABLATION_RESULTS[variant]
    # 5-fold cross-validation simulation
    fold_results = np.random.normal(mean / 100, std / 100, 5)
    return float(np.clip(fold_results, 0, 1).mean() * 100)


if __name__ == "__main__":
    print_ablation_table()
    print("\n[Simulated cross-validation results]")
    for variant in ABLATION_RESULTS:
        val = simulate_ablation_variant(variant)
        paper_val = ABLATION_RESULTS[variant][0]
        print(f"  {variant[:42]:<42} simulated={val:.1f}%  paper={paper_val:.1f}%")
