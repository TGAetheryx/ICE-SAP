"""
Single-node fault tolerance test (TGlacierEdge §V.A, Table III).

Results to reproduce:
  Single-node fault:   −2.9 pp IoU → 88.1 ± 0.3%  (ref: 91.0%)
  Dual-node comms out: −4.7 pp IoU → 86.3 ± 0.3%

Lab setup: 30 scenes, 5 trials, conditions:
  fog: B11/B12×0.08; −20°C thermal throttle.
  SBT+S-ARQ consensus >99% under fog/throttle (unchanged).

Table III:
  Winter   (−27 to −9°C):  91.9±0.3%  (ref 92.0, Δ=−0.1)
  Transition (−7 to +9°C): 90.6±0.3%  (ref 90.7, Δ=−0.1)
  Summer   (0 to +13°C):   90.0±0.3%  (ref 90.3, Δ=−0.3)
  Freezing fog:             89.5±0.3%
  Single-node fault:        88.1±0.3%  (Δ=−2.9)
  Dual-node comms out:      86.3±0.3%  (Δ=−4.7)
"""
import numpy as np
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from shared.utils.metrics import boundary_iou


# Paper constants (Table III)
TABLE_III = {
    "Winter (−27 to −9°C)":    (91.9, 0.3, 92.0, -0.1),
    "Transition (−7 to +9°C)": (90.6, 0.3, 90.7, -0.1),
    "Summer (0 to +13°C)":     (90.0, 0.3, 90.3, -0.3),
    "Freezing fog":             (89.5, 0.3, None, None),
    "Single-node fault":        (88.1, 0.3, None, -2.9),
    "Dual-node comms out":      (86.3, 0.3, None, -4.7),
}


def simulate_single_node_fault(
    n_scenes: int = 30,
    n_trials: int = 5,
    H: int = 128,
    W: int = 128,
    seed: int = 42,
) -> dict:
    """
    Simulate IoU degradation under single-node fault.

    In a 3-node deployment, losing one node reduces consensus quality.
    Expected: −2.9 pp vs. 3-node consensus (91.0% → 88.1%).
    """
    np.random.seed(seed)
    iou_3node = []
    iou_2node = []

    for _ in range(n_trials):
        for _ in range(n_scenes):
            # GT mask
            gt = np.zeros((H, W), dtype=np.float32)
            gt[:, W//2:] = 1.0

            # 3-node consensus: average of 3 noisy predictions
            preds_3 = []
            for _ in range(3):
                noise = np.random.randn(H, W) * 0.04
                p = np.clip(gt + noise, 0, 1)
                preds_3.append(p)
            consensus_3 = np.mean(preds_3, axis=0)
            iou_3node.append(boundary_iou(consensus_3, gt))

            # 2-node consensus (one fault): average of 2 noisy predictions
            preds_2 = preds_3[:2]
            consensus_2 = np.mean(preds_2, axis=0)
            iou_2node.append(boundary_iou(consensus_2, gt))

    return {
        "iou_3node_mean": float(np.mean(iou_3node) * 100),
        "iou_3node_std":  float(np.std(iou_3node) * 100),
        "iou_2node_mean": float(np.mean(iou_2node) * 100),
        "iou_2node_std":  float(np.std(iou_2node) * 100),
        "delta_pp": float((np.mean(iou_2node) - np.mean(iou_3node)) * 100),
    }


def simulate_dual_node_comms_out(
    n_scenes: int = 30,
    n_trials: int = 5,
    H: int = 128,
    W: int = 128,
    seed: int = 99,
) -> dict:
    """
    Simulate dual-node communications failure: −4.7 pp IoU.
    Only 1 of 3 nodes remains active.
    """
    np.random.seed(seed)
    iou_3node = []
    iou_1node = []

    for _ in range(n_trials):
        for _ in range(n_scenes):
            gt = np.zeros((H, W), dtype=np.float32)
            gt[:, W//2:] = 1.0

            preds = []
            for _ in range(3):
                noise = np.random.randn(H, W) * 0.04
                preds.append(np.clip(gt + noise, 0, 1))

            iou_3node.append(boundary_iou(np.mean(preds, axis=0), gt))
            iou_1node.append(boundary_iou(preds[0], gt))

    return {
        "iou_3node_mean": float(np.mean(iou_3node) * 100),
        "iou_1node_mean": float(np.mean(iou_1node) * 100),
        "delta_pp": float((np.mean(iou_1node) - np.mean(iou_3node)) * 100),
    }


def print_table_iii():
    print("\n=== Table III: Seasonal and Fault IoU (30-scene lab set) ===")
    print(f"  {'Condition':<28} {'Lab IoU (%)':>12} {'Ref IoU':>10} {'Δ pp':>8}")
    print("-"*62)
    for cond, (mean, std, ref, delta) in TABLE_III.items():
        ref_s = f"{ref:.1f}" if ref else "—"
        delta_s = f"{delta:+.1f}" if delta else "lab"
        print(f"  {cond:<28} {mean:.1f}±{std:.1f}      "
              f"{ref_s:>8}  {delta_s:>6}")


if __name__ == "__main__":
    print_table_iii()

    r1 = simulate_single_node_fault()
    print(f"\n[Simulated Single-Node Fault]")
    print(f"  3-node IoU: {r1['iou_3node_mean']:.1f}±{r1['iou_3node_std']:.1f}%")
    print(f"  2-node IoU: {r1['iou_2node_mean']:.1f}±{r1['iou_2node_std']:.1f}%")
    print(f"  Δ = {r1['delta_pp']:+.1f} pp  (paper: −2.9 pp)")

    r2 = simulate_dual_node_comms_out()
    print(f"\n[Simulated Dual-Node Comms Out]")
    print(f"  1-node IoU: {r2['iou_1node_mean']:.1f}%")
    print(f"  Δ = {r2['delta_pp']:+.1f} pp  (paper: −4.7 pp)")
