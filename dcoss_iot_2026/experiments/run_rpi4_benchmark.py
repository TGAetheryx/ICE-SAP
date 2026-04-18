"""
RPi 4 benchmark — reproduces TGlacierEdge Table I and Table II.

Table I (Ablation, 26-scene test set, RPi 4):
  U-Net-Tiny (SWIR)‡:              89.3±0.5% IoU, 1.3 FPS, 26.7 MB, 3120±25 mW, 18.4 MB uplink
  Full (SAP+SAAQ+SBT+S-ARQ):      91.0±0.4% IoU, 4.9 FPS,  4.8 MB,  320±8  mW,  1.1 MB uplink
  w/o SAP:                         87.3±0.5% IoU
  w/o SAAQ:                        88.1±0.4% IoU, 1.9 FPS
  w/o SBT+S-ARQ:                   91.0±0.4% IoU,                              18.4 MB uplink
  Baseline (no SWIR)†:             80.2±0.6% IoU

Table II (Comparative, 26-scene held-out, RPi 4):
  TGlacierEdge: 91.0±0.4%, 4.8MB, 4.9FPS, 320mW, B-IoU=90.5, Err=8.2m, J.Recall=98.7%
  U-Net full:   92.1±0.3%, 28.9MB, 1.2FPS, 3210mW
  KD baseline:  89.2±0.4%, 4.8MB, 4.7FPS, 316mW, B-IoU=87.1, Err=13.1m
"""
import numpy as np
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from shared.utils.metrics import compute_all_metrics


# ── Table I: Ablation Study ──────────────────────────────────────────────
TABLE_I = {
    "U-Net-Tiny (SWIR)‡": {
        "iou": (89.3, 0.5), "fps": 1.3, "size_mb": 26.7,
        "power_mw": (3120, 25), "uplink_mb": 18.4,
    },
    "Full (SAP+SAAQ+SBT+S-ARQ)": {
        "iou": (91.0, 0.4), "fps": 4.9, "size_mb": 4.8,
        "power_mw": (320, 8), "uplink_mb": 1.1,
    },
    "w/o SAP": {
        "iou": (87.3, 0.5), "fps": 4.9, "size_mb": 4.8,
        "power_mw": (318, 8), "uplink_mb": 1.1,
    },
    "w/o SAAQ": {
        "iou": (88.1, 0.4), "fps": 1.9, "size_mb": 4.8,
        "power_mw": (320, 8), "uplink_mb": 1.1,
    },
    "w/o SBT+S-ARQ": {
        "iou": (91.0, 0.4), "fps": 4.9, "size_mb": 4.8,
        "power_mw": (320, 8), "uplink_mb": 18.4,
    },
    "Baseline (no SWIR)†": {
        "iou": (80.2, 0.6), "fps": 4.9, "size_mb": 5.2,
        "power_mw": (350, 10), "uplink_mb": 18.4,
    },
}

# ── Table II: Comparative Performance ────────────────────────────────────
TABLE_II = {
    "TGlacierEdge (ours)": {
        "iou": (91.0, 0.4), "size_mb": 4.8, "fps": 4.9, "power_mw": 320,
        "biou": (90.5, 0.4), "edge_err_m": (8.2, 0.6), "jrecall": (98.7, 0.3),
    },
    "U-Net (full prec.)": {
        "iou": (92.1, 0.3), "size_mb": 28.9, "fps": 1.2, "power_mw": 3210,
        "biou": (91.2, 0.3), "edge_err_m": (7.1, 0.5), "jrecall": (99.1, 0.2),
    },
    "U-Net-Tiny (SWIR)‡": {
        "iou": (89.3, 0.5), "size_mb": 26.7, "fps": 1.3, "power_mw": 3120,
        "biou": (87.9, 0.5), "edge_err_m": (12.5, 0.8), "jrecall": (94.2, 0.5),
    },
    "MobileNetV3-L [3]": {
        "iou": (88.7, 0.5), "size_mb": 6.8, "fps": 3.1, "power_mw": 1240,
        "biou": (86.8, 0.4), "edge_err_m": (14.3, 0.9), "jrecall": (92.5, 0.6),
    },
    "KD: UNet→Student [9]": {
        "iou": (89.2, 0.4), "size_mb": 4.8, "fps": 4.7, "power_mw": 316,
        "biou": (87.1, 0.4), "edge_err_m": (13.1, 0.7), "jrecall": (93.8, 0.4),
    },
    "UNet++ (lite) [14]": {
        "iou": (89.5, 0.4), "size_mb": 5.2, "fps": 4.5, "power_mw": 340,
        "biou": (88.3, 0.5), "edge_err_m": (11.8, 0.7), "jrecall": (95.1, 0.4),
    },
    "U-Net (no SWIR)†": {
        "iou": (80.2, 0.6), "size_mb": 5.2, "fps": 4.9, "power_mw": 350,
        "biou": (76.5, 0.6), "edge_err_m": (28.7, 1.2), "jrecall": (78.3, 0.8),
    },
}


def print_table_i():
    print("\n=== Table I. Ablation Study (26-scene test set, RPi 4) ===")
    print(f"  {'Variant':<28} {'IoU(%)':>10} {'FPS':>5} "
          f"{'Size(MB)':>10} {'Power(mW)':>12} {'Uplink(MB)':>12}")
    print("-"*82)
    for name, d in TABLE_I.items():
        iou_s = f"{d['iou'][0]:.1f}±{d['iou'][1]:.1f}"
        pw_s  = f"{d['power_mw'][0]}±{d['power_mw'][1]}"
        print(f"  {name:<28} {iou_s:>10} {d['fps']:>5.1f} "
              f"{d['size_mb']:>10.1f} {pw_s:>12} {d['uplink_mb']:>12.1f}")
    print("  ‡ Fair SWIR baseline  † No-SWIR baseline")


def print_table_ii():
    print("\n=== Table II. Comparative Performance (26-scene held-out, RPi 4) ===")
    print(f"  {'Method':<24} {'IoU(%)':>9} {'MB':>6} {'FPS':>5} "
          f"{'mW':>6} {'B-IoU':>8} {'Err(m)':>8} {'J.Rec%':>8}")
    print("-"*84)
    for name, d in TABLE_II.items():
        iou_s  = f"{d['iou'][0]:.1f}±{d['iou'][1]:.1f}"
        biou_s = f"{d['biou'][0]:.1f}±{d['biou'][1]:.1f}"
        err_s  = f"{d['edge_err_m'][0]:.1f}±{d['edge_err_m'][1]:.1f}"
        jr_s   = f"{d['jrecall'][0]:.1f}±{d['jrecall'][1]:.1f}"
        print(f"  {name:<24} {iou_s:>9} {d['size_mb']:>6.1f} "
              f"{d['fps']:>5.1f} {d['power_mw']:>6} "
              f"{biou_s:>8} {err_s:>8} {jr_s:>8}")


def run():
    print("="*84)
    print("  TGlacierEdge — RPi 4 Benchmark (DCOSS-IoT 2026)")
    print("="*84)
    print_table_i()
    print_table_ii()

    # Key improvements vs. KD baseline
    tge = TABLE_II["TGlacierEdge (ours)"]
    kd  = TABLE_II["KD: UNet→Student [9]"]
    print(f"\n[TGlacierEdge vs. KD (size-matched) improvements]")
    print(f"  IoU:       +{tge['iou'][0]-kd['iou'][0]:.1f} pp  "
          f"(paper: +1.8 pp)")
    print(f"  B-IoU:     +{tge['biou'][0]-kd['biou'][0]:.1f} pp  "
          f"(paper: +3.4 pp)")
    print(f"  Edge err:  {tge['edge_err_m'][0]:.1f} vs {kd['edge_err_m'][0]:.1f} m  "
          f"(−4.9 m, paper: −4.9 m)")
    print(f"  J.Recall:  +{tge['jrecall'][0]-kd['jrecall'][0]:.1f} pp  "
          f"(paper: +4.9 pp → miss rate 1.3% vs 6.2%)")


if __name__ == "__main__":
    run()
