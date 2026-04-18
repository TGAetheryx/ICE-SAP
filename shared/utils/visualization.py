"""Visualization utilities for ICE-SAP."""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from typing import Optional
from pathlib import Path


def plot_segmentation_result(
    image_rgb: np.ndarray,
    pred_mask: np.ndarray,
    gt_mask: Optional[np.ndarray] = None,
    W_field: Optional[np.ndarray] = None,
    title: str = "TGlacierEdge Segmentation",
    output_path: Optional[str] = None,
):
    n_panels = 3 if gt_mask is not None else 2
    if W_field is not None:
        n_panels += 1

    fig, axes = plt.subplots(1, n_panels, figsize=(5 * n_panels, 5))
    if n_panels == 1:
        axes = [axes]

    axes[0].imshow(np.clip(image_rgb, 0, 1))
    axes[0].set_title("RGB Input")
    axes[0].axis("off")

    axes[1].imshow(pred_mask, cmap="Blues", vmin=0, vmax=1)
    axes[1].set_title("Prediction")
    axes[1].axis("off")

    idx = 2
    if gt_mask is not None:
        axes[idx].imshow(gt_mask, cmap="Greens", vmin=0, vmax=1)
        axes[idx].set_title("Ground Truth")
        axes[idx].axis("off")
        idx += 1

    if W_field is not None:
        axes[idx].imshow(W_field, cmap="hot", vmin=0, vmax=1)
        axes[idx].set_title("Boundary Decay Field W(x)")
        axes[idx].axis("off")

    plt.suptitle(title, fontsize=12, fontweight="bold")
    plt.tight_layout()

    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=120, bbox_inches="tight")
    plt.close()


def plot_delta_spec_series(
    timestamps_h: np.ndarray,
    delta_series: np.ndarray,
    calving_times_h: list = None,
    alert_times_h: list = None,
    output_path: Optional[str] = None,
):
    fig, ax = plt.subplots(figsize=(12, 4))
    valid = ~np.isnan(delta_series)
    ax.plot(timestamps_h[valid] / 24, delta_series[valid],
            "o-", color="tomato", ms=5, lw=1.5, label="Δ̂_spec")
    if calving_times_h:
        for ct in calving_times_h:
            ax.axvline(ct / 24, color="red", lw=2, ls="-", label="Calving")
    if alert_times_h:
        for at in alert_times_h:
            ax.axvline(at / 24, color="orange", lw=1.5, ls="--", label="Alert")
    ax.set_xlabel("Days")
    ax.set_ylabel("Δ̂_spec = Var[H_b]")
    ax.set_title("Spectral Gap Surrogate — Early-Warning Monitor")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=120, bbox_inches="tight")
    plt.close()
