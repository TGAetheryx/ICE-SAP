"""
Figure 2: Cascaded startup state machine + average power breakdown (DCOSS-IoT).
Reproduces ICE-SAP Fig. 5: L1 Ghost (56%), L2 Meta (27%), L3 Core (27%).
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path
import numpy as np


def plot_cascade_power(output_path: str = "figures/output/cascade_power.png"):
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle("Cascaded Startup State Machine & Average Power Breakdown\n"
                 "P_avg = 23.9 mW", fontsize=12, fontweight="bold")

    # ── Panel (a): State machine ─────────────────────────────────────────
    ax = axes[0]
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.axis("off")
    ax.set_title("(a) Cascaded Startup State Machine")

    boxes = [
        (5, 8.0, "#27ae60", "L1 Ghost Sensing",
         "15 mW | 1 Hz | 90% duty\n120 KB micro-model"),
        (5, 5.5, "#e67e22", "L2 Meta Inference",
         "80 mW | 0.1 Hz | 8% duty\nMeta-Net → σ_meta"),
        (5, 3.0, "#e74c3c", "L3 Core Execution",
         "320 mW | Triggered | 2% duty\nSAP-compressed U-Net"),
        (5, 0.8, "#7f8c8d", "Safety Fallback",
         "Life-Sign Mode"),
    ]
    for x, y, col, label, sub in boxes:
        rect = mpatches.FancyBboxPatch((x-2.5, y-0.6), 5, 1.2,
                                        boxstyle="round,pad=0.1",
                                        facecolor=col, alpha=0.9,
                                        edgecolor="white", lw=1.5)
        ax.add_patch(rect)
        ax.text(x, y + 0.1, label, ha="center", va="center",
                fontsize=8.5, fontweight="bold", color="white")
        ax.text(x, y - 0.25, sub, ha="center", va="center",
                fontsize=7, color="white")

    for (y1, y2, label) in [(7.4, 6.1, "θ_up=0.35"), (4.9, 3.6, "θ_up=0.50"),
                             (2.4, 1.4, "entropy>Φ_crit")]:
        ax.annotate("", xy=(5, y2), xytext=(5, y1),
                    arrowprops=dict(arrowstyle="->", color="grey", lw=1.5))
        ax.text(5.8, (y1+y2)/2, label, fontsize=7, color="grey")

    # ── Panel (b): Power pie chart ────────────────────────────────────────
    ax2 = axes[1]
    sizes   = [13.5, 6.4, 6.4, 0.0]    # L1, L2, L3 contributions, comms
    colors  = ["#27ae60", "#e67e22", "#e74c3c", "#3498db"]
    labels  = ["L1 Ghost\n13.5 mW (56%)",
               "L2 Meta\n6.4 mW (27%)",
               "L3 Core\n6.4 mW (27%)",
               ""]
    explode = (0.05, 0.05, 0.05, 0)
    wedges, texts = ax2.pie(
        [13.5, 6.4, 6.4],
        labels=["L1 Ghost\n13.5 mW (56%)",
                "L2 Meta\n6.4 mW (27%)",
                "L3 Core\n6.4 mW (27%)"],
        colors=colors[:3],
        explode=explode[:3],
        startangle=90,
        textprops={"fontsize": 9},
    )
    ax2.set_title("(b) Average Power Breakdown\nP_avg = 23.9 mW")

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"Cascade power figure saved → {output_path}")
    plt.close()


if __name__ == "__main__":
    plot_cascade_power()
