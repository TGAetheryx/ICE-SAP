"""
Generate Figure 1: ICE-SAP System Architecture diagram.

Reproduces ICE-SAP Fig. 1 (TG Aetheryx v5.0 System Architecture):
  Training Stage:   Boundary-Weighted Loss + Meta-Net + Tang Field W(x)
  Deployment Stage: L1 Ghost / L2 Meta / L3 Core + Safety Fallback
  Communication:    SBT tile selection + S-ARQ
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyArrowPatch
from pathlib import Path


def plot_system_architecture(output_path: str = "figures/output/architecture.png"):
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(16, 9))
    ax.set_xlim(0, 16)
    ax.set_ylim(0, 9)
    ax.axis("off")

    def box(x, y, w, h, label, sublabel="", color="steelblue", alpha=0.85,
            fontsize=9):
        rect = mpatches.FancyBboxPatch((x, y), w, h,
                                        boxstyle="round,pad=0.1",
                                        facecolor=color, alpha=alpha,
                                        edgecolor="white", linewidth=1.5)
        ax.add_patch(rect)
        ax.text(x + w/2, y + h/2 + (0.15 if sublabel else 0),
                label, ha="center", va="center",
                fontsize=fontsize, fontweight="bold", color="white")
        if sublabel:
            ax.text(x + w/2, y + h/2 - 0.25, sublabel, ha="center",
                    va="center", fontsize=7.5, color="white", style="italic")

    def arrow(x1, y1, x2, y2):
        ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                    arrowprops=dict(arrowstyle="->", color="grey",
                                   lw=1.5, connectionstyle="arc3,rad=0.1"))

    # Title
    ax.text(8, 8.5, "TG Aetheryx v5.0 System Architecture",
            ha="center", fontsize=14, fontweight="bold")

    # Training Stage
    ax.text(2.2, 7.7, "Training Stage", ha="center", fontsize=10,
            color="#2c3e50", fontweight="bold")
    box(0.3, 5.8, 3.8, 1.4, "Boundary-Weighted Loss",
        "L_geo = CE + Dice (W-weighted)", "#2980b9")
    box(0.3, 3.8, 3.8, 1.6, "Meta-Net",
        "NDSI/Thermal/GLCM→σ_meta\n1.2 KB, <2 mW", "#1abc9c")
    box(0.3, 2.0, 3.8, 1.5, "Tang Field W(x)",
        "exp(−d/σ_meta)·Φ(Entropy)\nMDL-optimal", "#8e44ad")

    # Deployment Stage
    ax.text(8.0, 7.7, "Deployment Stage", ha="center", fontsize=10,
            color="#2c3e50", fontweight="bold")
    box(5.8, 6.2, 4.5, 1.1, "L3 Core Execution",
        "320 mW | Triggered | 2% duty\nSAP-compressed U-Net", "#e74c3c")
    box(5.8, 4.6, 4.5, 1.2, "L2 Meta Inference",
        "80 mW | 0.1 Hz | 8% duty\nMeta-Net → σ_meta, W(x)", "#e67e22")
    box(5.8, 3.0, 4.5, 1.2, "L1 Ghost Sensing",
        "15 mW | 1 Hz | 90% duty\n120 KB micro-model", "#27ae60")
    box(5.8, 1.5, 4.5, 1.1, "Safety Fallback",
        "Life-Sign Mode (vector edge)", "#7f8c8d")

    # Communication
    ax.text(13.5, 7.7, "Communication", ha="center", fontsize=10,
            color="#2c3e50", fontweight="bold")
    box(11.5, 5.8, 4.0, 1.4, "SBT",
        "W(x)>τ tile selection\n93.8% uplink reduction", "#2980b9")
    box(11.5, 3.8, 4.0, 1.6, "S-ARQ",
        "RS(15,9) + 3 retransmits\n20% loss recovery", "#8e44ad")

    # Power summary
    ax.text(8.0, 1.0,
            "P_avg = 0.9×15 + 0.08×80 + 0.02×320 = 23.9 mW  →  24.6-day battery (4000 mAh)",
            ha="center", fontsize=9, color="#2c3e50",
            bbox=dict(boxstyle="round", facecolor="#ecf0f1", alpha=0.9))

    # Arrows: W(x) connects all stages
    arrow(4.1, 2.75, 5.8, 2.75)
    arrow(4.1, 5.0, 5.8, 5.0)
    arrow(10.3, 6.5, 11.5, 6.5)
    arrow(10.3, 4.5, 11.5, 4.5)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"Architecture figure saved → {output_path}")
    plt.close()


if __name__ == "__main__":
    plot_system_architecture()
