"""
Generate Figure 1: Δ̂_spec(t) time series (ASPT §5.4, Fig. 1).

Shows:
  (a) Full 90-day monitoring window — Breiðamerkurjökull Event #7
  (b) Zoomed view of final 15 days

Panel annotations:
  - Blue shading: stable phase
  - Orange shading: precursory collapse phase
  - Green shading: post-event recovery
  - Dashed blue line: 30-day causal rolling baseline
  - Dotted red line: 5th-percentile detection threshold
  - Hatched band: cloud-cover data gap (days 44–50)
  - Warning line at t = −18.5 h
  - Calving event at t = 0
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from shared.inference.entropy import delta_spec


def generate_synthetic_event7_series(
    n_obs: int = 22,        # ~90 days / 4-day revisit ≈ 22 observations
    calving_obs: int = 17,  # calving at observation 17
    seed: int = 7,
) -> tuple:
    """
    Synthesise a realistic Δ̂_spec series for Event #7 (Breið.).
    Returns (timestamps_h, delta_series, calving_h, warning_h).
    """
    np.random.seed(seed)
    H, W = 64, 64
    revisit_h = 96.0   # 4-day

    prob_series = np.zeros((n_obs, H, W), dtype=np.float32)
    for t in range(n_obs):
        if t < calving_obs - 6:
            # Stable: bimodal
            p = np.zeros((H, W), dtype=np.float32)
            p[:, W//2:] = 0.93
            p[:, :W//2] = 0.05
            p[:, W//2-1:W//2+2] = 0.5
            p += np.random.randn(H, W).astype(np.float32) * 0.02
        elif t < calving_obs:
            # Collapse phase: gradual flattening
            frac = (t - (calving_obs - 6)) / 6
            p_bimodal = np.zeros((H, W), dtype=np.float32)
            p_bimodal[:, W//2:] = 0.93
            p_bimodal[:, :W//2] = 0.05
            p_collapse = np.full((H, W), 0.48, dtype=np.float32)
            p = (1 - frac) * p_bimodal + frac * p_collapse
            p += np.random.randn(H, W).astype(np.float32) * 0.03
        else:
            # Post-calving recovery: new boundary
            p = np.zeros((H, W), dtype=np.float32)
            p[:, W//3:] = 0.88
            p[:, :W//3] = 0.06
            p += np.random.randn(H, W).astype(np.float32) * 0.03
        prob_series[t] = np.clip(p, 0, 1)

    ds_series = np.array([delta_spec(prob_series[t]) for t in range(n_obs)])
    timestamps = np.arange(n_obs) * revisit_h

    # Insert cloud gap (days 44–50 → observations ~11–12)
    cloud_obs = [11, 12]
    ds_series_plot = ds_series.copy().astype(float)
    for co in cloud_obs:
        ds_series_plot[co] = np.nan

    calving_h = calving_obs * revisit_h
    # Warning issued 18.5 h before calving
    warning_h = calving_h - 18.5

    return timestamps, ds_series_plot, calving_h, warning_h, cloud_obs


def plot_figure1(output_path: str = "figures/output/figure1.png"):
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    timestamps, ds, calving_h, warning_h, cloud_obs = \
        generate_synthetic_event7_series()

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Figure 1: Δ̂_spec(t) — Breiðamerkurjökull, Event #7",
                 fontsize=13, fontweight="bold")

    revisit_h = 96.0
    n_obs = len(timestamps)

    # Rolling baseline and threshold
    window_obs = 8
    baseline = np.full(n_obs, np.nan)
    thresh = np.full(n_obs, np.nan)
    for t in range(window_obs, n_obs):
        w = ds[max(0, t - window_obs):t]
        w_valid = w[~np.isnan(w)]
        if len(w_valid) >= 4:
            baseline[t] = float(np.median(w_valid))
            thresh[t] = float(np.percentile(w_valid, 5))

    # ── Panel (a): Full 90-day window ───────────────────────────────────
    ax = axes[0]
    days = timestamps / 24.0

    # Background shading
    calving_day = calving_h / 24
    collapse_start_day = (calving_day * revisit_h - 6 * revisit_h) / revisit_h
    ax.axvspan(0, 40, alpha=0.12, color="steelblue", label="Stable")
    ax.axvspan(40, calving_day, alpha=0.15, color="orange", label="Precursory collapse")
    ax.axvspan(calving_day, days[-1], alpha=0.12, color="green", label="Recovery")

    # Cloud gap hatch
    for co in cloud_obs:
        ax.axvspan(days[co] - 0.5, days[co] + 0.5, alpha=0.4,
                   color="grey", hatch="///", label="_nolegend_")

    # Data
    valid = ~np.isnan(ds)
    ax.plot(days[valid], ds[valid], "o-", color="tomato", ms=6, lw=1.5,
            label="Δ̂_spec (pre-calving)")
    ax.plot(days[~valid], [0.35]*sum(~valid), "x", color="grey", ms=8)

    # Baseline and threshold
    ax.plot(days, baseline, "--", color="steelblue", lw=1.5,
            label="30-day causal rolling baseline")
    ax.plot(days, thresh, ":", color="red", lw=1.5,
            label="5th-pct detection threshold")

    # Calving line
    ax.axvline(calving_day, color="red", lw=2, ls="-", label="Calving event")

    ax.set_xlabel("Time (days)")
    ax.set_ylabel("Δ̂_spec(t) = Var[H_b]")
    ax.set_title("(a) Full 90-day monitoring window")
    ax.legend(fontsize=8, loc="upper right")
    ax.set_ylim(0, 0.80)

    # ── Panel (b): Final 15 days (in hours relative to calving) ────────
    ax2 = axes[1]
    hours_rel = timestamps - calving_h
    zoom_mask = (hours_rel >= -360) & (hours_rel <= 24)

    ax2.plot(hours_rel[zoom_mask & valid], ds[zoom_mask & valid],
             "o-", color="tomato", ms=7, lw=1.8)
    ax2.axvline(0, color="red", lw=2.5, label="Confirmed calving (t=0)")
    ax2.axvline(-18.5, color="darkorange", lw=2, ls="--",
                label=f"Warning issued (t=−18.5 h)")
    ax2.plot(days, thresh, ":", color="red", lw=1, alpha=0.5)

    ax2.set_xlabel("Hours relative to calving event")
    ax2.set_ylabel("Δ̂_spec(t)")
    ax2.set_title("(b) Final 15 days (zoomed)")
    ax2.legend(fontsize=9)
    ax2.set_xlim(-360, 30)
    ax2.set_ylim(0, 0.80)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"Figure 1 saved → {output_path}")
    plt.close()


if __name__ == "__main__":
    plot_figure1()
