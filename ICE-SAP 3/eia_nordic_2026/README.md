# eia_nordic_2026

**Paper:** "ICE-SAP: An Energy-Centric Framework for High-Efficiency Sensing of
Polar Glacier Systems"

**Venue:** EIA Nordic 2026

---

## Overview

ICE-SAP addresses the **energy–accuracy–communication trilemma** for polar IoT:

    min   E_inference + E_comm
    s.t.  IoU(θ,m,q) ≥ τ_acc
          PDR(τ) ≥ τ_comm
          ‖θ‖₀ ≤ κ   (sparsity)
          q ∈ {8,16}

Three design principles unified by the Boundary Decay Field W(x):

1. **SAP** — Sensing-Aware Pruning with WSA: 83.4% compression, −0.5 pp IoU
2. **SAAQ** — Spectral-Aware Asymmetric Quantization: INT8, 4.9 FPS, zero loss
3. **SBT+S-ARQ** — Semantic Block Transmission: 93.8% uplink reduction

Plus:
- **Meta-Net** (1.2 KB, <2 mW): σ_meta from NDSI/thermal/GLCM
- **Cascaded startup** (L1 Ghost / L2 Meta / L3 Core): 23.9 mW average
- **56-site validation**: 500,000+ patches, thermal chamber −45°C to +13°C

---

## Key Results

| Metric | Value |
|--------|-------|
| Overall IoU (56 sites) | 91.0 ± 1.2% |
| Boundary IoU | 87.8 ± 2.1% |
| Peak inference power | 356 ± 16 mW (at −45°C) |
| Average system power | 27.0 mW (cascaded) |
| Battery life (4000 mAh) | 22.8 days |
| Uplink reduction (SBT) | 93.8 ± 0.6% |
| SAP compression | 83.4% (−0.5 pp IoU) |
| Jökulhlaup recall | 96.5 ± 1.4% |

---

## Usage

```bash
# Run full 56-site validation
python experiments/run_56_sites.py

# Thermal chamber test (−45°C to +13°C)
python experiments/thermal_chamber_test.py

# Cold-start reliability (50 cycles at −45°C)
python experiments/cold_start_test.py

# Battery life simulation
python experiments/battery_life_simulation.py

# Cross-site generalisation (Table V)
python experiments/cross_site_generalization.py

# SAP Pareto front analysis (Fig. 2)
python figures/generate_figure2.py
```

---

## Config

See `config.yaml` for all hyperparameters.
