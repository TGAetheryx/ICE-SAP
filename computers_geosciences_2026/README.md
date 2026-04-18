# computers_geosciences_2026

**Paper:** "Glacier calving front delineation and early warning using boundary-aware
deep learning with spectral surrogate monitoring"

**Journal:** Computers & Geosciences (2026)

---

## Overview

This directory contains all code for the ASPT paper, which introduces:

1. **Arithmetic Fisher Manifold** — models glacier boundary state space as a
   non-Euclidean manifold with Fisher Information Metric.

2. **Perceptual Operator Ω̂** — self-adjoint operator on L²(M, d vol_g) whose
   spectral gap Δspec = λ₁ − λ₀ encodes structural stability.

3. **Spectral Collapse Hypothesis** — Δspec(t) → 0 as t → t_c (calving event).

4. **Δ̂_spec = Var[H_b]** — computationally tractable surrogate for early-warning
   monitoring. Detects calving precursors 12–24 h in advance.

5. **ICE-SAP segmentation pipeline** with Meta-Net (convolutional variant).

---

## Key Results

| Metric | Breiðamerkurjökull | Skeiðarárjökull | Combined |
|--------|-------------------|-----------------|---------|
| Boundary-IoU | 91.0 ± 1.2% | 90.1 ± 1.6% | — |
| Improvement vs. U-Net | +8.7 pp | +8.9 pp | — |
| Detection rate (≥24h) | 11/14 (78.6%) | 10/12 (83.3%) | 21/26 (80.8%) |
| Median lead time | 18.3 h | 17.1 h | 17.8 h |
| False alarm rate | 4.3% | 2.4% | 3.4% |
| Fisher exact test | p = 0.003 | p = 0.004 | p < 0.001 |

---

## Usage

```bash
# Run Breiðamerkurjökull experiment
python experiments/run_breidamerkur.py

# Run Skeiðarárjökull experiment
python experiments/run_skeidarar.py

# Cross-glacier analysis (Table 1b)
python experiments/cross_glacier_analysis.py

# Ablation study (Table 2)
python experiments/ablation_study.py

# Sensitivity analysis (Table 3 sensitivity row)
python experiments/sensitivity_analysis.py

# Reproduce Figure 1 (Δ̂_spec time series)
python figures/generate_figure1.py

# Synthetic validation of Δ̂_spec vs Δspec (Supplementary S2)
python validation/synthetic_validation.py
```

---

## Data

- Sentinel-2 L2A: Google Earth Engine (Copernicus Programme)
- Calving events: NASA MEaSUREs ITS_LIVE glacier velocity data
- Ground truth: GLIMS glacier database (glims.org)
- Annotations: 2 independent annotators (inter-annotator B-IoU 96.4±0.8%)

---

## Config

See `config.yaml` for all hyperparameters.
