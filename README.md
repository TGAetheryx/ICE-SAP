# ICE-SAP: Integrated Cryosphere Edge Sensing — Adaptive Pipeline

This repository contains the code and data for three companion publications on
polar glacier monitoring using IoT edge AI:

---

## Configuration Matrix

This repository contains two deployment configurations of TGlacierEdge,
corresponding to **snapshots at different points along the SAP pruning Pareto
curve** (ICE-SAP §3.3, Fig. 2). Both are functionally valid operating points;
readers should use the directory matching the paper they are referencing.

| Directory | Model size | SAAQ calibration | Pareto operating point | Reported in |
|---|---|---|---|---|
| `dcoss_iot_2026/` | **4.8 MB** | 512 patches / 40–60 iter | Earlier compression variant (~50% channel retention) | Paper 3 — DCOSS-IoT 2026 (poster) |
| `eia_nordic_2026/` | **1.8 MB** | 2 000 patches / 200 iter | Final Pareto-optimal point (16.6% retention) | Paper 2 — EIA Nordic 2026 (long paper) |
| `computers_geosciences_2026/` | n/a (theory) | n/a | — | Paper 1 — Computers & Geosciences 2026 |

The 1.8 MB configuration is obtained from the 4.8 MB configuration by one
additional pass of WSA-guided pruning down to the Pareto-optimal 16.6%
channel-retention point, with correspondingly expanded SAAQ calibration. The
two sizes are therefore the same model at different stages of the compression
trajectory, not independent models.

---

## Repository Structure

```
ICE-SAP/
├── shared/                        # Core modules shared across all three papers
│   ├── model/                     # U-Net, MobileNet encoder, Meta-Net
│   ├── inference/                 # Boundary Decay Field, entropy, delta_spec
│   └── utils/                     # Preprocessing, metrics, visualisation
├── computers_geosciences_2026/    # Paper 1 — Computers & Geosciences
├── eia_nordic_2026/               # Paper 2 — EIA Nordic 2026
├── dcoss_iot_2026/                # Paper 3 — DCOSS-IoT 2026
├── examples/                      # Quick-start scripts and sample data
├── data/                          # Data download instructions
├── docs/                          # Installation, usage, hardware setup
│
├── unet_float32.onnx              # Paper 2/3 — U-Net FP32 ONNX export
├── unet_int8.tflite               # Paper 2/3 — U-Net INT8 TFLite (SAAQ)
├── meta_net_weights.pth           # Paper 2 — Meta-Net trained weights (1.2 KB)
├── unet_model_config.yaml         # Architecture / input / preprocessing config
└── unet_int8_quantization_params.json   # Per-channel INT8 scales & zero-points
```

---

## Papers

### Paper 1 — Computers & Geosciences 2026
**"Glacier calving front delineation and early warning using boundary-aware deep
learning with spectral surrogate monitoring"**

- `computers_geosciences_2026/`
- Introduces the **Arithmetic Fisher Manifold**, the **Perceptual Operator Ω̂**,
  and the **Spectral Collapse Hypothesis**
- Monitors **Δ̂_spec = Var[H_b]** as a calving early-warning index
- Validated on **Breiðamerkurjökull** (122 scenes, 14 events) and
  **Skeiðarárjökull** (106 scenes, 12 events)
- Combined detection: **21/26 events**, median lead time **17.8 h**, false alarm
  rate **3.4%** (Fisher exact test p < 0.001)
- Boundary-IoU improvement: **+8.7–8.9 pp** over standard U-Net

### Paper 2 — EIA Nordic 2026 (Long paper)
**"ICE-SAP: An Energy-Centric Framework for High-Efficiency Sensing of Polar
Glacier Systems"**

- `eia_nordic_2026/`
- Addresses the **energy–accuracy–communication trilemma**
- Three technical pillars unified by W(x):
  - **SAP** (Sensing-Aware Pruning with WSA): 83.4% compression, −0.5 pp IoU
  - **SAAQ** (Spectral-Aware Asymmetric Quantization): INT8, 4.9 FPS, zero loss
  - **SBT+S-ARQ** (Semantic Block Transmission): 93.8% uplink reduction
- **Meta-Net** (1.2 KB, <2 mW): predicts σ_meta from NDSI/thermal/GLCM
- Cascaded startup (L1/L2/L3): **23.9 mW average**, 24.6-day battery life
- Validated at **56 sites**, thermal chamber −45°C to +13°C, 500,000+ patches

### Paper 3 — DCOSS-IoT 2026 (Poster)
**"TGlacierEdge: Boundary-Aware Deep Compression for Persistent IoT Glacier
Delineation at the Polar Edge"**

- `dcoss_iot_2026/`
- RPi 4 hardware realisation of ICE-SAP
- **91.0% IoU at 320 mW peak (23.9 mW average)**
- **9.75× power reduction** vs. fair SWIR-preserving baseline
- 24.6-day battery life (4000 mAh, 30-min duty cycle)
- LoRa physical-layer validation: ≥98.3% 24-h link uptime at −27°C
- Multi-node SBT scaling: 3/4/8 nodes, IoU loss ≤0.3 pp

---

## Relationship Between Papers

```
                    Shared spatial prior: W(x) = exp(−d(x,∂G)/σ_meta) · Φ(Entropy)
                              ↑
        ┌─────────────────────┼──────────────────────┐
        │                     │                      │
  Paper 1 (Theory)      Paper 2 (System)       Paper 3 (Edge)
  Arithmetic Fisher     ICE-SAP framework      TGlacierEdge RPi4
  Manifold + ASPT       56-site validation     Deployment poster
  Δ̂_spec calving EW    Energy trilemma        LoRa + fault tolerance
```

**σ_meta** is the conformal scale field (Paper 1 theory) / bandwidth parameter
(Papers 2–3 engineering). Both interpretations modulate W(x) at boundary regions.

---

## Quick Start

```bash
git clone https://github.com/TGAetheryx/ICE-SAP.git
cd ICE-SAP
pip install -r requirements.txt

# Run example inference (uses a synthetic 6-channel 128x128 patch by default)
python examples/run_example.py
```

---

## Installation

See `docs/installation.md` for full instructions.

**Requirements (Python 3.9+):**
```
torch>=2.0.0
torchvision>=0.15.0
onnxruntime>=1.16.0
numpy>=1.24.0
scipy>=1.10.0
scikit-image>=0.20.0
rasterio>=1.3.0
matplotlib>=3.7.0
```

---

## Pre-trained Model Artifacts

The repository root ships the trained model artifacts used in Papers 2 and 3, so
you can reproduce inference without re-training:

| File | Purpose | Consumed by |
|---|---|---|
| `unet_float32.onnx` | U-Net FP32 ONNX export | ONNX Runtime baseline benchmark |
| `unet_int8.tflite` | U-Net INT8 TFLite (after SAAQ) | RPi 4 edge inference |
| `meta_net_weights.pth` | Meta-Net weights (1.2 KB, <2 mW at inference) | σ_meta predictor |
| `unet_model_config.yaml` | Architecture / input channels / preprocessing | Loader for both FP32 and INT8 |
| `unet_int8_quantization_params.json` | Per-channel INT8 scales & zero-points | SAAQ runtime dequantisation |

These are binary/config artefacts — treat them as read-only. Training code lives
in the per-paper directories.

---

## Data

All Sentinel-2 Level-2A imagery is publicly accessible via
[Google Earth Engine](https://earthengine.google.com/).
Ground-truth masks are from the [GLIMS glacier database](https://www.glims.org/).

See `data/README.md` for download instructions.

---

## Citation

If you use this code, please cite the relevant paper(s):

```bibtex
@inproceedings{tang2026tglacier,
  title={{TGlacierEdge}: Boundary-Aware Deep Compression for Persistent {IoT}
         Glacier Delineation at the Polar Edge},
  booktitle={22nd International Conference on Distributed Computing in Smart
             Systems and the Internet of Things (DCOSS-IoT)},
  year={2026}
}

@article{tang2026icesap,
  title={{ICE-SAP}: An Energy-Centric Framework for High-Efficiency Sensing of
         Polar Glacier Systems},
  journal={EIA Nordic 2026},
  year={2026}
}

@article{tang2026aspt,
  title={Glacier calving front delineation and early warning using
         boundary-aware deep learning with spectral surrogate monitoring},
  journal={Computers \& Geosciences},
  year={2026}
}
```

---

## License

MIT License. See `LICENSE` for details.

---

## Contact

School of Mathematics and Computational Science, Xiangtan University,
Xiangtan, Hunan, China 411105.
