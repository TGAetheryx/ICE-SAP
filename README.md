# ICE-SAP: Integrated Cryosphere Edge Sensing — Adaptive Pipeline

This repository contains the code and data for three companion publications on
polar glacier monitoring using IoT edge AI:

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
└── docs/                          # Installation, usage, hardware setup
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
git clone https://github.com/[to-be-confirmed]/ICE-SAP.git
cd ICE-SAP
pip install -r requirements.txt

# Run example inference
python examples/run_example.py --input examples/sample_input/test_tile_512x512.tif
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
