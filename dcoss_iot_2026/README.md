# dcoss_iot_2026

**Paper:** "TGlacierEdge: Boundary-Aware Deep Compression for Persistent IoT
Glacier Delineation at the Polar Edge"

**Venue:** DCOSS-IoT 2026 (22nd International Conference on Distributed Computing
in Smart Systems and the Internet of Things) — Poster

---

## Overview

TGlacierEdge is the **RPi 4 hardware realisation** of ICE-SAP, validated on
122 Sentinel-2 scenes (Vatnajökull, 2019–2023).

Key results vs. fair SWIR-preserving baseline (U-Net-Tiny float32):
  - **+1.7 pp IoU** (89.3% → 91.0%)
  - **9.75× power reduction** (3120 mW → 320 mW peak)
  - **4.1× throughput** (1.3 → 4.9 FPS)
  - **82.0% model compression**
  - **93.8% uplink reduction**

---

## Key Results (Table II, Physical RPi 4)

| Method | IoU (%) | Size (MB) | FPS | Power (mW) | B-IoU (%) | Edge Err (m) | J.Recall (%) |
|--------|---------|-----------|-----|-----------|-----------|--------------|--------------|
| TGlacierEdge (ours) | 91.0±0.4 | 4.8 | 4.9 | 320 | 90.5±0.4 | 8.2±0.6 | 98.7±0.3 |
| U-Net-Tiny (SWIR)‡ | 89.3±0.5 | 26.7 | 1.3 | 3120±25 | 87.9±0.5 | 12.5±0.8 | 94.2±0.5 |
| KD: UNet→Student | 89.2±0.4 | 4.8 | 4.7 | 316 | 87.1±0.4 | 13.1±0.7 | 93.8±0.4 |

---

## Usage

```bash
python experiments/run_rpi4_benchmark.py
python experiments/seasonal_validation.py
python experiments/cross_site_lab.py
python experiments/lora_vs_wifi.py
python experiments/battery_life_test.py
python fault_tolerance/single_node_fault.py
```

---

## Config

See `config.yaml` for all hyperparameters.
