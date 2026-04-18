# Installation

## Requirements

- Python **3.9+**
- ~4 GB disk space (source + pre-trained artefacts)
- For edge inference: Raspberry Pi 4 (4 GB), ONNX Runtime / TFLite Runtime

## Standard install (workstation / laptop)

```bash
git clone https://github.com/TGAetheryx/ICE-SAP.git
cd ICE-SAP
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install --upgrade pip
pip install -r requirements.txt
```

Verify the install by running the quick-start example:

```bash
python examples/run_example.py
```

## Edge install (Raspberry Pi 4)

The `eia_nordic_2026/` and `dcoss_iot_2026/` papers target RPi 4. A setup script
is provided:

```bash
sudo bash eia_nordic_2026/hardware/raspberry_pi/setup.sh
```

The script installs:
- ONNX Runtime ARM build
- TFLite Runtime
- LoRa stack dependencies (pyserial, spidev)

On-device inference uses `unet_int8.tflite` + `unet_int8_quantization_params.json`
from the repository root.

## Data

Sentinel-2 Level-2A imagery and GLIMS ground-truth masks must be downloaded
separately — see [`../data/README.md`](../data/README.md). None of the large
raster data (`*.tif`, `*.nc`, `*.h5`, `*.hdf5`) is tracked in git.

## Troubleshooting

- `rasterio` wheel fails on Apple Silicon → install GDAL via Homebrew first:
  `brew install gdal` then `pip install rasterio`.
- ONNX Runtime on RPi 4 → use the official pre-built wheel matching the kernel,
  do not `pip install onnxruntime-gpu`.
- LoRa hardware access requires adding the user to the `spi` and `gpio` groups.
