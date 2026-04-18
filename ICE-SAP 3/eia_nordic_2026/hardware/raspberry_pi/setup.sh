#!/bin/bash
# ICE-SAP / TGlacierEdge — Raspberry Pi 4 Setup Script
# Disables non-essential peripherals and configures for IoT deployment.
# Run as: sudo bash setup.sh

set -e

echo "=== ICE-SAP RPi 4 Setup ==="

# ── 1. System update ─────────────────────────────────────────────────────
apt-get update -qq
apt-get install -y python3-pip python3-venv git

# ── 2. Python environment ─────────────────────────────────────────────────
python3 -m venv /opt/icesap_env
source /opt/icesap_env/bin/activate
pip install --upgrade pip
pip install numpy scipy onnxruntime

# ── 3. Fix CPU frequency (1.5 GHz, no throttling) ────────────────────────
# Disable dynamic frequency scaling for reproducible benchmarks
if [ -f /sys/devices/system/cpu/cpu0/cpufreq/scaling_governor ]; then
    echo "performance" | tee /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor
    echo "CPU governor set to 'performance'"
fi

# ── 4. Disable non-essential hardware (see disable_peripherals.sh) ────────
bash "$(dirname "$0")/disable_peripherals.sh"

# ── 5. LoRa SX1276 / RFM95W SPI setup ───────────────────────────────────
# Enable SPI interface for LoRa module
if ! grep -q "dtparam=spi=on" /boot/config.txt; then
    echo "dtparam=spi=on" >> /boot/config.txt
    echo "SPI enabled (reboot required)"
fi

# ── 6. Create systemd service for autonomous deployment ──────────────────
cat > /etc/systemd/system/icesap.service << 'SERVICE'
[Unit]
Description=ICE-SAP Glacier Edge Sensing
After=network.target

[Service]
Type=simple
User=pi
WorkingDirectory=/home/pi/ICE-SAP
Environment=PYTHONPATH=/home/pi/ICE-SAP
ExecStart=/opt/icesap_env/bin/python dcoss_iot_2026/deployment/quantized_inference.py
Restart=on-failure
RestartSec=10

[Install]
WantedBy=multi-user.target
SERVICE

systemctl daemon-reload
echo "Systemd service 'icesap' created (enable with: systemctl enable icesap)"

echo ""
echo "=== Setup complete ==="
echo "Reboot required to apply SPI and governor settings."
echo "After reboot: source /opt/icesap_env/bin/activate"
