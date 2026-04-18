#!/bin/bash
# Disable non-essential peripherals on RPi 4 to achieve baseline 320 mW inference power.
# Matches ICE-SAP §4.1: "HDMI/Wi-Fi/BT/audio disabled"

set -e

CONFIG="/boot/config.txt"
echo "=== Disabling non-essential peripherals ==="

# Disable HDMI (saves ~25 mW)
if ! grep -q "hdmi_blanking=1" $CONFIG; then
    echo "hdmi_blanking=1" >> $CONFIG
fi
tvservice -o 2>/dev/null || true
echo "  HDMI disabled"

# Disable Wi-Fi (saves ~120 mW idle)
rfkill block wifi 2>/dev/null || true
if ! grep -q "dtoverlay=disable-wifi" $CONFIG; then
    echo "dtoverlay=disable-wifi" >> $CONFIG
fi
echo "  Wi-Fi disabled"

# Disable Bluetooth (saves ~5 mW)
rfkill block bluetooth 2>/dev/null || true
if ! grep -q "dtoverlay=disable-bt" $CONFIG; then
    echo "dtoverlay=disable-bt" >> $CONFIG
fi
echo "  Bluetooth disabled"

# Disable USB power (if no devices connected; saves ~50 mW)
# echo '1-1' | tee /sys/bus/usb/drivers/usb/unbind 2>/dev/null || true
echo "  (USB power: manual — only disable if no devices)"

# Disable audio (saves ~2 mW)
if ! grep -q "dtparam=audio=off" $CONFIG; then
    echo "dtparam=audio=off" >> $CONFIG
fi
echo "  Audio disabled"

echo ""
echo "Peripherals disabled. Reboot for full effect."
echo "Expected idle power reduction: ~150 mW."
echo "Target inference delta: 320 mW (paper §4.2)."
