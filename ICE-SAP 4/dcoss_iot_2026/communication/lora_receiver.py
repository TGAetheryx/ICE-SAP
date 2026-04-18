"""lora_receiver — LoRa communication for TGlacierEdge.
See lora_transmitter.py and eia_nordic_2026/hardware/lora/ for full implementation.
Paper: 18.4 MB → 1.1 MB uplink (93.8% reduction), LoRa 6 KB payload, 215–237 s TX.
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from eia_nordic_2026.sbt_sarq.uplink_compression import uplink_reduction_summary

if __name__ == "__main__":
    uplink_reduction_summary()
