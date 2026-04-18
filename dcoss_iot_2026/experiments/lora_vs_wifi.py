"""
LoRa vs. Wi-Fi comparison (TGlacierEdge §V.B, Table IV).

Physical lab: RPi 4 + RFM95W, SF12/125 kHz, 510 m, multipath emulated.
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from eia_nordic_2026.hardware.lora.rfm95w_config import print_lora_vs_wifi_table
from eia_nordic_2026.hardware.lora.packet_delivery_test import simulate_pdr


def run():
    print_lora_vs_wifi_table()

    print(f"\n[Simulated PDR with S-ARQ]")
    for name, loss in [("Winter −27°C", 0.021), ("Fog 0°C", 0.055),
                       ("Snow −10°C", 0.042)]:
        pdr = simulate_pdr(loss_rate=loss) * 100
        print(f"  {name:<18}: PDR = {pdr:.2f}%  (≥98.3% target)")


if __name__ == "__main__":
    run()
