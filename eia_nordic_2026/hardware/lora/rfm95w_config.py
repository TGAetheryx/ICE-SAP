"""
RFM95W (SX1276) LoRa configuration for Arctic IoT deployment.

Physical lab validation (ICE-SAP §V.B / TGlacierEdge §V.B):
  - SF12 / 125 kHz / +14 dBm
  - Channel emulator: 0.8–1.2 μs delay, ≥97% fidelity
  - Temperature chamber: −27°C to +13°C
  - 510 m node separation
  - LoRa maintains ≥98.3% 24-h uptime under all Arctic conditions

LoRa vs Wi-Fi (Table IV):
  LoRa Winter −27°C: 2.1±0.3% loss, 100% uptime, 3.2±0.1 mW
  LoRa Fog    0°C:   5.5±0.4% loss, 98.3±0.5% uptime, 3.3±0.1 mW
  LoRa Snow −10°C:   4.2±0.3% loss, 99.1±0.3% uptime, 3.2±0.1 mW
  Wi-Fi Winter −27°C: 99.9% loss, <0.5h uptime, 120±5 mW
"""
import numpy as np


# RFM95W / SX1276 configuration
LORA_CONFIG = {
    "spreading_factor": 12,
    "bandwidth_khz":    125,
    "tx_power_dbm":     14,
    "coding_rate":      "4/5",
    "frequency_mhz":    868,        # EU 868 MHz band
    "preamble_len":     8,
    "sync_word":        0x34,       # LoRaWAN default
}

# Physical lab results (Table IV)
LORA_VALIDATION = {
    "winter_m27c": {
        "packet_loss_pct": (2.1, 0.3),
        "distance_m": 510,
        "uptime_24h_pct": 100.0,
        "tx_power_mw": (3.2, 0.1),
    },
    "fog_0c": {
        "packet_loss_pct": (5.5, 0.4),
        "distance_m": 510,
        "uptime_24h_pct": (98.3, 0.5),
        "tx_power_mw": (3.3, 0.1),
    },
    "snow_m10c": {
        "packet_loss_pct": (4.2, 0.3),
        "distance_m": 510,
        "uptime_24h_pct": (99.1, 0.3),
        "tx_power_mw": (3.2, 0.1),
    },
    "wifi_m27c": {
        "packet_loss_pct": (99.9, 0.1),
        "distance_m": 510,
        "uptime_24h_h": "<0.5h",
        "tx_power_mw": (120, 5),
    },
}


def configure_rfm95w_arctic() -> dict:
    """Return recommended RFM95W config for Arctic polar deployment."""
    return {
        **LORA_CONFIG,
        "notes": [
            "SF12 for maximum link budget (Arctic multipath)",
            "+14 dBm for 510m range with margin",
            "RS(15,9) S-ARQ for packet loss recovery",
            "Duty cycle: TX 0.23s/tile, RX 0.5s/cycle",
        ]
    }


def print_lora_vs_wifi_table():
    print("\n=== Table IV: LoRa vs Wi-Fi Link Reliability ===")
    print(f"  {'Scheme/Cond.':<22} {'Loss(%)':>10} {'Dist(m)':>8} "
          f"{'24h Uptime':>12} {'Pwr(mW)':>10}")
    print("-"*66)
    rows = [
        ("LoRa Winter −27°C", "2.1±0.3", 510, "100%",       "3.2±0.1"),
        ("LoRa Fog 0°C",      "5.5±0.4", 510, "98.3±0.5%",  "3.3±0.1"),
        ("LoRa Snow −10°C",   "4.2±0.3", 510, "99.1±0.3%",  "3.2±0.1"),
        ("Wi-Fi Winter −27°C","99.9±0.1",510, "<0.5 h",     "120±5"),
    ]
    for r in rows:
        print(f"  {r[0]:<22} {r[1]:>10} {r[2]:>8} {r[3]:>12} {r[4]:>10}")
    print("\n  LoRa IoU loss under multipath: ≤0.5 pp")


if __name__ == "__main__":
    cfg = configure_rfm95w_arctic()
    print("RFM95W Arctic Config:", cfg)
    print_lora_vs_wifi_table()
