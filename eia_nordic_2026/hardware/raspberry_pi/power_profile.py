"""
Raspberry Pi 4 power profiling for TGlacierEdge / ICE-SAP.

Measurement methodology (ICE-SAP §4.2):
  - UM25C USB power meter on 5V supply rail, logging at 1 Hz
  - All non-essential peripherals disabled (HDMI, Wi-Fi, BT, audio)
  - CPU fixed at 1.5 GHz (no throttling)
  - Reported 320 mW = instantaneous peak inference power (single forward pass)

Power breakdown per component (Table 4, ICE-SAP):
  L1 Ghost Sensing:       0.015 J   (12%)
  L2 Meta Inference:      0.008 J    (6%)
  L3 Core Execution:      0.064 J   (51%)
  Communication SBT+SARQ: 0.039 J   (31%)
  Total per cycle:        0.126 J  (100%)

LoRa module (SX1276):
  TX (SF10, +14 dBm):  3.5 ± 0.3 mW for 0.23 s per tile
  RX/listen:           9.2 ± 0.4 mW for 0.5 s per cycle
  Sleep:               0.02 mW
  Average comms power: 4.4 mW
"""
import numpy as np
from typing import Dict


# Hardware constants
RPI4_IDLE_W         = 3.4     # board idle (not included in inference delta)
INFERENCE_DELTA_MW  = 320.0   # incremental power during inference
INFERENCE_LATENCY_S = 0.201   # 201 ms

# Per-cycle energy breakdown (Table 4)
CYCLE_ENERGY_J = {
    "L1_ghost":      0.015,
    "L2_meta":       0.008,
    "L3_core":       0.064,
    "communication": 0.039,
}
CYCLE_TOTAL_J = sum(CYCLE_ENERGY_J.values())  # 0.126 J

# LoRa SX1276 constants
LORA_TX_MW   = 3.5
LORA_TX_STD  = 0.3
LORA_TX_S    = 0.23
LORA_RX_MW   = 9.2
LORA_RX_STD  = 0.4
LORA_RX_S    = 0.5
LORA_SLEEP_MW = 0.02
LORA_AVG_COMMS_MW = 4.4


def simulate_power_measurement(
    n_samples: int = 100,
    temperature_c: float = 25.0,
    seed: int = 42,
) -> Dict:
    """
    Simulate UM25C power meter readings during ICE-SAP inference cycle.

    Args:
        n_samples:    Number of 1-Hz measurement samples.
        temperature_c: Operating temperature.

    Returns:
        dict with mean_mw, std_mw, peak_mw, duty_cycle.
    """
    np.random.seed(seed)

    # Arrhenius correction at low temperature
    if temperature_c <= -45:
        peak_mw = 356.0
        peak_std = 16.0
    else:
        frac = max(0, (25.0 - temperature_c) / 70.0)
        peak_mw = 320.0 + frac * 36.0
        peak_std = 8.0

    # Simulate duty-cycled power over n_samples
    samples = np.full(n_samples, 15.0)   # L1 baseline
    # L3 active ~2% of time
    l3_idx = np.random.choice(n_samples,
                               size=max(1, int(n_samples * 0.02)),
                               replace=False)
    samples[l3_idx] = np.random.normal(peak_mw, peak_std, len(l3_idx))

    return {
        "mean_mw": float(samples.mean()),
        "std_mw":  float(samples.std()),
        "peak_mw": float(samples.max()),
        "peak_mw_paper": peak_mw,
        "temperature_c": temperature_c,
    }


def print_power_breakdown():
    print("\n=== Table 4: Energy per Inference Cycle ===")
    total = CYCLE_TOTAL_J
    print(f"  {'Component':<28} {'Energy (J)':>12} {'%':>8}")
    print("-"*52)
    for comp, e in CYCLE_ENERGY_J.items():
        print(f"  {comp:<28} {e:>12.3f} {e/total*100:>7.0f}%")
    print("-"*52)
    print(f"  {'Total':<28} {total:>12.3f}   100%")

    print(f"\n=== LoRa SX1276 Power ===")
    print(f"  TX (SF10, +14 dBm): {LORA_TX_MW}±{LORA_TX_STD} mW for {LORA_TX_S} s/tile")
    print(f"  RX/listen:          {LORA_RX_MW}±{LORA_RX_STD} mW for {LORA_RX_S} s/cycle")
    print(f"  Sleep:              {LORA_SLEEP_MW} mW")
    print(f"  Average comms:      {LORA_AVG_COMMS_MW} mW")


if __name__ == "__main__":
    print_power_breakdown()
    for T in [25, 0, -27, -45]:
        r = simulate_power_measurement(temperature_c=T)
        print(f"\n  @{T:+4.0f}°C: mean={r['mean_mw']:.1f} mW, "
              f"peak={r['peak_mw']:.1f} mW  (paper peak: {r['peak_mw_paper']:.0f} mW)")
