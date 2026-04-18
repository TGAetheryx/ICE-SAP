"""
Battery life simulation (ICE-SAP §4.2, §3.7).

P_avg = 0.9×15 + 0.08×80 + 0.02×320 = 23.9 mW  (@+25°C)
P_avg = 0.9×15 + 0.08×80 + 0.02×356 = 27.0 mW  (@−45°C)

Battery life = 4000 mAh × 3.6 J/mAh / P_avg
  @+25°C: 24.6 days
  @−45°C: 22.8 days

Non-cascaded (L3 always-on): 356 mW → ≈1.7 days at −45°C (13.2× worse).

LoRa TX adds: 3.5 mW (SF10, +14 dBm) for 0.23 s per tile → negligible.
"""
import numpy as np
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from eia_nordic_2026.cascaded_startup.l3_core import average_system_power


BATTERY_MAH = 4000
JOULES_PER_MAH = 3.6


def battery_life_days(avg_power_mw: float, capacity_mah: float = BATTERY_MAH) -> float:
    """Compute battery life in days."""
    energy_j = capacity_mah * JOULES_PER_MAH
    power_w = avg_power_mw / 1000.0
    return energy_j / power_w / 86400


def run():
    print("\n" + "="*60)
    print("  Battery Life Simulation")
    print("="*60)

    configs = {
        "Cascaded @+25°C": average_system_power(25.0),
        "Cascaded @−45°C": average_system_power(-45.0),
        "Always-on (L3) @−45°C": 356.0,
    }
    papers = {
        "Cascaded @+25°C": 24.6,
        "Cascaded @−45°C": 22.8,
        "Always-on (L3) @−45°C": 1.7,
    }

    print(f"\n  {'Configuration':<28} {'P_avg (mW)':>12} "
          f"{'Life (days)':>14} {'Paper':>10}")
    print("-"*68)
    for name, p in configs.items():
        life = battery_life_days(p)
        paper = papers.get(name, "—")
        print(f"  {name:<28} {p:>12.1f} {life:>14.1f}   {paper:>8}")

    cascade_life = battery_life_days(average_system_power(-45.0))
    always_on_life = battery_life_days(356.0)
    ratio = cascade_life / always_on_life
    print(f"\n  Cascade advantage: {ratio:.1f}×  (paper: 13.2×)")


if __name__ == "__main__":
    run()
