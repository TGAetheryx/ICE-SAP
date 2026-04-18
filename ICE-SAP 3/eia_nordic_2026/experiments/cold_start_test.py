"""
Cold-start reliability test (ICE-SAP §5.2).

50 consecutive cold-start cycles at −45°C after 2-hour thermal soak.
Paper result: 100% success rate.

The cascaded startup architecture suppresses the instantaneous voltage sag
caused by increased battery internal resistance at low temperatures,
preventing brown-out reset during cold-start phase.
"""
import numpy as np
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from eia_nordic_2026.experiments.thermal_chamber_test import cold_start_test


def run():
    print("\n" + "="*60)
    print("  Cold-Start Reliability Test @ −45°C")
    print("="*60)
    result = cold_start_test(n_cycles=50, temperature_c=-45.0, soak_hours=2.0)
    print(f"\n  Cycles tested:   {result['n_cycles']}")
    print(f"  Successful:      {result['n_success']} ({result['success_rate_pct']:.1f}%)")
    print(f"  Temperature:     {result['temperature_c']}°C")
    print(f"  Soak duration:   {result['soak_hours']} h")
    print(f"\n  Paper result: 100% success across 50 cycles")
    print(f"  Mechanism: cascaded startup suppresses voltage sag "
          f"(increased battery internal resistance at −45°C)")


if __name__ == "__main__":
    run()
