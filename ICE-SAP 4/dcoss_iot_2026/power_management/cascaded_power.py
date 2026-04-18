"""
Cascaded power management for TGlacierEdge.

Wraps cascaded_startup machinery for DCOSS-IoT paper.
P_avg = 0.9×15 + 0.08×80 + 0.02×320 = 23.9 mW → 24.6-day battery.
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from eia_nordic_2026.cascaded_startup.power_state_machine import (
    CascadedPowerManager, EXPECTED_AVG_POWER_MW, EXPECTED_BATTERY_LIFE_DAYS,
    PowerState, POWER_MW, DUTY_CYCLE,
)
import numpy as np


def power_budget_summary():
    print("\n=== TGlacierEdge Cascaded Power Budget ===")
    for state, pw in POWER_MW.items():
        if state in DUTY_CYCLE:
            duty = DUTY_CYCLE[state]
            contrib = pw * duty
            print(f"  {state.value:<16}: {pw:>6.0f} mW × {duty:.0%} = {contrib:.1f} mW")
    print(f"  {'P_avg':<16}: {EXPECTED_AVG_POWER_MW:.1f} mW")
    print(f"  {'Battery life':<16}: {EXPECTED_BATTERY_LIFE_DAYS:.1f} days "
          f"(4000 mAh)  [paper: 24.6 days]")


def simulate_power_session(n_cycles: int = 100, seed: int = 42) -> dict:
    """Simulate a full deployment session and compute average power."""
    np.random.seed(seed)
    mgr = CascadedPowerManager()
    for i in range(n_cycles):
        sc = 0.2 + np.random.randn() * 0.15
        if 40 <= i <= 50:
            sc = 0.55
        phi = abs(np.random.randn()) * 0.2
        result = mgr.process_cycle(sc, phi, sigma_meta=5.0)
    return {
        "avg_power_mw": result["avg_power_mw"],
        "l3_rate": result["l3_rate"],
        "final_state": result["state"],
    }


if __name__ == "__main__":
    power_budget_summary()
    r = simulate_power_session()
    print(f"\n  Simulated avg power: {r['avg_power_mw']:.1f} mW")
    print(f"  L3 activation rate:  {r['l3_rate']*100:.1f}%  (target: 2%)")
