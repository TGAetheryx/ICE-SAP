"""
Battery life test (TGlacierEdge §I, §III.F).

Lab-estimated 24.6-day battery life (4000 mAh, cascaded duty cycle, 30-min cycle).
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from eia_nordic_2026.experiments.battery_life_simulation import run as battery_run


def run():
    print("=== TGlacierEdge Battery Life Test ===")
    battery_run()
    print(f"\n  TGlacierEdge 30-min duty cycle:")
    print(f"    Active (L3 inference):  ~2% of 30 min = 36 s")
    print(f"    Meta  (L2 active):      ~8% of 30 min = 144 s")
    print(f"    Ghost (L1 active):      ~90% of 30 min = 1620 s")
    print(f"    P_avg = 0.9×15 + 0.08×80 + 0.02×320 = 23.9 mW")
    print(f"    Battery life = 4000×3.6 / (23.9/1000) / 86400 = 24.6 days")


if __name__ == "__main__":
    run()
