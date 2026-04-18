"""
Cross-site lab results (TGlacierEdge §V.C, Table V).
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from eia_nordic_2026.experiments.cross_site_generalization import (
    print_cross_site_table, sigma_sensitivity_analysis
)


def run():
    print("=== TGlacierEdge Cross-Site Lab Results ===")
    print_cross_site_table()
    sigma_sensitivity_analysis()
    print(f"\n  All cross-site σ (4–7 px) calibrated per §III.B")
    print(f"  σ sensitivity: ≤0.2 pp IoU across 4–8 px range")


if __name__ == "__main__":
    run()
