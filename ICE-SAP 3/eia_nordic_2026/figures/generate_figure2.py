"""
Figure 2: SAP Pareto pruning curve (ICE-SAP Fig. 2).
Wraps eia_nordic_2026/sap_pruning/pareto_front.py.
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from eia_nordic_2026.sap_pruning.pareto_front import plot_pareto_curve

if __name__ == "__main__":
    plot_pareto_curve("figures/output/pareto_pruning.png")
