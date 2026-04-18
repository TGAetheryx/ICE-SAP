"""
Figure 1: TGlacierEdge system architecture (DCOSS-IoT poster).
Three RPi 4 nodes, 490–510 m grid, boundary tile exchange, LoRa uplink.
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from eia_nordic_2026.figures.generate_figure1 import plot_system_architecture

if __name__ == "__main__":
    plot_system_architecture("figures/output/tglacier_edge_architecture.png")
