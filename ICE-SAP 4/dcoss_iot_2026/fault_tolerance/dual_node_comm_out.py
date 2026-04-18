"""
Dual-node communications-out fault tolerance.

When both adjacent nodes lose LoRa connectivity, only the standalone node
continues inference. IoU degrades by −4.7 pp (86.3±0.3%).
"""
import numpy as np
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from dcoss_iot_2026.fault_tolerance.single_node_fault import simulate_dual_node_comms_out


def run():
    result = simulate_dual_node_comms_out()
    print("=== Dual-Node Communications Out ===")
    print(f"  Single-node IoU: {result['iou_1node_mean']:.1f}%")
    print(f"  Delta vs 3-node: {result['delta_pp']:+.1f} pp  (paper: −4.7 pp)")
    print(f"  Paper result:    86.3±0.3%")


if __name__ == "__main__":
    run()
