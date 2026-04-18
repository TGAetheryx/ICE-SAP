"""Tests for fault tolerance."""
import numpy as np
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from dcoss_iot_2026.fault_tolerance.single_node_fault import (
    simulate_single_node_fault, simulate_dual_node_comms_out
)


def test_single_fault_degrades_iou():
    r = simulate_single_node_fault(n_scenes=10, n_trials=2)
    assert r["delta_pp"] < 0, "Single fault should degrade IoU"
    assert r["delta_pp"] > -10, "Degradation should be bounded"


def test_dual_fault_degrades_more():
    r1 = simulate_single_node_fault(n_scenes=10, n_trials=2)
    r2 = simulate_dual_node_comms_out(n_scenes=10, n_trials=2)
    assert r2["delta_pp"] < r1["delta_pp"], \
        "Dual fault should degrade more than single fault"
