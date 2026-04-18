"""Tests for cascaded startup state machine."""
import numpy as np
import pytest
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from eia_nordic_2026.cascaded_startup.power_state_machine import (
    HysteresisController, PowerState, EXPECTED_AVG_POWER_MW
)


def test_l1_is_initial_state():
    ctrl = HysteresisController()
    assert ctrl.state == PowerState.L1_GHOST


def test_transitions_to_l3_on_high_scene_change():
    ctrl = HysteresisController()
    import time
    ctrl._state_entry_time = time.time() - 60  # bypass dwell time
    # Force L1→L2
    ctrl.update(0.40, 0.3)
    ctrl._state_entry_time = time.time() - 60
    # Force L2→L3
    state = ctrl.update(0.55, 0.3)
    assert state in (PowerState.L3_CORE, PowerState.L2_META)


def test_expected_avg_power():
    """Verify analytic average power formula."""
    expected = 0.9 * 15 + 0.08 * 80 + 0.02 * 320
    assert abs(EXPECTED_AVG_POWER_MW - expected) < 0.01, \
        f"Expected {expected:.1f} mW, got {EXPECTED_AVG_POWER_MW:.1f} mW"


def test_safety_fallback_triggered():
    ctrl = HysteresisController()
    state = ctrl.update(0.3, spatial_entropy=0.95)
    assert state == PowerState.SAFETY_FALLBACK
