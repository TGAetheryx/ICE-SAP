"""Tests for duty cycle manager."""
import time
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from dcoss_iot_2026.deployment.duty_cycle_manager import DutyCycleManager


def test_trigger_after_cycle():
    mgr = DutyCycleManager(cycle_minutes=0.0001)
    time.sleep(0.01)
    assert mgr.should_trigger()


def test_no_trigger_before_cycle():
    mgr = DutyCycleManager(cycle_minutes=60.0)
    assert not mgr.should_trigger()
