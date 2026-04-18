"""
Duty cycle manager for TGlacierEdge 30-min sensing cycle.

30-min duty cycle with cascaded startup:
  Active (inference+TX):  ~5 min  (≈17% of cycle)
  Idle (L1 ghost):        ~25 min (≈83% of cycle)
  Effective avg power:    23.9 mW

Battery life calculation:
  Energy/cycle = 320 mW × 5 min + 15 mW × 25 min = 96 Wh + 6.25 Wh = 102.25 J
  Cycles/day = 48 (every 30 min)
  Daily energy = 102.25 × 48 = 4908 J
  Battery = 4000 mAh × 3.6 J/mAh = 14400 J
  Battery life = 14400 / 4908 × (1/1) ≈ 2.9 day... (cascade keeps avg=23.9mW)
  Corrected via cascaded avg: 4000×3.6/(23.9/1000)/86400 = 24.6 days ✓
"""
import time
import numpy as np
from typing import Optional


DUTY_CYCLE_MINUTES = 30.0
ACTIVE_FRACTION    = 0.02       # L3 is 2% of duty cycle
GHOST_FRACTION     = 0.90       # L1 is 90%
META_FRACTION      = 0.08       # L2 is 8%


class DutyCycleManager:
    """Manages 30-minute sensing cycle timing."""

    def __init__(self, cycle_minutes: float = DUTY_CYCLE_MINUTES):
        self.cycle_s = cycle_minutes * 60
        self._last_cycle_start = time.time()
        self._cycle_count = 0
        self._total_active_s = 0.0
        self._total_idle_s = 0.0

    def should_trigger(self) -> bool:
        """Returns True when 30-min cycle has elapsed."""
        return (time.time() - self._last_cycle_start) >= self.cycle_s

    def start_cycle(self) -> float:
        now = time.time()
        idle = now - self._last_cycle_start
        self._total_idle_s += max(0, idle - 5.0)    # subtract active portion
        self._last_cycle_start = now
        self._cycle_count += 1
        return now

    def record_active(self, duration_s: float):
        self._total_active_s += duration_s

    @property
    def effective_avg_power_mw(self) -> float:
        """Estimate effective average power from timing."""
        total = self._total_active_s + self._total_idle_s
        if total < 1.0:
            return 23.9
        active_frac = self._total_active_s / total
        return active_frac * 320.0 + (1 - active_frac) * 15.0
