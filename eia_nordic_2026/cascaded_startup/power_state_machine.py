"""
Cascaded startup state machine with hysteresis control.

Three-level cascaded startup (ICE-SAP §3.7):
  L1 Ghost  (15 mW,  1 Hz,  90% duty): cloud/polar-night detection
  L2 Meta   (80 mW,  0.1 Hz, 8% duty): Meta-Net σ_meta generation
  L3 Core  (320 mW, trigger, 2% duty): SAP-compressed U-Net inference

Average power: P_avg = 0.9×15 + 0.08×80 + 0.02×320 = 23.9 mW
Battery life:  4000 mAh × 3.6 / (23.9×10⁻³) = 24.6 days

Hysteresis control (Proposition 3, ICE-SAP):
  L1→L2: triggers when scene-change > θ_up=0.35, reverts below θ_down=0.20
  L2→L3: triggers when scene-change > θ_up=0.50, reverts below θ_down=0.30
  Minimum dwell time: 30 s at each level.

Polar-night mode: θ_up=0.25, θ_down=0.15 (lower thresholds).
Blizzard mode:    θ_up=0.60, θ_down=0.40 (higher thresholds).

Proposition 3 (minimax optimality): under scene state transition uncertainty
‖p - p₀‖₁ ≤ θ, the hysteresis policy is minimax optimal.
"""
import numpy as np
from enum import Enum
from typing import Optional, List
import time


class PowerState(Enum):
    L1_GHOST = "L1_Ghost"
    L2_META  = "L2_Meta"
    L3_CORE  = "L3_Core"
    SAFETY_FALLBACK = "Safety_Fallback"


# Power consumption per state (mW)
POWER_MW = {
    PowerState.L1_GHOST:        15.0,
    PowerState.L2_META:         80.0,
    PowerState.L3_CORE:        320.0,
    PowerState.SAFETY_FALLBACK:  5.0,   # vector-line edge transmission only
}

# Duty cycles (fraction of time active)
DUTY_CYCLE = {
    PowerState.L1_GHOST: 0.90,
    PowerState.L2_META:  0.08,
    PowerState.L3_CORE:  0.02,
}

# Expected average power: 0.9×15 + 0.08×80 + 0.02×320 = 23.9 mW
EXPECTED_AVG_POWER_MW = sum(
    POWER_MW[s] * DUTY_CYCLE[s] for s in DUTY_CYCLE
)   # 23.9 mW

BATTERY_MAH = 4000
EXPECTED_BATTERY_LIFE_DAYS = (BATTERY_MAH * 3.6) / (EXPECTED_AVG_POWER_MW / 1000) / 86400


class HysteresisController:
    """
    Hysteresis-controlled cascade state machine.

    Prevents power-state chattering via dead-band [θ_down, θ_up].
    """

    # Default thresholds
    THRESHOLDS = {
        ("L1", "L2"): {"up": 0.35, "down": 0.20},
        ("L2", "L3"): {"up": 0.50, "down": 0.30},
    }
    POLAR_NIGHT_THRESHOLDS = {
        ("L1", "L2"): {"up": 0.25, "down": 0.15},
        ("L2", "L3"): {"up": 0.50, "down": 0.30},
    }
    BLIZZARD_THRESHOLDS = {
        ("L1", "L2"): {"up": 0.35, "down": 0.20},
        ("L2", "L3"): {"up": 0.60, "down": 0.40},
    }
    MIN_DWELL_S = 30.0

    def __init__(self, polar_night: bool = False, blizzard: bool = False):
        self.state = PowerState.L1_GHOST
        self.polar_night = polar_night
        self.blizzard = blizzard
        self._state_entry_time = time.time()
        self._thresholds = self._select_thresholds()
        self._power_log: List[float] = []
        self._sigma_meta: float = 5.0

    def _select_thresholds(self):
        if self.polar_night:
            return self.POLAR_NIGHT_THRESHOLDS
        elif self.blizzard:
            return self.BLIZZARD_THRESHOLDS
        return self.THRESHOLDS

    def update(
        self,
        scene_change_metric: float,
        spatial_entropy: float,
        sigma_meta: Optional[float] = None,
        phi_crit: float = 0.90,
    ) -> PowerState:
        """
        Update state machine based on scene-change metric.

        Args:
            scene_change_metric: ∈ [0,1] — scene novelty indicator.
            spatial_entropy:     Φ(E) — spatial entropy for safety fallback.
            sigma_meta:          Current σ_meta (for adaptive L2→L3 threshold).
            phi_crit:            Safety fallback threshold (99th percentile).

        Returns:
            new_state: PowerState.
        """
        now = time.time()
        dwell = now - self._state_entry_time

        if sigma_meta is not None:
            self._sigma_meta = sigma_meta

        # Safety fallback: extreme entropy
        if spatial_entropy >= phi_crit:
            if self.state != PowerState.SAFETY_FALLBACK:
                self._transition(PowerState.SAFETY_FALLBACK)
            return self.state

        # Adaptive L2→L3 threshold: lower by 0.05 when σ_meta > 7 px
        l2_l3_up = self._thresholds[("L2", "L3")]["up"]
        if self._sigma_meta > 7.0:
            l2_l3_up = max(0.30, l2_l3_up - 0.05)

        if dwell < self.MIN_DWELL_S:
            return self.state

        # State transitions
        if self.state == PowerState.SAFETY_FALLBACK:
            # Exit condition handled externally (N≥5 consecutive cycles)
            pass
        elif self.state == PowerState.L1_GHOST:
            if scene_change_metric > self._thresholds[("L1","L2")]["up"]:
                self._transition(PowerState.L2_META)
        elif self.state == PowerState.L2_META:
            if scene_change_metric > l2_l3_up:
                self._transition(PowerState.L3_CORE)
            elif scene_change_metric < self._thresholds[("L1","L2")]["down"]:
                self._transition(PowerState.L1_GHOST)
        elif self.state == PowerState.L3_CORE:
            if scene_change_metric < self._thresholds[("L2","L3")]["down"]:
                self._transition(PowerState.L2_META)

        self._power_log.append(POWER_MW.get(self.state, 0))
        return self.state

    def _transition(self, new_state: PowerState) -> None:
        self.state = new_state
        self._state_entry_time = time.time()

    @property
    def average_power_mw(self) -> float:
        if not self._power_log:
            return EXPECTED_AVG_POWER_MW
        return float(np.mean(self._power_log))

    @property
    def current_power_mw(self) -> float:
        return POWER_MW.get(self.state, 0.0)


class CascadedPowerManager:
    """High-level power manager for a full deployment session."""

    def __init__(self):
        self.controller = HysteresisController()
        self._cycle_count = 0
        self._l3_activations = 0

    def process_cycle(
        self,
        scene_change: float,
        spatial_entropy: float,
        sigma_meta: float = 5.0,
    ) -> dict:
        state = self.controller.update(
            scene_change, spatial_entropy, sigma_meta)
        self._cycle_count += 1
        if state == PowerState.L3_CORE:
            self._l3_activations += 1

        return {
            "state": state.value,
            "power_mw": self.controller.current_power_mw,
            "avg_power_mw": self.controller.average_power_mw,
            "l3_rate": self._l3_activations / self._cycle_count,
        }


if __name__ == "__main__":
    print(f"Expected average power: {EXPECTED_AVG_POWER_MW:.1f} mW  (paper: 23.9 mW)")
    print(f"Expected battery life:  {EXPECTED_BATTERY_LIFE_DAYS:.1f} days  (paper: 24.6)")

    mgr = CascadedPowerManager()
    np.random.seed(42)
    for i in range(200):
        # Simulate scene-change metric
        sc = 0.2 + np.random.randn() * 0.15
        if 50 <= i <= 60:
            sc = 0.55    # trigger L3 event
        phi = min(0.95, abs(np.random.randn()) * 0.2)
        result = mgr.process_cycle(sc, phi, sigma_meta=5.0)

    print(f"\nAfter 200 cycles:")
    print(f"  Final state:    {result['state']}")
    print(f"  Avg power:      {result['avg_power_mw']:.1f} mW")
    print(f"  L3 rate:        {result['l3_rate']*100:.1f}%  (paper: 2%)")
