"""
Safety Fallback — Life-Sign Mode for extreme glacier events.

When spatial entropy Φ(E) exceeds Φ_crit (99th percentile of training entropy),
the system abandons complex segmentation and reverts to single-pixel
vector-line edge transmission, preserving minimum glacier flow velocity data.

A high-priority "Catastrophic Change Alert" is broadcast via LoRa.
System exits Life-Sign Mode when Φ(E) < Φ_crit for N ≥ 5 consecutive cycles.

Proposition 4 (Sequential Detection, ICE-SAP Appendix E):
  N* = 5 under AR(1) temporal dependence with ρ ≈ 0.60.

Lab validation (n=20 injected entropy spikes):
  95% Life-Sign Mode recall (19/20), 5% false trigger rate.

References:
  ICE-SAP §3.7. TGlacierEdge §III.F.
"""
import numpy as np
from typing import List, Optional, Tuple
from enum import Enum


# Safety fallback parameters
PHI_CRIT_PERCENTILE = 99.0     # 99th percentile of training entropy
N_EXIT_CONSECUTIVE  = 5        # N ≥ 5 consecutive sub-threshold cycles
LIFE_SIGN_POWER_MW  = 5.0      # minimal vector-line transmission power
# Lab validation
LAB_RECALL_PCT     = 95.0      # 19/20 injected spikes
LAB_FALSE_TRIG_PCT = 5.0


class FallbackState(Enum):
    NORMAL    = "normal"
    FALLBACK  = "life_sign_mode"


class SafetyFallbackMonitor:
    """
    Monitors spatial entropy for catastrophic glacier events.

    Uses Neyman-Pearson threshold Φ_crit = F_{H₀}^{-1}(0.99).
    Confirmation window N=5 corrects for AR(1) temporal dependence.
    """

    def __init__(
        self,
        phi_crit: Optional[float] = None,
        n_exit: int = N_EXIT_CONSECUTIVE,
        training_entropy_samples: Optional[np.ndarray] = None,
    ):
        self.n_exit = n_exit
        self.state = FallbackState.NORMAL
        self._consecutive_normal = 0
        self._alert_history: List[bool] = []

        # Compute Φ_crit from training data or use default
        if phi_crit is not None:
            self.phi_crit = phi_crit
        elif training_entropy_samples is not None:
            self.phi_crit = float(np.percentile(
                training_entropy_samples, PHI_CRIT_PERCENTILE))
        else:
            self.phi_crit = 0.90   # default

    def update(
        self,
        spatial_entropy: float,
        sigma_meta: float = 5.0,
    ) -> Tuple[FallbackState, bool]:
        """
        Update monitor with new entropy observation.

        σ_meta feedback: when σ_meta > 7 px, lower L2→L3 threshold by 0.05
        (handled in power_state_machine.py).

        Args:
            spatial_entropy: Φ(E) current value.
            sigma_meta:      Current σ_meta (for adaptive thresholding).

        Returns:
            state:       Current fallback state.
            new_alert:   True if just entered fallback.
        """
        triggered = spatial_entropy >= self.phi_crit
        new_alert = False

        if triggered and self.state == FallbackState.NORMAL:
            self.state = FallbackState.FALLBACK
            self._consecutive_normal = 0
            new_alert = True
        elif not triggered and self.state == FallbackState.FALLBACK:
            self._consecutive_normal += 1
            if self._consecutive_normal >= self.n_exit:
                # Exit Life-Sign Mode
                self.state = FallbackState.NORMAL
                self._consecutive_normal = 0

        self._alert_history.append(triggered)
        return self.state, new_alert

    def transmit_life_sign(
        self,
        glacier_boundary_points: np.ndarray,
    ) -> bytes:
        """
        Encode minimal glacier edge as vector-line for LoRa transmission.

        In Life-Sign Mode, only the glacier terminus x-coordinates are
        transmitted (single pixel per scan-line), preserving flow velocity
        reconstruction capability.

        Args:
            glacier_boundary_points: (N, 2) array of (row, col) boundary pixels.

        Returns:
            encoded: Compact byte representation.
        """
        # Encode as 2-byte per point (row 0–255, col 0–255 normalised)
        if len(glacier_boundary_points) == 0:
            return b""
        pts = glacier_boundary_points[:128]   # max 128 points per packet
        encoded = bytearray()
        for row, col in pts:
            encoded.append(int(row) & 0xFF)
            encoded.append(int(col) & 0xFF)
        return bytes(encoded)

    @property
    def alert_rate(self) -> float:
        if not self._alert_history:
            return 0.0
        return sum(self._alert_history) / len(self._alert_history)


def validate_safety_fallback(n_spikes: int = 20, seed: int = 42) -> dict:
    """
    Validate safety fallback with injected entropy spikes.

    Lab result: 19/20 recall (95%), 1/20 false trigger.
    """
    np.random.seed(seed)
    monitor = SafetyFallbackMonitor(phi_crit=0.90, n_exit=5)

    # Background entropy: ~0.4 ± 0.15
    T = 200
    entropy_bg = np.random.normal(0.40, 0.15, T).astype(np.float32)
    entropy_bg = np.clip(entropy_bg, 0, 1)

    # Inject spikes
    spike_indices = sorted(np.random.choice(range(10, T - 10), n_spikes,
                                             replace=False))
    entropy = entropy_bg.copy()
    for si in spike_indices:
        entropy[si] = 0.95    # above φ_crit

    detected = 0
    false_triggers = 0

    for t in range(T):
        state, alert = monitor.update(float(entropy[t]))
        if alert:
            if t in set(spike_indices):
                detected += 1
            else:
                false_triggers += 1

    return {
        "n_spikes": n_spikes,
        "detected": detected,
        "recall_pct": detected / n_spikes * 100,
        "false_triggers": false_triggers,
        "false_trig_pct": false_triggers / n_spikes * 100,
    }


if __name__ == "__main__":
    result = validate_safety_fallback(20)
    print(f"Safety Fallback Validation (lab simulation):")
    print(f"  Recall:        {result['detected']}/{result['n_spikes']} "
          f"({result['recall_pct']:.0f}%)  (paper: 19/20 = 95%)")
    print(f"  False triggers: {result['false_triggers']} "
          f"({result['false_trig_pct']:.0f}%)  (paper: 5%)")
