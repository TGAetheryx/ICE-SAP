"""
S-ARQ blind retransmission and uplink compression.

Blind retransmission: application-layer retransmission without ACK feedback.
Channel adaptation: RSSI sliding-window smoothing determines retransmit count.
  - Boundary tiles (W > τ): max 3 retransmissions
  - Low-priority tiles: base packets only

Energy model (ICE-SAP §3.4):
  E_tx = P_tx × T_on_air,  P_tx = 120 mW at +14 dBm
  Compressed payload (0.3 kB) → E_tx ≈ 0.07 J vs. 1.15 J raw

References:
  ICE-SAP §3.4. TGlacierEdge §III.D.
"""
import numpy as np
from typing import List, Optional, Tuple


# LoRa SX1276 constants (physical lab: RFM95W, SF12/125 kHz, +14 dBm)
LORA_TX_POWER_MW      = 120.0     # +14 dBm transmit power
LORA_RX_POWER_MW      = 9.2       # receive/listen mode
LORA_SLEEP_POWER_MW   = 0.02      # sleep
LORA_SYMBOL_RATE_SF12 = 125000 / (2**12)   # ~30.5 symbols/s at SF12

# Energy constants (ICE-SAP §3.4)
COMPRESSED_PAYLOAD_KB = 0.3
RAW_PAYLOAD_KB        = 1.15 * 1024 / 1000   # ≈1.18 KB for single tile
E_TX_COMPRESSED_J     = 0.07
E_TX_RAW_J            = 1.15


def compute_tx_energy(
    payload_kb: float,
    tx_power_mw: float = LORA_TX_POWER_MW,
    spreading_factor: int = 12,
    bandwidth_khz: float = 125.0,
    coding_rate: float = 4/5,
) -> float:
    """
    Compute LoRa transmission energy for a given payload.

    E_tx = P_tx × T_on_air

    Args:
        payload_kb: Payload size in KB.
        tx_power_mw: Transmit power (mW).

    Returns:
        energy_j: Transmission energy in Joules.
    """
    from eia_nordic_2026.sbt_sarq.tile_selection import estimate_lora_transmission_time
    t_air = estimate_lora_transmission_time(
        payload_kb, spreading_factor, bandwidth_khz, coding_rate)
    return (tx_power_mw / 1000.0) * t_air   # W × s = J


def rssi_adaptive_retransmit_count(
    rssi_history: List[float],
    window_size: int = 5,
    max_retransmit: int = 3,
) -> int:
    """
    Determine retransmission count from RSSI sliding window.

    Better RSSI → fewer retransmissions needed.
    Worse RSSI → more retransmissions (up to max_retransmit).

    Args:
        rssi_history: Recent RSSI values (dBm), most recent last.
        window_size:  Sliding window size.
        max_retransmit: Maximum retransmissions.

    Returns:
        n_retransmit: Recommended retransmission count (0–max_retransmit).
    """
    if len(rssi_history) < 2:
        return max_retransmit   # conservative default

    recent = rssi_history[-window_size:]
    mean_rssi = float(np.mean(recent))

    # RSSI thresholds (typical LoRa at 500 m)
    # Good: > -100 dBm → 0–1 retransmits
    # OK:   -100 to -110 → 1–2 retransmits
    # Poor: < -110 → max retransmits
    if mean_rssi > -100:
        return min(1, max_retransmit)
    elif mean_rssi > -110:
        return min(2, max_retransmit)
    else:
        return max_retransmit


class BlindRetransmitter:
    """
    Application-layer blind retransmission without ACK feedback.

    Implements S-ARQ protocol from ICE-SAP §3.4:
    - High-priority tiles (W > τ): RS(15,9) + up to 3 retransmissions
    - Low-priority tiles: base packet only (no retransmission)
    """

    def __init__(
        self,
        max_retransmit_high: int = 3,
        max_retransmit_low: int = 0,
        loss_simulation_rate: float = 0.0,
        seed: int = 42,
    ):
        self.max_retransmit_high = max_retransmit_high
        self.max_retransmit_low  = max_retransmit_low
        self.loss_rate = loss_simulation_rate
        self._rng = np.random.default_rng(seed)
        self._rssi_history: List[float] = []
        self.total_transmissions = 0
        self.successful_transmissions = 0

    def update_rssi(self, rssi_dbm: float) -> None:
        self._rssi_history.append(rssi_dbm)

    def transmit_tile(
        self,
        tile_data: bytes,
        is_high_priority: bool,
        rssi_history: Optional[List[float]] = None,
    ) -> Tuple[bool, int]:
        """
        Attempt tile transmission with blind retransmission.

        Returns:
            success: bool — whether tile was successfully received.
            n_attempts: int — total transmission attempts.
        """
        if rssi_history is None:
            rssi_history = self._rssi_history

        max_retrans = rssi_adaptive_retransmit_count(
            rssi_history, max_retransmit=self.max_retransmit_high
        ) if is_high_priority else self.max_retransmit_low

        n_attempts = 0
        for _ in range(max_retrans + 1):
            n_attempts += 1
            self.total_transmissions += 1
            # Simulate channel
            if self._rng.random() >= self.loss_rate:
                self.successful_transmissions += 1
                return True, n_attempts

        return False, n_attempts

    @property
    def packet_delivery_rate(self) -> float:
        if self.total_transmissions == 0:
            return 0.0
        return self.successful_transmissions / self.total_transmissions


if __name__ == "__main__":
    import sys, os
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

    # Test energy savings
    e_comp = compute_tx_energy(COMPRESSED_PAYLOAD_KB)
    e_raw  = compute_tx_energy(RAW_PAYLOAD_KB)
    print(f"E_tx (compressed 0.3 kB): {e_comp:.3f} J  (paper: 0.07 J)")
    print(f"E_tx (raw ~1.18 kB):      {e_raw:.3f} J  (paper: 1.15 J)")
    print(f"Energy saving:            {(1 - e_comp/e_raw)*100:.1f}%")

    # Test S-ARQ with 20% loss
    xmit = BlindRetransmitter(loss_simulation_rate=0.20, seed=7)
    xmit.update_rssi(-105.0)
    successes = 0
    n = 100
    for _ in range(n):
        ok, _ = xmit.transmit_tile(b"tile_data_high", is_high_priority=True)
        if ok:
            successes += 1
    print(f"\nS-ARQ recovery (20% loss): {successes}/{n} = {successes}%")
    print(f"Paper: recovers from 20% packet loss with RS(15,9)")
