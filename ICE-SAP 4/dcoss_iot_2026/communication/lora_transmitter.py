"""
LoRa transmitter for TGlacierEdge uplink (DCOSS-IoT §III.D).

Wraps SBT+S-ARQ pipeline with LoRa SX1276 hardware interface.
TX power: 3.5±0.3 mW (SF10, +14 dBm) for 0.23 s per tile.
"""
import numpy as np
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from eia_nordic_2026.sbt_sarq.semantic_block import run_sbt_sarq_pipeline
from eia_nordic_2026.hardware.lora.sx1276_driver import SX1276Driver


class LoRaTransmitter:
    """TGlacierEdge LoRa uplink transmitter."""

    def __init__(self, simulation: bool = True, node_id: int = 0,
                 tau: float = 0.5):
        self.node_id = node_id
        self.tau = tau
        self._driver = SX1276Driver(simulation=simulation)
        self._total_bytes_tx = 0
        self._tx_count = 0

    def transmit_mask(
        self,
        pred_mask: np.ndarray,
        W_field: np.ndarray,
        spreading_factor: int = 12,
    ) -> dict:
        """
        Run SBT+S-ARQ and transmit selected tiles via LoRa.

        Returns:
            dict with uplink_mb, reduction_pct, pdr, tx_time_s.
        """
        sbt_result = run_sbt_sarq_pipeline(pred_mask, W_field, tau=self.tau)

        # Encode selected tiles for LoRa
        payload = self._encode_selected_tiles(pred_mask, W_field)
        ok, tx_time = self._driver.transmit(
            payload, spreading_factor=spreading_factor)

        self._total_bytes_tx += len(payload)
        self._tx_count += 1

        return {
            **sbt_result,
            "payload_bytes": len(payload),
            "tx_success": ok,
            "tx_time_s": tx_time,
            "tx_count": self._tx_count,
        }

    def _encode_selected_tiles(
        self,
        pred_mask: np.ndarray,
        W_field: np.ndarray,
        tile_size: int = 16,
    ) -> bytes:
        """Pack selected boundary tiles into LoRa payload."""
        from eia_nordic_2026.sbt_sarq.tile_selection import select_boundary_tiles
        selected, _, _ = select_boundary_tiles(W_field, self.tau, tile_size)

        payload = bytearray()
        for (r, c) in selected[:20]:    # cap at 20 tiles for LoRa 6 KB budget
            tile = pred_mask[r*tile_size:(r+1)*tile_size,
                             c*tile_size:(c+1)*tile_size]
            # Pack as 1-bit per pixel (16×16 = 32 bytes per tile)
            flat = (tile.flatten() >= 0.5).astype(np.uint8)
            bits = np.packbits(flat)
            payload.extend([r, c])    # tile coordinates
            payload.extend(bits)

        return bytes(payload)


if __name__ == "__main__":
    np.random.seed(0)
    H, W = 128, 128
    mask = (np.random.rand(H, W) > 0.5).astype(np.float32)
    Wf = np.zeros((H, W), dtype=np.float32)
    Wf[:, W//2-10:W//2+10] = 0.9

    tx = LoRaTransmitter(simulation=True)
    result = tx.transmit_mask(mask, Wf)
    print(f"Payload: {result['payload_bytes']} bytes")
    print(f"Reduction: {result['reduction_pct']:.1f}%  (paper: 93.8%)")
    print(f"TX success: {result['tx_success']}")
