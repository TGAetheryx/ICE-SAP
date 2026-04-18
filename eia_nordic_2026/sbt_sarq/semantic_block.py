"""
Full SBT+S-ARQ pipeline: semantic tile selection + perception-aware retransmission.

Combines tile_selection, reed_solomon, and blind_retransmission into one pipeline.
"""
import numpy as np
from typing import List, Tuple, Optional
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from eia_nordic_2026.sbt_sarq.tile_selection import select_boundary_tiles
from eia_nordic_2026.sbt_sarq.reed_solomon import rs_encode, rs_decode, simulate_packet_loss
from eia_nordic_2026.sbt_sarq.blind_retransmission import BlindRetransmitter


def run_sbt_sarq_pipeline(
    pred_mask: np.ndarray,
    W_field: np.ndarray,
    tau: float = 0.5,
    tile_size: int = 16,
    loss_rate: float = 0.05,
    rssi_dbm: float = -105.0,
    seed: int = 42,
) -> dict:
    """
    Full SBT+S-ARQ pipeline for one inference cycle.

    Args:
        pred_mask:  (H, W) binary segmentation mask.
        W_field:    (H, W) Boundary Decay Field.
        tau:        Tile selection threshold.
        tile_size:  Tile size (px).
        loss_rate:  Simulated packet loss rate.
        rssi_dbm:   Current RSSI (dBm) for adaptive retransmission.

    Returns:
        dict with uplink_mb, reduction_pct, pdr, n_selected, n_total.
    """
    H, W = pred_mask.shape
    n_total = (H // tile_size) * (W // tile_size)

    # Step 1: Select high-priority tiles
    selected, tile_mask, uplink_frac = select_boundary_tiles(
        W_field, tau=tau, tile_size=tile_size)
    n_selected = len(selected)

    # Step 2: Encode + transmit selected tiles
    xmit = BlindRetransmitter(loss_simulation_rate=loss_rate, seed=seed)
    xmit.update_rssi(rssi_dbm)

    n_success = 0
    for (r, c) in selected:
        tile = pred_mask[r*tile_size:(r+1)*tile_size,
                         c*tile_size:(c+1)*tile_size]
        tile_bytes = tile.tobytes()
        # RS encode
        enc = rs_encode(tile_bytes[:9] if len(tile_bytes) >= 9
                        else tile_bytes.ljust(9, b'\x00'))
        ok, _ = xmit.transmit_tile(enc, is_high_priority=True,
                                   rssi_history=[rssi_dbm])
        if ok:
            n_success += 1

    pdr = n_success / max(n_selected, 1)

    # Uplink size estimate (bytes per tile: 16×16×1 = 256 B + overhead)
    bytes_per_tile = tile_size * tile_size + 64
    uplink_mb = n_selected * bytes_per_tile / (1024**2)
    full_mb   = n_total * bytes_per_tile / (1024**2)
    reduction = (1 - uplink_mb / full_mb) * 100 if full_mb > 0 else 0

    return {
        "n_selected": n_selected,
        "n_total": n_total,
        "uplink_fraction": uplink_frac,
        "uplink_mb": uplink_mb,
        "full_mb": full_mb,
        "reduction_pct": reduction,
        "pdr": pdr,
        "n_success": n_success,
    }


if __name__ == "__main__":
    np.random.seed(42)
    H, W = 128, 128
    pred = (np.random.rand(H, W) > 0.5).astype(np.float32)
    Wf = np.zeros((H, W), dtype=np.float32)
    Wf[:, W//2-10:W//2+10] = 0.9
    Wf[:, W//2-20:W//2-10] = 0.5
    Wf[:, W//2+10:W//2+20] = 0.5

    result = run_sbt_sarq_pipeline(pred, Wf, loss_rate=0.05)
    print(f"Selected: {result['n_selected']}/{result['n_total']} tiles")
    print(f"Reduction: {result['reduction_pct']:.1f}%  (paper: 93.8%)")
    print(f"PDR: {result['pdr']*100:.1f}%")
