"""
Semantic Block Transmission (SBT) — tile selection.

SBT selects only tiles satisfying W(x) > τ for uplink transmission.
Result: 18.4 MB → 1.1 MB (93.8% reduction).

Each 128×128 patch is divided into 16×16 tiles. Only tiles whose
mean W(x) exceeds the threshold τ (default 0.5) are transmitted.

SBT with τ-passing sessions:
  - Adjacent nodes exchange 16-px boundary tiles
  - τ-passing sessions uplink compressed masks (18.4→1.1 MB, LoRa-ready)
  - LoRa 6 KB payload: 215–237 s under multipath

References:
  ICE-SAP §3.4, TGlacierEdge §III.D.
"""
import numpy as np
from typing import List, Tuple, Optional


# Paper constants
UPLINK_ORIGINAL_MB   = 18.4
UPLINK_COMPRESSED_MB = 1.1
UPLINK_REDUCTION_PCT = 93.8
TAU_DEFAULT          = 0.5
TILE_SIZE_PX         = 16
LORA_PAYLOAD_KB      = 6


def select_boundary_tiles(
    W_field: np.ndarray,
    tau: float = TAU_DEFAULT,
    tile_size: int = TILE_SIZE_PX,
) -> Tuple[List[Tuple[int, int]], np.ndarray, float]:
    """
    Select tiles where mean W(x) > τ for transmission.

    Args:
        W_field:   (H, W) Boundary Decay Field ∈ [0, 1].
        tau:       Selection threshold (default 0.5).
        tile_size: Tile size in pixels (default 16).

    Returns:
        selected_tiles:  List of (row, col) tile indices.
        tile_mask:       (n_tiles_h, n_tiles_w) binary selection mask.
        uplink_fraction: Fraction of tiles selected (target: ~6.2%).
    """
    H, W = W_field.shape
    n_h = H // tile_size
    n_w = W // tile_size
    tile_mask = np.zeros((n_h, n_w), dtype=bool)
    selected = []

    for r in range(n_h):
        for c in range(n_w):
            tile = W_field[r*tile_size:(r+1)*tile_size,
                           c*tile_size:(c+1)*tile_size]
            if tile.mean() > tau:
                tile_mask[r, c] = True
                selected.append((r, c))

    total = n_h * n_w
    frac = len(selected) / max(total, 1)
    return selected, tile_mask, frac


def compute_uplink_size_mb(
    n_selected_tiles: int,
    tile_size: int = TILE_SIZE_PX,
    bits_per_pixel: float = 8.0,
    overhead_bytes: int = 64,
) -> float:
    """
    Estimate uplink data size for selected tiles.

    Args:
        n_selected_tiles: Number of tiles to transmit.
        tile_size:        Tile spatial size (px).
        bits_per_pixel:   Encoded bits per pixel (INT8 = 8).
        overhead_bytes:   Per-tile header overhead.

    Returns:
        size_mb: Estimated uplink size in MB.
    """
    pixels_per_tile = tile_size * tile_size
    bytes_per_tile = (pixels_per_tile * bits_per_pixel / 8) + overhead_bytes
    total_bytes = n_selected_tiles * bytes_per_tile
    return total_bytes / (1024 ** 2)


def estimate_lora_transmission_time(
    payload_kb: float,
    spreading_factor: int = 12,
    bandwidth_khz: float = 125.0,
    coding_rate: float = 4/5,
) -> float:
    """
    Estimate LoRa transmission time for a given payload.

    Paper result: LoRa 6 KB payload → 215–237 s under multipath (SF12/125 kHz).

    Args:
        payload_kb:      Payload size in KB.
        spreading_factor: LoRa SF (default 12 for long range).
        bandwidth_khz:   LoRa bandwidth (default 125 kHz).
        coding_rate:     LoRa coding rate (4/5 = 0.8).

    Returns:
        time_s: Estimated transmission time in seconds.
    """
    # Symbol rate
    symbol_rate = bandwidth_khz * 1000 / (2 ** spreading_factor)
    # Bits per symbol (with coding rate)
    bits_per_symbol = spreading_factor * coding_rate
    # Total payload bits
    total_bits = payload_kb * 1024 * 8
    # Add LoRa overhead (~13 symbol preamble + header)
    total_bits += 13 * bits_per_symbol + 20 * 8
    # Time estimate
    time_s = total_bits / (symbol_rate * bits_per_symbol)
    # Multipath adds 0–10% overhead
    return time_s * 1.05


def extract_boundary_tiles_for_exchange(
    W_field: np.ndarray,
    pred_mask: np.ndarray,
    tile_size: int = 16,
    border_px: int = 16,
) -> np.ndarray:
    """
    Extract 16-px boundary tiles for inter-node exchange (TGlacierEdge §III.A).

    Adjacent RPi 4 nodes exchange boundary tiles to ensure consistent
    glacier boundary delineation across overlapping spatial regions.

    Args:
        W_field:    (H, W) Boundary Decay Field.
        pred_mask:  (H, W) binary segmentation mask.
        tile_size:  Tile size for exchange (default 16 px).
        border_px:  Border width to extract (default 16 px).

    Returns:
        border_tiles: (H, border_px) left + right border tiles concatenated.
    """
    H, W = pred_mask.shape
    left_border  = pred_mask[:, :border_px].copy()
    right_border = pred_mask[:, W-border_px:].copy()
    return np.concatenate([left_border, right_border], axis=1)


if __name__ == "__main__":
    np.random.seed(42)
    H, W = 128, 128
    # Simulate BDF: high near vertical boundary
    W_field = np.zeros((H, W), dtype=np.float32)
    W_field[:, W//2-8:W//2+8] = 0.95
    W_field[:, W//2-20:W//2-8] = 0.6
    W_field[:, W//2+8:W//2+20] = 0.6

    tiles, mask, frac = select_boundary_tiles(W_field, tau=TAU_DEFAULT)
    n_total = (H // TILE_SIZE_PX) * (W // TILE_SIZE_PX)
    uplink = compute_uplink_size_mb(len(tiles))
    full_uplink = compute_uplink_size_mb(n_total)
    reduction = 1 - uplink / full_uplink

    print(f"Selected tiles:    {len(tiles)}/{n_total} ({frac*100:.1f}%)")
    print(f"Uplink:            {uplink:.3f} MB / {full_uplink:.3f} MB full")
    print(f"Reduction:         {reduction*100:.1f}%  (paper: {UPLINK_REDUCTION_PCT}%)")

    lora_t = estimate_lora_transmission_time(LORA_PAYLOAD_KB)
    print(f"LoRa TX time:      {lora_t:.0f} s  (paper: 215–237 s)")
