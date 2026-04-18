"""
Life-Sign Mode — minimal edge transmission during extreme events.

When spatial entropy exceeds Φ_crit, system reverts to single-pixel
vector-line edge transmission preserving minimum glacier flow velocity data.
A high-priority "Catastrophic Change Alert" is broadcast via LoRa.
"""
import numpy as np
from typing import Optional


PHI_CRIT_DEFAULT    = 0.90
LIFE_SIGN_POWER_MW  = 5.0
ALERT_MESSAGE       = b"ICESAP_CATASTROPHIC_CHANGE_ALERT"


def extract_glacier_edge_vectorline(
    pred_mask: np.ndarray,
) -> np.ndarray:
    """
    Extract glacier terminus as a 1D vector line (one point per row).

    Used in Life-Sign Mode: minimal data preserving flow velocity reconstruction.

    Args:
        pred_mask: (H, W) binary glacier mask.

    Returns:
        edge_x: (H,) x-coordinates of glacier terminus per row.
                Values are column indices; -1 if no glacier in row.
    """
    H, W = pred_mask.shape
    edge_x = np.full(H, -1, dtype=np.int16)

    for row in range(H):
        glacier_cols = np.where(pred_mask[row] > 0.5)[0]
        if len(glacier_cols) > 0:
            # Terminus = rightmost glacier pixel (ice→water boundary)
            edge_x[row] = int(glacier_cols[-1])

    return edge_x


def encode_lifesign_packet(
    edge_x: np.ndarray,
    timestamp_s: Optional[int] = None,
    node_id: int = 0,
) -> bytes:
    """
    Encode Life-Sign Mode packet for LoRa transmission.

    Format: [ALERT_HEADER(4B)] [node_id(1B)] [timestamp(4B)] [edge_x compressed]

    Args:
        edge_x:      (H,) vector-line edge coordinates.
        timestamp_s: Unix timestamp (optional).
        node_id:     Node identifier.

    Returns:
        packet: bytes (target < 256 B for LoRa SF12/125 kHz efficiency).
    """
    import struct

    header = b"ICLS"   # ICE-SAP Life-Sign
    ts = timestamp_s or 0

    # Compress edge_x: only non-negative values, delta-encoded
    valid = [(i, int(x)) for i, x in enumerate(edge_x) if x >= 0]
    compressed = bytearray()
    prev_row, prev_x = 0, 0
    for row, x in valid[:60]:    # max 60 points for LoRa payload budget
        dr = min(row - prev_row, 255)
        dx = int(x - prev_x) + 128   # offset to unsigned
        dx = max(0, min(255, dx))
        compressed.extend([dr, dx])
        prev_row, prev_x = row, x

    return (header
            + struct.pack(">BIH", node_id, ts, len(valid))
            + bytes(compressed))


def is_life_sign_mode_active(
    spatial_entropy: float,
    phi_crit: float = PHI_CRIT_DEFAULT,
) -> bool:
    """Returns True if spatial entropy exceeds Φ_crit."""
    return spatial_entropy >= phi_crit


if __name__ == "__main__":
    np.random.seed(42)
    mask = (np.random.rand(128, 128) > 0.45).astype(np.float32)
    edge = extract_glacier_edge_vectorline(mask)
    packet = encode_lifesign_packet(edge, timestamp_s=1735689600, node_id=0)
    print(f"Life-Sign packet: {len(packet)} bytes  (target: <256 B for LoRa)")
    print(f"Glacier edge points detected: {(edge >= 0).sum()}")
    print(f"Life-Sign power: {LIFE_SIGN_POWER_MW} mW  (vs 320 mW normal)")
