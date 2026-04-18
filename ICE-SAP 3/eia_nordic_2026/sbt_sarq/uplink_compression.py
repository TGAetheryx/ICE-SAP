"""
Uplink compression statistics for SBT.

Paper results:
  Original mask:   18.4 MB
  Compressed:       1.1 MB
  Reduction:       93.8 ± 0.6%
  LoRa payload:    6 KB (215–237 s under multipath SF12/125 kHz)
"""
import numpy as np


UPLINK_ORIGINAL_MB   = 18.4
UPLINK_COMPRESSED_MB = 1.1
UPLINK_REDUCTION_PCT = 93.8
UPLINK_REDUCTION_STD = 0.6


def compute_uplink_reduction(
    original_mb: float,
    compressed_mb: float,
) -> float:
    """Compute uplink reduction percentage."""
    return (1 - compressed_mb / original_mb) * 100


def uplink_reduction_summary():
    red = compute_uplink_reduction(UPLINK_ORIGINAL_MB, UPLINK_COMPRESSED_MB)
    print(f"SBT Uplink Compression:")
    print(f"  Original:   {UPLINK_ORIGINAL_MB} MB")
    print(f"  Compressed: {UPLINK_COMPRESSED_MB} MB")
    print(f"  Reduction:  {red:.1f}%  (paper: {UPLINK_REDUCTION_PCT}±{UPLINK_REDUCTION_STD}%)")


if __name__ == "__main__":
    uplink_reduction_summary()
