"""
τ-passing consensus threshold for multi-node TGlacierEdge.

τ-passing sessions: nodes only transmit uplink when consensus metric > τ.
This further reduces unnecessary transmissions beyond SBT tile selection.

SBT+S-ARQ consensus: >99% under fog/thermal throttle (§V.A).
"""
import numpy as np
from typing import List, Optional


TAU_DEFAULT = 0.5


def consensus_vote(
    node_probs: List[np.ndarray],
    tau: float = TAU_DEFAULT,
    weights: Optional[List[float]] = None,
) -> np.ndarray:
    """
    Weighted consensus vote across node probability maps.

    Args:
        node_probs: List of (H, W) probability maps from active nodes.
        tau:        Binarisation threshold.
        weights:    Optional per-node weights (default: uniform).

    Returns:
        consensus_mask: (H, W) binary mask.
    """
    if not node_probs:
        return np.zeros((128, 128), dtype=np.float32)

    if weights is None:
        weights = [1.0 / len(node_probs)] * len(node_probs)

    stacked = np.stack(node_probs, axis=0)      # (N, H, W)
    w = np.array(weights, dtype=np.float32)
    avg = (stacked * w[:, None, None]).sum(axis=0)
    return (avg >= tau).astype(np.float32)


def tau_passing_criterion(
    W_field_mean: float,
    tau: float = TAU_DEFAULT,
) -> bool:
    """
    Determine whether to uplink based on τ-passing criterion.

    Only transmit if mean boundary weight exceeds τ.
    Reduces redundant uplinks during stable (no-change) periods.

    Args:
        W_field_mean: Mean W(x) over selected tiles.
        tau:          Threshold (default 0.5).

    Returns:
        should_uplink: bool.
    """
    return W_field_mean > tau


if __name__ == "__main__":
    np.random.seed(0)
    probs = [np.random.rand(64, 64).astype(np.float32) for _ in range(3)]
    consensus = consensus_vote(probs)
    print(f"Consensus mask: {consensus.shape}, "
          f"glacier fraction: {consensus.mean():.2%}")
    print(f"τ-passing (W_mean=0.6): {tau_passing_criterion(0.6)}")
    print(f"τ-passing (W_mean=0.3): {tau_passing_criterion(0.3)}")
