"""
Multi-node synchronisation for 3-node TGlacierEdge deployment.

Three RPi 4 nodes spaced 490–510 m apart independently infer 128×128
six-band patches. Adjacent nodes exchange 16-px boundary tiles.
τ-passing sessions uplink compressed masks (18.4→1.1 MB, LoRa-ready).

Node consensus (§III.A):
  Boundary IoU improvement from consensus: reduces single-node uncertainty.
  Single-node fault:    −2.9 pp IoU (88.1±0.3%)
  Dual-node comms out:  −4.7 pp IoU (86.3±0.3%)
"""
import numpy as np
from typing import List, Optional, Dict, Tuple


NODE_SPACING_M = [490, 510]   # 490–510 m apart
N_NODES        = 3
BOUNDARY_TILE_PX = 16         # pixels exchanged between adjacent nodes


class NodeSynchroniser:
    """
    Coordinates boundary tile exchange between 3 RPi 4 nodes.

    Node topology: Node 0 ↔ Node 1 ↔ Node 2  (linear arrangement)
    """

    def __init__(self, n_nodes: int = N_NODES):
        self.n_nodes = n_nodes
        self._node_masks: Dict[int, Optional[np.ndarray]] = {
            i: None for i in range(n_nodes)
        }
        self._node_active: Dict[int, bool] = {i: True for i in range(n_nodes)}

    def submit_mask(self, node_id: int, pred_mask: np.ndarray) -> None:
        """Node submits its segmentation mask for consensus."""
        if 0 <= node_id < self.n_nodes:
            self._node_masks[node_id] = pred_mask.astype(np.float32)

    def set_node_status(self, node_id: int, active: bool) -> None:
        """Mark a node as active or faulted."""
        self._node_active[node_id] = active
        if not active:
            self._node_masks[node_id] = None

    def compute_consensus_mask(
        self, tau: float = 0.5
    ) -> Tuple[np.ndarray, int]:
        """
        Compute consensus mask by averaging available node predictions.

        Args:
            tau: Binarisation threshold.

        Returns:
            consensus: (H, W) binary consensus mask.
            n_active:  Number of active nodes used.
        """
        available = [
            self._node_masks[i]
            for i in range(self.n_nodes)
            if self._node_active[i] and self._node_masks[i] is not None
        ]
        n_active = len(available)

        if n_active == 0:
            # Safety fallback: return empty mask
            H, W = 128, 128
            return np.zeros((H, W), dtype=np.float32), 0

        stacked = np.stack(available, axis=0)
        avg = stacked.mean(axis=0)
        return (avg >= tau).astype(np.float32), n_active

    def exchange_boundary_tiles(self) -> Dict[Tuple[int, int], np.ndarray]:
        """
        Extract and exchange 16-px boundary tiles between adjacent nodes.

        Returns:
            dict mapping (sender_id, receiver_id) → boundary_tile.
        """
        exchanges = {}
        for i in range(self.n_nodes - 1):
            if not (self._node_active[i] and self._node_active[i+1]):
                continue
            mask_i   = self._node_masks[i]
            mask_ip1 = self._node_masks[i+1]

            if mask_i is not None:
                H, W = mask_i.shape
                # Right border of node i → sent to node i+1
                tile = mask_i[:, W-BOUNDARY_TILE_PX:]
                exchanges[(i, i+1)] = tile

            if mask_ip1 is not None:
                H, W = mask_ip1.shape
                # Left border of node i+1 → sent to node i
                tile = mask_ip1[:, :BOUNDARY_TILE_PX]
                exchanges[(i+1, i)] = tile

        return exchanges


if __name__ == "__main__":
    np.random.seed(42)
    sync = NodeSynchroniser(n_nodes=3)

    for node_id in range(3):
        mask = (np.random.rand(128, 128) > 0.5).astype(np.float32)
        sync.submit_mask(node_id, mask)

    consensus, n_active = sync.compute_consensus_mask()
    print(f"Consensus from {n_active} nodes, shape={consensus.shape}")

    # Test single-node fault
    sync.set_node_status(1, False)
    consensus_fault, n_active_fault = sync.compute_consensus_mask()
    print(f"After single-node fault: {n_active_fault} nodes active")
    print(f"Expected IoU loss: −2.9 pp  (paper Table III)")
