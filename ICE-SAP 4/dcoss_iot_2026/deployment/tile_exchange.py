"""
16-px boundary tile exchange between adjacent TGlacierEdge nodes.
"""
import numpy as np
from typing import Tuple


BOUNDARY_TILE_PX = 16


def extract_boundary_tiles(
    pred_mask: np.ndarray,
    W_field: np.ndarray,
    tile_px: int = BOUNDARY_TILE_PX,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Extract left and right 16-px boundary tiles for inter-node exchange.

    Args:
        pred_mask: (H, W) binary segmentation mask.
        W_field:   (H, W) Boundary Decay Field.
        tile_px:   Tile width in pixels.

    Returns:
        left_tile:  (H, tile_px) left border.
        right_tile: (H, tile_px) right border.
    """
    H, W = pred_mask.shape
    left_tile  = pred_mask[:, :tile_px].copy()
    right_tile = pred_mask[:, W - tile_px:].copy()
    return left_tile, right_tile


def merge_boundary_tiles(
    own_mask: np.ndarray,
    neighbor_left: np.ndarray = None,
    neighbor_right: np.ndarray = None,
    tile_px: int = BOUNDARY_TILE_PX,
) -> np.ndarray:
    """
    Merge received boundary tiles from adjacent nodes into own mask.

    Where neighbor tiles are available, replaces own border predictions
    with the consensus of own + neighbor values.

    Args:
        own_mask:       (H, W) own prediction.
        neighbor_left:  (H, tile_px) tile from left neighbor.
        neighbor_right: (H, tile_px) tile from right neighbor.
        tile_px:        Tile width.

    Returns:
        merged: (H, W) updated mask.
    """
    merged = own_mask.copy().astype(np.float32)
    H, W = merged.shape

    if neighbor_left is not None:
        merged[:, :tile_px] = (merged[:, :tile_px] +
                                neighbor_left.astype(np.float32)) / 2.0

    if neighbor_right is not None:
        merged[:, W - tile_px:] = (merged[:, W - tile_px:] +
                                    neighbor_right.astype(np.float32)) / 2.0

    return (merged >= 0.5).astype(np.float32)


if __name__ == "__main__":
    np.random.seed(0)
    H, W = 128, 128
    mask = (np.random.rand(H, W) > 0.5).astype(np.float32)
    Wf   = np.ones((H, W), dtype=np.float32) * 0.5

    left, right = extract_boundary_tiles(mask, Wf)
    print(f"Left tile: {left.shape}, Right tile: {right.shape}")

    merged = merge_boundary_tiles(mask, neighbor_left=left)
    print(f"Merged mask shape: {merged.shape}")
