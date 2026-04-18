"""Post-processing utilities: morphological cleanup, boundary extraction."""
import numpy as np
from scipy.ndimage import binary_erosion, binary_dilation, label


def clean_segmentation_mask(
    mask: np.ndarray,
    min_area_px: int = 64,
    apply_morphology: bool = True,
) -> np.ndarray:
    """
    Clean binary segmentation mask: remove small components, fill holes.
    """
    mask = mask.astype(bool)
    if apply_morphology:
        mask = binary_dilation(binary_erosion(mask, iterations=2), iterations=2)
    # Remove small connected components
    labeled, n = label(mask)
    for i in range(1, n + 1):
        if (labeled == i).sum() < min_area_px:
            mask[labeled == i] = False
    return mask.astype(np.float32)


def extract_calving_front(
    mask: np.ndarray,
    method: str = "erosion",
) -> np.ndarray:
    """Extract glacier calving front boundary from binary mask."""
    m = mask.astype(bool)
    if method == "erosion":
        return (m & ~binary_erosion(m)).astype(np.float32)
    else:
        return (binary_dilation(m) & ~m).astype(np.float32)
