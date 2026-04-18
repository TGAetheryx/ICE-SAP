"""
Evaluation metrics for glacier segmentation and early-warning detection.

Metrics used across the three papers:

Segmentation:
  - IoU (Intersection over Union) — overall
  - Boundary IoU — restricted to boundary pixels (key metric in all papers)
  - Pixel Accuracy
  - Ice-edge error (metres) — median Euclidean distance to GLIMS ground truth
  - Jökulhlaup recall — detection rate for glacial outburst flood events

Early-warning:
  - Lead time (hours) — time between Δ̂_spec alert and confirmed calving
  - False alarm rate — alerts in non-calving windows / total non-calving windows
  - Detection rate — fraction of calving events with ≥24h precursor

References:
  ICE-SAP Table I, II, III (EIA Nordic 2026).
  TGlacierEdge Table I, II, III (DCOSS-IoT 2026).
  ASPT Tables 1–3 (Computers & Geosciences 2026).
"""

import numpy as np
from typing import Optional, List, Tuple, Dict
from scipy.ndimage import distance_transform_edt


# ---------------------------------------------------------------------------
# Segmentation metrics
# ---------------------------------------------------------------------------

def iou(pred: np.ndarray, gt: np.ndarray, threshold: float = 0.5) -> float:
    """
    Intersection over Union (Overall IoU).

    Args:
        pred: (H, W) probability map ∈ [0,1] or binary array.
        gt:   (H, W) binary ground truth.
        threshold: binarisation threshold for probability maps.

    Returns:
        iou_score: float ∈ [0, 1].
    """
    p = (pred >= threshold).astype(bool)
    g = gt.astype(bool)
    intersection = (p & g).sum()
    union = (p | g).sum()
    if union == 0:
        return 1.0 if intersection == 0 else 0.0
    return float(intersection) / float(union)


def boundary_iou(
    pred: np.ndarray,
    gt: np.ndarray,
    boundary_width: int = 5,
    threshold: float = 0.5,
) -> float:
    """
    Boundary IoU — IoU restricted to boundary regions.

    This is the key operational metric: it measures accuracy on the <3%
    of pixels near glacier termini that govern >95% of IoU variance
    (ICE-SAP §III.B, TGlacierEdge §III.B).

    At 10 m/px Sentinel-2 resolution, boundary_width=5 px = 50 m zone.

    Args:
        pred:           (H, W) probability map or binary array.
        gt:             (H, W) binary ground truth.
        boundary_width: pixel width of boundary zone (default 5 = 50 m).
        threshold:      binarisation threshold.

    Returns:
        boundary_iou_score: float ∈ [0, 1].
    """
    from scipy.ndimage import binary_dilation

    p = (pred >= threshold).astype(bool)
    g = gt.astype(bool)

    # Boundary region of ground truth
    g_boundary = binary_dilation(g ^ binary_dilation(g)) | \
                 (binary_dilation(g, iterations=boundary_width) & ~g)
    # Simpler: pixels within boundary_width of the GT boundary
    gt_dist = distance_transform_edt(g) + distance_transform_edt(~g)
    boundary_zone = gt_dist <= boundary_width

    if boundary_zone.sum() == 0:
        return float(iou(pred, gt, threshold))

    p_b = p[boundary_zone]
    g_b = g[boundary_zone]
    intersection = (p_b & g_b).sum()
    union = (p_b | g_b).sum()
    if union == 0:
        return 1.0
    return float(intersection) / float(union)


def pixel_accuracy(pred: np.ndarray, gt: np.ndarray,
                   threshold: float = 0.5) -> float:
    """
    Pixel-level accuracy.
    """
    p = (pred >= threshold).astype(bool)
    g = gt.astype(bool)
    return float((p == g).mean())


def ice_edge_error(
    pred: np.ndarray,
    gt: np.ndarray,
    pixel_size_m: float = 10.0,
    threshold: float = 0.5,
) -> float:
    """
    Median Euclidean ice-edge localisation error (metres).

    For each predicted boundary pixel, finds the distance to the nearest
    ground-truth boundary pixel. Reports the median over all predicted pixels.

    At Sentinel-2 (10 m/px), a 1-pixel error = 10 m.
    Target: < 10 m (≤1 px) for sub-diurnal ablation scale monitoring.
    TGlacierEdge achieves 8.2 m (Table II); KD baseline is 13.1 m.

    Args:
        pred:         (H, W) probability or binary map.
        gt:           (H, W) binary ground truth.
        pixel_size_m: Pixel size in metres (10 m for Sentinel-2).
        threshold:    Binarisation threshold.

    Returns:
        median_error_m: float.
    """
    p = (pred >= threshold).astype(bool)
    g = gt.astype(bool)

    # Extract prediction boundary
    from scipy.ndimage import binary_erosion
    pred_boundary = p & ~binary_erosion(p)
    gt_boundary   = g & ~binary_erosion(g)

    if pred_boundary.sum() == 0 or gt_boundary.sum() == 0:
        return float("nan")

    # Distance from GT boundary (map gives distance at each non-GT-boundary pixel)
    gt_dist_map = distance_transform_edt(~gt_boundary)

    # Distances at predicted boundary pixels
    errors_px = gt_dist_map[pred_boundary]
    return float(np.median(errors_px)) * pixel_size_m


def jokulhlaup_recall(
    pred: np.ndarray,
    gt_events: List[np.ndarray],
    fpr_threshold: float = 0.017,
    threshold: float = 0.5,
) -> float:
    """
    Jökulhlaup (glacial lake outburst flood) early-warning recall.

    Recall = TP / (TP + FN) for glacial outburst flood events.
    Target: ≥98% recall (GLIMS/RGI standards, ICE-SAP §IV.B).
    TGlacierEdge achieves 98.7% (Table II).

    An event is detected if the predicted mask shows ≥10 m ice-margin
    displacement within a 24-h detection window.

    Args:
        pred:         (H, W) probability map for the event scene.
        gt_events:    List of (H, W) binary event ground-truth masks.
        fpr_threshold: Maximum acceptable FPR (default 1.7%).
        threshold:    Binarisation threshold.

    Returns:
        recall: float.

    Note: This is a simplified version. Full evaluation uses 23
    consensus-annotated events (ASPT §3.1).
    """
    p = (pred >= threshold).astype(bool)
    detected = 0
    for gt_event in gt_events:
        g = gt_event.astype(bool)
        iou_score = iou(p, g)
        if iou_score > 0.3:    # event detected
            detected += 1
    return detected / len(gt_events) if gt_events else 0.0


def compute_all_metrics(
    pred: np.ndarray,
    gt: np.ndarray,
    pixel_size_m: float = 10.0,
    threshold: float = 0.5,
) -> Dict[str, float]:
    """Compute all segmentation metrics in one call."""
    return {
        "iou": iou(pred, gt, threshold),
        "boundary_iou": boundary_iou(pred, gt, threshold=threshold),
        "pixel_accuracy": pixel_accuracy(pred, gt, threshold),
        "ice_edge_error_m": ice_edge_error(pred, gt, pixel_size_m, threshold),
    }


# ---------------------------------------------------------------------------
# Early-warning metrics
# ---------------------------------------------------------------------------

def compute_lead_time(
    delta_series: np.ndarray,
    calving_time_idx: int,
    threshold: float = None,
    percentile: float = 5.0,
    window_days: int = 30,
    revisit_hours: float = 4.0,
    min_consecutive: int = 3,
) -> Optional[float]:
    """
    Compute lead time (hours) between Δ̂_spec anomaly and calving event.

    Detection criterion (ASPT §5.4):
      Sustained drop below the 5th percentile of the 30-day rolling baseline,
      maintained for ≥3 consecutive observations.

    Args:
        delta_series:     (T,) array of Δ̂_spec values.
        calving_time_idx: Index of confirmed calving event.
        threshold:        Fixed threshold; if None, computed from percentile.
        percentile:       Rolling baseline percentile (default 5).
        window_days:      Rolling window size (days).
        revisit_hours:    Sentinel-2 revisit cadence (hours, default ~4 days).
        min_consecutive:  Minimum consecutive observations below threshold.

    Returns:
        lead_time_hours: float, or None if not detected.
    """
    T = len(delta_series)
    window_obs = int(window_days * 24 / revisit_hours)   # observations in window

    # Compute causal rolling baseline (no future data leakage)
    alert_idx = None
    consecutive = 0

    for t in range(min_consecutive, calving_time_idx + 1):
        # Rolling baseline: [t−window, t−1]
        t_start = max(0, t - window_obs)
        baseline = delta_series[t_start:t]
        if len(baseline) < 4:
            continue

        thresh = np.percentile(baseline, percentile) if threshold is None \
            else threshold

        if delta_series[t] < thresh:
            consecutive += 1
        else:
            consecutive = 0

        if consecutive >= min_consecutive and alert_idx is None:
            alert_idx = t - min_consecutive + 1    # first observation below thresh

    if alert_idx is None:
        return None

    lead_obs = calving_time_idx - alert_idx
    return float(lead_obs * revisit_hours)


def false_alarm_rate(
    delta_series: np.ndarray,
    calving_indices: List[int],
    non_calving_windows: List[Tuple[int, int]],
    percentile: float = 5.0,
    window_days: int = 30,
    revisit_hours: float = 4.0,
    min_consecutive: int = 3,
) -> float:
    """
    False alarm rate = alerts in non-calving windows / total non-calving windows.

    ASPT paper: 3/88 = 3.4% combined; 2/47 = 4.3% Breiðamerkurjökull.
    TGlacierEdge: FPR ≤ 1.7%.

    Args:
        delta_series:         (T,) Δ̂_spec time series.
        calving_indices:      Indices of confirmed calving events.
        non_calving_windows:  List of (start, end) index tuples for non-calving
                              30-day windows.
        percentile, window_days, revisit_hours, min_consecutive: same as above.

    Returns:
        far: float ∈ [0, 1].
    """
    false_alarms = 0
    for (start, end) in non_calving_windows:
        window_series = delta_series[start:end]
        if len(window_series) < 4:
            continue
        # Any alert in this non-calving window
        lead = compute_lead_time(
            delta_series=delta_series[start:end + 1],
            calving_time_idx=len(window_series),
            percentile=percentile,
            window_days=window_days,
            revisit_hours=revisit_hours,
            min_consecutive=min_consecutive,
        )
        if lead is not None:
            false_alarms += 1

    total = len(non_calving_windows)
    return false_alarms / total if total > 0 else 0.0


# ---------------------------------------------------------------------------
# Multi-run statistics
# ---------------------------------------------------------------------------

def mean_std(values: List[float]) -> Tuple[float, float]:
    """Return (mean, std) of a list of values."""
    a = np.array(values, dtype=np.float64)
    return float(a.mean()), float(a.std())


if __name__ == "__main__":
    # Quick test with random arrays
    np.random.seed(0)
    H, W = 128, 128
    pred = np.random.rand(H, W).astype(np.float32)
    gt = (np.random.rand(H, W) > 0.5).astype(np.float32)

    metrics = compute_all_metrics(pred, gt)
    for k, v in metrics.items():
        print(f"  {k:25s}: {v:.4f}")
