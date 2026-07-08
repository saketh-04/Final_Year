"""
HumanMM ByteTrack V2 Matching Utilities.

Implements data association utilities used by ByteTrack.

Responsibilities
----------------
- IoU computation
- Hungarian assignment (LAPJV)
- Cost matrix construction
- Kalman gating
- Motion fusion
- Score fusion

Compatible with:
- Detection
- BaseTrack
- KalmanFilter
"""

from __future__ import annotations

from typing import List, Tuple

import lap
import numpy as np

from models.bytetrack.kalman_filter import KalmanFilter

__all__ = [
    "linear_assignment",
    "bbox_iou",
    "iou_distance",
    "fuse_score",
    "gate_cost_matrix",
    "fuse_motion",
]

def bbox_iou(
    boxes_a: np.ndarray,
    boxes_b: np.ndarray,
) -> np.ndarray:
    """
    Compute pairwise IoU between two sets of bounding boxes.

    Args:
        boxes_a:
            Array of shape (N, 4) in XYXY format.
        boxes_b:
            Array of shape (M, 4) in XYXY format.

    Returns:
        IoU matrix of shape (N, M).
    """
    if len(boxes_a) == 0 or len(boxes_b) == 0:
        return np.zeros((len(boxes_a), len(boxes_b)), dtype=np.float32)

    boxes_a = boxes_a.astype(np.float32)
    boxes_b = boxes_b.astype(np.float32)

    # Top-left corner of intersection
    tl = np.maximum(
        boxes_a[:, None, :2],
        boxes_b[None, :, :2],
    )

    # Bottom-right corner of intersection
    br = np.minimum(
        boxes_a[:, None, 2:],
        boxes_b[None, :, 2:],
    )

    wh = np.clip(br - tl, a_min=0.0, a_max=None)

    inter = wh[..., 0] * wh[..., 1]

    area_a = (
        (boxes_a[:, 2] - boxes_a[:, 0]) *
        (boxes_a[:, 3] - boxes_a[:, 1])
    )[:, None]

    area_b = (
        (boxes_b[:, 2] - boxes_b[:, 0]) *
        (boxes_b[:, 3] - boxes_b[:, 1])
    )[None, :]

    union = area_a + area_b - inter

    return inter / np.clip(union, 1e-6, None)