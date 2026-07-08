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

def iou_distance(
    tracks: List,
    detections: List,
) -> np.ndarray:
    """
    Compute IoU-based cost matrix.

    Cost = 1 - IoU

    Compatible with:
        - Track objects
        - Detection objects
        - Raw numpy XYXY arrays

    Args:
        tracks:
            Existing tracks.

        detections:
            Current detections.

    Returns:
        Cost matrix of shape (N_tracks, N_detections).
    """

    if len(tracks) == 0 or len(detections) == 0:
        return np.zeros(
            (len(tracks), len(detections)),
            dtype=np.float32,
        )

    # Already numpy arrays
    if isinstance(tracks[0], np.ndarray):
        track_boxes = np.asarray(tracks, dtype=np.float32)

    else:
        track_boxes = np.asarray(
            [
                t.tlbr if hasattr(t, "tlbr")
                else t.bbox
                for t in tracks
            ],
            dtype=np.float32,
        )

    if isinstance(detections[0], np.ndarray):
        det_boxes = np.asarray(
            detections,
            dtype=np.float32,
        )

    else:
        det_boxes = np.asarray(
            [
                d.bbox
                for d in detections
            ],
            dtype=np.float32,
        )

    iou = bbox_iou(track_boxes, det_boxes)

    return 1.0 - iou

def linear_assignment(
    cost_matrix: np.ndarray,
    thresh: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Solve the Linear Assignment Problem using LAPJV.

    Args:
        cost_matrix:
            Cost matrix of shape (N_tracks, N_detections).

        thresh:
            Maximum allowed matching cost.

    Returns:
        matches:
            ndarray of shape (K,2)

        unmatched_tracks:
            ndarray

        unmatched_detections:
            ndarray
    """

    if cost_matrix.size == 0:
        return (
            np.empty((0, 2), dtype=np.int32),
            np.arange(cost_matrix.shape[0]),
            np.arange(cost_matrix.shape[1]),
        )

    _, x, y = lap.lapjv(
        cost_matrix,
        extend_cost=True,
        cost_limit=thresh,
    )

    matches = [
        [i, j]
        for i, j in enumerate(x)
        if j >= 0
    ]

    unmatched_tracks = np.where(x < 0)[0]
    unmatched_detections = np.where(y < 0)[0]

    return (
        np.asarray(matches, dtype=np.int32),
        unmatched_tracks,
        unmatched_detections,
    )