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