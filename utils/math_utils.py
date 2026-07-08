"""
HumanMM — General Math Utility Module.

Generic numeric helpers (clamping, interpolation, smoothing, distance
metrics) used by the visualization and exporter layers.  This module is
distinct from :mod:`utils.geometry`, which is dedicated to 3D rotation
representations and Procrustes alignment used by the trajectory aligner.

Example:
    >>> from utils.math_utils import clamp, moving_average_1d, lerp
    >>> v = clamp(1.5, 0.0, 1.0)
    >>> smoothed = moving_average_1d(signal, window=5)
"""

from __future__ import annotations

from typing import Sequence, Tuple, Union

import numpy as np

from utils.logger import get_logger

log = get_logger(__name__)

Number = Union[int, float]


def clamp(value: Number, low: Number, high: Number) -> Number:
    """Clamp ``value`` to the inclusive range ``[low, high]``.

    Args:
        value: Input scalar.
        low: Lower bound.
        high: Upper bound.

    Returns:
        Clamped value.

    Example:
        >>> clamp(1.5, 0.0, 1.0)
        1.0
    """
    return max(low, min(high, value))


def lerp(a: Number, b: Number, t: float) -> float:
    """Linearly interpolate between ``a`` and ``b``.

    Args:
        a: Start value.
        b: End value.
        t: Interpolation factor (not restricted to ``[0, 1]``).

    Returns:
        Interpolated value ``a + (b - a) * t``.
    """
    return a + (b - a) * t


def safe_divide(numerator: Number, denominator: Number, default: float = 0.0) -> float:
    """Divide two numbers, returning ``default`` if the denominator is ~zero.

    Args:
        numerator: Numerator value.
        denominator: Denominator value.
        default: Value to return when ``denominator`` is near zero.

    Returns:
        ``numerator / denominator``, or ``default`` on near-zero division.
    """
    if abs(denominator) < 1e-9:
        return default
    return numerator / denominator


def moving_average_1d(signal: np.ndarray, window: int = 5) -> np.ndarray:
    """Smooth a 1D signal with a centred moving-average filter.

    Edge-replicated padding is used so the output has the same length as
    the input.

    Args:
        signal: 1D array of shape ``(N,)``.
        window: Window size (odd values give a symmetric kernel).

    Returns:
        Smoothed array of shape ``(N,)``.

    Example:
        >>> smoothed = moving_average_1d(np.array([1, 5, 2, 8, 3]), window=3)
    """
    if window <= 1 or signal.size == 0:
        return signal

    pad = window // 2
    padded = np.pad(signal, pad, mode="edge")
    kernel = np.ones(window) / window
    return np.convolve(padded, kernel, mode="valid")[: signal.size]


def exponential_smooth(signal: np.ndarray, alpha: float = 0.3) -> np.ndarray:
    """Apply exponential moving-average smoothing to a 1D signal.

    Args:
        signal: 1D array of shape ``(N,)``.
        alpha: Smoothing factor in ``(0, 1]``; higher = less smoothing.

    Returns:
        Smoothed array of shape ``(N,)``.
    """
    if signal.size == 0:
        return signal

    out = np.empty_like(signal, dtype=np.float64)
    out[0] = signal[0]
    for i in range(1, signal.size):
        out[i] = alpha * signal[i] + (1.0 - alpha) * out[i - 1]
    return out


def euclidean_distance(p1: np.ndarray, p2: np.ndarray) -> float:
    """Compute the Euclidean distance between two N-dimensional points.

    Args:
        p1: Point of shape ``(D,)``.
        p2: Point of shape ``(D,)``.

    Returns:
        Scalar distance.
    """
    return float(np.linalg.norm(np.asarray(p1) - np.asarray(p2)))


def compute_velocity(positions: np.ndarray, fps: float = 30.0) -> np.ndarray:
    """Estimate per-step velocity from a position time series.

    Args:
        positions: Array of shape ``(T, D)``.
        fps: Frame rate used to convert per-frame deltas to per-second units.

    Returns:
        Velocity array of shape ``(T-1, D)``. Empty if ``T < 2``.
    """
    if positions.shape[0] < 2:
        return np.zeros((0, positions.shape[-1] if positions.ndim > 1 else 1))
    return np.diff(positions, axis=0) * fps


def compute_acceleration(positions: np.ndarray, fps: float = 30.0) -> np.ndarray:
    """Estimate per-step acceleration from a position time series.

    Args:
        positions: Array of shape ``(T, D)``.
        fps: Frame rate used for unit conversion.

    Returns:
        Acceleration array of shape ``(T-2, D)``. Empty if ``T < 3``.
    """
    velocity = compute_velocity(positions, fps=fps)
    if velocity.shape[0] < 2:
        return np.zeros((0, velocity.shape[-1] if velocity.ndim > 1 else 1))
    return np.diff(velocity, axis=0) * fps


def normalize_vector(v: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    """Normalise a vector to unit length, guarding against the zero vector.

    Args:
        v: Input vector of arbitrary shape.
        eps: Small constant preventing division by zero.

    Returns:
        Unit vector (or the zero vector unchanged if its norm is ~0).
    """
    norm = float(np.linalg.norm(v))
    if norm < eps:
        return v
    return v / norm


def fps_from_timestamps(timestamps: Sequence[float]) -> float:
    """Compute the average FPS implied by a sequence of timestamps.

    Args:
        timestamps: Monotonically increasing timestamps in seconds.

    Returns:
        Estimated frames per second, or ``0.0`` if insufficient data.
    """
    if len(timestamps) < 2:
        return 0.0
    span = timestamps[-1] - timestamps[0]
    return safe_divide(len(timestamps) - 1, span, default=0.0)


def bbox_iou(box_a: Sequence[float], box_b: Sequence[float]) -> float:
    """Compute the IoU between two ``[x1, y1, x2, y2]`` boxes.

    Args:
        box_a: First bounding box.
        box_b: Second bounding box.

    Returns:
        IoU score in ``[0, 1]``.
    """
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b

    inter_x1, inter_y1 = max(ax1, bx1), max(ay1, by1)
    inter_x2, inter_y2 = min(ax2, bx2), min(ay2, by2)

    inter_w = max(0.0, inter_x2 - inter_x1)
    inter_h = max(0.0, inter_y2 - inter_y1)
    inter_area = inter_w * inter_h

    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter_area

    return safe_divide(inter_area, union, default=0.0)


def bbox_center(bbox: Sequence[float]) -> Tuple[float, float]:
    """Return the ``(cx, cy)`` centre of a ``[x1, y1, x2, y2]`` box.

    Args:
        bbox: Bounding box.

    Returns:
        Centre coordinates.
    """
    x1, y1, x2, y2 = bbox
    return (x1 + x2) / 2.0, (y1 + y2) / 2.0


def percentile_clip(arr: np.ndarray, low_pct: float = 1.0, high_pct: float = 99.0) -> np.ndarray:
    """Clip an array's values to a percentile range (robust to outliers).

    Useful before colour-mapping confidence/depth values for visualization.

    Args:
        arr: Input array of any shape.
        low_pct: Lower percentile bound.
        high_pct: Upper percentile bound.

    Returns:
        Clipped array of the same shape.
    """
    if arr.size == 0:
        return arr
    lo, hi = np.percentile(arr, [low_pct, high_pct])
    if hi - lo < 1e-9:
        return arr
    return np.clip(arr, lo, hi)


def min_max_normalize(arr: np.ndarray, eps: float = 1e-9) -> np.ndarray:
    """Rescale an array's values to ``[0, 1]`` using min-max normalisation.

    Args:
        arr: Input array of any shape.
        eps: Small constant guarding against a zero-range array.

    Returns:
        Normalised array of the same shape.
    """
    if arr.size == 0:
        return arr
    lo, hi = float(arr.min()), float(arr.max())
    if hi - lo < eps:
        return np.zeros_like(arr, dtype=np.float64)
    return (arr - lo) / (hi - lo)
