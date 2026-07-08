"""
HumanMM — Image Drawing & Colour Utility Module.

Higher-level drawing primitives built on top of :mod:`utils.frame_utils`.
While ``frame_utils`` handles raw I/O and colour-space conversion, this
module focuses on the drawing operations shared across every renderer in
:mod:`visualization`: bounding boxes, skeletons, trails, colour assignment,
and overlay blending.

Example:
    >>> from utils.image_utils import draw_bbox_with_label, color_for_id
    >>> color = color_for_id(track_id=3)
    >>> draw_bbox_with_label(frame, (10, 10, 100, 200), "person 3", color)
"""

from __future__ import annotations

import colorsys
from typing import Iterable, List, Optional, Sequence, Tuple

import cv2
import numpy as np

from utils.frame_utils import Frame, draw_text_with_background
from utils.logger import get_logger

log = get_logger(__name__)

BGRColor = Tuple[int, int, int]


# ---------------------------------------------------------------------------
# Colour assignment
# ---------------------------------------------------------------------------

def color_for_id(track_id: int, saturation: float = 0.85, value: float = 0.95) -> BGRColor:
    """Deterministically map an integer ID to a distinct, stable BGR colour.

    Uses the golden-ratio hue stepping technique so that consecutive IDs
    receive visually distinct (not gradually-shifting) colours.

    Args:
        track_id: Integer identifier (e.g. a tracking ID).
        saturation: HSV saturation in ``[0, 1]``.
        value: HSV value/brightness in ``[0, 1]``.

    Returns:
        BGR colour tuple with each channel in ``[0, 255]``.

    Example:
        >>> color_for_id(1)
        (37, 201, 245)
    """
    golden_ratio_conjugate = 0.618033988749895
    hue = (track_id * golden_ratio_conjugate) % 1.0
    r, g, b = colorsys.hsv_to_rgb(hue, saturation, value)
    return int(b * 255), int(g * 255), int(r * 255)


def heatmap_color(value: float, vmin: float = 0.0, vmax: float = 1.0) -> BGRColor:
    """Map a scalar value to a BGR colour using a blue→red heatmap ramp.

    Args:
        value: Scalar value to colour-map.
        vmin: Minimum of the expected value range.
        vmax: Maximum of the expected value range.

    Returns:
        BGR colour tuple.
    """
    span = max(vmax - vmin, 1e-9)
    t = float(np.clip((value - vmin) / span, 0.0, 1.0))
    # Apply OpenCV's TURBO colormap for a smoother CVPR-style ramp
    gray = np.array([[int(t * 255)]], dtype=np.uint8)
    bgr = cv2.applyColorMap(gray, cv2.COLORMAP_TURBO)[0, 0]
    return int(bgr[0]), int(bgr[1]), int(bgr[2])


# ---------------------------------------------------------------------------
# Bounding boxes
# ---------------------------------------------------------------------------

def draw_bbox_with_label(
    frame: Frame,
    bbox: Sequence[float],
    label: str,
    color: BGRColor = (0, 255, 0),
    thickness: int = 2,
    font_scale: float = 0.6,
) -> Frame:
    """Draw a rectangle bounding box with a text label above it.

    Args:
        frame: Image to draw on (modified in-place).
        bbox: Box as ``(x1, y1, x2, y2)``.
        label: Text label to render above the box.
        color: BGR box and label background colour.
        thickness: Rectangle line thickness.
        font_scale: Label font scale.

    Returns:
        The modified frame (same array reference).
    """
    x1, y1, x2, y2 = [int(v) for v in bbox]
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)

    if label:
        text_y = max(y1 - 6, 14)
        draw_text_with_background(
            frame,
            label,
            (x1, text_y),
            font_scale=font_scale,
            text_color=(255, 255, 255),
            bg_color=color,
        )
    return frame


def draw_trail(
    frame: Frame,
    points: Sequence[Tuple[int, int]],
    color: BGRColor = (0, 255, 0),
    thickness: int = 2,
    fade: bool = True,
) -> Frame:
    """Draw a polyline trail through a sequence of ``(x, y)`` points.

    Optionally fades older segments by blending toward the frame's existing
    pixel values (older points become progressively more transparent).

    Args:
        frame: Image to draw on (modified in-place).
        points: Ordered list of ``(x, y)`` pixel coordinates, oldest first.
        color: BGR trail colour.
        thickness: Line thickness.
        fade: If ``True``, older trail segments fade toward transparency.

    Returns:
        The modified frame (same array reference).
    """
    n = len(points)
    if n < 2:
        return frame

    for i in range(1, n):
        if fade:
            alpha = i / n
            overlay = frame.copy()
            cv2.line(overlay, points[i - 1], points[i], color, thickness, cv2.LINE_AA)
            cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, dst=frame)
        else:
            cv2.line(frame, points[i - 1], points[i], color, thickness, cv2.LINE_AA)

    return frame


# ---------------------------------------------------------------------------
# Skeletons
# ---------------------------------------------------------------------------

def draw_keypoints(
    frame: Frame,
    keypoints_xy: np.ndarray,
    confidences: Optional[np.ndarray] = None,
    radius: int = 5,
    color: BGRColor = (0, 255, 255),
    conf_threshold: float = 0.0,
) -> Frame:
    """Draw circular markers at each 2D keypoint location.

    Args:
        frame: Image to draw on (modified in-place).
        keypoints_xy: Array of shape ``(J, 2)`` with pixel ``[x, y]`` coords.
        confidences: Optional ``(J,)`` confidence array used to skip
            low-confidence joints and (if ``color`` is ``None``) heatmap
            their colour.
        radius: Marker radius in pixels.
        color: BGR colour for the markers.
        conf_threshold: Minimum confidence required to draw a joint.

    Returns:
        The modified frame (same array reference).
    """
    for j, (x, y) in enumerate(keypoints_xy):
        conf = float(confidences[j]) if confidences is not None else 1.0
        if conf < conf_threshold:
            continue
        cv2.circle(frame, (int(x), int(y)), radius, color, thickness=-1, lineType=cv2.LINE_AA)
    return frame


def draw_skeleton_lines(
    frame: Frame,
    keypoints_xy: np.ndarray,
    skeleton_pairs: Iterable[Tuple[int, int]],
    confidences: Optional[np.ndarray] = None,
    color: BGRColor = (255, 165, 0),
    thickness: int = 2,
    conf_threshold: float = 0.0,
) -> Frame:
    """Draw skeleton connection lines between keypoint pairs.

    Args:
        frame: Image to draw on (modified in-place).
        keypoints_xy: Array of shape ``(J, 2)`` with pixel ``[x, y]`` coords.
        skeleton_pairs: Iterable of ``(joint_a, joint_b)`` index pairs.
        confidences: Optional ``(J,)`` confidence array; a connection is
            skipped if either endpoint is below ``conf_threshold``.
        color: BGR line colour.
        thickness: Line thickness.
        conf_threshold: Minimum joint confidence required to draw the edge.

    Returns:
        The modified frame (same array reference).
    """
    J = keypoints_xy.shape[0]
    for a, b in skeleton_pairs:
        if a >= J or b >= J:
            continue
        if confidences is not None:
            if confidences[a] < conf_threshold or confidences[b] < conf_threshold:
                continue
        pt_a = (int(keypoints_xy[a, 0]), int(keypoints_xy[a, 1]))
        pt_b = (int(keypoints_xy[b, 0]), int(keypoints_xy[b, 1]))
        cv2.line(frame, pt_a, pt_b, color, thickness, cv2.LINE_AA)
    return frame


# ---------------------------------------------------------------------------
# Overlays
# ---------------------------------------------------------------------------

def alpha_blend(base: Frame, overlay: Frame, alpha: float = 0.6) -> Frame:
    """Alpha-blend an overlay frame onto a base frame.

    Args:
        base: Background frame ``(H, W, 3)``.
        overlay: Foreground frame of the same shape.
        alpha: Overlay opacity in ``[0, 1]``.

    Returns:
        New blended frame (does not modify inputs).
    """
    return cv2.addWeighted(overlay, alpha, base, 1.0 - alpha, 0)


def draw_fps_overlay(
    frame: Frame,
    fps: float,
    position: str = "top_right",
    text_color: BGRColor = (0, 255, 0),
    bg_color: BGRColor = (0, 0, 0),
) -> Frame:
    """Draw an FPS counter overlay in a corner of the frame.

    Args:
        frame: Image to draw on (modified in-place).
        fps: Current frames-per-second value to display.
        position: One of ``"top_left"``, ``"top_right"``, ``"bottom_left"``,
            ``"bottom_right"``.
        text_color: BGR text colour.
        bg_color: BGR background rectangle colour.

    Returns:
        The modified frame (same array reference).
    """
    h, w = frame.shape[:2]
    label = f"FPS: {fps:.1f}"

    margin = 10
    if position == "top_left":
        pos = (margin, margin + 18)
    elif position == "bottom_left":
        pos = (margin, h - margin)
    elif position == "bottom_right":
        text_w = 9 * len(label)
        pos = (w - text_w - margin, h - margin)
    else:  # top_right (default)
        text_w = 9 * len(label)
        pos = (w - text_w - margin, margin + 18)

    return draw_text_with_background(
        frame, label, pos, font_scale=0.6, text_color=text_color, bg_color=bg_color
    )


def draw_shot_flash(frame: Frame, color: BGRColor = (255, 0, 255), alpha: float = 0.25) -> Frame:
    """Overlay a translucent full-frame colour flash (shot-cut indicator).

    Args:
        frame: Image to draw on (modified in-place).
        color: BGR flash colour.
        alpha: Flash opacity in ``[0, 1]``.

    Returns:
        The modified frame (same array reference).
    """
    overlay = np.full_like(frame, color, dtype=np.uint8)
    blended = alpha_blend(frame, overlay, alpha=alpha)
    frame[:] = blended
    return frame


def draw_header_bar(
    frame: Frame,
    text: str,
    height: int = 40,
    bg_color: BGRColor = (20, 20, 20),
    text_color: BGRColor = (220, 220, 220),
) -> Frame:
    """Draw a labelled header bar across the top of a frame (in-place).

    Useful for comparison-panel labelling (e.g. "Detection", "Pose").

    Args:
        frame: Image to draw on (modified in-place).
        text: Header label text.
        height: Header bar height in pixels.
        bg_color: BGR header background colour.
        text_color: BGR header text colour.

    Returns:
        The modified frame (same array reference).
    """
    w = frame.shape[1]
    cv2.rectangle(frame, (0, 0), (w, height), bg_color, thickness=-1)
    cv2.putText(
        frame,
        text,
        (10, int(height * 0.68)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        text_color,
        2,
        cv2.LINE_AA,
    )
    return frame


def label_colors_for_tracks(track_ids: Iterable[int]) -> List[BGRColor]:
    """Generate a deterministic colour list, one per track ID.

    Args:
        track_ids: Iterable of integer track identifiers.

    Returns:
        List of BGR colours in the same order as ``track_ids``.
    """
    return [color_for_id(tid) for tid in track_ids]
