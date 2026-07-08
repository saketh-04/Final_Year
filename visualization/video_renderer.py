"""
HumanMM — CVPR-Style Detection & Tracking Renderer (v4).

Research-quality visualization matching professional CV demo standards.

Stage 02 — Detection  (DetectionRenderer, unchanged)
    Tight YOLO bounding boxes on the original video with confidence labels.

Stage 03 + 07_final — Tracking  (TrackingRenderer, v4)
    User-specified elements ONLY (per requirements):
        ✓  Solid green bounding box  (CVPR green, all persons same colour)
        ✓  Track ID  (#1, #2, …)
        ✓  Confidence  (0.94)
        ✓  FPS  (HUD top-left)
        ✓  Frame Number  (HUD top-left)
        ✗  No trails  (removed)
        ✗  No dashed predicted-state boxes  (removed)
        ✗  No boxes away from the real detected person  (only Confirmed shown)

HUD (top-left semi-transparent panel):
    FPS | Frame | Persons

Public API unchanged: DetectionRenderer, TrackingRenderer with .render() methods.
draw_skeleton_on_canvas() is also unchanged.
"""

from __future__ import annotations

import colorsys
from collections import defaultdict, deque
from typing import Any, Deque, Dict, List, Optional, Tuple

import cv2
import numpy as np

from models.bytetrack_tracker import Track, TrackState
from models.yolo_detector import Detection
from utils.frame_utils import Frame
from utils.logger import get_logger

log = get_logger(__name__)

BGRColor = Tuple[int, int, int]

# ---------------------------------------------------------------------------
# CVPR-standard colour palette
# ---------------------------------------------------------------------------

_CVPR_GREEN:   BGRColor = (0, 230, 50)     # primary tracking box colour
_CVPR_TEXT_BG: BGRColor = (10, 10, 10)     # label background
_CVPR_TEXT_FG: BGRColor = (240, 240, 240)  # label text
_CVPR_HUD_BG:  BGRColor = (0, 0, 0)        # HUD background (semi-transparent)
_WHITE:        BGRColor = (255, 255, 255)

# COCO-17 skeleton connectivity
_SKELETON_PAIRS = [
    (0, 1), (0, 2), (1, 3), (2, 4),          # head
    (5, 6),                                    # shoulders
    (5, 7), (7, 9),                            # left arm
    (6, 8), (8, 10),                           # right arm
    (5, 11), (6, 12),                          # torso-hip
    (11, 12),                                  # hips
    (11, 13), (13, 15),                        # left leg
    (12, 14), (14, 16),                        # right leg
]

# Per-limb colours (BGR) — matches standard CVPR pose visualisation
_LIMB_COLORS: List[BGRColor] = [
    (0, 215, 255),   # head
    (0, 215, 255),
    (0, 215, 255),
    (0, 215, 255),
    (255, 128, 0),   # shoulders
    (0, 255, 0),     # left arm
    (0, 255, 0),
    (0, 128, 255),   # right arm
    (0, 128, 255),
    (255, 0, 128),   # torso
    (255, 0, 128),
    (128, 0, 255),   # hips
    (255, 200, 0),   # left leg
    (255, 200, 0),
    (0, 200, 255),   # right leg
    (0, 200, 255),
]

_JOINT_COLOR:   BGRColor = (255, 255, 255)  # white joints
_JOINT_OUTLINE: BGRColor = (0, 0, 0)        # black outline for visibility


def _person_color(track_id: int) -> BGRColor:
    """Deterministic, visually distinct colour for each track ID.
    Used by DetectionRenderer; TrackingRenderer v4 ignores this and uses green.
    """
    golden = 0.618033988749895
    hue = (track_id * golden) % 1.0
    r, g, b = colorsys.hsv_to_rgb(hue, 0.90, 1.0)
    return int(b * 255), int(g * 255), int(r * 255)


# ---------------------------------------------------------------------------
# Drawing primitives
# ---------------------------------------------------------------------------

def _draw_bbox(
    img: Frame,
    x1: int, y1: int, x2: int, y2: int,
    color: BGRColor,
    thickness: int = 2,
) -> None:
    """Draw a clean, thin bounding box (no rounded corners — YOLO demo style)."""
    cv2.rectangle(img, (x1, y1), (x2, y2), color, thickness, cv2.LINE_AA)


def _draw_label(
    img: Frame,
    text: str,
    x: int, y: int,
    text_color: BGRColor = _CVPR_TEXT_FG,
    bg_color: BGRColor   = _CVPR_TEXT_BG,
    font_scale: float    = 0.50,
    thickness: int       = 1,
) -> None:
    """Draw a compact label tag above a bounding box."""
    font = cv2.FONT_HERSHEY_SIMPLEX
    (tw, th), baseline = cv2.getTextSize(text, font, font_scale, thickness)
    pad = 3
    bx1, by1 = x,          y - th - pad * 2
    bx2, by2 = x + tw + pad * 2, y + baseline - pad

    cv2.rectangle(img, (bx1, by1), (bx2, by2), bg_color, -1)
    cv2.putText(img, text, (x + pad, y - pad), font, font_scale,
                text_color, thickness, cv2.LINE_AA)


def _draw_hud(
    img:       Frame,
    fps:       float,
    frame_idx: int,
    n_persons: int,
    latency_ms: float = 0.0,
    show_latency: bool = False,
) -> None:
    """Draw a minimal top-left HUD with key stats (semi-transparent panel).

    Per user requirements, the HUD shows ONLY:
        FPS | Frame | Persons (active track count)
    Latency is optional and hidden by default.
    """
    lines: List[str] = []
    if fps > 0.0:
        lines.append(f"FPS {fps:.1f}")
    lines.append(f"Frame {frame_idx}")
    lines.append(f"Persons {n_persons}")
    if show_latency and latency_ms > 0.0:
        lines.append(f"Lat {latency_ms:.0f}ms")

    font    = cv2.FONT_HERSHEY_SIMPLEX
    fscale  = 0.55
    fthick  = 1
    pad     = 6
    line_h  = 22

    if not lines:
        return

    max_w   = max(cv2.getTextSize(l, font, fscale, fthick)[0][0] for l in lines)
    block_h = line_h * len(lines) + pad
    block_w = max_w + pad * 2

    # Semi-transparent dark background
    overlay = img.copy()
    cv2.rectangle(overlay, (6, 6), (6 + block_w, 6 + block_h), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.55, img, 0.45, 0, img)

    for i, line in enumerate(lines):
        y = 6 + pad + (i + 1) * line_h - 4
        cv2.putText(img, line, (6 + pad, y), font, fscale,
                    (220, 220, 220), fthick, cv2.LINE_AA)


def _draw_trail(
    img:    Frame,
    points: List[Tuple[int, int]],
    color:  BGRColor,
) -> None:
    """Draw a fading foot-point trail (used when trail_enabled=True)."""
    n = len(points)
    if n < 2:
        return
    for i in range(1, n):
        alpha     = (i / n) * 0.7
        thickness = max(1, int(3 * i / n))
        overlay   = img.copy()
        cv2.line(overlay, points[i - 1], points[i], color, thickness, cv2.LINE_AA)
        cv2.addWeighted(overlay, alpha, img, 1.0 - alpha, 0, img)


# ---------------------------------------------------------------------------
# Skeleton canvas renderer (right panel — white background)
# ---------------------------------------------------------------------------

def draw_skeleton_on_canvas(
    canvas:        Frame,
    keypoints_xy:  np.ndarray,
    confidences:   Optional[np.ndarray],
    color:         BGRColor,
    conf_threshold: float = 0.3,
) -> None:
    """Draw a stick-figure skeleton on a white canvas.

    Args:
        canvas:        White BGR canvas to draw on (modified in-place).
        keypoints_xy:  Array of shape ``(J, 2)`` with pixel ``[x, y]`` coords.
        confidences:   Optional ``(J,)`` confidence scores.
        color:         Per-person identity colour.
        conf_threshold: Skip joints below this confidence.
    """
    J = keypoints_xy.shape[0]

    # Draw limbs first (behind joints)
    for idx, (a, b) in enumerate(_SKELETON_PAIRS):
        if a >= J or b >= J:
            continue
        if confidences is not None:
            if confidences[a] < conf_threshold or confidences[b] < conf_threshold:
                continue
        limb_color = _LIMB_COLORS[idx] if idx < len(_LIMB_COLORS) else color
        pt_a = (int(keypoints_xy[a, 0]), int(keypoints_xy[a, 1]))
        pt_b = (int(keypoints_xy[b, 0]), int(keypoints_xy[b, 1]))
        cv2.line(canvas, pt_a, pt_b, limb_color, 3, cv2.LINE_AA)

    # Draw joints
    for j in range(J):
        if confidences is not None and confidences[j] < conf_threshold:
            continue
        cx, cy = int(keypoints_xy[j, 0]), int(keypoints_xy[j, 1])
        cv2.circle(canvas, (cx, cy), 5, _JOINT_OUTLINE, -1, cv2.LINE_AA)
        cv2.circle(canvas, (cx, cy), 3, _JOINT_COLOR, -1, cv2.LINE_AA)


# ---------------------------------------------------------------------------
# Detection renderer — Stage 02 (single-panel)
# ---------------------------------------------------------------------------

class DetectionRenderer:
    """Draws raw YOLO detection boxes with CVPR-style minimal overlay.

    Args:
        config: Detection visualization config dict.

    Example:
        >>> renderer = DetectionRenderer(config=cfg.visualization.detection)
        >>> out = renderer.render(frame, detections, frame_idx=0)
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        self._cfg: Dict[str, Any] = config or {}
        self._color:      BGRColor = tuple(self._cfg.get("bbox_color", list(_CVPR_GREEN)))
        self._thickness:  int      = self._cfg.get("bbox_thickness", 2)
        self._show_conf:  bool     = self._cfg.get("show_confidence", True)
        self._show_frame: bool     = self._cfg.get("show_frame_number", True)

    def render(
        self,
        frame:      Frame,
        detections: List[Detection],
        frame_idx:  int   = 0,
        fps:        float = 0.0,
        latency_ms: float = 0.0,
    ) -> Frame:
        """Draw detections onto the frame.

        Args:
            frame:      BGR input image (modified in-place).
            detections: YOLO detections to render.
            frame_idx:  Current frame index.
            fps:        Pipeline FPS (for HUD).
            latency_ms: Detection latency ms (for HUD).

        Returns:
            Annotated frame.
        """
        for det in detections:
            x1, y1, x2, y2 = int(det.x1), int(det.y1), int(det.x2), int(det.y2)
            h_img, w_img   = frame.shape[:2]
            x1 = max(0, x1);          y1 = max(0, y1)
            x2 = min(w_img - 1, x2);  y2 = min(h_img - 1, y2)

            _draw_bbox(frame, x1, y1, x2, y2, self._color, self._thickness)

            if self._show_conf:
                label = f"person {det.confidence:.2f}"
                _draw_label(frame, label, x1, y1,
                            bg_color=self._color, text_color=(10, 10, 10))

        _draw_hud(frame, fps, frame_idx, len(detections), latency_ms,
                  show_latency=True)
        return frame


# ---------------------------------------------------------------------------
# Tracking renderer — Stage 03 + 07_final  (v4 — green-only, confirmed-only)
# ---------------------------------------------------------------------------

class TrackingRenderer:
    """Draws ByteTrack confirmed tracking boxes — CVPR Research Demo style.

    v4 changes:
        • Renders ONLY ``Confirmed`` tracks (and very-briefly ``Predicted``
          tracks with ``_predicted_frames ≤ 1``).  Tentative / Lost tracks
          are NEVER drawn.
        • All boxes are solid CVPR green — no per-person colour variation.
        • Label format: ``#ID  0.XX`` (track ID + confidence).
        • HUD shows: FPS, Frame, Persons.
        • Trails disabled by default (not in user's "Show only" list).
        • No dashed/transparent predicted-state overlay.

    Args:
        config: Tracking visualization config dict.

    Example:
        >>> renderer = TrackingRenderer(config=cfg.visualization.tracking)
        >>> out = renderer.render(frame, tracks, frame_idx=42)
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        self._cfg: Dict[str, Any] = config or {}

        # Label elements
        self._show_id:       bool  = self._cfg.get("show_track_id",   True)
        self._show_conf:     bool  = self._cfg.get("show_confidence",  True)
        self._show_fps:      bool  = self._cfg.get("show_fps",         True)
        self._show_latency:  bool  = self._cfg.get("show_latency",     False)
        self._show_frame:    bool  = self._cfg.get("show_frame_number", True)

        # Box style
        self._thickness: int      = self._cfg.get("bbox_thickness", 2)
        # Always green — override any per-person colour setting
        _raw_color              = self._cfg.get("fallback_bbox_color", list(_CVPR_GREEN))
        self._box_color: BGRColor = tuple(int(c) for c in _raw_color)

        # Trails — disabled by default per user's "Show only" requirements
        self._trail_enabled: bool  = self._cfg.get("trail_enabled",  False)
        self._trail_length:  int   = self._cfg.get("trail_length",   20)

        # Internal trail history (kept for backward-compat even when disabled)
        self._trails: Dict[int, Deque[Tuple[int, int]]] = defaultdict(
            lambda: deque(maxlen=self._trail_length)
        )

    # ------------------------------------------------------------------
    # Main render method (unchanged public signature)
    # ------------------------------------------------------------------

    def render(
        self,
        frame:      Frame,
        tracks:     List[Track],
        frame_idx:  int   = 0,
        fps:        float = 0.0,
        latency_ms: float = 0.0,
    ) -> Frame:
        """Draw all confirmed tracks onto the frame.

        Draws ONLY:
          • Solid green bounding box  (``Confirmed`` state, aligned to detector)
          • Track ID and confidence label above the box
          • HUD: FPS, Frame Number, Persons count

        Tracks in ``Tentative``, ``Lost``, or ``Deleted`` state are silently
        skipped — they must NEVER be drawn, as their positions may be stale.
        ``Predicted`` tracks (brief Kalman occlusion fill) are shown only when
        ``_predicted_frames == 1`` (the very first frame of occlusion).

        Args:
            frame:      BGR input image (modified in-place).
            tracks:     Active tracks returned by ByteTrackTracker.run().
            frame_idx:  Current frame index.
            fps:        Pipeline FPS (for HUD).
            latency_ms: Tracking latency ms (for HUD, if show_latency=True).

        Returns:
            Annotated frame.
        """
        h_img, w_img = frame.shape[:2]
        drawn_count  = 0

        for track in tracks:
            # ── Filter: only draw Confirmed tracks ────────────────────────────
            # Also allow Predicted tracks for the first frame of occlusion only.
            # After that the Kalman box might drift — skip it.
            if track.state == TrackState.Tentative:
                continue   # Never draw tentative (potential ghost)
            if track.state == TrackState.Lost:
                continue   # Never draw lost (stale Kalman position)
            if track.state == TrackState.Deleted:
                continue
            if track.state == TrackState.Predicted:
                # Allow ONLY frame 1 of Kalman prediction (essentially zero lag)
                if getattr(track, "_predicted_frames", 0) > 1:
                    continue

            # ── Clamp box coordinates to frame bounds ─────────────────────────
            x1, y1, x2, y2 = [int(v) for v in track.bbox.tolist()]
            x1 = max(0, x1);          y1 = max(0, y1)
            x2 = min(w_img - 1, x2);  y2 = min(h_img - 1, y2)

            # Skip degenerate boxes (< 3 px in either dimension)
            if x2 <= x1 + 2 or y2 <= y1 + 2:
                log.debug(
                    "TrackingRenderer: skipped degenerate box for track #{}",
                    track.track_id,
                )
                continue

            # ── Draw solid green bounding box ─────────────────────────────────
            _draw_bbox(frame, x1, y1, x2, y2, self._box_color, self._thickness)

            # ── Label: "#ID  conf" above the top-left corner ──────────────────
            label_parts: List[str] = []
            if self._show_id:
                label_parts.append(f"#{track.track_id}")
            if self._show_conf:
                label_parts.append(f"{track.score:.2f}")
            if label_parts:
                label_text = "  ".join(label_parts)
                _draw_label(
                    frame, label_text, x1, y1,
                    bg_color=self._box_color,
                    text_color=(10, 10, 10),   # dark text on green background
                )

            # ── Optional trail (disabled by default) ──────────────────────────
            if self._trail_enabled:
                foot = (int((x1 + x2) / 2), y2)
                self._trails[track.track_id].append(foot)
                if len(self._trails[track.track_id]) >= 2:
                    _draw_trail(
                        frame,
                        list(self._trails[track.track_id]),
                        self._box_color,
                    )

            drawn_count += 1

        # ── HUD: FPS | Frame | Persons ────────────────────────────────────────
        if self._show_fps or self._show_frame:
            _draw_hud(
                frame, fps, frame_idx,
                n_persons=drawn_count,
                latency_ms=latency_ms,
                show_latency=self._show_latency,
            )

        return frame

    # ------------------------------------------------------------------
    # Utility
    # ------------------------------------------------------------------

    def reset(self) -> None:
        """Clear trail history (call at shot boundaries)."""
        self._trails.clear()
