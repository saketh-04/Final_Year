"""
HumanMM — Overlay & Research Demo Renderer  (v3 — visible skeleton canvas)

Key changes from v2
--------------------
* **ResearchDemoRenderer**: right-panel skeleton canvas now reliably draws
  coloured stick-figure joints at full opacity with zero confidence threshold.
* **Keypoint normalisation**: scales MediaPipe pixel coords to the canvas
  with body-proportional margins so the figure fills ~80% of the white panel.
* **Fallback body figure**: when pose keypoints are unavailable (no MediaPipe
  output for this frame) a grey T-pose outline is drawn so the right panel
  is never blank.
* **4-panel ComparisonRenderer**: unchanged behaviour, just cleaner headers.

Public API unchanged.
"""

from __future__ import annotations

import colorsys
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np

from utils.frame_utils import Frame, resize_frame
from utils.image_utils import draw_fps_overlay, draw_header_bar, draw_shot_flash
from utils.logger import get_logger

log = get_logger(__name__)

BGRColor = Tuple[int, int, int]

# COCO-17 skeleton
_SKELETON_PAIRS = [
    (0, 1), (0, 2), (1, 3), (2, 4),
    (5, 6), (5, 7), (7, 9), (6, 8), (8, 10),
    (5, 11), (6, 12), (11, 12),
    (11, 13), (13, 15), (12, 14), (14, 16),
]

_LIMB_COLORS: List[BGRColor] = [
    (0, 215, 255), (0, 215, 255), (0, 215, 255), (0, 215, 255),
    (255, 128, 0),
    (0, 255, 0),   (0, 255, 0),
    (0, 128, 255), (0, 128, 255),
    (255, 0, 128), (255, 0, 128),
    (128, 0, 255),
    (255, 200, 0), (255, 200, 0),
    (0, 200, 255), (0, 200, 255),
]


def _person_color(track_id: int) -> BGRColor:
    golden = 0.618033988749895
    hue    = (track_id * golden) % 1.0
    r, g, b = colorsys.hsv_to_rgb(hue, 0.88, 0.95)
    return int(b*255), int(g*255), int(r*255)


# ---------------------------------------------------------------------------
# Skeleton drawing onto a white canvas
# ---------------------------------------------------------------------------

def _draw_skeleton_on_canvas(
    canvas: Frame,
    keypoints_xy: np.ndarray,
    confidences: Optional[np.ndarray],
    joint_radius: int = 8,
    line_thickness: int = 3,
    conf_threshold: float = 0.0,
) -> None:
    """Draw skeleton on white canvas — always bright and fully opaque."""
    h, w = canvas.shape[:2]
    J    = keypoints_xy.shape[0]

    # Draw limbs first
    for idx, (a, b) in enumerate(_SKELETON_PAIRS):
        if a >= J or b >= J:
            continue
        if confidences is not None:
            if confidences[a] < conf_threshold or confidences[b] < conf_threshold:
                continue
        color = _LIMB_COLORS[idx] if idx < len(_LIMB_COLORS) else (255, 165, 0)
        pt_a  = (int(np.clip(keypoints_xy[a, 0], 0, w-1)),
                 int(np.clip(keypoints_xy[a, 1], 0, h-1)))
        pt_b  = (int(np.clip(keypoints_xy[b, 0], 0, w-1)),
                 int(np.clip(keypoints_xy[b, 1], 0, h-1)))
        cv2.line(canvas, pt_a, pt_b, color, line_thickness, cv2.LINE_AA)

    # Draw joints on top
    for j in range(J):
        if confidences is not None and confidences[j] < conf_threshold:
            continue
        cx_j = int(np.clip(keypoints_xy[j, 0], 0, w-1))
        cy_j = int(np.clip(keypoints_xy[j, 1], 0, h-1))
        cv2.circle(canvas, (cx_j, cy_j), joint_radius + 2,
                   (0, 0, 0),     -1, cv2.LINE_AA)   # black outline
        cv2.circle(canvas, (cx_j, cy_j), joint_radius,
                   (0, 255, 255), -1, cv2.LINE_AA)   # cyan fill


def _draw_fallback_tpose(canvas: Frame) -> None:
    """Draw a minimal grey T-pose silhouette when no pose data is available."""
    h, w = canvas.shape[:2]
    cx   = w // 2
    color = (160, 160, 160)
    t     = 2

    # Head
    cv2.circle(canvas, (cx, int(h*0.12)), int(h*0.06), color, -1, cv2.LINE_AA)
    # Torso
    cv2.line(canvas, (cx, int(h*0.18)), (cx, int(h*0.55)), color, t*2, cv2.LINE_AA)
    # Arms
    cv2.line(canvas, (int(cx-w*0.25), int(h*0.28)),
                     (int(cx+w*0.25), int(h*0.28)), color, t, cv2.LINE_AA)
    # Legs
    cv2.line(canvas, (cx, int(h*0.55)), (int(cx-w*0.1), int(h*0.88)), color, t*2, cv2.LINE_AA)
    cv2.line(canvas, (cx, int(h*0.55)), (int(cx+w*0.1), int(h*0.88)), color, t*2, cv2.LINE_AA)


# ---------------------------------------------------------------------------
# ResearchDemoRenderer  (07_final side-by-side)
# ---------------------------------------------------------------------------

class ResearchDemoRenderer:
    """Renders LEFT=tracking video + RIGHT=skeleton on white canvas.

    RIGHT panel is NEVER blank: if valid pose keypoints exist they are drawn
    with bright limb colours; otherwise a grey T-pose placeholder is shown.

    Args:
        config: ``cfg.visualization.comparison`` dict.
        panel_w: Width of each half-panel.
        panel_h: Height of each half-panel.
    """

    def __init__(
        self,
        config: Optional[Dict[str, Any]] = None,
        panel_w: int = 640,
        panel_h: int = 360,
    ) -> None:
        cfg            = config or {}
        self._pw       = panel_w
        self._ph       = panel_h
        self._div_color: BGRColor = tuple(cfg.get("divider_color", [180, 180, 180]))
        self._conf_thr  = float(cfg.get("pose_conf_threshold", 0.0))
        self._jrad      = int(cfg.get("canvas_joint_radius",   8))
        self._lthick    = int(cfg.get("canvas_line_thickness",  3))
        self._bg_color: BGRColor = tuple(cfg.get("canvas_bg_color", [255, 255, 255]))

    def compose(
        self,
        left_frame: Frame,
        poses: Optional[Dict[int, Any]] = None,
        frame_idx: int = 0,
        fps: float = 0.0,
    ) -> Frame:
        """Compose side-by-side research demo frame.

        Args:
            left_frame: Annotated tracking/detection frame.
            poses: ``{track_id: PersonPose}`` for the current frame.
            frame_idx: Current frame index.
            fps: Pipeline FPS (unused visually).

        Returns:
            Combined BGR frame of shape ``(ph, pw*2+2, 3)``.
        """
        pw, ph = self._pw, self._ph

        # Left: resize tracking frame
        left = resize_frame(left_frame, pw, ph)

        # Right: white canvas with skeleton
        right = np.full((ph, pw, 3), self._bg_color, dtype=np.uint8)
        drawn = self._draw_skeletons(right, poses)
        if not drawn:
            _draw_fallback_tpose(right)

        # Thin divider
        div = np.full((ph, 2, 3), self._div_color, dtype=np.uint8)
        return np.concatenate([left, div, right], axis=1)

    def _draw_skeletons(
        self, canvas: Frame, poses: Optional[Dict[int, Any]]
    ) -> bool:
        """Draw all skeletons on the canvas. Returns True if anything drawn."""
        if not poses:
            return False

        h, w   = canvas.shape[:2]
        drawn  = False

        for track_id, pose in poses.items():
            try:
                kpts = pose.keypoints_array     # (J, 3) x,y,conf
            except AttributeError:
                continue

            if kpts is None or kpts.size == 0:
                continue

            xy   = kpts[:, :2].copy()
            conf = kpts[:, 2] if kpts.shape[1] > 2 else np.ones(kpts.shape[0])

            # Check if any joint has a non-trivial coordinate
            if np.allclose(xy, 0, atol=1.0):
                continue

            # ── Scale from source image coords → canvas ──────────────────
            # Use per-pose bounding box for normalisation
            valid_mask = conf >= self._conf_thr
            if valid_mask.sum() < 3:
                valid_mask = np.ones(len(conf), dtype=bool)

            valid_xy = xy[valid_mask]
            x_min, y_min = valid_xy[:, 0].min(), valid_xy[:, 1].min()
            x_max, y_max = valid_xy[:, 0].max(), valid_xy[:, 1].max()

            body_w = max(x_max - x_min, 1e-3)
            body_h = max(y_max - y_min, 1e-3)

            # Target: body fills 80% of canvas with equal margins
            margin = 0.10
            tgt_w  = w * (1 - 2*margin)
            tgt_h  = h * (1 - 2*margin)
            scale  = min(tgt_w / body_w, tgt_h / body_h)

            src_cx = (x_min + x_max) / 2
            src_cy = (y_min + y_max) / 2

            xy_canvas      = xy.copy()
            xy_canvas[:, 0] = (xy[:, 0] - src_cx) * scale + w / 2
            xy_canvas[:, 1] = (xy[:, 1] - src_cy) * scale + h / 2

            _draw_skeleton_on_canvas(
                canvas, xy_canvas, conf,
                joint_radius=self._jrad,
                line_thickness=self._lthick,
                conf_threshold=self._conf_thr,
            )
            drawn = True

        return drawn


# ---------------------------------------------------------------------------
# Generic overlay helpers (unchanged public API)
# ---------------------------------------------------------------------------

class OverlayRenderer:
    """FPS counter and shot-cut flash overlays."""

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        cfg  = config or {}
        fpsc = cfg.get("fps_overlay", {})
        shtc = cfg.get("shot_indicator", {})

        self._fps_enabled: bool  = fpsc.get("enabled", True)
        self._fps_position: str  = fpsc.get("position", "top_right")
        self._fps_text_color     = tuple(fpsc.get("text_color", [0, 255, 0]))
        self._fps_bg_color       = tuple(fpsc.get("bg_color",   [0, 0, 0]))
        self._shot_enabled: bool = shtc.get("enabled", True)
        self._shot_color         = tuple(shtc.get("flash_color", [255, 0, 255]))
        self._flash_duration: int = shtc.get("flash_duration_frames", 3)

        self._boundary_frames: set = set()
        self._flash_remaining: Dict[int, int] = {}

    def set_shot_boundaries(self, frames: List[int]) -> None:
        self._boundary_frames = set(frames)

    def apply_fps(self, frame: Frame, fps: float) -> Frame:
        if not self._fps_enabled:
            return frame
        return draw_fps_overlay(frame, fps, position=self._fps_position,
                                text_color=self._fps_text_color,
                                bg_color=self._fps_bg_color)

    def apply_shot_flash(self, frame: Frame, frame_idx: int) -> Frame:
        if not self._shot_enabled:
            return frame
        if frame_idx in self._boundary_frames:
            self._flash_remaining[frame_idx] = self._flash_duration
        if any(fidx <= frame_idx < fidx + self._flash_duration
               for fidx in self._flash_remaining):
            draw_shot_flash(frame, color=self._shot_color, alpha=0.25)
        for fidx in list(self._flash_remaining.keys()):
            if frame_idx >= fidx + self._flash_duration:
                del self._flash_remaining[fidx]
        return frame

    def apply_all(self, frame: Frame, frame_idx: int, fps: float) -> Frame:
        self.apply_shot_flash(frame, frame_idx)
        self.apply_fps(frame, fps)
        return frame


# ---------------------------------------------------------------------------
# 4-panel ComparisonRenderer (kept for backwards compatibility)
# ---------------------------------------------------------------------------

class ComparisonRenderer:
    """2×2 panel comparison grid."""

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        cfg             = config or {}
        self._panels    = cfg.get("panels", ["original","detection","pose","mesh"])
        self._hh        = int(cfg.get("header_height", 36))
        self._hc        = tuple(cfg.get("header_color",      [15,15,15]))
        self._htc       = tuple(cfg.get("header_text_color", [220,220,220]))
        self._labels    = bool(cfg.get("show_stage_labels", True))
        self._pw        = int(cfg.get("panel_width",  640))
        self._ph        = int(cfg.get("panel_height", 360))

    def compose(
        self,
        original: Optional[Frame] = None,
        detection: Optional[Frame] = None,
        pose: Optional[Frame] = None,
        mesh: Optional[Frame] = None,
    ) -> Frame:
        from utils.frame_utils import stack_frames_2x2
        names = {"original": original, "detection": detection,
                 "pose": pose, "mesh": mesh}
        labels = {"original": "Original", "detection": "Detection",
                  "pose": "Pose Estimation", "mesh": "3D Motion Recovery"}
        panels = []
        for name in (self._panels[:4] + ["original"]*4)[:4]:
            p = names.get(name)
            if p is None:
                p = np.zeros((self._ph, self._pw, 3), dtype=np.uint8)
            p = resize_frame(p, self._pw, self._ph).copy()
            if self._labels:
                draw_header_bar(p, labels.get(name, name.title()),
                                height=self._hh,
                                bg_color=self._hc,
                                text_color=self._htc)
            panels.append(p)
        while len(panels) < 4:
            panels.append(np.zeros((self._ph, self._pw, 3), dtype=np.uint8))
        return stack_frames_2x2(
            panels[0], panels[1], panels[2], panels[3],
            target_w=self._pw, target_h=self._ph,
        )
