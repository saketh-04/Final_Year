"""
HumanMM — 2D Skeleton Renderer  (v2 — high-visibility, research-quality)

Key changes from v1
--------------------
* **Solid bright colours** — joints drawn as cyan filled circles with black
  outline; limb lines drawn in orange/colour-coded.  No more heatmap scheme
  that blended into the background.
* **Full opacity** — skeleton is drawn at alpha=1.0 by default.
* **Thick joints (r=8) and thick limb lines (thickness=3)** — clearly visible
  against any background colour.
* **Zero confidence threshold** — every MediaPipe keypoint is drawn regardless
  of its visibility score, because MediaPipe visibility ≠ COCO confidence and
  even low-visibility joints are usually in the correct position.
* Config-driven — all values still read from ``cfg.visualization.pose``.

Public API unchanged.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np

from models.mediapipe_pose import PersonPose
from utils.frame_utils import Frame
from utils.logger import get_logger

log = get_logger(__name__)

# COCO-17 skeleton connectivity (used when no custom pairs supplied)
_DEFAULT_SKELETON_PAIRS: List[Tuple[int, int]] = [
    (0, 1), (0, 2), (1, 3), (2, 4),          # head
    (5, 6),                                    # shoulders
    (5, 7), (7, 9),                            # left arm
    (6, 8), (8, 10),                           # right arm
    (5, 11), (6, 12), (11, 12),               # torso
    (11, 13), (13, 15),                        # left leg
    (12, 14), (14, 16),                        # right leg
]

# Per-limb colour palette (BGR) — colour-coded for left/right body sides
_LIMB_COLORS_BGR: List[Tuple[int, int, int]] = [
    (0, 215, 255), (0, 215, 255), (0, 215, 255), (0, 215, 255),  # head — gold
    (255, 128, 0),                                                  # shoulders — blue
    (0, 255, 0),   (0, 255, 0),                                   # left arm — green
    (0, 128, 255), (0, 128, 255),                                  # right arm — orange
    (255, 0, 128), (255, 0, 128), (128, 0, 255),                  # torso — pink/purple
    (255, 200, 0), (255, 200, 0),                                  # left leg — cyan
    (0, 200, 255), (0, 200, 255),                                  # right leg — yellow
]


class SkeletonRenderer:
    """Renders bright, visible 2D skeleton overlays onto video frames.

    Args:
        config: Visualization config dict (mirrors ``cfg.visualization.pose``).
        skeleton_pairs: Joint connectivity pairs.  Defaults to COCO-17.

    Example:
        >>> renderer = SkeletonRenderer(config=cfg.visualization.pose)
        >>> frame = renderer.render(frame, poses_for_frame)
    """

    def __init__(
        self,
        config: Optional[Dict[str, Any]] = None,
        skeleton_pairs: Optional[List[Tuple[int, int]]] = None,
    ) -> None:
        cfg = config or {}
        self._pairs = skeleton_pairs or _DEFAULT_SKELETON_PAIRS

        self._joint_radius: int   = int(cfg.get("joint_radius", 8))
        self._joint_color: tuple  = tuple(cfg.get("joint_solid_color", [0, 255, 255]))
        self._limb_thickness: int = int(cfg.get("skeleton_thickness", 3))
        self._limb_color: tuple   = tuple(cfg.get("skeleton_color", [255, 165, 0]))
        self._conf_threshold: float = float(cfg.get("confidence_threshold", 0.0))
        self._use_track_color: bool = bool(cfg.get("use_track_color", False))
        self._color_coded_limbs: bool = True  # always use per-limb colours

    # ------------------------------------------------------------------
    # Core rendering helpers
    # ------------------------------------------------------------------

    def _draw_limbs(
        self,
        frame: Frame,
        xy: np.ndarray,
        conf: np.ndarray,
    ) -> None:
        """Draw all skeleton limb connections.

        Args:
            frame: BGR image (modified in-place).
            xy:    ``(J, 2)`` joint pixel coordinates.
            conf:  ``(J,)`` confidence / visibility scores.
        """
        J = xy.shape[0]
        for idx, (a, b) in enumerate(self._pairs):
            if a >= J or b >= J:
                continue
            # Draw even if confidence is zero — MediaPipe visibility is
            # unreliable for partially-visible joints.
            if conf[a] < self._conf_threshold or conf[b] < self._conf_threshold:
                continue
            color = _LIMB_COLORS_BGR[idx] if idx < len(_LIMB_COLORS_BGR) \
                    else self._limb_color
            pt_a = (int(xy[a, 0]), int(xy[a, 1]))
            pt_b = (int(xy[b, 0]), int(xy[b, 1]))
            cv2.line(frame, pt_a, pt_b, color, self._limb_thickness, cv2.LINE_AA)

    def _draw_joints(
        self,
        frame: Frame,
        xy: np.ndarray,
        conf: np.ndarray,
    ) -> None:
        """Draw filled joint circles with a thin black outline.

        Args:
            frame: BGR image (modified in-place).
            xy:    ``(J, 2)`` joint pixel coordinates.
            conf:  ``(J,)`` confidence / visibility scores.
        """
        J = xy.shape[0]
        for j in range(J):
            if conf[j] < self._conf_threshold:
                continue
            cx, cy = int(xy[j, 0]), int(xy[j, 1])
            # Skip clearly out-of-frame joints
            h, w = frame.shape[:2]
            if not (0 <= cx < w and 0 <= cy < h):
                continue
            # Black outline for contrast on any background
            cv2.circle(frame, (cx, cy), self._joint_radius + 2,
                       (0, 0, 0), -1, cv2.LINE_AA)
            # Coloured fill
            cv2.circle(frame, (cx, cy), self._joint_radius,
                       self._joint_color, -1, cv2.LINE_AA)

    def render_person(
        self,
        frame: Frame,
        pose: PersonPose,
        color_override: Optional[Tuple[int, int, int]] = None,
    ) -> Frame:
        """Render a single person's skeleton onto a frame.

        Args:
            frame: BGR image (modified in-place).
            pose:  :class:`~models.mediapipe_pose.PersonPose` to render.
            color_override: If given, overrides joint/limb colour.

        Returns:
            Modified frame.
        """
        kpts = pose.keypoints_array   # (J, 3) — x, y, conf
        if kpts is None or kpts.size == 0:
            return frame
        xy   = kpts[:, :2]
        conf = kpts[:, 2] if kpts.shape[1] > 2 else np.ones(kpts.shape[0])

        if color_override:
            self._joint_color = color_override
            self._limb_color  = color_override

        # Draw limbs first (behind joints)
        self._draw_limbs(frame, xy, conf)
        # Draw joints on top
        self._draw_joints(frame, xy, conf)
        return frame

    def render(
        self,
        frame: Frame,
        poses: Dict[int, PersonPose],
    ) -> Frame:
        """Render skeletons for all persons in one frame.

        Args:
            frame: BGR image (modified in-place).
            poses: ``{track_id: PersonPose}`` for this frame.

        Returns:
            Modified frame.
        """
        for track_id, pose in poses.items():
            try:
                self.render_person(frame, pose)
            except Exception as exc:
                log.warning("SkeletonRenderer: failed for track {}: {}", track_id, exc)
        return frame
