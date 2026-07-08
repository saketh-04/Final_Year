"""
HumanMM — MediaPipe Pose Estimation Backend.

Implements the ``PoseBackend`` strategy for 2D human pose estimation using
Google MediaPipe.  Works on CPU without any CUDA dependency.

Returns ``PersonPose`` dataclasses with 17 COCO-format 2D keypoints extracted
from MediaPipe's 33-landmark output.

Example:
    >>> from models.mediapipe_pose import MediaPipePoseBackend
    >>> backend = MediaPipePoseBackend(config=cfg.pose.mediapipe)
    >>> backend.initialize()
    >>> pose = backend.run(frame_rgb, bbox=(x1, y1, x2, y2))
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from models.base_model import BaseModel
from utils.logger import get_logger

log = get_logger(__name__)


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class Keypoint:
    """A single 2D body keypoint.

    Attributes:
        x: Horizontal position in pixels.
        y: Vertical position in pixels.
        confidence: Visibility / confidence score in ``[0, 1]``.
        name: Joint name (e.g. ``"left_shoulder"``).
    """

    x: float
    y: float
    confidence: float
    name: str = ""

    def as_array(self) -> np.ndarray:
        """Return ``[x, y, confidence]`` as a NumPy array."""
        return np.array([self.x, self.y, self.confidence], dtype=np.float32)


@dataclass
class PersonPose:
    """Pose estimation result for a single person in one frame.

    Attributes:
        track_id: Person tracking identifier.
        frame_idx: Frame index.
        keypoints: List of 17 COCO keypoints.
        bbox: Bounding box ``(x1, y1, x2, y2)`` used for this estimate.
        confidence: Overall pose detection confidence.
        backend: Name of the pose estimation backend used.
    """

    track_id: int
    frame_idx: int
    keypoints: List[Keypoint]
    bbox: Tuple[float, float, float, float] = field(default_factory=lambda: (0, 0, 0, 0))
    confidence: float = 0.0
    backend: str = "mediapipe"

    @property
    def keypoints_array(self) -> np.ndarray:
        """Return keypoints as a ``(J, 3)`` array ``[x, y, conf]``."""
        return np.stack([kp.as_array() for kp in self.keypoints], axis=0)

    def to_dict(self) -> Dict[str, Any]:
        """Serialise to a plain dictionary for JSON export."""
        return {
            "track_id": self.track_id,
            "frame_idx": self.frame_idx,
            "confidence": round(self.confidence, 4),
            "backend": self.backend,
            "bbox": [round(v, 2) for v in self.bbox],
            "keypoints": [
                {"name": kp.name, "x": round(kp.x, 2), "y": round(kp.y, 2), "conf": round(kp.confidence, 4)}
                for kp in self.keypoints
            ],
        }


# ---------------------------------------------------------------------------
# MediaPipe → COCO-17 landmark mapping
# ---------------------------------------------------------------------------
# MediaPipe has 33 landmarks; we extract the 17 that correspond to COCO.
_MP_TO_COCO17: List[int] = [
    0,   # nose
    2,   # left_eye
    5,   # right_eye
    7,   # left_ear
    8,   # right_ear
    11,  # left_shoulder
    12,  # right_shoulder
    13,  # left_elbow
    14,  # right_elbow
    15,  # left_wrist
    16,  # right_wrist
    23,  # left_hip
    24,  # right_hip
    25,  # left_knee
    26,  # right_knee
    27,  # left_ankle
    28,  # right_ankle
]

_COCO17_NAMES: List[str] = [
    "nose", "left_eye", "right_eye", "left_ear", "right_ear",
    "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
    "left_wrist", "right_wrist", "left_hip", "right_hip",
    "left_knee", "right_knee", "left_ankle", "right_ankle",
]


# ---------------------------------------------------------------------------
# Backend
# ---------------------------------------------------------------------------

class MediaPipePoseBackend(BaseModel):
    """MediaPipe Holistic/Pose backend for 2D pose estimation.

    Args:
        device: Ignored (MediaPipe runs on CPU only). Kept for API consistency.
        config: Dict with MediaPipe-specific settings (mirrors ``pose.mediapipe``
            in ``configs/pose.yaml``).

    Example:
        >>> backend = MediaPipePoseBackend(config={"model_complexity": 1})
        >>> backend.initialize()
        >>> pose = backend.run(rgb_frame, track_id=1, frame_idx=0, bbox=(x1,y1,x2,y2))
    """

    def __init__(
        self,
        device: str = "cpu",
        config: Optional[Dict[str, Any]] = None,
    ) -> None:
        super().__init__(name="MediaPipePose", device="cpu", config=config)
        self._mp_pose = None
        self._pose_instance = None

        # Config defaults
        self._model_complexity: int = self.config.get("model_complexity", 1)
        self._smooth_landmarks: bool = self.config.get("smooth_landmarks", True)
        self._enable_segmentation: bool = self.config.get("enable_segmentation", False)
        self._min_det_conf: float = self.config.get("min_detection_confidence", 0.50)
        self._min_track_conf: float = self.config.get("min_tracking_confidence", 0.50)
        self._static_image_mode: bool = self.config.get("static_image_mode", False)
        self._conf_threshold: float = self.config.get("confidence_threshold", 0.30)

    def load(self) -> None:
        """Initialise the MediaPipe Pose solution.

        Raises:
            ImportError: If ``mediapipe`` is not installed.
        """
        try:
            import mediapipe as mp  # noqa: F401
        except ImportError as exc:
            raise ImportError("mediapipe is required: pip install mediapipe") from exc

        import mediapipe as mp

        self._mp_pose = mp.solutions.pose
        self._pose_instance = self._mp_pose.Pose(
            static_image_mode=self._static_image_mode,
            model_complexity=self._model_complexity,
            smooth_landmarks=self._smooth_landmarks,
            enable_segmentation=self._enable_segmentation,
            min_detection_confidence=self._min_det_conf,
            min_tracking_confidence=self._min_track_conf,
        )
        log.debug("MediaPipe Pose initialised (complexity={})", self._model_complexity)

    def run(
        self,
        frame_rgb: np.ndarray,
        track_id: int = 0,
        frame_idx: int = 0,
        bbox: Optional[Tuple[float, float, float, float]] = None,
        **kwargs: Any,
    ) -> Optional[PersonPose]:
        """Estimate 2D pose for a single person in an RGB frame.

        Args:
            frame_rgb: RGB image array of shape ``(H, W, 3)``.
            track_id: Person tracking ID for this detection.
            frame_idx: Frame index.
            bbox: Optional ``(x1, y1, x2, y2)`` bounding box.  If provided,
                the frame is cropped before pose estimation.
            **kwargs: Ignored.

        Returns:
            :class:`PersonPose` with 17 COCO keypoints, or ``None`` if
            MediaPipe cannot detect a pose.

        Raises:
            RuntimeError: If ``initialize()`` has not been called.
        """
        self._require_loaded()

        h, w = frame_rgb.shape[:2]

        # Optionally crop to the person bounding box
        crop_frame = frame_rgb
        x_offset, y_offset = 0, 0
        if bbox is not None:
            x1, y1, x2, y2 = [int(v) for v in bbox]
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w, x2), min(h, y2)
            if x2 > x1 and y2 > y1:
                crop_frame = frame_rgb[y1:y2, x1:x2]
                x_offset, y_offset = x1, y1

        result = self._pose_instance.process(crop_frame)

        if result.pose_landmarks is None:
            log.debug("MediaPipe: no pose detected (track={}, frame={})", track_id, frame_idx)
            return None

        ch, cw = crop_frame.shape[:2]
        landmarks = result.pose_landmarks.landmark

        keypoints: List[Keypoint] = []
        visibilities: List[float] = []

        for coco_idx, mp_idx in enumerate(_MP_TO_COCO17):
            lm = landmarks[mp_idx]
            # MediaPipe returns normalised [0,1] coordinates
            px = lm.x * cw + x_offset
            py = lm.y * ch + y_offset
            vis = float(lm.visibility)
            visibilities.append(vis)
            keypoints.append(
                Keypoint(x=px, y=py, confidence=vis, name=_COCO17_NAMES[coco_idx])
            )

        mean_confidence = float(np.mean(visibilities))

        if mean_confidence < self._conf_threshold:
            log.debug(
                "MediaPipe: pose confidence {:.3f} below threshold {:.3f}",
                mean_confidence,
                self._conf_threshold,
            )
            return None

        return PersonPose(
            track_id=track_id,
            frame_idx=frame_idx,
            keypoints=keypoints,
            bbox=bbox or (0.0, 0.0, float(w), float(h)),
            confidence=mean_confidence,
            backend="mediapipe",
        )

    def release(self) -> None:
        """Close the MediaPipe Pose instance and free resources."""
        if self._pose_instance is not None:
            self._pose_instance.close()
            self._pose_instance = None
        super().release()
