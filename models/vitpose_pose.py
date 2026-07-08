"""
HumanMM — ViTPose Pose Estimation Backend (Optional).

Implements the ``PoseBackend`` strategy using ViTPose via the ``mmpose``
library.  This backend provides higher accuracy than MediaPipe but requires
a CUDA-capable GPU and a more complex installation (mmcv + mmpose).

Falls back gracefully if ``mmpose`` is not installed — the pipeline will
automatically use MediaPipe in that case.

Example:
    >>> from models.vitpose_pose import ViTPosePoseBackend
    >>> backend = ViTPosePoseBackend(device="cuda", config=cfg.pose.vitpose)
    >>> backend.initialize()
    >>> pose = backend.run(frame_rgb, track_id=1, frame_idx=0, bbox=(x1,y1,x2,y2))
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from models.base_model import BaseModel
from models.mediapipe_pose import PersonPose, Keypoint, _COCO17_NAMES
from utils.logger import get_logger

log = get_logger(__name__)

# ViTPose outputs 17 COCO keypoints natively
_VITPOSE_COCO17_NAMES = _COCO17_NAMES


class ViTPosePoseBackend(BaseModel):
    """ViTPose 2D pose estimation backend using mmpose.

    Requires ``mmpose>=1.3.0`` and ``mmcv>=2.1.0`` with matching CUDA.
    See ``docs/installation.md`` for setup instructions.

    Args:
        device: Compute device (``"cuda"`` or ``"cpu"``).
        config: Dict with ViTPose settings (mirrors ``pose.vitpose`` in
            ``configs/pose.yaml``).

    Example:
        >>> backend = ViTPosePoseBackend(device="cuda")
        >>> backend.initialize()
        >>> pose = backend.run(frame_rgb, track_id=0, frame_idx=10,
        ...                    bbox=(100, 50, 400, 600))
    """

    def __init__(
        self,
        device: str = "cuda",
        config: Optional[Dict[str, Any]] = None,
    ) -> None:
        super().__init__(name="ViTPose", device=device, config=config)
        self._inferencer = None

        # Config defaults
        self._model_name: str = self.config.get("model_name", "ViTPose-H")
        self._fallback_model: str = self.config.get("fallback_model", "RTMPose-X")
        self._config_path: Optional[str] = self.config.get("config_path")
        self._checkpoint_path: Optional[str] = self.config.get("checkpoint_path")
        self._input_size: Tuple[int, int] = tuple(self.config.get("input_size", [192, 256]))
        self._bbox_thr: float = self.config.get("bbox_thr", 0.50)
        self._conf_threshold: float = self.config.get("confidence_threshold", 0.30)

    def load(self) -> None:
        """Load ViTPose model via mmpose PoseInferencer.

        Raises:
            ImportError: If ``mmpose`` is not installed.
            RuntimeError: If model files cannot be found or loaded.
        """
        try:
            from mmpose.apis import PoseInferencer
        except ImportError as exc:
            raise ImportError(
                "mmpose is required for ViTPose: pip install mmpose mmcv\n"
                "See docs/installation.md for full instructions."
            ) from exc

        log.info("Loading primary pose model: {}", self._model_name)

        # mmpose PoseInferencer supports model alias strings
        model_alias = self._resolve_model_alias(self._model_name)

        try:
            self._inferencer = PoseInferencer(
                pose2d=model_alias,
                pose2d_weights=self._checkpoint_path,
                device=self.device,
            )
            log.debug("Primary pose model loaded: {}", model_alias)
        except Exception as exc:
            log.warning("Primary model {} load failed: {}. Attempting fallback...", self._model_name, exc)
            fallback_alias = self._resolve_model_alias(self._fallback_model)
            try:
                self._inferencer = PoseInferencer(
                    pose2d=fallback_alias,
                    pose2d_weights=None,
                    device=self.device,
                )
                log.info("Fallback pose model loaded: {}", fallback_alias)
            except Exception as fallback_exc:
                log.error("Fallback model {} also failed: {}", self._fallback_model, fallback_exc)
                raise RuntimeError(f"Pose initialization failed. Both primary and fallback models failed.") from fallback_exc

    def run(
        self,
        frame_rgb: np.ndarray,
        track_id: int = 0,
        frame_idx: int = 0,
        bbox: Optional[Tuple[float, float, float, float]] = None,
        **kwargs: Any,
    ) -> Optional[PersonPose]:
        """Estimate 2D pose using ViTPose.

        Args:
            frame_rgb: RGB image array ``(H, W, 3)``.
            track_id: Person tracking ID.
            frame_idx: Frame index.
            bbox: Person bounding box ``(x1, y1, x2, y2)``.
            **kwargs: Ignored.

        Returns:
            :class:`PersonPose` with 17 COCO keypoints, or ``None`` on failure.

        Raises:
            RuntimeError: If ``initialize()`` has not been called.
        """
        self._require_loaded()

        try:
            # Build bboxes in XYXY format expected by mmpose
            bboxes = None
            if bbox is not None:
                bboxes = [[bbox[0], bbox[1], bbox[2], bbox[3], 1.0]]

            result_gen = self._inferencer(
                inputs=frame_rgb,
                bboxes=bboxes,
                return_datasamples=True,
                batch_size=1,
            )

            results = list(result_gen)
            if not results:
                return None

            # Extract first person's predictions
            pred = results[0]["predictions"][0]
            keypoints_raw = pred.pred_instances.keypoints[0]  # (17, 2)
            scores_raw = pred.pred_instances.keypoint_scores[0]  # (17,)

        except Exception as exc:
            log.warning("ViTPose inference error (track={}, frame={}): {}", track_id, frame_idx, exc)
            return None

        keypoints: List[Keypoint] = []
        for i, (xy, score) in enumerate(zip(keypoints_raw, scores_raw)):
            name = _VITPOSE_COCO17_NAMES[i] if i < len(_VITPOSE_COCO17_NAMES) else f"joint_{i}"
            keypoints.append(
                Keypoint(
                    x=float(xy[0]),
                    y=float(xy[1]),
                    confidence=float(score),
                    name=name,
                )
            )

        mean_confidence = float(np.mean([kp.confidence for kp in keypoints]))

        if mean_confidence < self._conf_threshold:
            log.debug("ViTPose: low confidence {:.3f}", mean_confidence)
            return None

        h, w = frame_rgb.shape[:2]
        return PersonPose(
            track_id=track_id,
            frame_idx=frame_idx,
            keypoints=keypoints,
            bbox=bbox or (0.0, 0.0, float(w), float(h)),
            confidence=mean_confidence,
            backend="vitpose",
        )

    def release(self) -> None:
        """Release ViTPose model and CUDA memory."""
        if self._inferencer is not None:
            del self._inferencer
            self._inferencer = None
            try:
                import torch
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except ImportError:
                pass
        super().release()

    def _resolve_model_alias(self, model_name: str) -> str:
        """Map model_name config to an mmpose alias string.

        Returns:
            mmpose model alias for PoseInferencer.
        """
        alias_map = {
            "ViTPose-S": "td-hm_ViTPose-small_8xb64-210e_coco-256x192",
            "ViTPose-B": "td-hm_ViTPose-base_8xb64-210e_coco-256x192",
            "ViTPose-L": "td-hm_ViTPose-large_8xb64-210e_coco-256x192",
            "ViTPose-H": "td-hm_ViTPose-huge_8xb64-210e_coco-256x192",
            "RTMPose-X": "rtmpose-x_8xb256-420e_coco-384x288",
            "RTMPose-L": "rtmpose-l_8xb256-420e_coco-384x288",
        }
        alias = alias_map.get(model_name, alias_map["ViTPose-H"])
        log.debug("Pose model alias resolved: {} → {}", model_name, alias)
        return alias
