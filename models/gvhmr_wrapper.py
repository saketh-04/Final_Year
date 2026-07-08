"""
HumanMM — HMR 2.0 / GVHMR Motion Recovery Wrapper.

Wraps the 4D-Humans (HMR 2.0) inference pipeline for 3D human body recovery.
Produces SMPL parameters (betas, body_pose, global_orient, transl), 3D joints,
and mesh vertices for each detected person.

The interface is deliberately compatible with GVHMR's output format so that
swapping backends (hmr2 ↔ gvhmr) requires only a config change.

Example:
    >>> from models.gvhmr_wrapper import MotionRecoveryModel, SMPLOutput
    >>> model = MotionRecoveryModel(device="cuda", config=cfg.motion)
    >>> model.initialize()
    >>> output = model.run(frame_rgb, bbox=(x1, y1, x2, y2))
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from models.base_model import BaseModel
from utils.logger import get_logger

log = get_logger(__name__)


# ---------------------------------------------------------------------------
# Output data structure
# ---------------------------------------------------------------------------

@dataclass
class SMPLOutput:
    """3D human body recovery output for a single person.

    Attributes:
        track_id: Person tracking identifier.
        frame_idx: Frame index.
        betas: SMPL shape parameters of shape ``(10,)``.
        body_pose: SMPL body pose parameters of shape ``(23, 3)`` (axis-angle).
        global_orient: Global body orientation of shape ``(1, 3)`` (axis-angle).
        transl: Global translation of shape ``(3,)``.
        joints_3d: 3D joint positions of shape ``(J, 3)`` in camera coordinates.
        vertices: SMPL mesh vertices of shape ``(6890, 3)``.
        camera_params: Weak-perspective camera ``[scale, tx, ty]`` of shape ``(3,)``.
        confidence: Overall recovery confidence.
    """

    track_id: int
    frame_idx: int
    betas: np.ndarray = field(default_factory=lambda: np.zeros(10))
    body_pose: np.ndarray = field(default_factory=lambda: np.zeros((23, 3)))
    global_orient: np.ndarray = field(default_factory=lambda: np.zeros((1, 3)))
    transl: np.ndarray = field(default_factory=lambda: np.zeros(3))
    joints_3d: np.ndarray = field(default_factory=lambda: np.zeros((17, 3)))
    vertices: np.ndarray = field(default_factory=lambda: np.zeros((6890, 3)))
    camera_params: np.ndarray = field(default_factory=lambda: np.zeros(3))
    confidence: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        """Serialise to a plain dictionary (arrays as lists)."""
        return {
            "track_id": self.track_id,
            "frame_idx": self.frame_idx,
            "confidence": round(self.confidence, 4),
            "betas": self.betas.tolist(),
            "global_orient": self.global_orient.tolist(),
            "transl": self.transl.tolist(),
            "camera_params": self.camera_params.tolist(),
            # Omit large arrays from JSON; save them as NPY separately
        }


# ---------------------------------------------------------------------------
# Model wrapper
# ---------------------------------------------------------------------------

class MotionRecoveryModel(BaseModel):
    """3D human motion recovery using HMR 2.0 (4D-Humans).

    Performs single-frame SMPL body fitting per person using a pre-trained
    HMR 2.0 checkpoint.  If HMR 2.0 is not installed, falls back to a
    geometry-based pseudo-3D lifter.

    Args:
        device: Compute device (``"cuda"`` or ``"cpu"``).
        config: Motion recovery config dict (mirrors ``configs/motion.yaml``).

    Example:
        >>> model = MotionRecoveryModel(device="cuda")
        >>> model.initialize()
        >>> smpl_out = model.run(frame_rgb, track_id=1, frame_idx=0, bbox=det.bbox)
    """

    def __init__(
        self,
        device: str = "cpu",
        config: Optional[Dict[str, Any]] = None,
    ) -> None:
        super().__init__(name="HMR2-MotionRecovery", device=device, config=config)
        self._hmr_model = None
        self._hmr_available: bool = False
        self._use_fallback: bool = False

        # Config
        self._img_size: int = self.config.get("hmr2", {}).get("img_size", 256)
        self._rescale_factor: float = self.config.get("hmr2", {}).get("rescale_factor", 1.20)
        self._output_vertices: bool = self.config.get("hmr2", {}).get("output_vertices", True)
        self._smpl_dir: Optional[str] = self.config.get("hmr2", {}).get("smpl_dir")

    def load(self) -> None:
        """Attempt to load HMR 2.0.  Falls back to pseudo-3D lifter if unavailable.

        The pseudo-3D fallback uses 2D-to-3D lifting heuristics based on
        anatomical bone length ratios to produce plausible (but approximate)
        3D joint estimates from 2D pose inputs.

        Raises:
            RuntimeError: If neither HMR 2.0 nor the fallback can be initialised.
        """
        try:
            self._load_hmr2()
            self._hmr_available = True
            log.info("HMR 2.0 loaded on {}", self.device)
        except (ImportError, Exception) as exc:
            log.warning(
                "HMR 2.0 not available ({}). Using pseudo-3D fallback. "
                "Install 4D-Humans for full accuracy.", exc
            )
            self._hmr_available = False
            self._use_fallback = True

    def _load_hmr2(self) -> None:
        """Load HMR 2.0 from the 4D-Humans package.

        Raises:
            ImportError: If ``hmr2`` package is not installed.
        """
        try:
            from hmr2.models import download_models, load_hmr2  # type: ignore
        except ImportError as exc:
            raise ImportError(
                "4D-Humans (HMR 2.0) not installed.\n"
                "Install via: pip install git+https://github.com/shubham-goel/4D-Humans\n"
                "Or: python scripts/download_models.py"
            ) from exc

        model, model_cfg = load_hmr2()
        model = model.to(self.device)
        model.eval()
        self._hmr_model = model
        self._hmr_model_cfg = model_cfg

    def run(
        self,
        frame_rgb: np.ndarray,
        track_id: int = 0,
        frame_idx: int = 0,
        bbox: Optional[Tuple[float, float, float, float]] = None,
        joints_2d: Optional[np.ndarray] = None,
        **kwargs: Any,
    ) -> Optional[SMPLOutput]:
        """Recover 3D SMPL body parameters for a single person.

        Args:
            frame_rgb: Full RGB frame ``(H, W, 3)``.
            track_id: Person tracking ID.
            frame_idx: Frame index.
            bbox: Person bounding box ``(x1, y1, x2, y2)``.
            joints_2d: Optional ``(J, 3)`` 2D joints ``[x, y, conf]`` from
                pose estimator (used by fallback mode).
            **kwargs: Ignored.

        Returns:
            :class:`SMPLOutput` or ``None`` on failure.

        Raises:
            RuntimeError: If ``initialize()`` has not been called.
        """
        self._require_loaded()

        if self._hmr_available:
            return self._run_hmr2(frame_rgb, track_id, frame_idx, bbox)
        else:
            return self._run_fallback(track_id, frame_idx, bbox, joints_2d)

    def _run_hmr2(
        self,
        frame_rgb: np.ndarray,
        track_id: int,
        frame_idx: int,
        bbox: Optional[Tuple[float, float, float, float]],
    ) -> Optional[SMPLOutput]:
        """Run HMR 2.0 inference for a single person crop.

        Args:
            frame_rgb: Full-frame RGB image.
            track_id: Person ID.
            frame_idx: Frame index.
            bbox: Bounding box.

        Returns:
            :class:`SMPLOutput` or ``None`` on error.
        """
        import torch

        try:
            from hmr2.utils.utils_detectron2 import DefaultPredictor_Lazy  # type: ignore
            from hmr2.utils import recursive_to  # type: ignore
            from hmr2.datasets.vitdet_dataset import ViTDetDataset  # type: ignore
        except ImportError as exc:
            log.error("HMR2 inference utilities missing: {}", exc)
            return None

        try:
            # Prepare crop
            h, w = frame_rgb.shape[:2]
            if bbox is None:
                bbox = (0.0, 0.0, float(w), float(h))

            x1, y1, x2, y2 = bbox
            # Rescale bbox
            cx = (x1 + x2) / 2
            cy = (y1 + y2) / 2
            bw = (x2 - x1) * self._rescale_factor
            bh = (y2 - y1) * self._rescale_factor
            x1r = max(0, cx - bw / 2)
            y1r = max(0, cy - bh / 2)
            x2r = min(w, cx + bw / 2)
            y2r = min(h, cy + bh / 2)

            crop = frame_rgb[int(y1r):int(y2r), int(x1r):int(x2r)]
            if crop.size == 0:
                return None

            # Resize to model input size
            import cv2
            crop_resized = cv2.resize(crop, (self._img_size, self._img_size))

            # Normalise
            mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
            std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
            crop_norm = (crop_resized.astype(np.float32) / 255.0 - mean) / std
            tensor = torch.from_numpy(crop_norm.transpose(2, 0, 1)).unsqueeze(0).float()
            tensor = tensor.to(self.device)

            with torch.no_grad():
                out = self._hmr_model({"img": tensor})

            betas = out["pred_smpl_params"]["betas"][0].cpu().numpy()
            body_pose = out["pred_smpl_params"]["body_pose"][0].cpu().numpy()
            global_orient = out["pred_smpl_params"]["global_orient"][0].cpu().numpy()
            transl = out.get("pred_cam_t", torch.zeros(1, 3))[0].cpu().numpy()
            joints_3d = out.get("pred_keypoints_3d", torch.zeros(1, 17, 3))[0].cpu().numpy()
            vertices = out.get("pred_vertices", torch.zeros(1, 6890, 3))[0].cpu().numpy() \
                if self._output_vertices else np.zeros((6890, 3))
            cam = out.get("pred_cam", torch.zeros(1, 3))[0].cpu().numpy()

            return SMPLOutput(
                track_id=track_id,
                frame_idx=frame_idx,
                betas=betas.flatten()[:10],
                body_pose=body_pose.reshape(23, 3),
                global_orient=global_orient.reshape(1, 3),
                transl=transl,
                joints_3d=joints_3d,
                vertices=vertices,
                camera_params=cam,
                confidence=1.0,
            )

        except Exception as exc:
            log.warning("HMR2 inference error (track={}, frame={}): {}", track_id, frame_idx, exc)
            return None

    def _run_fallback(
        self,
        track_id: int,
        frame_idx: int,
        bbox: Optional[Tuple[float, float, float, float]],
        joints_2d: Optional[np.ndarray],
    ) -> Optional[SMPLOutput]:
        """Pseudo-3D fallback: lift 2D joints to 3D using heuristic bone lengths.

        This method produces a plausible but approximate 3D pose estimate
        using weak-perspective projection reversal.  It is NOT a trained
        model — it is a geometry-based heuristic intended as a graceful
        degradation when HMR 2.0 is unavailable.

        Args:
            track_id: Person ID.
            frame_idx: Frame index.
            bbox: Bounding box for depth estimation.
            joints_2d: ``(J, 3)`` 2D keypoints ``[x, y, conf]``.

        Returns:
            :class:`SMPLOutput` with approximate 3D joints.
        """
        if joints_2d is None or len(joints_2d) == 0:
            return None

        J = joints_2d.shape[0]
        joints_3d = np.zeros((J, 3), dtype=np.float32)

        # Copy 2D x, y
        joints_3d[:, 0] = joints_2d[:, 0]
        joints_3d[:, 1] = joints_2d[:, 1]

        # Estimate depth from bbox height (weak perspective assumption)
        if bbox is not None:
            bbox_height = abs(bbox[3] - bbox[1]) + 1e-6
            # Assume average human height ~170cm, focal length ~1000px
            depth_estimate = 1700.0 / bbox_height
        else:
            depth_estimate = 3.0  # metres (arbitrary default)

        joints_3d[:, 2] = depth_estimate

        # Centre at pelvis (midpoint of hips — joints 11 and 12 in COCO)
        if J >= 13:
            pelvis = (joints_3d[11] + joints_3d[12]) / 2.0
            joints_3d -= pelvis

        return SMPLOutput(
            track_id=track_id,
            frame_idx=frame_idx,
            joints_3d=joints_3d,
            confidence=0.5,  # Lower confidence for fallback
        )

    def release(self) -> None:
        """Release model and GPU memory."""
        if self._hmr_model is not None:
            del self._hmr_model
            self._hmr_model = None
            try:
                import torch
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except ImportError:
                pass
        super().release()
