"""
HumanMM — Model Factory.

Implements the Factory Pattern for instantiating all model backends.
Pipeline modules never directly ``import`` or instantiate model classes.
Instead they call :func:`ModelFactory.create_detector`, etc., which reads
the configuration and returns the appropriate concrete implementation.

This keeps the pipeline decoupled from any specific framework choice and
makes backend swapping a one-line config change.

Example:
    >>> from models.model_factory import ModelFactory
    >>> factory = ModelFactory(cfg, device="cuda")
    >>> detector = factory.create_detector()
    >>> tracker  = factory.create_tracker()
    >>> pose_est = factory.create_pose_estimator()
    >>> motion   = factory.create_motion_recovery()
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Union

from omegaconf import DictConfig, OmegaConf

from models.base_model import BaseModel
from utils.logger import get_logger

log = get_logger(__name__)


def _cfg_to_dict(cfg: Any) -> Dict[str, Any]:
    """Safely convert an OmegaConf node or plain dict to a plain dict."""
    if isinstance(cfg, DictConfig):
        return OmegaConf.to_container(cfg, resolve=True)  # type: ignore[return-value]
    if isinstance(cfg, dict):
        return cfg
    return {}


class ModelFactory:
    """Factory that constructs model instances from a Hydra configuration.

    Args:
        cfg: The master ``DictConfig`` (or a plain dict for testing).
        device: Compute device string (``"cuda"`` or ``"cpu"``).

    Example:
        >>> factory = ModelFactory(cfg, device="cuda")
        >>> detector = factory.create_detector()
        >>> detector.initialize()
    """

    def __init__(
        self,
        cfg: Union[DictConfig, Dict[str, Any]],
        device: str = "cpu",
    ) -> None:
        self._cfg = cfg
        self._device = device

    # ------------------------------------------------------------------
    # Detector
    # ------------------------------------------------------------------

    def create_detector(self) -> BaseModel:
        """Create and return a YOLOv8 human detector.

        Reads ``cfg.detector`` for backend settings.

        Returns:
            Initialised :class:`~models.yolo_detector.YOLODetector`.

        Raises:
            RuntimeError: If the detector cannot be constructed.
        """
        from models.yolo_detector import YOLODetector

        det_cfg = _cfg_to_dict(self._cfg.get("detector", {})) if isinstance(self._cfg, dict) \
            else _cfg_to_dict(getattr(self._cfg, "detector", {}))

        log.info("ModelFactory: creating YOLODetector (device={})", self._device)
        detector = YOLODetector(device=self._device, config=det_cfg)
        detector.initialize()
        return detector

    # ------------------------------------------------------------------
    # Tracker
    # ------------------------------------------------------------------

    def create_tracker(self) -> BaseModel:
        """Create and return a person tracker based on config.

        Reads ``cfg.tracker.backend`` to choose between ``bytetrack``
        (default) and ``deepsort``.

        Returns:
            Initialised tracker (:class:`~models.bytetrack_tracker.ByteTrackTracker`
            or :class:`~models.deepsort_tracker.DeepSORTTracker`).

        Raises:
            ValueError: If an unknown tracker backend is specified.
        """
        tracker_cfg = _cfg_to_dict(self._cfg.get("tracker", {})) if isinstance(self._cfg, dict) \
            else _cfg_to_dict(getattr(self._cfg, "tracker", {}))

        backend = tracker_cfg.get("backend", "bytetrack").lower()
        sub_cfg = tracker_cfg.get(backend, {})

        log.info("ModelFactory: creating tracker backend={}", backend)

        if backend == "bytetrack":
            from models.bytetrack_tracker import ByteTrackTracker
            tracker = ByteTrackTracker(device="cpu", config=sub_cfg)

        elif backend == "deepsort":
            from models.deepsort_tracker import DeepSORTTracker
            tracker = DeepSORTTracker(device=self._device, config=sub_cfg)

        else:
            raise ValueError(
                f"Unknown tracker backend: '{backend}'. "
                "Choose 'bytetrack' or 'deepsort'."
            )

        tracker.initialize()
        return tracker

    # ------------------------------------------------------------------
    # Pose Estimator
    # ------------------------------------------------------------------

    def create_pose_estimator(self) -> BaseModel:
        """Create and return a 2D pose estimator based on config.

        Reads ``cfg.pose.backend`` to choose between ``mediapipe``
        (default, CPU-friendly) and ``vitpose`` (GPU, higher accuracy).

        Returns:
            Initialised pose estimator backend.

        Raises:
            ValueError: If an unknown pose backend is specified.
            ImportError: If ViTPose is selected but mmpose is not installed.
        """
        pose_cfg = _cfg_to_dict(self._cfg.get("pose", {})) if isinstance(self._cfg, dict) \
            else _cfg_to_dict(getattr(self._cfg, "pose", {}))

        backend = pose_cfg.get("backend", "mediapipe").lower()
        sub_cfg = pose_cfg.get(backend, {})
        # Merge common settings
        sub_cfg.update(pose_cfg.get("common", {}))

        log.info("ModelFactory: creating pose estimator backend={}", backend)

        if backend == "mediapipe":
            from models.mediapipe_pose import MediaPipePoseBackend
            estimator = MediaPipePoseBackend(device="cpu", config=sub_cfg)

        elif backend == "vitpose":
            from models.vitpose_pose import ViTPosePoseBackend
            estimator = ViTPosePoseBackend(device=self._device, config=sub_cfg)

        else:
            raise ValueError(
                f"Unknown pose backend: '{backend}'. "
                "Choose 'mediapipe' or 'vitpose'."
            )

        estimator.initialize()
        return estimator

    # ------------------------------------------------------------------
    # Motion Recovery
    # ------------------------------------------------------------------

    def create_motion_recovery(self) -> BaseModel:
        """Create and return a 3D motion recovery model.

        Uses :class:`~models.smpl_mesh_model.SMPLMeshModel` which implements
        a 3-tier strategy:
          - Tier 1: HMR 2.0 (4D-Humans) if installed — full SMPL prediction
          - Tier 2: GeometrySMPL — rigid-fit of neutral SMPL template (always works)
          - Tier 3: Stick-figure — only if pyrender is also missing

        Reads ``cfg.motion`` for backend settings.

        Returns:
            Initialised :class:`~models.smpl_mesh_model.SMPLMeshModel`.
        """
        motion_cfg = _cfg_to_dict(self._cfg.get("motion", {})) if isinstance(self._cfg, dict) \
            else _cfg_to_dict(getattr(self._cfg, "motion", {}))

        log.info(
            "ModelFactory: creating SMPLMeshModel (3-tier HMR2→GeometrySMPL→Fallback, device={})",
            self._device,
        )

        from models.smpl_mesh_model import SMPLMeshModel
        model = SMPLMeshModel(device=self._device, config=motion_cfg)
        model.initialize()
        log.info(
            "ModelFactory: SMPLMeshModel ready — active_tier={}", model.active_tier
        )
        return model

    # ------------------------------------------------------------------
    # Convenience: create & initialise all models at once
    # ------------------------------------------------------------------

    def create_all(self) -> Dict[str, BaseModel]:
        """Create and initialise all pipeline models in one call.

        Returns:
            Dictionary with keys ``"detector"``, ``"tracker"``,
            ``"pose_estimator"``, and ``"motion_recovery"``, each mapped
            to its initialised model instance.

        Example:
            >>> models = factory.create_all()
            >>> detector = models["detector"]
        """
        log.info("ModelFactory: initializing all pipeline models")
        models = {
            "detector": self.create_detector(),
            "tracker": self.create_tracker(),
            "pose_estimator": self.create_pose_estimator(),
            "motion_recovery": self.create_motion_recovery(),
        }
        log.info("ModelFactory: all models ready")
        return models
