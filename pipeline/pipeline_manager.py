"""
HumanMM — Pipeline Manager (Master Orchestrator).

Wires together every pipeline stage — video loading, shot detection,
human detection, tracking, pose estimation, motion recovery, trajectory
alignment, visualization, and export — into a single, configuration-driven
run.  This is the class that ``main.py`` instantiates and calls ``run()``
on.

Design Pattern: Dependency Injection — all concrete model backends are
constructed once via :class:`models.model_factory.ModelFactory` and handed
to the thin pipeline-stage wrappers (``HumanDetector``, ``PersonTracker``,
etc.), which remain agnostic to the specific backend implementation.

Example:
    >>> from pipeline.pipeline_manager import PipelineManager
    >>> manager = PipelineManager(cfg)
    >>> report = manager.run()
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Dict, Optional, Union

from omegaconf import DictConfig, OmegaConf

from models.model_factory import ModelFactory
from pipeline.human_detector import HumanDetector
from pipeline.motion_recovery import MotionRecovery
from pipeline.output_writer import OutputWriter
from pipeline.pose_estimator import PoseEstimator
from pipeline.shot_detector import ShotDetector
from pipeline.tracker import PersonTracker
from pipeline.trajectory_aligner import TrajectoryAligner
from pipeline.video_loader import VideoLoader
from pipeline.visualizer import Visualizer
from utils.config_loader import get_nested, resolve_device, save_config
from utils.logger import get_logger
from utils.metrics import MetricsTracker

log = get_logger(__name__)


def _to_plain_dict(cfg_node: Any) -> Dict[str, Any]:
    """Convert an OmegaConf node (or plain dict) to a plain Python dict."""
    if isinstance(cfg_node, DictConfig):
        return OmegaConf.to_container(cfg_node, resolve=True)  # type: ignore[return-value]
    if isinstance(cfg_node, dict):
        return cfg_node
    return {}


class PipelineManager:
    """Master orchestrator for the full HumanMM pipeline.

    Reads ``cfg.pipeline.run_*`` flags to enable/disable individual stages,
    constructs all models via :class:`~models.model_factory.ModelFactory`,
    and runs every stage in sequence, accumulating intermediate results
    that later stages depend on.

    Args:
        cfg: The master Hydra ``DictConfig`` (or an equivalent plain dict).

    Attributes:
        metrics: :class:`~utils.metrics.MetricsTracker` instance shared
            across all stages.

    Example:
        >>> manager = PipelineManager(cfg)
        >>> report = manager.run()
        >>> print(report["summary"])
    """

    def __init__(self, cfg: Union[DictConfig, Dict[str, Any]]) -> None:
        self._cfg = cfg
        self._pipeline_cfg = _to_plain_dict(get_nested(cfg, "pipeline", {}) if not isinstance(cfg, dict) else cfg.get("pipeline", {}))
        self._output_cfg = _to_plain_dict(get_nested(cfg, "output", {}) if not isinstance(cfg, dict) else cfg.get("output", {}))
        self._video_cfg = _to_plain_dict(get_nested(cfg, "video", {}) if not isinstance(cfg, dict) else cfg.get("video", {}))
        self._alignment_cfg = _to_plain_dict(get_nested(cfg, "alignment", {}) if not isinstance(cfg, dict) else cfg.get("alignment", {}))
        self._viz_cfg = _to_plain_dict(get_nested(cfg, "visualization", {}) if not isinstance(cfg, dict) else cfg.get("visualization", {}))
        self._pose_cfg = _to_plain_dict(get_nested(cfg, "pose", {}) if not isinstance(cfg, dict) else cfg.get("pose", {}))
        self._metrics_cfg = _to_plain_dict(get_nested(cfg, "metrics", {}) if not isinstance(cfg, dict) else cfg.get("metrics", {}))

        self._fail_fast: bool = self._pipeline_cfg.get("fail_fast", False)
        self._max_persons: int = self._pipeline_cfg.get("max_persons", 10)
        self._output_root = Path(self._output_cfg.get("root_dir", "outputs"))
        self._output_root.mkdir(parents=True, exist_ok=True)

        self._device: str = resolve_device(cfg) if not isinstance(cfg, dict) else cfg.get("device", {}).get("backend", "cpu")

        self.metrics = MetricsTracker(
            enabled=self._metrics_cfg.get("enabled", True),
            track_gpu=self._metrics_cfg.get("track_gpu", True),
            track_cpu=self._metrics_cfg.get("track_cpu", True),
            sample_interval_sec=self._metrics_cfg.get("sample_interval_sec", 1.0),
        )

        self._model_factory = ModelFactory(cfg, device=self._device)

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------

    def run(self) -> Dict[str, Any]:
        """Execute the full pipeline end-to-end and return the metrics report.

        Returns:
            Performance metrics report (see
            :meth:`utils.metrics.MetricsTracker.get_report`).

        Raises:
            FileNotFoundError: If the configured input video does not exist.
            RuntimeError: If a required stage fails and ``fail_fast`` is
                enabled in config.
        """
        video_path = self._video_cfg.get("path")
        if not video_path or not Path(video_path).exists():
            raise FileNotFoundError(f"Input video not found: {video_path}")

        self.metrics.start()
        t_start = time.perf_counter()

        try:
            video_loader, frames = self._load_video(video_path)
            shots = self._run_shot_detection(video_path, video_loader)

            detections_per_frame = self._run_detection(frames)
            tracks_per_frame = self._run_tracking(detections_per_frame, frames)
            poses_per_frame = self._run_pose_estimation(tracks_per_frame, frames)
            smpl_results = self._run_motion_recovery(tracks_per_frame, poses_per_frame, frames)
            aligned_results = self._run_trajectory_alignment(smpl_results, shots, tracks_per_frame)

            self._run_visualization(
                frames, detections_per_frame, tracks_per_frame,
                poses_per_frame, smpl_results, aligned_results, shots,
            )

            self._run_export(
                detections_per_frame, tracks_per_frame, poses_per_frame,
                smpl_results, shots, aligned_results, video_loader,
            )

        finally:
            self.metrics.stop()
            video_loader_local = locals().get("video_loader")
            if video_loader_local is not None:
                video_loader_local.release()

        report = self.metrics.get_report()
        report["summary"]["total_wall_clock_sec"] = round(time.perf_counter() - t_start, 3)

        try:
            save_config(self._cfg, self._output_root / "config_used.yaml")
        except Exception as exc:  # pylint: disable=broad-except
            log.debug("Could not save resolved config (non-DictConfig input): {}", exc)

        log.info(
            "Pipeline run complete in {:.1f}s", report["summary"]["total_wall_clock_sec"]
        )
        return report

    # ------------------------------------------------------------------
    # Stage 1: Video loading
    # ------------------------------------------------------------------

    def _load_video(self, video_path: str):
        """Open the input video and read all selected frames into memory."""
        with self.metrics.stage("video_loading"):
            loader = VideoLoader(video_path, config=self._video_cfg)
            loader.load()
            frames = loader.read_all_frames(
                save_dir=self._output_root / "frames" if self._output_cfg.get("save_frames") else None
            )
        log.info("Loaded {} frames from {}", len(frames), video_path)
        return loader, frames

    # ------------------------------------------------------------------
    # Stage 2: Shot detection
    # ------------------------------------------------------------------

    def _run_shot_detection(self, video_path: str, loader: VideoLoader):
        """Detect shot boundaries, or fall back to a single full-video shot."""
        if not self._pipeline_cfg.get("run_shot_detection", True):
            from pipeline.shot_detector import ShotSegment

            total = loader.metadata.frame_count if loader.metadata else 0
            return [ShotSegment(0, 0, max(0, total - 1), 0.0, loader.metadata.duration_sec if loader.metadata else 0.0)]

        with self.metrics.stage("shot_detection"):
            detector = ShotDetector(config=self._pipeline_cfg)
            shots = detector.detect(
                video_path,
                fps=loader.metadata.fps if loader.metadata else 30.0,
                total_frames=loader.metadata.frame_count if loader.metadata else 0,
            )
        return shots

    # ------------------------------------------------------------------
    # Stage 3: Human detection
    # ------------------------------------------------------------------

    def _run_detection(self, frames):
        """Run YOLOv8 human detection across all frames."""
        if not self._pipeline_cfg.get("run_tracking", True) and not self._pipeline_cfg.get("run_pose_estimation", True):
            return {}

        try:
            with self.metrics.stage("detection", frames=len(frames)):
                model = self._model_factory.create_detector()
                stage = HumanDetector(model=model, max_persons=self._max_persons)
                return stage.run_on_frames(frames)
        except Exception as exc:
            self._handle_stage_failure("detection", exc)
            return {}

    # ------------------------------------------------------------------
    # Stage 4: Tracking
    # ------------------------------------------------------------------

    def _run_tracking(self, detections_per_frame, frames):
        """Assign consistent person identities across frames."""
        if not self._pipeline_cfg.get("run_tracking", True):
            return {}

        try:
            with self.metrics.stage("tracking", frames=len(frames)):
                model = self._model_factory.create_tracker()
                stage = PersonTracker(model=model)
                return stage.run_on_frames(detections_per_frame, frames)
        except Exception as exc:
            self._handle_stage_failure("tracking", exc)
            return {}

    # ------------------------------------------------------------------
    # Stage 5: Pose estimation
    # ------------------------------------------------------------------

    def _run_pose_estimation(self, tracks_per_frame, frames):
        """Estimate 2D pose for every tracked person in every frame."""
        if not self._pipeline_cfg.get("run_pose_estimation", True):
            return {}

        try:
            with self.metrics.stage("pose_estimation", frames=len(frames)):
                model = self._model_factory.create_pose_estimator()
                stage = PoseEstimator(model=model)
                return stage.run_on_frames(tracks_per_frame, frames)
        except Exception as exc:
            self._handle_stage_failure("pose_estimation", exc)
            return {}

    # ------------------------------------------------------------------
    # Stage 6: Motion recovery
    # ------------------------------------------------------------------

    def _run_motion_recovery(self, tracks_per_frame, poses_per_frame, frames):
        """Recover 3D SMPL body parameters for every tracked person."""
        if not self._pipeline_cfg.get("run_motion_recovery", True):
            return {}

        try:
            with self.metrics.stage("motion_recovery", frames=len(frames)):
                model = self._model_factory.create_motion_recovery()
                stage = MotionRecovery(model=model)
                return stage.run_on_frames(tracks_per_frame, poses_per_frame, frames)
        except Exception as exc:
            self._handle_stage_failure("motion_recovery", exc)
            return {}

    # ------------------------------------------------------------------
    # Stage 7: Trajectory alignment
    # ------------------------------------------------------------------

    def _run_trajectory_alignment(self, smpl_results, shots, tracks_per_frame):
        """Smooth and cross-shot align all recovered 3D trajectories."""
        if not self._pipeline_cfg.get("run_trajectory_alignment", True) or not smpl_results:
            return {}

        track_ids = sorted({tid for (_, tid) in smpl_results.keys()})

        try:
            with self.metrics.stage("trajectory_alignment"):
                aligner = TrajectoryAligner(config=self._alignment_cfg)
                return aligner.align(smpl_results, shots, track_ids)
        except Exception as exc:
            self._handle_stage_failure("trajectory_alignment", exc)
            return {}

    # ------------------------------------------------------------------
    # Stage 8: Visualization
    # ------------------------------------------------------------------

    def _run_visualization(
        self, frames, detections_per_frame, tracks_per_frame,
        poses_per_frame, smpl_results, aligned_results, shots,
    ) -> None:
        """Render all visualization stage videos."""
        if not self._pipeline_cfg.get("run_visualization", True):
            return

        skeleton_pairs = self._pose_cfg.get("mediapipe", {}).get("skeleton_pairs")

        try:
            with self.metrics.stage("visualization", frames=len(frames)):
                viz = Visualizer(
                    config=self._viz_cfg,
                    output_dir=self._output_root,
                    pose_skeleton_pairs=skeleton_pairs,
                )
                viz.render_all(
                    frames, detections_per_frame, tracks_per_frame,
                    poses_per_frame, smpl_results, aligned_results, shots,
                )
        except Exception as exc:
            self._handle_stage_failure("visualization", exc)

    # ------------------------------------------------------------------
    # Stage 9: Export
    # ------------------------------------------------------------------

    def _run_export(
        self, detections_per_frame, tracks_per_frame, poses_per_frame,
        smpl_results, shots, aligned_results, video_loader,
    ) -> None:
        """Write all final results (JSON/CSV/NPY) to the output directory."""
        if not self._pipeline_cfg.get("run_export", True):
            return

        fps = video_loader.metadata.fps if video_loader.metadata else 30.0

        try:
            with self.metrics.stage("export"):
                writer = OutputWriter(config=self._output_cfg, output_dir=self._output_root)
                writer.write_all(
                    detections_per_frame=detections_per_frame,
                    tracks_per_frame=tracks_per_frame,
                    poses_per_frame=poses_per_frame,
                    smpl_results=smpl_results,
                    shot_segments=shots,
                    aligned_results=aligned_results,
                    metrics_report=None,  # written after run() finalises
                    fps=fps,
                )
        except Exception as exc:
            self._handle_stage_failure("export", exc)

    # ------------------------------------------------------------------
    # Error handling
    # ------------------------------------------------------------------

    def _handle_stage_failure(self, stage_name: str, exc: Exception) -> None:
        """Log (and optionally re-raise) a pipeline stage failure.

        Args:
            stage_name: Name of the failed stage, for logging.
            exc: The caught exception.

        Raises:
            RuntimeError: If ``cfg.pipeline.fail_fast`` is ``True``.
        """
        log.error("Pipeline stage '{}' failed: {}", stage_name, exc)
        if self._fail_fast:
            raise RuntimeError(f"Pipeline stage '{stage_name}' failed") from exc
        log.warning("fail_fast=False — continuing pipeline despite '{}' failure", stage_name)
