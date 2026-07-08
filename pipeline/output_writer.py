"""
HumanMM — Output Writer Pipeline Module.

Facade over the four exporter back-ends (JSON, CSV, NPY, video/GIF) that
gates each export behind the corresponding ``cfg.output.save_*`` flag and
reports the full set of written file paths back to the caller
(:class:`pipeline.pipeline_manager.PipelineManager`).

Design Pattern: Facade — callers do not need to know which exporter
module handles which file format; they call :meth:`OutputWriter.write_all`
once, after every other stage has produced its results.

Example:
    >>> from pipeline.output_writer import OutputWriter
    >>> writer = OutputWriter(config=cfg.output, output_dir="outputs")
    >>> paths = writer.write_all(detections, tracks, poses, smpl, shots, aligned, metrics)
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

from models.bytetrack_tracker import Track
from models.gvhmr_wrapper import SMPLOutput
from models.mediapipe_pose import PersonPose
from models.yolo_detector import Detection
from pipeline.shot_detector import ShotSegment
from pipeline.trajectory_aligner import AlignedTrackResult
from utils.logger import get_logger

from exporters.csv_exporter import CSVExporter
from exporters.json_exporter import JSONExporter
from exporters.npy_exporter import NPYExporter

log = get_logger(__name__)


class OutputWriter:
    """Facade that gates and dispatches all final-results export operations.

    Args:
        config: Output config dict (mirrors ``cfg.output``).
        output_dir: Root output directory (mirrors ``cfg.output.root_dir``).

    Example:
        >>> writer = OutputWriter(config=cfg.output, output_dir="outputs")
        >>> written_paths = writer.write_all(
        ...     detections_per_frame=det, tracks_per_frame=trk,
        ...     poses_per_frame=poses, smpl_results=smpl,
        ...     shot_segments=shots, aligned_results=aligned,
        ...     metrics_report=metrics, fps=30.0,
        ... )
    """

    def __init__(
        self,
        config: Optional[Dict[str, Any]] = None,
        output_dir: Union[str, Path] = "outputs",
    ) -> None:
        self._cfg: Dict[str, Any] = config or {}
        self._root = Path(output_dir)
        self._root.mkdir(parents=True, exist_ok=True)

        self._save_json: bool = self._cfg.get("save_json", True)
        self._save_csv: bool = self._cfg.get("save_csv", True)
        self._save_npy: bool = self._cfg.get("save_npy", True)
        self._save_mesh: bool = self._cfg.get("save_mesh", False)

        self._json_exporter = JSONExporter(output_dir=self._root)
        self._csv_exporter = CSVExporter(output_dir=self._root)
        self._npy_exporter = NPYExporter(output_dir=self._root)

    def write_all(
        self,
        detections_per_frame: Optional[Dict[int, List[Detection]]] = None,
        tracks_per_frame: Optional[Dict[int, List[Track]]] = None,
        poses_per_frame: Optional[Dict[int, Dict[int, PersonPose]]] = None,
        smpl_results: Optional[Dict[Tuple[int, int], SMPLOutput]] = None,
        shot_segments: Optional[List[ShotSegment]] = None,
        aligned_results: Optional[Dict[int, AlignedTrackResult]] = None,
        metrics_report: Optional[Dict[str, Any]] = None,
        fps: float = 30.0,
    ) -> Dict[str, Path]:
        """Export all available results, gated by ``cfg.output.save_*`` flags.

        Any argument left as ``None`` is treated as "stage was disabled
        upstream" and silently skipped.

        Args:
            detections_per_frame: Dict ``frame_idx → List[Detection]``.
            tracks_per_frame: Dict ``frame_idx → List[Track]``.
            poses_per_frame: Nested dict ``frame_idx → {track_id → PersonPose}``.
            smpl_results: Dict ``(frame_idx, track_id) → SMPLOutput``.
            shot_segments: Detected shot boundaries.
            aligned_results: Dict ``track_id → AlignedTrackResult``.
            metrics_report: Performance metrics report dict.
            fps: Video frame rate, used for CSV ``time_sec`` columns.

        Returns:
            Dict mapping export name → written file path.

        Example:
            >>> written = writer.write_all(
            ...     detections_per_frame=dets, tracks_per_frame=tracks,
            ...     aligned_results=aligned, fps=29.97,
            ... )
        """
        written: Dict[str, Path] = {}

        if self._save_json:
            written.update(
                self._json_exporter.export_all(
                    detections_per_frame=detections_per_frame,
                    tracks_per_frame=tracks_per_frame,
                    poses_per_frame=poses_per_frame,
                    smpl_results=smpl_results,
                    shot_segments=shot_segments,
                    aligned_results=aligned_results,
                    metrics_report=metrics_report,
                )
            )
        else:
            log.debug("OutputWriter: JSON export disabled via config")

        if self._save_csv and aligned_results:
            try:
                written["trajectory_csv"] = self._csv_exporter.export_trajectory(
                    aligned_results, fps=fps
                )
            except Exception as exc:  # pylint: disable=broad-except
                log.warning("CSV trajectory export failed: {}", exc)

            if detections_per_frame:
                try:
                    written["detection_summary_csv"] = self._csv_exporter.export_detection_summary(
                        detections_per_frame
                    )
                except Exception as exc:  # pylint: disable=broad-except
                    log.warning("CSV detection summary export failed: {}", exc)
        elif not self._save_csv:
            log.debug("OutputWriter: CSV export disabled via config")

        if self._save_npy and aligned_results:
            try:
                written["joints_npz"] = self._npy_exporter.export_aligned_joints(aligned_results)
            except Exception as exc:  # pylint: disable=broad-except
                log.warning("NPY joints export failed: {}", exc)

            if smpl_results:
                try:
                    written["smpl_params_npz"] = self._npy_exporter.export_smpl_params(smpl_results)
                except Exception as exc:  # pylint: disable=broad-except
                    log.warning("NPY SMPL params export failed: {}", exc)

                if self._save_mesh:
                    try:
                        written["mesh_vertices_npz"] = self._npy_exporter.export_mesh_vertices(
                            smpl_results
                        )
                    except Exception as exc:  # pylint: disable=broad-except
                        log.warning("NPY mesh vertices export failed: {}", exc)
        elif not self._save_npy:
            log.debug("OutputWriter: NPY export disabled via config")

        log.info("OutputWriter: wrote {} result files → {}", len(written), self._root)
        return written
