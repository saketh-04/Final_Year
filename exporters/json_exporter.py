"""
HumanMM — JSON Exporter.

Serialises every pipeline stage's results (detections, tracks, poses,
SMPL summaries, shot segments, aligned-trajectory metadata, and
performance metrics) to JSON files under the configured output directory.

Example:
    >>> from exporters.json_exporter import JSONExporter
    >>> exporter = JSONExporter(output_dir="outputs")
    >>> exporter.export_detections(detections_per_frame)
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
from utils.io_utils import save_json
from utils.logger import get_logger

log = get_logger(__name__)


class JSONExporter:
    """Writes structured pipeline results to JSON files.

    Args:
        output_dir: Root output directory (mirrors ``cfg.output.root_dir``).

    Example:
        >>> exporter = JSONExporter(output_dir="outputs")
        >>> exporter.export_shots(shot_segments)
    """

    def __init__(self, output_dir: Union[str, Path] = "outputs") -> None:
        self._root = Path(output_dir)

    def export_detections(self, detections_per_frame: Dict[int, List[Detection]]) -> Path:
        """Export per-frame detection results to ``detections.json``.

        Args:
            detections_per_frame: Dict mapping ``frame_idx → List[Detection]``.

        Returns:
            Path to the written JSON file.
        """
        payload = {
            str(fidx): [d.to_dict() for d in dets]
            for fidx, dets in sorted(detections_per_frame.items())
        }
        path = self._root / "detections.json"
        save_json(payload, path)
        return path

    def export_tracks(self, tracks_per_frame: Dict[int, List[Track]]) -> Path:
        """Export per-frame tracking results to ``tracks.json``.

        Args:
            tracks_per_frame: Dict mapping ``frame_idx → List[Track]``.

        Returns:
            Path to the written JSON file.
        """
        payload = {
            str(fidx): [t.to_dict() for t in tracks]
            for fidx, tracks in sorted(tracks_per_frame.items())
        }
        path = self._root / "tracks.json"
        save_json(payload, path)
        return path

    def export_poses(self, poses_per_frame: Dict[int, Dict[int, PersonPose]]) -> Path:
        """Export per-frame, per-person 2D pose results to ``poses.json``.

        Args:
            poses_per_frame: Nested dict ``frame_idx → {track_id → PersonPose}``.

        Returns:
            Path to the written JSON file.
        """
        payload = {
            str(fidx): {str(tid): pose.to_dict() for tid, pose in poses.items()}
            for fidx, poses in sorted(poses_per_frame.items())
        }
        path = self._root / "poses.json"
        save_json(payload, path)
        return path

    def export_smpl_summary(
        self, smpl_results: Dict[Tuple[int, int], SMPLOutput]
    ) -> Path:
        """Export lightweight SMPL parameter summaries to ``smpl_params.json``.

        Large arrays (vertices, joints) are intentionally omitted — use
        :class:`exporters.npy_exporter.NPYExporter` for those.

        Args:
            smpl_results: Dict mapping ``(frame_idx, track_id) → SMPLOutput``.

        Returns:
            Path to the written JSON file.
        """
        payload: Dict[str, Dict[str, Any]] = {}
        for (fidx, tid), out in sorted(smpl_results.items()):
            payload.setdefault(str(fidx), {})[str(tid)] = out.to_dict()
        path = self._root / "smpl_params.json"
        save_json(payload, path)
        return path

    def export_shots(self, shot_segments: List[ShotSegment]) -> Path:
        """Export detected shot boundaries to ``shots.json``.

        Args:
            shot_segments: List of detected :class:`ShotSegment` objects.

        Returns:
            Path to the written JSON file.
        """
        payload = [shot.to_dict() for shot in shot_segments]
        path = self._root / "shots.json"
        save_json(payload, path)
        return path

    def export_aligned_summary(
        self, aligned_results: Dict[int, AlignedTrackResult]
    ) -> Path:
        """Export aligned-trajectory metadata to ``aligned_trajectory.json``.

        Args:
            aligned_results: Dict mapping ``track_id → AlignedTrackResult``.

        Returns:
            Path to the written JSON file.
        """
        payload = {str(tid): result.to_dict() for tid, result in aligned_results.items()}
        path = self._root / "aligned_trajectory.json"
        save_json(payload, path)
        return path

    def export_metrics(self, metrics_report: Dict[str, Any]) -> Path:
        """Export the performance metrics report to ``metrics.json``.

        Args:
            metrics_report: Report dict produced by
                :meth:`utils.metrics.MetricsTracker.get_report`.

        Returns:
            Path to the written JSON file.
        """
        path = self._root / "metrics.json"
        save_json(metrics_report, path)
        return path

    def export_all(
        self,
        detections_per_frame: Optional[Dict[int, List[Detection]]] = None,
        tracks_per_frame: Optional[Dict[int, List[Track]]] = None,
        poses_per_frame: Optional[Dict[int, Dict[int, PersonPose]]] = None,
        smpl_results: Optional[Dict[Tuple[int, int], SMPLOutput]] = None,
        shot_segments: Optional[List[ShotSegment]] = None,
        aligned_results: Optional[Dict[int, AlignedTrackResult]] = None,
        metrics_report: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Path]:
        """Export every available pipeline result in a single call.

        Any argument left as ``None`` is skipped silently — this allows
        ``main.py`` to call ``export_all`` regardless of which optional
        pipeline stages were enabled.

        Returns:
            Dict mapping export name → written file path.
        """
        written: Dict[str, Path] = {}

        if detections_per_frame is not None:
            written["detections"] = self.export_detections(detections_per_frame)
        if tracks_per_frame is not None:
            written["tracks"] = self.export_tracks(tracks_per_frame)
        if poses_per_frame is not None:
            written["poses"] = self.export_poses(poses_per_frame)
        if smpl_results is not None:
            written["smpl_params"] = self.export_smpl_summary(smpl_results)
        if shot_segments is not None:
            written["shots"] = self.export_shots(shot_segments)
        if aligned_results is not None:
            written["aligned_trajectory"] = self.export_aligned_summary(aligned_results)
        if metrics_report is not None:
            written["metrics"] = self.export_metrics(metrics_report)

        log.info("JSONExporter: wrote {} files → {}", len(written), self._root)
        return written
