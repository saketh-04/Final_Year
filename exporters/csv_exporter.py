"""
HumanMM — CSV Exporter.

Flattens per-frame, per-person 3D trajectory data (aligned root translation,
global rotation, and joint positions) into tabular CSV files suitable for
spreadsheet analysis or downstream tooling.

Example:
    >>> from exporters.csv_exporter import CSVExporter
    >>> exporter = CSVExporter(output_dir="outputs")
    >>> exporter.export_trajectory(aligned_results, fps=30.0)
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Union

from pipeline.trajectory_aligner import AlignedTrackResult
from utils.io_utils import save_csv
from utils.logger import get_logger

log = get_logger(__name__)


class CSVExporter:
    """Writes flattened trajectory and detection tables to CSV.

    Args:
        output_dir: Root output directory (mirrors ``cfg.output.root_dir``).

    Example:
        >>> exporter = CSVExporter(output_dir="outputs")
        >>> exporter.export_trajectory(aligned_results, fps=30.0)
    """

    def __init__(self, output_dir: Union[str, Path] = "outputs") -> None:
        self._root = Path(output_dir)

    def export_trajectory(
        self,
        aligned_results: Dict[int, AlignedTrackResult],
        fps: float = 30.0,
    ) -> Path:
        """Export per-frame root translation/rotation to ``trajectory.csv``.

        One row per ``(track_id, frame_idx)`` with columns:
        ``frame_idx, time_sec, track_id, tx, ty, tz, rx, ry, rz``.

        Args:
            aligned_results: Dict mapping ``track_id → AlignedTrackResult``.
            fps: Frame rate used to compute the ``time_sec`` column.

        Returns:
            Path to the written CSV file.
        """
        rows: List[Dict[str, Any]] = []
        for track_id, result in aligned_results.items():
            for i, frame_idx in enumerate(result.frame_indices):
                tx, ty, tz = result.translations_aligned[i]
                rx, ry, rz = result.rotations_aligned[i]
                rows.append({
                    "frame_idx": frame_idx,
                    "time_sec": round(frame_idx / fps, 4) if fps > 0 else 0.0,
                    "track_id": track_id,
                    "tx": float(tx), "ty": float(ty), "tz": float(tz),
                    "rx": float(rx), "ry": float(ry), "rz": float(rz),
                })

        rows.sort(key=lambda r: (r["frame_idx"], r["track_id"]))
        path = self._root / "trajectory.csv"
        save_csv(rows, path)
        return path

    def export_joints_long_format(
        self,
        aligned_results: Dict[int, AlignedTrackResult],
        fps: float = 30.0,
    ) -> Path:
        """Export every 3D joint, per frame and person, in long/tidy format.

        One row per ``(track_id, frame_idx, joint_idx)`` with columns:
        ``frame_idx, time_sec, track_id, joint_idx, x, y, z``.  This format
        is convenient for pivoting in pandas/Excel but considerably larger
        than the root-only :meth:`export_trajectory` table.

        Args:
            aligned_results: Dict mapping ``track_id → AlignedTrackResult``.
            fps: Frame rate used to compute the ``time_sec`` column.

        Returns:
            Path to the written CSV file.
        """
        rows: List[Dict[str, Any]] = []
        for track_id, result in aligned_results.items():
            joints = result.joints_3d_aligned  # (T, J, 3)
            for i, frame_idx in enumerate(result.frame_indices):
                for j in range(joints.shape[1]):
                    x, y, z = joints[i, j]
                    rows.append({
                        "frame_idx": frame_idx,
                        "time_sec": round(frame_idx / fps, 4) if fps > 0 else 0.0,
                        "track_id": track_id,
                        "joint_idx": j,
                        "x": float(x), "y": float(y), "z": float(z),
                    })

        path = self._root / "joints_long.csv"
        save_csv(rows, path)
        return path

    def export_detection_summary(
        self, detections_per_frame: Dict[int, List[Any]]
    ) -> Path:
        """Export a per-frame detection-count summary to ``detection_summary.csv``.

        Args:
            detections_per_frame: Dict mapping ``frame_idx → List[Detection]``.

        Returns:
            Path to the written CSV file.
        """
        rows = [
            {
                "frame_idx": fidx,
                "num_persons": len(dets),
                "mean_confidence": (
                    round(sum(d.confidence for d in dets) / len(dets), 4) if dets else 0.0
                ),
            }
            for fidx, dets in sorted(detections_per_frame.items())
        ]
        path = self._root / "detection_summary.csv"
        save_csv(rows, path)
        return path
