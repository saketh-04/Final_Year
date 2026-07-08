"""
HumanMM — Motion Recovery Pipeline Module.

Orchestrates 3D human body recovery across all tracked persons using the
HMR 2.0 (or GVHMR) wrapper.  For each frame and each confirmed track,
crops the person from the full frame and runs SMPL parameter estimation.

Integrates 2D pose results (from ``PoseEstimator``) as auxiliary input
for the pseudo-3D fallback mode.

Design Pattern: Strategy — motion recovery model is injected.

Example:
    >>> from pipeline.motion_recovery import MotionRecovery
    >>> mr = MotionRecovery(model=hmr2_model)
    >>> smpl_results = mr.run_on_frames(tracks_dict, poses_dict, frames_list)
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np

from models.base_model import BaseModel
from models.bytetrack_tracker import Track
from models.mediapipe_pose import PersonPose
from models.gvhmr_wrapper import SMPLOutput
from utils.logger import get_logger
from utils.frame_utils import Frame, bgr_to_rgb

log = get_logger(__name__)


class MotionRecovery:
    """Pipeline stage for 3D SMPL body parameter recovery.

    Processes each (frame, track) pair and stores per-person SMPL outputs
    indexed by ``(frame_idx, track_id)``.

    Args:
        model: Initialised motion recovery model (HMR 2.0 or compatible).
        min_bbox_area: Skip crops smaller than this area (pixels²).

    Example:
        >>> mr = MotionRecovery(model=hmr2_model)
        >>> results = mr.run_on_frames(tracks_dict, poses_dict, frame_list)
    """

    def __init__(
        self,
        model: BaseModel,
        min_bbox_area: float = 2000.0,
    ) -> None:
        self._model = model
        self._min_bbox_area = min_bbox_area
        # results[(frame_idx, track_id)] = SMPLOutput
        self._results: Dict[Tuple[int, int], SMPLOutput] = {}

    def recover_frame(
        self,
        frame: Frame,
        tracks: List[Track],
        poses: Dict[int, PersonPose],
        frame_idx: int,
    ) -> Dict[int, SMPLOutput]:
        """Run 3D recovery for all tracks in a single frame.

        Args:
            frame: BGR image array ``(H, W, 3)``.
            tracks: Confirmed tracks in this frame.
            poses: Dict ``track_id → PersonPose`` for this frame.
            frame_idx: Zero-based frame index.

        Returns:
            Dict mapping ``track_id → SMPLOutput`` for successful recoveries.
        """
        frame_rgb = bgr_to_rgb(frame)
        frame_results: Dict[int, SMPLOutput] = {}

        for track in tracks:
            x1, y1, x2, y2 = track.bbox
            bbox_area = (x2 - x1) * (y2 - y1)

            if bbox_area < self._min_bbox_area:
                log.debug(
                    "Skipping recovery (track={}, area={:.0f} < {})",
                    track.track_id,
                    bbox_area,
                    self._min_bbox_area,
                )
                continue

            # Provide 2D joints to fallback mode
            joints_2d = None
            if track.track_id in poses:
                joints_2d = poses[track.track_id].keypoints_array  # (J, 3)

            try:
                smpl_out = self._model.run(
                    frame_rgb=frame_rgb,
                    track_id=track.track_id,
                    frame_idx=frame_idx,
                    bbox=tuple(track.bbox.tolist()),
                    joints_2d=joints_2d,
                )
                if smpl_out is not None:
                    frame_results[track.track_id] = smpl_out
                    self._results[(frame_idx, track.track_id)] = smpl_out

            except Exception as exc:
                log.warning(
                    "Motion recovery failed (track={}, frame={}): {}",
                    track.track_id,
                    frame_idx,
                    exc,
                )

        return frame_results

    def run_on_frames(
        self,
        tracks_per_frame: Dict[int, List[Track]],
        poses_per_frame: Dict[int, Dict[int, PersonPose]],
        frames: List[Tuple[Frame, int]],
    ) -> Dict[Tuple[int, int], SMPLOutput]:
        """Run 3D motion recovery across all frames.

        Args:
            tracks_per_frame: Dict mapping ``frame_idx → List[Track]``.
            poses_per_frame: Nested dict ``frame_idx → {track_id → PersonPose}``.
            frames: List of ``(BGR frame, frame_idx)`` tuples.

        Returns:
            Dict mapping ``(frame_idx, track_id) → SMPLOutput``.
        """
        total = len(frames)
        total_recovered = 0

        for frame, frame_idx in frames:
            tracks = tracks_per_frame.get(frame_idx, [])
            poses = poses_per_frame.get(frame_idx, {})
            recovered = self.recover_frame(frame, tracks, poses, frame_idx)
            total_recovered += len(recovered)

            if frame_idx % 30 == 0:
                log.debug(
                    "Motion recovery: frame {}/{} — {} persons",
                    frame_idx,
                    total,
                    len(recovered),
                )

        log.info(
            "Motion recovery complete: {} frames | {} total SMPL estimates",
            total,
            total_recovered,
        )
        return self._results

    def get_results(self) -> Dict[Tuple[int, int], SMPLOutput]:
        """Return all accumulated motion recovery results."""
        return self._results

    def get_track_sequence(self, track_id: int) -> Dict[int, SMPLOutput]:
        """Return all SMPL outputs for a specific track (chronological).

        Args:
            track_id: Track ID to retrieve.

        Returns:
            Dict mapping ``frame_idx → SMPLOutput``.
        """
        result: Dict[int, SMPLOutput] = {}
        for (fidx, tid), out in self._results.items():
            if tid == track_id:
                result[fidx] = out
        return dict(sorted(result.items()))

    def build_joints3d_array(self, track_id: int) -> Tuple[np.ndarray, List[int]]:
        """Build a time-series 3D joint array for a specific track.

        Args:
            track_id: Track ID to build array for.

        Returns:
            Tuple of:
                - ``joints_3d``: Array of shape ``(T, J, 3)``
                - ``frame_indices``: List of corresponding frame indices

        Example:
            >>> joints, frames = mr.build_joints3d_array(track_id=1)
            >>> print(joints.shape)  # (T, 17, 3)
        """
        seq = self.get_track_sequence(track_id)
        if not seq:
            return np.zeros((0, 17, 3)), []

        frame_indices = sorted(seq.keys())
        J = seq[frame_indices[0]].joints_3d.shape[0]
        joints_3d = np.stack([seq[f].joints_3d for f in frame_indices], axis=0)

        return joints_3d, frame_indices
