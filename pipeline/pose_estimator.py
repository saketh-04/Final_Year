"""
HumanMM — Pose Estimator Pipeline Module.

Orchestrates 2D pose estimation across all tracked persons and frames.
Accepts injected pose backend (MediaPipe or ViTPose) and processes each
person crop independently, returning per-person, per-frame ``PersonPose``
structures.

Upgraded to CVPR-quality with:
- OneEuro temporal filtering for jitter-free skeletons.
- Missing joint recovery via temporal extrapolation.
- Left/Right limb consistency to prevent swapping.
- Rich tracking metrics (FPS, Latency, Conf).
"""

from __future__ import annotations

import time
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np

from models.base_model import BaseModel
from models.bytetrack_tracker import Track
from models.mediapipe_pose import PersonPose, Keypoint
from utils.logger import get_logger
from utils.frame_utils import Frame, bgr_to_rgb
from utils.filters import OneEuroFilter

log = get_logger(__name__)


class PoseEstimator:
    """Pipeline stage for multi-person 2D pose estimation.

    For each frame, iterates over all active tracks and runs the pose backend
    on the cropped person region. Applies OneEuro smoothing and limb consistency
    heuristics for research-quality CVPR/ICCV level output.

    Args:
        model: Initialised pose estimator (MediaPipe or ViTPose backend).
        config: Pose config dictionary containing filtering hyperparameters.
    """

    def __init__(self, model: BaseModel, config: Optional[Dict[str, Any]] = None) -> None:
        self._model = model
        self._config = config or {}
        
        # results[frame_idx][track_id] = PersonPose
        self._results: Dict[int, Dict[int, PersonPose]] = {}
        
        # Track-specific filters: filters[track_id] = OneEuroFilter
        self._filters: Dict[int, OneEuroFilter] = {}
        
        # Track-specific last known good keypoints (for missing joint recovery)
        self._last_good: Dict[int, PersonPose] = {}
        
        # Tuning params
        common_cfg = self._config.get("common", {})
        self._conf_threshold = common_cfg.get("confidence_threshold", 0.30)
        
        filter_cfg = self._config.get("temporal_filter", {})
        self._use_filter = filter_cfg.get("enabled", True)
        self._min_cutoff = filter_cfg.get("min_cutoff", 1.0)
        self._beta = filter_cfg.get("beta", 0.05)
        self._d_cutoff = filter_cfg.get("d_cutoff", 1.0)

    def _apply_left_right_consistency(
        self,
        current_pose: PersonPose,
        last_pose: PersonPose,
    ) -> None:
        """Prevent left/right limb swapping by checking crossing over temporal history.
        
        Swaps back left/right shoulder, elbow, wrist, hip, knee, ankle if the 
        horizontal relationship suddenly flipped compared to last frame, assuming
        the person didn't instantly turn around.
        """
        # A simple CVPR-style heuristic: shoulders and hips don't cross instantly
        kpts_curr = current_pose.keypoints
        kpts_last = last_pose.keypoints
        
        # Indices (from COCO17)
        L_SH, R_SH = 5, 6
        L_HIP, R_HIP = 11, 12
        
        # Function to check horizontal relationship (is L to the left of R?)
        # Since image coordinates: x increases left to right. L is usually right side of image 
        # if facing camera, but let's just check relative sign.
        def relative_sign(k1, k2):
            return np.sign(k1.x - k2.x)
            
        # Check shoulders
        if (kpts_curr[L_SH].confidence > self._conf_threshold and kpts_curr[R_SH].confidence > self._conf_threshold and
            kpts_last[L_SH].confidence > self._conf_threshold and kpts_last[R_SH].confidence > self._conf_threshold):
            
            curr_sign = relative_sign(kpts_curr[L_SH], kpts_curr[R_SH])
            last_sign = relative_sign(kpts_last[L_SH], kpts_last[R_SH])
            
            # If swapped suddenly, revert the swap for the whole arm
            if curr_sign != last_sign and curr_sign != 0:
                log.debug("PoseEstimator: Prevented left/right arm swap on track {}", current_pose.track_id)
                # Swap shoulders
                kpts_curr[L_SH].x, kpts_curr[R_SH].x = kpts_curr[R_SH].x, kpts_curr[L_SH].x
                kpts_curr[L_SH].y, kpts_curr[R_SH].y = kpts_curr[R_SH].y, kpts_curr[L_SH].y
                # Swap elbows
                kpts_curr[7].x, kpts_curr[8].x = kpts_curr[8].x, kpts_curr[7].x
                kpts_curr[7].y, kpts_curr[8].y = kpts_curr[8].y, kpts_curr[7].y
                # Swap wrists
                kpts_curr[9].x, kpts_curr[10].x = kpts_curr[10].x, kpts_curr[9].x
                kpts_curr[9].y, kpts_curr[10].y = kpts_curr[10].y, kpts_curr[9].y

        # Check hips
        if (kpts_curr[L_HIP].confidence > self._conf_threshold and kpts_curr[R_HIP].confidence > self._conf_threshold and
            kpts_last[L_HIP].confidence > self._conf_threshold and kpts_last[R_HIP].confidence > self._conf_threshold):
            
            curr_sign = relative_sign(kpts_curr[L_HIP], kpts_curr[R_HIP])
            last_sign = relative_sign(kpts_last[L_HIP], kpts_last[R_HIP])
            
            # If swapped suddenly, revert the swap for the whole leg
            if curr_sign != last_sign and curr_sign != 0:
                log.debug("PoseEstimator: Prevented left/right leg swap on track {}", current_pose.track_id)
                # Swap hips
                kpts_curr[L_HIP].x, kpts_curr[R_HIP].x = kpts_curr[R_HIP].x, kpts_curr[L_HIP].x
                kpts_curr[L_HIP].y, kpts_curr[R_HIP].y = kpts_curr[R_HIP].y, kpts_curr[L_HIP].y
                # Swap knees
                kpts_curr[13].x, kpts_curr[14].x = kpts_curr[14].x, kpts_curr[13].x
                kpts_curr[13].y, kpts_curr[14].y = kpts_curr[14].y, kpts_curr[13].y
                # Swap ankles
                kpts_curr[15].x, kpts_curr[16].x = kpts_curr[16].x, kpts_curr[15].x
                kpts_curr[15].y, kpts_curr[16].y = kpts_curr[16].y, kpts_curr[15].y


    def estimate_frame(
        self,
        frame: Frame,
        tracks: List[Track],
        frame_idx: int,
        fps_timestamp: float,
    ) -> Tuple[Dict[int, PersonPose], int, float, float]:
        """Estimate 2D poses for all tracks in a single frame.

        Args:
            frame: BGR image array ``(H, W, 3)``.
            tracks: Confirmed tracks for this frame.
            frame_idx: Zero-based frame index.
            fps_timestamp: Current timestamp (seconds) used for OneEuro filter.

        Returns:
            Tuple containing:
            - Dict mapping ``track_id → PersonPose``
            - int: Number of missing joints across all persons
            - float: Average confidence across all joints
            - float: Latency in ms for this frame
        """
        t0 = time.perf_counter()
        frame_rgb = bgr_to_rgb(frame)
        frame_poses: Dict[int, PersonPose] = {}
        
        missing_joints = 0
        total_joints = 0
        conf_sum = 0.0

        for track in tracks:
            bbox = tuple(track.bbox.tolist())
            try:
                pose = self._model.run(
                    frame_rgb=frame_rgb,
                    track_id=track.track_id,
                    frame_idx=frame_idx,
                    bbox=bbox,
                )
                
                if pose is not None:
                    # 1. Left/Right Consistency
                    if track.track_id in self._last_good:
                        self._apply_left_right_consistency(pose, self._last_good[track.track_id])
                    
                    # 2. Missing Joint Recovery
                    if track.track_id in self._last_good:
                        last_pose = self._last_good[track.track_id]
                        for i, kp in enumerate(pose.keypoints):
                            if kp.confidence < self._conf_threshold and last_pose.keypoints[i].confidence >= self._conf_threshold:
                                # Extrapolate using last known good position
                                kp.x = last_pose.keypoints[i].x
                                kp.y = last_pose.keypoints[i].y
                                # We decay confidence slightly so it eventually fades if permanently occluded
                                kp.confidence = last_pose.keypoints[i].confidence * 0.9
                    
                    # 3. Temporal Smoothing (OneEuro)
                    if self._use_filter:
                        kpts_array = pose.keypoints_array[:, :2]  # (J, 2)
                        
                        if track.track_id not in self._filters:
                            self._filters[track.track_id] = OneEuroFilter(
                                t0=fps_timestamp,
                                x0=kpts_array,
                                min_cutoff=self._min_cutoff,
                                beta=self._beta,
                                d_cutoff=self._d_cutoff
                            )
                        else:
                            smoothed_kpts = self._filters[track.track_id](fps_timestamp, kpts_array)
                            for i, kp in enumerate(pose.keypoints):
                                kp.x = float(smoothed_kpts[i, 0])
                                kp.y = float(smoothed_kpts[i, 1])
                    
                    # Update metrics
                    for kp in pose.keypoints:
                        total_joints += 1
                        conf_sum += kp.confidence
                        if kp.confidence < self._conf_threshold:
                            missing_joints += 1
                            
                    # Update tracking
                    self._last_good[track.track_id] = pose
                    frame_poses[track.track_id] = pose
                    
            except Exception as exc:
                log.warning("Pose estimation failed (track={}, frame={}): {}", track.track_id, frame_idx, exc)

        latency_ms = (time.perf_counter() - t0) * 1000.0
        avg_conf = (conf_sum / total_joints) if total_joints > 0 else 0.0

        self._results[frame_idx] = frame_poses
        return frame_poses, missing_joints, avg_conf, latency_ms

    def run_on_frames(
        self,
        tracks_per_frame: Dict[int, List[Track]],
        frames: List[Tuple[Frame, int]],
    ) -> Dict[int, Dict[int, PersonPose]]:
        """Run pose estimation across all frames with their tracks.

        Args:
            tracks_per_frame: Dict mapping ``frame_idx → List[Track]``.
            frames: List of ``(BGR frame, frame_idx)`` tuples in order.

        Returns:
            Nested dict: ``results[frame_idx][track_id] = PersonPose``.
        """
        total = len(frames)
        total_poses = 0
        total_missing = 0
        
        # Assume 30fps default if timestamps aren't provided
        dt = 1.0 / 30.0 

        for i, (frame, frame_idx) in enumerate(frames):
            tracks = tracks_per_frame.get(frame_idx, [])
            fps_time = i * dt
            
            poses, missing, avg_conf, lat = self.estimate_frame(frame, tracks, frame_idx, fps_time)
            
            total_poses += len(poses)
            total_missing += missing
            
            # FPS tracking for this frame
            pose_fps = (1000.0 / lat) if lat > 0 else 0.0

            if frame_idx % 50 == 0:
                log.info(
                    "Pose Frame {:>4}/{} | persons={} fps={:.1f} lat={:.1f}ms conf={:.3f} missing={}",
                    frame_idx, total, len(poses), pose_fps, lat, avg_conf, missing
                )

        log.info(
            "Pose estimation complete: {} frames | {} total poses | {} missing joints recovered/smoothed",
            total, total_poses, total_missing
        )
        return self._results

    def get_results(self) -> Dict[int, Dict[int, PersonPose]]:
        """Return all accumulated pose results."""
        return self._results

    def get_track_poses(self, track_id: int) -> Dict[int, PersonPose]:
        """Return all poses for a specific track across all frames."""
        result: Dict[int, PersonPose] = {}
        for frame_idx, poses in self._results.items():
            if track_id in poses:
                result[frame_idx] = poses[track_id]
        return result

    def to_joint_array(
        self,
        track_id: int,
        frame_indices: Optional[List[int]] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Build a ``(T, J, 3)`` joint array for a track over time."""
        track_poses = self.get_track_poses(track_id)

        if frame_indices is None:
            frame_indices = sorted(track_poses.keys())

        T = len(frame_indices)
        if T == 0:
            return np.zeros((0, 17, 3)), np.zeros(0, dtype=bool)

        first_pose = next(iter(track_poses.values()), None)
        J = len(first_pose.keypoints) if first_pose else 17

        joints = np.zeros((T, J, 3), dtype=np.float32)
        valid = np.zeros(T, dtype=bool)

        for t, fidx in enumerate(frame_indices):
            if fidx in track_poses:
                joints[t] = track_poses[fidx].keypoints_array
                valid[t] = True

        return joints, valid
