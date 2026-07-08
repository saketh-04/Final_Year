"""
HumanMM — Trajectory Aligner (Original Research Implementation).

This module is the **core intellectual contribution** of the HumanMM project.
It solves the fundamental problem of temporal discontinuity at shot boundaries
in cinematically edited videos by implementing a suite of trajectory
smoothing and cross-shot alignment algorithms.

All algorithms are implemented from scratch using NumPy, SciPy, and filterpy.
No code has been copied from any external repository.

Algorithms implemented:
    1. ``interpolate_missing_joints``   — Cubic spline interpolation for occluded frames
    2. ``moving_average_smooth``        — Sliding window joint trajectory smoothing
    3. ``gaussian_temporal_smooth``     — Gaussian-weighted temporal smoothing
    4. ``savitzky_golay_smooth``        — SG filter for translation trajectories
    5. ``smooth_rotations_slerp``       — SLERP-based rotation sequence smoothing
    6. ``smooth_translations``          — Combined SG + Gaussian smoothing
    7. ``align_cross_shot``             — Procrustes cross-shot rigid alignment
    8. ``blend_shot_boundary``          — Cosine-weighted boundary blending
    9. ``kalman_filter_trajectory``     — Optional Kalman filter post-processing
    10. ``align_all_shots``             — Full multi-shot alignment pipeline

Example:
    >>> from pipeline.trajectory_aligner import TrajectoryAligner
    >>> aligner = TrajectoryAligner(config=cfg.alignment)
    >>> aligned = aligner.align(smpl_results, shot_segments, track_ids)
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from scipy.interpolate import CubicSpline, interp1d
from scipy.ndimage import gaussian_filter1d
from scipy.signal import savgol_filter
from scipy.spatial.transform import Rotation, Slerp

from models.gvhmr_wrapper import SMPLOutput
from pipeline.shot_detector import ShotSegment
from utils.geometry import procrustes_align
from utils.logger import get_logger

log = get_logger(__name__)


# ---------------------------------------------------------------------------
# Output data structure
# ---------------------------------------------------------------------------

@dataclass
class AlignedTrackResult:
    """Aligned 3D trajectory for a single person track.

    Attributes:
        track_id: Person tracking identifier.
        frame_indices: Chronologically ordered list of frame indices.
        joints_3d_aligned: Aligned joint positions ``(T, J, 3)``.
        translations_aligned: Aligned root translations ``(T, 3)``.
        rotations_aligned: Aligned global orientations ``(T, 3)`` (axis-angle).
        shot_boundaries: Frame indices where shot cuts occurred.
        smoothing_applied: List of smoothing algorithm names applied.
    """

    track_id: int
    frame_indices: List[int]
    joints_3d_aligned: np.ndarray  # (T, J, 3)
    translations_aligned: np.ndarray  # (T, 3)
    rotations_aligned: np.ndarray  # (T, 3) axis-angle
    shot_boundaries: List[int] = field(default_factory=list)
    smoothing_applied: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Serialise metadata (not the large arrays) for JSON export."""
        return {
            "track_id": self.track_id,
            "num_frames": len(self.frame_indices),
            "frame_start": self.frame_indices[0] if self.frame_indices else 0,
            "frame_end": self.frame_indices[-1] if self.frame_indices else 0,
            "shot_boundaries": self.shot_boundaries,
            "smoothing_applied": self.smoothing_applied,
        }


# ---------------------------------------------------------------------------
# Trajectory Aligner — Main Class
# ---------------------------------------------------------------------------

class TrajectoryAligner:
    """Cross-shot trajectory smoother and aligner.

    Processes 3D motion recovery results for all tracks and all frames,
    applies a configurable smoothing pipeline, then aligns trajectories
    across shot boundaries using Procrustes-based rigid registration.

    Args:
        config: Alignment config dict (mirrors ``cfg.alignment``).

    Attributes:
        smoothing_log: List of operations applied (for reporting).

    Example:
        >>> aligner = TrajectoryAligner(config={"savgol_window": 11})
        >>> aligned_results = aligner.align(smpl_results, shots, track_ids=[1, 2])
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        self._cfg: Dict[str, Any] = config or {}

        # Interpolation
        self._interp_method: str = self._cfg.get("interpolation_method", "cubic")
        self._max_gap: int = self._cfg.get("max_gap_frames", 30)

        # Smoothing parameters
        self._ma_window: int = self._cfg.get("moving_average_window", 5)
        self._gauss_sigma: float = self._cfg.get("gaussian_sigma", 1.5)
        self._savgol_win: int = self._cfg.get("savgol_window", 11)
        self._savgol_poly: int = self._cfg.get("savgol_polyorder", 3)

        # Rotation smoothing
        self._rot_smooth_win: int = self._cfg.get("rotation_smooth_window", 7)

        # Cross-shot alignment
        self._overlap: int = self._cfg.get("shot_overlap_frames", 15)
        self._blend_type: str = self._cfg.get("blend_weight_type", "cosine")
        self._allow_scale: bool = self._cfg.get("procrustes_allow_scale", False)

        # Kalman filter
        self._use_kalman: bool = self._cfg.get("use_kalman", False)
        self._kf_process_noise: float = self._cfg.get("kalman_process_noise", 0.001)
        self._kf_meas_noise: float = self._cfg.get("kalman_measurement_noise", 0.1)

        self.smoothing_log: List[str] = []

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------

    def align(
        self,
        smpl_results: Dict[Tuple[int, int], SMPLOutput],
        shot_segments: List[ShotSegment],
        track_ids: List[int],
    ) -> Dict[int, AlignedTrackResult]:
        """Run the complete alignment pipeline for all tracks.

        Pipeline:
            1. Build raw joint/translation/rotation arrays per track
            2. Interpolate missing frames
            3. Apply joint smoothing (MA + Gaussian)
            4. Smooth translations (Savitzky-Golay + Gaussian)
            5. Smooth rotations (SLERP windowed)
            6. Align across shot boundaries (Procrustes)
            7. Optional Kalman filter post-processing

        Args:
            smpl_results: Dict ``(frame_idx, track_id) → SMPLOutput``.
            shot_segments: Detected shot segments.
            track_ids: List of track IDs to process.

        Returns:
            Dict mapping ``track_id → AlignedTrackResult``.
        """
        log.info("TrajectoryAligner: processing {} tracks, {} shots", len(track_ids), len(shot_segments))

        aligned_results: Dict[int, AlignedTrackResult] = {}

        # Collect all frame indices observed globally
        all_frame_indices = sorted({f for (f, _) in smpl_results.keys()})

        for track_id in track_ids:
            log.debug("Aligning track #{}", track_id)

            # --- Step 1: Extract raw time series for this track ---
            track_data = self._extract_track_data(smpl_results, track_id, all_frame_indices)
            if track_data is None:
                log.warning("Track #{} has no SMPL data — skipping", track_id)
                continue

            frame_indices, joints_3d, translations, rotations, valid_mask = track_data

            # --- Step 2: Interpolate missing frames ---
            joints_3d, translations, rotations = self.interpolate_missing_joints(
                joints_3d, translations, rotations, valid_mask, frame_indices
            )
            self.smoothing_log.append("interpolation")

            # --- Step 3: Smooth joints (moving average + Gaussian) ---
            joints_3d = self.moving_average_smooth(joints_3d, window=self._ma_window)
            joints_3d = self.gaussian_temporal_smooth(joints_3d, sigma=self._gauss_sigma)
            self.smoothing_log.extend(["moving_average", "gaussian_smooth"])

            # --- Step 4: Smooth translations ---
            translations = self.smooth_translations(
                translations,
                savgol_win=self._savgol_win,
                savgol_poly=self._savgol_poly,
                gauss_sigma=self._gauss_sigma,
            )
            self.smoothing_log.append("savgol+gauss_translation")

            # --- Step 5: Smooth rotations (SLERP) ---
            rotations = self.smooth_rotations_slerp(rotations, window=self._rot_smooth_win)
            self.smoothing_log.append("slerp_rotation_smooth")

            # --- Step 6: Cross-shot alignment ---
            if len(shot_segments) > 1:
                joints_3d, translations, rotations = self.align_all_shots(
                    joints_3d, translations, rotations,
                    frame_indices, shot_segments
                )
                self.smoothing_log.append("cross_shot_procrustes")

            # --- Step 7: Optional Kalman filter ---
            if self._use_kalman:
                translations = self.kalman_filter_trajectory(translations)
                self.smoothing_log.append("kalman_filter")

            # Collect shot boundaries within this track's frame range
            shot_boundaries = [
                s.start_frame for s in shot_segments
                if frame_indices[0] < s.start_frame < frame_indices[-1]
            ]

            aligned_results[track_id] = AlignedTrackResult(
                track_id=track_id,
                frame_indices=list(frame_indices),
                joints_3d_aligned=joints_3d,
                translations_aligned=translations,
                rotations_aligned=rotations,
                shot_boundaries=shot_boundaries,
                smoothing_applied=list(set(self.smoothing_log)),
            )

        log.info("TrajectoryAligner: completed {} tracks", len(aligned_results))
        return aligned_results

    # ------------------------------------------------------------------
    # Algorithm 1: Interpolation
    # ------------------------------------------------------------------

    def interpolate_missing_joints(
        self,
        joints_3d: np.ndarray,
        translations: np.ndarray,
        rotations: np.ndarray,
        valid_mask: np.ndarray,
        frame_indices: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Interpolate missing joint data across short temporal gaps.

        For frames where no person was detected (``valid_mask[t] == False``),
        interpolates joint positions, translations, and rotations using cubic
        spline (or linear) interpolation, provided the gap is shorter than
        ``max_gap_frames``.

        Args:
            joints_3d: Joint array ``(T, J, 3)``.
            translations: Translation array ``(T, 3)``.
            rotations: Rotation array ``(T, 3)`` (axis-angle).
            valid_mask: Boolean array ``(T,)`` — True where data exists.
            frame_indices: Frame index array ``(T,)``.

        Returns:
            Tuple of interpolated ``(joints_3d, translations, rotations)``.
        """
        T = len(valid_mask)
        valid_idx = np.where(valid_mask)[0]

        if len(valid_idx) < 2:
            log.debug("Insufficient valid frames for interpolation")
            return joints_3d, translations, rotations

        # Identify contiguous gaps below max_gap_frames
        invalid_idx = np.where(~valid_mask)[0]
        if len(invalid_idx) == 0:
            return joints_3d, translations, rotations

        t_valid = frame_indices[valid_idx].astype(float)
        t_all = frame_indices.astype(float)

        # --- Interpolate translations ---
        for dim in range(3):
            vals = translations[valid_idx, dim]
            if self._interp_method == "cubic" and len(valid_idx) >= 4:
                cs = CubicSpline(t_valid, vals, extrapolate=True)
                interp_vals = cs(t_all)
            else:
                f = interp1d(t_valid, vals, kind="linear", fill_value="extrapolate")
                interp_vals = f(t_all)

            # Only fill gaps within max_gap_frames
            for inv_i in invalid_idx:
                gap_ok = self._check_gap_size(inv_i, valid_idx)
                if gap_ok:
                    translations[inv_i, dim] = interp_vals[inv_i]

        # --- Interpolate joints (per-joint, per-dim) ---
        J = joints_3d.shape[1]
        for j in range(J):
            for dim in range(3):
                vals = joints_3d[valid_idx, j, dim]
                if self._interp_method == "cubic" and len(valid_idx) >= 4:
                    cs = CubicSpline(t_valid, vals, extrapolate=True)
                    interp_vals = cs(t_all)
                else:
                    f = interp1d(t_valid, vals, kind="linear", fill_value="extrapolate")
                    interp_vals = f(t_all)

                for inv_i in invalid_idx:
                    if self._check_gap_size(inv_i, valid_idx):
                        joints_3d[inv_i, j, dim] = interp_vals[inv_i]

        # --- Interpolate rotations (SLERP) ---
        rot_vals = rotations[valid_idx]  # (M, 3)
        rots_scipy = Rotation.from_rotvec(rot_vals)
        slerp = Slerp(t_valid, rots_scipy)

        for inv_i in invalid_idx:
            if self._check_gap_size(inv_i, valid_idx):
                t_q = float(t_all[inv_i])
                t_q_clamped = float(np.clip(t_q, t_valid[0], t_valid[-1]))
                rotations[inv_i] = slerp([t_q_clamped])[0].as_rotvec()

        return joints_3d, translations, rotations

    # ------------------------------------------------------------------
    # Algorithm 2: Moving Average Smoothing
    # ------------------------------------------------------------------

    def moving_average_smooth(
        self,
        joints_3d: np.ndarray,
        window: int = 5,
    ) -> np.ndarray:
        """Apply sliding-window moving average to joint trajectories.

        Each joint's x, y, z are smoothed independently using a uniform
        rectangular kernel of size ``window``.

        Args:
            joints_3d: Joint array ``(T, J, 3)``.
            window: Smoothing window size (must be odd for symmetric).

        Returns:
            Smoothed joint array ``(T, J, 3)``.
        """
        if window <= 1:
            return joints_3d

        T, J, D = joints_3d.shape
        smoothed = joints_3d.copy()
        kernel = np.ones(window) / window

        for j in range(J):
            for d in range(D):
                signal = joints_3d[:, j, d]
                # Use 'same' convolution with edge replication padding
                pad_w = window // 2
                padded = np.pad(signal, pad_w, mode="edge")
                conv_result = np.convolve(padded, kernel, mode="valid")
                smoothed[:, j, d] = conv_result[:T]

        return smoothed

    # ------------------------------------------------------------------
    # Algorithm 3: Gaussian Temporal Smoothing
    # ------------------------------------------------------------------

    def gaussian_temporal_smooth(
        self,
        joints_3d: np.ndarray,
        sigma: float = 1.5,
    ) -> np.ndarray:
        """Apply Gaussian temporal smoothing to joint trajectories.

        Uses a 1D Gaussian kernel along the time axis for each joint and
        spatial dimension independently.  Unlike moving average, this gives
        more weight to nearby frames.

        Args:
            joints_3d: Joint array ``(T, J, 3)``.
            sigma: Standard deviation of the Gaussian kernel in frames.

        Returns:
            Smoothed joint array ``(T, J, 3)``.
        """
        if sigma <= 0:
            return joints_3d

        T, J, D = joints_3d.shape
        smoothed = joints_3d.copy()

        for j in range(J):
            for d in range(D):
                smoothed[:, j, d] = gaussian_filter1d(joints_3d[:, j, d], sigma=sigma, mode="nearest")

        return smoothed

    # ------------------------------------------------------------------
    # Algorithm 4 + 5: Translation Smoothing (SG + Gaussian)
    # ------------------------------------------------------------------

    def smooth_translations(
        self,
        translations: np.ndarray,
        savgol_win: int = 11,
        savgol_poly: int = 3,
        gauss_sigma: float = 1.5,
    ) -> np.ndarray:
        """Smooth 3D root translations using Savitzky-Golay followed by Gaussian.

        Savitzky-Golay preserves polynomial trends (natural human motion) while
        removing high-frequency noise.  The subsequent Gaussian pass removes
        any residual jitter from the SG boundary effects.

        Args:
            translations: Translation array ``(T, 3)``.
            savgol_win: Savitzky-Golay window length (must be odd, > polyorder).
            savgol_poly: Polynomial order for SG filter.
            gauss_sigma: Gaussian sigma for secondary smoothing.

        Returns:
            Smoothed translation array ``(T, 3)``.
        """
        T = translations.shape[0]
        smoothed = translations.copy()

        # Ensure window is valid
        win = savgol_win
        if win > T:
            win = T if T % 2 == 1 else T - 1
        if win <= savgol_poly:
            win = savgol_poly + 2
            if win % 2 == 0:
                win += 1

        for d in range(3):
            signal = translations[:, d]
            try:
                sg_smooth = savgol_filter(signal, window_length=win, polyorder=savgol_poly)
                smoothed[:, d] = gaussian_filter1d(sg_smooth, sigma=gauss_sigma, mode="nearest")
            except Exception as exc:
                log.warning("SG filter failed (dim={}): {} — using Gaussian only", d, exc)
                smoothed[:, d] = gaussian_filter1d(signal, sigma=gauss_sigma, mode="nearest")

        return smoothed

    # ------------------------------------------------------------------
    # Algorithm 6: Rotation Smoothing (SLERP windowed)
    # ------------------------------------------------------------------

    def smooth_rotations_slerp(
        self,
        rotations: np.ndarray,
        window: int = 7,
    ) -> np.ndarray:
        """Smooth a rotation sequence using windowed SLERP averaging.

        For each time step t, averages the rotations in a window around t
        by iteratively averaging pairs via SLERP.  This avoids gimbal lock
        and discontinuities that Euler-angle smoothing would introduce.

        Args:
            rotations: Axis-angle rotation array ``(T, 3)``.
            window: Smoothing window size (half-width = ``window // 2``).

        Returns:
            Smoothed axis-angle rotation array ``(T, 3)``.
        """
        T = rotations.shape[0]
        if T < 2 or window <= 1:
            return rotations

        smoothed = rotations.copy()
        hw = window // 2

        for t in range(T):
            start = max(0, t - hw)
            end = min(T, t + hw + 1)
            window_rots = rotations[start:end]  # (W, 3)
            smoothed[t] = self._rotation_mean_slerp(window_rots)

        return smoothed

    # ------------------------------------------------------------------
    # Algorithm 7: Cross-Shot Alignment (Procrustes)
    # ------------------------------------------------------------------

    def align_cross_shot(
        self,
        joints_ref: np.ndarray,
        joints_tgt: np.ndarray,
        translations_ref: np.ndarray,
        translations_tgt: np.ndarray,
        rotations_tgt: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Align a target shot's 3D pose to a reference shot using Procrustes.

        Computes the optimal rigid transform (rotation + translation, no scale)
        that maps the **overlap window** of the target shot to the reference
        shot's tail.  Then applies this transform to ALL frames of the target.

        Args:
            joints_ref: Reference shot overlap joints ``(N, J, 3)``
                (last N frames of the reference shot).
            joints_tgt: Target shot overlap joints ``(N, J, 3)``
                (first N frames of the target shot).
            translations_ref: Reference overlap translations ``(N, 3)``.
            translations_tgt: Target shot ALL translations ``(T_tgt, 3)``.
            rotations_tgt: Target shot ALL rotations ``(T_tgt, 3)`` axis-angle.

        Returns:
            Tuple of aligned ``(joints_tgt_all, translations_tgt, rotations_tgt)``
            covering all frames of the target shot.
        """
        N = min(len(joints_ref), len(joints_tgt), self._overlap)
        if N < 3:
            log.warning("Insufficient overlap frames ({}) for Procrustes alignment", N)
            return joints_tgt, translations_tgt, rotations_tgt

        # Flatten joint clouds for Procrustes: (N*J, 3)
        J = joints_ref.shape[1]
        src = joints_tgt[:N].reshape(N * J, 3)
        tgt = joints_ref[:N].reshape(N * J, 3)

        try:
            R_align, t_align, scale, _ = procrustes_align(
                source=src, target=tgt, allow_scale=self._allow_scale
            )
        except Exception as exc:
            log.warning("Procrustes alignment failed: {}", exc)
            return joints_tgt, translations_tgt, rotations_tgt

        # Apply rigid transform to ALL frames of the target shot's joints
        T_tgt = joints_tgt.shape[0]
        joints_aligned = np.zeros_like(joints_tgt)
        for t in range(T_tgt):
            joints_aligned[t] = (scale * (joints_tgt[t] @ R_align.T)) + t_align

        # Apply to translations
        translations_aligned = (scale * (translations_tgt @ R_align.T)) + t_align

        # Apply rotation component to axis-angle rotations
        rot_matrix = R_align  # (3, 3)
        rots_scipy = Rotation.from_rotvec(rotations_tgt)
        align_rot = Rotation.from_matrix(rot_matrix)
        rotations_aligned = (align_rot * rots_scipy).as_rotvec()

        log.debug(
            "Procrustes alignment applied: scale={:.4f} |t|={:.4f}",
            scale,
            float(np.linalg.norm(t_align)),
        )

        return joints_aligned, translations_aligned, rotations_aligned

    # ------------------------------------------------------------------
    # Algorithm 8: Boundary Blending
    # ------------------------------------------------------------------

    def blend_shot_boundary(
        self,
        joints_prev: np.ndarray,
        joints_next: np.ndarray,
        n_blend: int,
        blend_type: str = "cosine",
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Blend the N frames around a shot boundary for seamless transition.

        Replaces the last ``n_blend // 2`` frames of the previous shot and
        the first ``n_blend // 2`` frames of the next shot with a weighted
        average that transitions smoothly from 100% previous to 100% next.

        Args:
            joints_prev: Previous shot joints ``(T_prev, J, 3)``.
            joints_next: Next shot joints ``(T_next, J, 3)``.
            n_blend: Number of frames to blend (split evenly across boundary).
            blend_type: Weight schedule — ``"cosine"`` (recommended) or
                ``"linear"``.

        Returns:
            Tuple of blended ``(joints_prev, joints_next)`` arrays.
        """
        half = n_blend // 2
        if half < 1:
            return joints_prev, joints_next

        half_prev = min(half, len(joints_prev))
        half_next = min(half, len(joints_next))
        actual = min(half_prev, half_next)

        if actual < 1:
            return joints_prev, joints_next

        # Build weights
        if blend_type == "cosine":
            t = np.linspace(0, 1, actual)
            alpha = 0.5 * (1 - np.cos(np.pi * t))  # 0 → 1 cosine curve
        else:
            alpha = np.linspace(0, 1, actual)

        # Blend: tail of prev + head of next
        for i in range(actual):
            w = float(alpha[i])
            prev_frame = joints_prev[len(joints_prev) - actual + i]
            next_frame = joints_next[i]
            blended = (1.0 - w) * prev_frame + w * next_frame
            joints_prev[len(joints_prev) - actual + i] = blended
            joints_next[i] = blended

        return joints_prev, joints_next

    # ------------------------------------------------------------------
    # Algorithm 9: Kalman Filter (optional)
    # ------------------------------------------------------------------

    def kalman_filter_trajectory(
        self,
        translations: np.ndarray,
    ) -> np.ndarray:
        """Apply a Kalman filter to 3D root translations.

        Models the trajectory as a constant-velocity linear dynamical system.
        Reduces measurement noise while preserving the overall motion dynamics.

        Args:
            translations: Root translation array ``(T, 3)``.

        Returns:
            Filtered translation array ``(T, 3)``.
        """
        try:
            from filterpy.kalman import KalmanFilter  # type: ignore
        except ImportError:
            log.warning("filterpy not installed — skipping Kalman filter. pip install filterpy")
            return translations

        T = translations.shape[0]
        if T < 3:
            return translations

        smoothed = translations.copy()

        # Separate Kalman filter for each spatial dimension
        for dim in range(3):
            kf = KalmanFilter(dim_x=2, dim_z=1)
            kf.F = np.array([[1.0, 1.0], [0.0, 1.0]])  # Constant velocity
            kf.H = np.array([[1.0, 0.0]])               # Observe position only
            kf.Q = np.eye(2) * self._kf_process_noise
            kf.R = np.array([[self._kf_meas_noise]])
            kf.P = np.eye(2) * 1.0
            kf.x = np.array([[translations[0, dim]], [0.0]])

            filtered = []
            for t in range(T):
                kf.predict()
                kf.update(np.array([[translations[t, dim]]]))
                filtered.append(kf.x[0, 0])

            smoothed[:, dim] = np.array(filtered)

        log.debug("Kalman filter applied to translations")
        return smoothed

    # ------------------------------------------------------------------
    # Algorithm 10: Full Multi-Shot Alignment
    # ------------------------------------------------------------------

    def align_all_shots(
        self,
        joints_3d: np.ndarray,
        translations: np.ndarray,
        rotations: np.ndarray,
        frame_indices: np.ndarray,
        shot_segments: List[ShotSegment],
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Align trajectories across all detected shot boundaries.

        Iterates over consecutive shot pairs and applies :meth:`align_cross_shot`
        followed by :meth:`blend_shot_boundary` at each cut.

        Args:
            joints_3d: Joint array ``(T, J, 3)``.
            translations: Translation array ``(T, 3)``.
            rotations: Rotation array ``(T, 3)`` axis-angle.
            frame_indices: Frame index array ``(T,)``.
            shot_segments: List of detected shot segments.

        Returns:
            Aligned ``(joints_3d, translations, rotations)`` tuple.
        """
        if len(shot_segments) < 2:
            return joints_3d, translations, rotations

        frame_idx_arr = np.array(frame_indices)
        joints_aligned = joints_3d.copy()
        trans_aligned = translations.copy()
        rots_aligned = rotations.copy()

        for i in range(len(shot_segments) - 1):
            shot_a = shot_segments[i]
            shot_b = shot_segments[i + 1]

            # Find time indices corresponding to each shot
            mask_a = (frame_idx_arr >= shot_a.start_frame) & (frame_idx_arr <= shot_a.end_frame)
            mask_b = (frame_idx_arr >= shot_b.start_frame) & (frame_idx_arr <= shot_b.end_frame)

            idx_a = np.where(mask_a)[0]
            idx_b = np.where(mask_b)[0]

            if len(idx_a) < self._overlap or len(idx_b) < self._overlap:
                log.debug(
                    "Shot {}/{}: insufficient frames for alignment (A={}, B={})",
                    i, i + 1, len(idx_a), len(idx_b)
                )
                continue

            # Reference: last `overlap` frames of shot A
            ref_joints = joints_aligned[idx_a[-self._overlap:]]
            ref_trans = trans_aligned[idx_a[-self._overlap:]]

            # Target: ALL frames of shot B (aligned globally)
            tgt_joints_overlap = joints_aligned[idx_b[:self._overlap]]
            tgt_trans_b = trans_aligned[idx_b]
            tgt_rots_b = rots_aligned[idx_b]

            # Run Procrustes alignment
            joints_b_aligned, trans_b_aligned, rots_b_aligned = self.align_cross_shot(
                joints_ref=ref_joints,
                joints_tgt=tgt_joints_overlap,
                translations_ref=ref_trans,
                translations_tgt=tgt_trans_b,
                rotations_tgt=tgt_rots_b,
            )

            # Write aligned shot B back
            joints_aligned[idx_b] = joints_b_aligned
            trans_aligned[idx_b] = trans_b_aligned
            rots_aligned[idx_b] = rots_b_aligned

            # Blend boundary
            n_blend = min(self._overlap, len(idx_a), len(idx_b))
            prev_joints = joints_aligned[idx_a].copy()
            next_joints = joints_aligned[idx_b].copy()
            prev_joints, next_joints = self.blend_shot_boundary(
                prev_joints, next_joints, n_blend=n_blend, blend_type=self._blend_type
            )
            joints_aligned[idx_a] = prev_joints
            joints_aligned[idx_b] = next_joints

            log.debug(
                "Cross-shot alignment applied: shots {} → {}",
                shot_a.shot_id,
                shot_b.shot_id,
            )

        return joints_aligned, trans_aligned, rots_aligned

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _extract_track_data(
        self,
        smpl_results: Dict[Tuple[int, int], SMPLOutput],
        track_id: int,
        all_frame_indices: List[int],
    ) -> Optional[Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]]:
        """Extract raw arrays for a single track from the SMPL result dict.

        Args:
            smpl_results: Full results dictionary.
            track_id: Track to extract.
            all_frame_indices: Sorted list of all frame indices.

        Returns:
            Tuple of ``(frame_indices, joints_3d, translations, rotations, valid_mask)``
            or ``None`` if no data exists for this track.
        """
        track_frames = sorted(
            [f for (f, tid) in smpl_results.keys() if tid == track_id]
        )
        if not track_frames:
            return None

        T = len(all_frame_indices)
        frame_idx_arr = np.array(all_frame_indices)

        # Sample joint shape from first available output
        first_out = smpl_results[(track_frames[0], track_id)]
        J = first_out.joints_3d.shape[0]

        joints_3d = np.zeros((T, J, 3), dtype=np.float32)
        translations = np.zeros((T, 3), dtype=np.float32)
        rotations = np.zeros((T, 3), dtype=np.float32)
        valid_mask = np.zeros(T, dtype=bool)

        frame_to_t = {f: t for t, f in enumerate(all_frame_indices)}

        for f in track_frames:
            if f not in frame_to_t:
                continue
            t = frame_to_t[f]
            out = smpl_results[(f, track_id)]
            joints_3d[t] = out.joints_3d
            translations[t] = out.transl
            rotations[t] = out.global_orient.flatten()[:3]
            valid_mask[t] = True

        if not valid_mask.any():
            return None

        return frame_idx_arr, joints_3d, translations, rotations, valid_mask

    def _check_gap_size(self, inv_i: int, valid_idx: np.ndarray) -> bool:
        """Check whether the gap around an invalid frame is within the max gap.

        Args:
            inv_i: Index of the invalid (missing) frame.
            valid_idx: Array of valid frame indices.

        Returns:
            ``True`` if the gap should be interpolated.
        """
        if len(valid_idx) == 0:
            return False
        # Find nearest valid neighbours
        lower = valid_idx[valid_idx < inv_i]
        upper = valid_idx[valid_idx > inv_i]
        if len(lower) == 0 or len(upper) == 0:
            return False
        gap = int(upper[0]) - int(lower[-1])
        return gap <= self._max_gap

    @staticmethod
    def _rotation_mean_slerp(rotations: np.ndarray) -> np.ndarray:
        """Compute the mean of a set of rotations via iterative SLERP averaging.

        Args:
            rotations: Axis-angle array ``(W, 3)``.

        Returns:
            Mean rotation as axis-angle ``(3,)``.
        """
        if len(rotations) == 1:
            return rotations[0]

        scipy_rots = Rotation.from_rotvec(rotations)
        # Use scipy's mean rotation (Chordal mean on SO(3))
        mean_rot = scipy_rots.mean()
        return mean_rot.as_rotvec()
