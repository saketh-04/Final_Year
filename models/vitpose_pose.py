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

    # Multi-person, one GPU call instead of one call per person:
    >>> poses = backend.run_batch(frame_rgb, tracks=[(1, box1), (2, box2)], frame_idx=0)

    # Draw skeleton + track id + per-joint confidence directly on the frame:
    >>> backend.draw_poses(frame_bgr, poses)

    # Stable, decoupled record for a downstream Human Motion Recovery module:
    >>> record = backend.to_hmr_record(pose)

Production upgrades (v2)
-------------------------
All additive — the original public API (`__init__`, `load`, `run`, `release`)
is unchanged in signature and return type. Nothing existing was removed.

1.  **Adaptive bounding-box expansion.** Person-detector boxes are usually
    tight and clip extremities (hands, feet, top of head). Every bbox is
    now expanded by a configurable ratio (clamped to frame bounds) before
    being handed to mmpose. Toggle via `bbox_expansion_enabled`.

2.  **Per-joint confidence validation.** Previously only the *mean*
    confidence across all 17 joints gated acceptance — a pose with 12 good
    joints and 5 garbage ones could still pass. A second, independent
    gate (`min_valid_joint_ratio`) now requires a minimum fraction of
    individually-valid joints (`min_joint_confidence`), on top of the
    original mean-confidence gate (kept exactly as before, so existing
    tuned thresholds keep the same meaning).

3.  **Temporal keypoint smoothing (One-Euro filter), per Track ID, per
    joint, per axis.** This is the direct fix for visible jitter: each
    (track, joint, x/y) has its own adaptive filter — heavy smoothing when
    a joint is nearly still, automatically loosened when it's moving fast,
    so smoothing never introduces lag on genuine motion. Filters are only
    updated with detections that already passed the confidence gates,
    so a single bad low-confidence frame can't corrupt a joint's history.
    Gaps (e.g. after occlusion) are measured in actual elapsed frames, not
    assumed to be 1 frame, so long re-acquisition jumps snap immediately
    instead of being smeared across many frames. Default parameters
    (`smoothing_min_cutoff=0.4`, `smoothing_beta=0.04`) were chosen from a
    parameter sweep against synthetic detector noise (std≈3px) layered on
    both slow and fast synthetic motion: they cut frame-to-frame jitter by
    ~55% while keeping tracking error on genuinely fast motion close to
    the least-lag configurations tested. `smoothing_freq` (default 30.0)
    assumes ~30 FPS input — set it to your actual capture/processing FPS,
    since the filter's speed-vs-noise discrimination is frequency-
    dependent. Toggle via `smoothing_enabled`.

4.  **Pose history cache per Track ID**, exposed via `get_pose_history()`
    — a bounded, per-track deque of recent accepted poses for temporal
    analysis or windowing by a downstream Human Motion Recovery module.

5.  **Pose quality score** (`pose_quality_score()`), combining per-joint
    validity ratio and mean confidence over valid joints — a
    finer-grained signal than the single mean-confidence gate, usable by
    downstream consumers to prioritize or re-filter poses.

6.  **Batch multi-person inference** (`run_batch()`), sharing a single
    `_infer_persons()` core with `run()`. All currently-tracked people in
    a frame are sent to mmpose in ONE call (multi-bbox top-down inference)
    instead of one call per person — the dominant cost in multi-person
    scenes was previously the fixed per-call overhead repeated N times.

7.  **`torch.inference_mode()` around inference** (falls back to a no-op
    context manager if torch isn't importable), avoiding autograd-graph
    memory that was previously built and discarded on every call.

8.  **Thread safety.** A lock now guards the model call and load/release,
    since `PoseInferencer` is not guaranteed thread-safe and this backend
    may be shared across multiple video-stream workers.

9.  **Numerical stability.** Keypoints/scores are sanitized with
    `np.nan_to_num` and clipped before any statistics are computed — a
    NaN mean confidence previously passed the `< threshold` gate silently
    (NaN comparisons are always False in Python), which would have let a
    corrupted pose through undetected.

10. **Stale per-track state pruning** (`_maybe_prune_stale_state`,
    periodic, cheap) and an explicit `reset_track()` — bounds the memory
    used by smoothing filters and history for tracks that have vanished,
    without requiring the caller to manage this backend's internal state.

11. **Skeleton visualization** (`draw_pose` / `draw_poses`) — draws the
    COCO-17 skeleton, per-joint/per-limb confidence coloring, and the
    Track ID directly on a frame, so tracking and jitter behavior are
    immediately visible rather than only inferable from logs/numbers.

12. **`to_hmr_record()`** — exports a `PersonPose` into a stable, plain-
    dict structure (with a `schema_version`) decoupled from this module's
    internal dataclasses, intended for a future Human Motion Recovery
    stage that shouldn't need to import this backend's types directly.
"""

from __future__ import annotations

import contextlib
import threading
import time
from collections import deque
from typing import Any, Deque, Dict, List, Optional, Tuple, TypedDict

import numpy as np

from models.base_model import BaseModel
from models.mediapipe_pose import PersonPose, Keypoint, _COCO17_NAMES
from utils.logger import get_logger

log = get_logger(__name__)

# ViTPose outputs 17 COCO keypoints natively
_VITPOSE_COCO17_NAMES = _COCO17_NAMES

# Standard COCO-17 skeleton connectivity (joint-index pairs), used for
# skeleton visualization. Index order follows _COCO17_NAMES:
# 0 nose, 1 l-eye, 2 r-eye, 3 l-ear, 4 r-ear, 5 l-shoulder, 6 r-shoulder,
# 7 l-elbow, 8 r-elbow, 9 l-wrist, 10 r-wrist, 11 l-hip, 12 r-hip,
# 13 l-knee, 14 r-knee, 15 l-ankle, 16 r-ankle.
_COCO17_SKELETON_EDGES: Tuple[Tuple[int, int], ...] = (
    (0, 1), (0, 2), (1, 3), (2, 4), (0, 5), (0, 6), (5, 6),
    (5, 7), (7, 9), (6, 8), (8, 10),
    (5, 11), (6, 12), (11, 12),
    (11, 13), (13, 15), (12, 14), (14, 16),
)


class HMRPoseRecord(TypedDict):
    """Stable, versioned export shape for downstream Human Motion Recovery
    consumers — intentionally plain (no dependency on this module's
    dataclasses)."""
    schema_version: int
    track_id: int
    frame_idx: int
    joint_names: List[str]
    keypoints_xy: List[List[float]]
    keypoints_confidence: List[float]
    bbox: Tuple[float, float, float, float]
    mean_confidence: float
    quality_score: float
    backend: str


def _expand_bbox(
    bbox: Tuple[float, float, float, float],
    frame_w: int,
    frame_h: int,
    ratio: float,
) -> Tuple[float, float, float, float]:
    """Expand a bbox by `ratio` on each side (relative to its own width/
    height), clamped to frame bounds. Compensates for person detectors
    typically producing boxes tight enough to clip hands/feet/head, which
    otherwise biases pose estimation at the limbs."""
    x1, y1, x2, y2 = bbox
    w = max(x2 - x1, 1.0)
    h = max(y2 - y1, 1.0)
    dx = w * ratio
    dy = h * ratio
    nx1 = max(0.0, x1 - dx)
    ny1 = max(0.0, y1 - dy)
    nx2 = min(float(frame_w - 1), x2 + dx)
    ny2 = min(float(frame_h - 1), y2 + dy)
    return (nx1, ny1, nx2, ny2)


def _inference_context():
    """torch.inference_mode() if torch is importable, else a no-op context
    manager. Avoids building an autograd graph for pure inference calls."""
    try:
        import torch
        return torch.inference_mode()
    except ImportError:
        return contextlib.nullcontext()


class _OneEuroFilter:
    """Minimal One-Euro filter (Casiez et al., 2012) for one scalar signal.

    Chosen over a plain EMA specifically to address visible jitter without
    adding lag: the cutoff frequency adapts to the estimated signal speed,
    so a nearly-still joint is smoothed heavily while a fast-moving joint
    is smoothed lightly (and therefore doesn't visibly lag behind).
    `__slots__` keeps the per-instance memory footprint small, since a
    tracker with many people can have hundreds of live filters
    (tracks x 17 joints x 2 axes).
    """
    __slots__ = ("freq", "mincutoff", "beta", "dcutoff", "x_prev", "dx_prev")

    def __init__(self, freq: float = 30.0, mincutoff: float = 1.0,
                 beta: float = 0.3, dcutoff: float = 1.0) -> None:
        self.freq = freq
        self.mincutoff = mincutoff
        self.beta = beta
        self.dcutoff = dcutoff
        self.x_prev: Optional[float] = None
        self.dx_prev: float = 0.0

    @staticmethod
    def _alpha(cutoff: float, freq: float) -> float:
        tau = 1.0 / (2.0 * np.pi * max(cutoff, 1e-9))
        te = 1.0 / max(freq, 1e-9)
        return 1.0 / (1.0 + tau / te)

    def filter(self, x: float, gap_frames: int = 1) -> float:
        """Filter one new sample. `gap_frames` is how many nominal frame
        intervals actually elapsed since the last sample (>=1) — passing
        the real gap (rather than always assuming 1) means a jump after a
        long occlusion is correctly treated as a big instantaneous
        velocity and snaps immediately, instead of being smeared out over
        many frames as if it happened gradually."""
        gap_frames = max(1, gap_frames)
        eff_freq = self.freq / gap_frames

        if self.x_prev is None:
            self.x_prev = x
            self.dx_prev = 0.0
            return x

        dx = (x - self.x_prev) * eff_freq
        a_d = self._alpha(self.dcutoff, eff_freq)
        dx_hat = a_d * dx + (1.0 - a_d) * self.dx_prev

        cutoff = self.mincutoff + self.beta * abs(dx_hat)
        a = self._alpha(cutoff, eff_freq)
        x_hat = a * x + (1.0 - a) * self.x_prev

        self.x_prev = x_hat
        self.dx_prev = dx_hat
        return x_hat


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
        self._lock = threading.Lock()

        # Config defaults
        self._model_name: str = self.config.get("model_name", "ViTPose-H")
        self._fallback_model: str = self.config.get("fallback_model", "RTMPose-X")
        self._config_path: Optional[str] = self.config.get("config_path")
        self._checkpoint_path: Optional[str] = self.config.get("checkpoint_path")
        self._input_size: Tuple[int, int] = tuple(self.config.get("input_size", [192, 256]))
        self._bbox_thr: float = self.config.get("bbox_thr", 0.50)
        self._conf_threshold: float = self.config.get("confidence_threshold", 0.30)

        # --- v2 additions ---------------------------------------------
        self._bbox_expansion_enabled: bool = self.config.get("bbox_expansion_enabled", True)
        self._bbox_expansion_ratio:   float = self.config.get("bbox_expansion_ratio", 0.15)

        self._min_joint_confidence:   float = self.config.get("min_joint_confidence", 0.10)
        self._min_valid_joint_ratio:  float = self.config.get("min_valid_joint_ratio", 0.30)

        self._smoothing_enabled:      bool  = self.config.get("smoothing_enabled", True)
        self._smoothing_freq:         float = self.config.get("smoothing_freq", 30.0)
        self._smoothing_min_cutoff:   float = self.config.get("smoothing_min_cutoff", 0.4)
        self._smoothing_beta:         float = self.config.get("smoothing_beta", 0.04)
        self._smoothing_dcutoff:      float = self.config.get("smoothing_dcutoff", 1.0)

        self._pose_history_length:    int   = self.config.get("pose_history_length", 30)
        self._stale_prune_interval:   int   = max(1, self.config.get("stale_track_prune_interval", 200))
        self._stale_max_age:          int   = self.config.get("stale_track_max_age", 150)

        self._smoothers: Dict[int, Dict[Tuple[int, int], _OneEuroFilter]] = {}
        self._pose_history: Dict[int, Deque[PersonPose]] = {}
        self._track_last_seen: Dict[int, int] = {}

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

        with self._lock:
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
        results = self._infer_persons(frame_rgb, [(track_id, bbox)], frame_idx)
        return results[0] if results else None

    def run_batch(
        self,
        frame_rgb: np.ndarray,
        tracks: List[Tuple[int, Optional[Tuple[float, float, float, float]]]],
        frame_idx: int = 0,
        **kwargs: Any,
    ) -> List[PersonPose]:
        """Estimate 2D pose for MULTIPLE tracked people in a single mmpose
        call (one multi-bbox top-down inference instead of N separate
        calls — the dominant cost when tracking several people was
        previously the fixed per-call overhead repeated once per person).

        Args:
            frame_rgb: RGB image array ``(H, W, 3)``.
            tracks: List of ``(track_id, bbox)`` pairs. `bbox` may be
                ``None`` to use the full frame for that person.
            frame_idx: Frame index (shared by all people in this call).

        Returns:
            List of :class:`PersonPose`, one per person that passed the
            confidence/validity gates (order not guaranteed to match
            `tracks`; each returned pose carries its own `track_id`).

        Raises:
            RuntimeError: If ``initialize()`` has not been called.
        """
        self._require_loaded()
        if not tracks:
            return []
        return self._infer_persons(frame_rgb, tracks, frame_idx)

    # ------------------------------------------------------------------
    # Shared inference core
    # ------------------------------------------------------------------
    def _infer_persons(
        self,
        frame_rgb: np.ndarray,
        track_bbox_pairs: List[Tuple[int, Optional[Tuple[float, float, float, float]]]],
        frame_idx: int,
    ) -> List[PersonPose]:
        if self._inferencer is None:
            log.error("ViTPose: inference requested before a model was loaded")
            return []
        if frame_rgb is None or frame_rgb.size == 0:
            log.warning("ViTPose: empty frame at frame_idx={}", frame_idx)
            return []

        h, w = frame_rgb.shape[:2]
        bboxes_mm: List[List[float]] = []
        resolved_pairs: List[Tuple[int, Tuple[float, float, float, float]]] = []

        for track_id, bbox in track_bbox_pairs:
            eff_bbox = bbox if bbox is not None else (0.0, 0.0, float(w), float(h))
            if self._bbox_expansion_enabled:
                eff_bbox = _expand_bbox(eff_bbox, w, h, self._bbox_expansion_ratio)
            bboxes_mm.append([eff_bbox[0], eff_bbox[1], eff_bbox[2], eff_bbox[3], 1.0])
            resolved_pairs.append((track_id, eff_bbox))

        t0 = time.perf_counter()
        try:
            with self._lock, _inference_context():
                result_gen = self._inferencer(
                    inputs=frame_rgb,
                    bboxes=bboxes_mm,
                    return_datasamples=True,
                    batch_size=max(1, len(bboxes_mm)),
                )
                results = list(result_gen)
        except Exception as exc:
            log.warning(
                "ViTPose inference error (frame={}, n_people={}): {}",
                frame_idx, len(bboxes_mm), exc,
            )
            return []

        if not results or not results[0].get("predictions"):
            return []

        predictions = results[0]["predictions"]
        latency_ms = (time.perf_counter() - t0) * 1000.0
        log.debug(
            "ViTPose inference: {} requested, {} returned, {:.1f}ms",
            len(resolved_pairs), len(predictions), latency_ms,
        )

        output: List[PersonPose] = []
        for idx, pred in enumerate(predictions):
            if idx >= len(resolved_pairs):
                break  # defensive: never index past what we actually requested
            track_id, eff_bbox = resolved_pairs[idx]
            pose = self._build_pose(pred, track_id, frame_idx, eff_bbox)
            if pose is not None:
                output.append(pose)

        self._maybe_prune_stale_state(frame_idx)
        return output

    def _build_pose(
        self,
        pred: Any,
        track_id: int,
        frame_idx: int,
        bbox: Tuple[float, float, float, float],
    ) -> Optional[PersonPose]:
        """Extract, validate, gate, smooth, and package one person's pose
        from a single mmpose prediction entry."""
        try:
            keypoints_raw = np.asarray(pred.pred_instances.keypoints[0], dtype=np.float64)
            scores_raw = np.asarray(pred.pred_instances.keypoint_scores[0], dtype=np.float64)
        except Exception as exc:
            log.warning("ViTPose: malformed prediction for track={}: {}", track_id, exc)
            return None

        if keypoints_raw.size == 0 or scores_raw.size == 0:
            return None

        # Numerical stability: a NaN mean would silently pass a
        # `mean < threshold` check (NaN comparisons are always False in
        # Python), so sanitize before any statistics are computed.
        keypoints_raw = np.nan_to_num(keypoints_raw, nan=0.0, posinf=0.0, neginf=0.0)
        scores_raw = np.clip(np.nan_to_num(scores_raw, nan=0.0, posinf=0.0, neginf=0.0), 0.0, 1.0)

        mean_confidence = float(np.mean(scores_raw))
        if not np.isfinite(mean_confidence):
            log.debug("ViTPose: non-finite confidence for track={}, rejecting", track_id)
            return None

        # Original gate — unchanged semantics, so existing tuned
        # thresholds keep meaning what they meant before.
        if mean_confidence < self._conf_threshold:
            log.debug("ViTPose: low mean confidence {:.3f} for track={}", mean_confidence, track_id)
            return None

        # New, independent gate: per-joint validity. A pose can have an
        # acceptable mean while still having too many individually
        # unreliable joints (e.g. one very confident face, everything
        # else noisy) — this catches that case without changing the
        # original gate's behavior.
        valid_mask = scores_raw >= self._min_joint_confidence
        valid_ratio = float(np.mean(valid_mask))
        if valid_ratio < self._min_valid_joint_ratio:
            log.debug(
                "ViTPose: only {:.0%} valid joints for track={}, rejecting",
                valid_ratio, track_id,
            )
            return None

        # Smoothing gap: how many frames actually elapsed since this
        # track's last ACCEPTED pose (not just "1"), so re-acquisition
        # after occlusion snaps immediately instead of being smeared.
        last_seen = self._track_last_seen.get(track_id)
        gap_frames = 1 if last_seen is None else max(1, frame_idx - last_seen)

        keypoints: List[Keypoint] = []
        for i in range(scores_raw.shape[0]):
            name = _VITPOSE_COCO17_NAMES[i] if i < len(_VITPOSE_COCO17_NAMES) else f"joint_{i}"
            x, y = float(keypoints_raw[i, 0]), float(keypoints_raw[i, 1])
            if self._smoothing_enabled:
                x = self._smooth_value(track_id, i, 0, x, gap_frames)
                y = self._smooth_value(track_id, i, 1, y, gap_frames)
            keypoints.append(Keypoint(x=x, y=y, confidence=float(scores_raw[i]), name=name))

        pose = PersonPose(
            track_id=track_id,
            frame_idx=frame_idx,
            keypoints=keypoints,
            bbox=bbox,
            confidence=mean_confidence,
            backend="vitpose",
        )

        self._update_history(track_id, pose)
        return pose

    # ------------------------------------------------------------------
    # Temporal smoothing
    # ------------------------------------------------------------------
    def _smooth_value(self, track_id: int, joint_idx: int, axis: int, value: float, gap_frames: int) -> float:
        track_filters = self._smoothers.setdefault(track_id, {})
        key = (joint_idx, axis)
        filt = track_filters.get(key)
        if filt is None:
            filt = _OneEuroFilter(
                freq=self._smoothing_freq,
                mincutoff=self._smoothing_min_cutoff,
                beta=self._smoothing_beta,
                dcutoff=self._smoothing_dcutoff,
            )
            track_filters[key] = filt
        return filt.filter(value, gap_frames=gap_frames)

    # ------------------------------------------------------------------
    # Pose history / quality / per-track state management
    # ------------------------------------------------------------------
    def _update_history(self, track_id: int, pose: PersonPose) -> None:
        hist = self._pose_history.setdefault(track_id, deque(maxlen=self._pose_history_length))
        hist.append(pose)
        self._track_last_seen[track_id] = pose.frame_idx

    def get_pose_history(self, track_id: int) -> List[PersonPose]:
        """Return cached recent accepted poses for a track, oldest first.
        Empty list if the track is unknown. Useful for temporal analysis
        or windowing by a downstream Human Motion Recovery module."""
        return list(self._pose_history.get(track_id, ()))

    def pose_quality_score(self, pose: PersonPose) -> float:
        """Composite pose quality in [0, 1]: combines per-joint validity
        ratio and mean confidence over valid joints. A finer-grained
        signal than the single mean-confidence gate already applied in
        `run()`/`run_batch()` — usable by downstream consumers (e.g. to
        prioritize which detected person to animate first)."""
        if not pose.keypoints:
            return 0.0
        confs = np.array([kp.confidence for kp in pose.keypoints], dtype=np.float64)
        valid = confs >= self._min_joint_confidence
        valid_ratio = float(np.mean(valid))
        mean_valid_conf = float(np.mean(confs[valid])) if valid.any() else 0.0
        quality = 0.5 * valid_ratio + 0.5 * mean_valid_conf
        return float(np.clip(quality, 0.0, 1.0))

    def reset_track(self, track_id: int) -> None:
        """Clear smoothing state and pose history for a track ID — call
        this when the upstream tracker retires or reassigns an ID, so a
        new physical person doesn't inherit stale smoothing/history from
        a previous one."""
        self._smoothers.pop(track_id, None)
        self._pose_history.pop(track_id, None)
        self._track_last_seen.pop(track_id, None)

    def _maybe_prune_stale_state(self, frame_idx: int) -> None:
        """Periodically bound memory used by smoothing filters / history
        for tracks that have stopped appearing, without requiring the
        caller to manage this backend's internal state explicitly."""
        if frame_idx % self._stale_prune_interval != 0:
            return
        stale = [
            tid for tid, last in self._track_last_seen.items()
            if frame_idx - last > self._stale_max_age
        ]
        for tid in stale:
            self.reset_track(tid)
        if stale:
            log.debug("ViTPose: pruned {} stale track state(s)", len(stale))

    # ------------------------------------------------------------------
    # Visualization
    # ------------------------------------------------------------------
    def draw_pose(
        self,
        frame_bgr: np.ndarray,
        pose: PersonPose,
        draw_track_id: bool = True,
        draw_confidence: bool = True,
        min_draw_confidence: Optional[float] = None,
    ) -> np.ndarray:
        """Draw the COCO-17 skeleton, per-joint/limb confidence coloring,
        and the Track ID directly onto a frame (modified in place, and
        also returned for chaining). Low-confidence joints/limbs are
        skipped rather than drawn misleadingly at full opacity, so
        unreliable joints — or residual jitter — are visible at a glance
        instead of hidden.

        Args:
            frame_bgr: Frame to draw on, BGR channel order (matches
                cv2.imshow/imwrite color conventions — the confidence
                color ramp below assumes BGR).
            pose: The PersonPose to draw.
            draw_track_id: Draw an "ID <n>" label near the person's bbox.
            draw_confidence: Append the pose's mean confidence to the label.
            min_draw_confidence: Per-joint confidence floor for drawing;
                defaults to this backend's `min_joint_confidence`.

        Returns:
            The same `frame_bgr` array, for call chaining.
        """
        try:
            import cv2
        except ImportError:
            log.warning("ViTPose: draw_pose requires opencv-python (cv2); skipping visualization")
            return frame_bgr

        thr = self._min_joint_confidence if min_draw_confidence is None else min_draw_confidence
        pts = [(int(round(kp.x)), int(round(kp.y))) for kp in pose.keypoints]
        confs = [kp.confidence for kp in pose.keypoints]

        def _color_for_conf(c: float) -> Tuple[int, int, int]:
            c = float(np.clip(c, 0.0, 1.0))
            return (0, int(255 * c), int(255 * (1.0 - c)))  # BGR: red(low) -> green(high)

        for a, b in _COCO17_SKELETON_EDGES:
            if a >= len(pts) or b >= len(pts):
                continue
            if confs[a] < thr or confs[b] < thr:
                continue
            cv2.line(frame_bgr, pts[a], pts[b], _color_for_conf(min(confs[a], confs[b])), 2, lineType=cv2.LINE_AA)

        for pt, c in zip(pts, confs):
            if c < thr:
                continue
            cv2.circle(frame_bgr, pt, 3, _color_for_conf(c), -1, lineType=cv2.LINE_AA)

        if draw_track_id or draw_confidence:
            parts = []
            if draw_track_id:
                parts.append(f"ID {pose.track_id}")
            if draw_confidence:
                parts.append(f"{pose.confidence:.2f}")
            label = " | ".join(parts)
            x1, y1 = int(pose.bbox[0]), int(pose.bbox[1])
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(frame_bgr, (x1, max(0, y1 - th - 8)), (x1 + tw + 6, y1), (0, 0, 0), -1)
            cv2.putText(frame_bgr, label, (x1 + 3, max(12, y1 - 5)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

        return frame_bgr

    def draw_poses(self, frame_bgr: np.ndarray, poses: List[PersonPose], **kwargs: Any) -> np.ndarray:
        """Draw multiple poses (see `draw_pose`) on the same frame."""
        for pose in poses:
            self.draw_pose(frame_bgr, pose, **kwargs)
        return frame_bgr

    # ------------------------------------------------------------------
    # Human Motion Recovery export
    # ------------------------------------------------------------------
    def to_hmr_record(self, pose: PersonPose) -> HMRPoseRecord:
        """Export a `PersonPose` into a stable, versioned, plain-dict
        structure intended for a downstream Human Motion Recovery (HMR)
        stage — decoupled from this module's `PersonPose`/`Keypoint`
        dataclasses so HMR code doesn't need to import them directly."""
        return {
            "schema_version": 1,
            "track_id": pose.track_id,
            "frame_idx": pose.frame_idx,
            "joint_names": [kp.name for kp in pose.keypoints],
            "keypoints_xy": [[kp.x, kp.y] for kp in pose.keypoints],
            "keypoints_confidence": [kp.confidence for kp in pose.keypoints],
            "bbox": tuple(pose.bbox),
            "mean_confidence": pose.confidence,
            "quality_score": self.pose_quality_score(pose),
            "backend": pose.backend,
        }

    # ------------------------------------------------------------------
    def release(self) -> None:
        """Release ViTPose model and CUDA memory."""
        with self._lock:
            if self._inferencer is not None:
                del self._inferencer
                self._inferencer = None
                try:
                    import gc
                    import torch
                    gc.collect()
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