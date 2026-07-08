"""
HumanMM — DeepSORT Multi-Person Tracker.

Wraps a lightweight DeepSORT implementation for appearance-based
multi-person tracking.  When ``deep_sort_realtime`` is not installed,
falls back to the IoU-only mode (bypassing re-identification).

This wrapper provides the same ``Track``-returning interface as
:class:`~models.bytetrack_tracker.ByteTrackTracker` so that both trackers
are interchangeable via the Strategy Pattern in ``pipeline/tracker.py``.

Example:
    >>> from models.deepsort_tracker import DeepSORTTracker
    >>> tracker = DeepSORTTracker(device="cuda", config=cfg.tracker.deepsort)
    >>> tracker.initialize()
    >>> tracks = tracker.run(detections, frame_rgb=frame, frame_idx=5)
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np

from models.base_model import BaseModel
from models.yolo_detector import Detection
from models.bytetrack_tracker import Track, TrackState
from utils.logger import get_logger

log = get_logger(__name__)


class DeepSORTTracker(BaseModel):
    """DeepSORT tracker with appearance re-identification.

    Uses ``deep_sort_realtime`` (pip package) if available.  Gracefully
    degrades to IoU-only tracking if the package is absent.

    Args:
        device: Compute device (``"cuda"`` or ``"cpu"``).
        config: DeepSORT config dict (mirrors ``tracker.deepsort``).

    Example:
        >>> tracker = DeepSORTTracker(device="cuda")
        >>> tracker.initialize()
        >>> tracks = tracker.run(detections, frame_rgb=frame, frame_idx=0)
    """

    def __init__(
        self,
        device: str = "cpu",
        config: Optional[Dict[str, Any]] = None,
    ) -> None:
        super().__init__(name="DeepSORT", device=device, config=config)
        self._tracker = None
        self._use_deep_sort: bool = False

        # Config defaults
        self._max_dist: float = self.config.get("max_dist", 0.30)
        self._max_iou_distance: float = self.config.get("max_iou_distance", 0.70)
        self._max_age: int = self.config.get("max_age", 30)
        self._n_init: int = self.config.get("n_init", 3)
        self._nn_budget: int = self.config.get("nn_budget", 100)

        # Internal ID mapping: DeepSORT id → our Track object
        self._track_registry: Dict[int, Track] = {}
        self._frame_idx: int = 0

    def load(self) -> None:
        """Attempt to load DeepSORT.  Falls back to IoU-only on import failure."""
        try:
            from deep_sort_realtime.deepsort_tracker import DeepSort  # type: ignore

            self._tracker = DeepSort(
                max_iou_distance=self._max_iou_distance,
                max_age=self._max_age,
                n_init=self._n_init,
                nn_budget=self._nn_budget,
                embedder="mobilenet",
                half=True,
                embedder_gpu=(self.device == "cuda"),
            )
            self._use_deep_sort = True
            log.info("DeepSORT loaded with appearance embedding (mobilenet)")

        except ImportError:
            log.warning(
                "deep_sort_realtime not found — DeepSORT running in IoU-only mode. "
                "Install via: pip install deep-sort-realtime"
            )
            self._use_deep_sort = False

    def run(
        self,
        detections: List[Detection],
        frame_rgb: Optional[np.ndarray] = None,
        frame_idx: int = 0,
        **kwargs: Any,
    ) -> List[Track]:
        """Update tracker with new detections and return active tracks.

        Args:
            detections: List of :class:`Detection` for the current frame.
            frame_rgb: Full RGB frame (required for appearance embedding).
            frame_idx: Current frame index.
            **kwargs: Ignored.

        Returns:
            List of confirmed :class:`Track` objects.

        Raises:
            RuntimeError: If ``initialize()`` has not been called.
        """
        self._require_loaded()
        self._frame_idx = frame_idx

        if self._use_deep_sort and frame_rgb is not None and self._tracker is not None:
            return self._run_deepsort(detections, frame_rgb, frame_idx)
        else:
            return self._run_iou_fallback(detections, frame_idx)

    def _run_deepsort(
        self,
        detections: List[Detection],
        frame_rgb: np.ndarray,
        frame_idx: int,
    ) -> List[Track]:
        """Full DeepSORT pass with re-identification embedding.

        Args:
            detections: Current frame detections.
            frame_rgb: RGB frame for embedding extraction.
            frame_idx: Frame index.

        Returns:
            List of confirmed tracks.
        """
        # Convert detections to deep_sort_realtime format: [[ltwh, conf, class_id], ...]
        ds_dets = []
        for det in detections:
            x1, y1, x2, y2 = det.x1, det.y1, det.x2, det.y2
            w, h = x2 - x1, y2 - y1
            ds_dets.append(([x1, y1, w, h], det.confidence, det.class_id))

        tracks_raw = self._tracker.update_tracks(ds_dets, frame=frame_rgb)

        result: List[Track] = []
        for t in tracks_raw:
            if not t.is_confirmed():
                continue

            tid = t.track_id
            ltrb = t.to_ltrb()
            bbox = np.array([ltrb[0], ltrb[1], ltrb[2], ltrb[3]], dtype=np.float64)

            if tid not in self._track_registry:
                self._track_registry[tid] = Track(
                    track_id=tid,
                    state=TrackState.Confirmed,
                    frame_idx=frame_idx,
                    bbox=bbox,
                )
            else:
                self._track_registry[tid].bbox = bbox
                self._track_registry[tid].frame_idx = frame_idx
                self._track_registry[tid].hits += 1
                self._track_registry[tid].frames_since_update = 0
                self._track_registry[tid].history.append((frame_idx, bbox.copy()))

            result.append(self._track_registry[tid])

        log.debug("Frame {}: {} DeepSORT tracks", frame_idx, len(result))
        return result

    def _run_iou_fallback(
        self, detections: List[Detection], frame_idx: int
    ) -> List[Track]:
        """Simple greedy IoU-based tracking (no appearance model).

        Matches detections to existing tracks using IoU only.  Used when
        ``deep_sort_realtime`` is not installed.

        Args:
            detections: Current detections.
            frame_idx: Frame index.

        Returns:
            List of active tracks.
        """
        from models.bytetrack_tracker import _iou_matrix, _linear_assignment  # noqa: F401

        active_tracks = [
            t for t in self._track_registry.values()
            if t.frames_since_update < self._max_age
        ]

        # Predict (age all tracks)
        for t in active_tracks:
            t.frames_since_update += 1

        if not detections or not active_tracks:
            # No match possible — init new tracks or return empty
            next_id = max((t.track_id for t in self._track_registry.values()), default=0) + 1
            for det in detections:
                new_t = Track(
                    track_id=next_id,
                    state=TrackState.Confirmed,
                    frame_idx=frame_idx,
                    bbox=det.bbox_xyxy.copy(),
                    score=det.confidence,
                )
                self._track_registry[next_id] = new_t
                next_id += 1
            return list(self._track_registry.values())

        track_boxes = np.array([t.bbox for t in active_tracks])
        det_boxes = np.array([d.bbox_xyxy for d in detections])
        iou = _iou_matrix(track_boxes, det_boxes)
        cost = 1.0 - iou
        row_ind, col_ind = _linear_assignment(cost)

        matched_t, matched_d = set(), set()
        for r, c in zip(row_ind, col_ind):
            if cost[r, c] <= (1.0 - 0.3):  # IoU threshold 0.3
                active_tracks[r].bbox = detections[c].bbox_xyxy.copy()
                active_tracks[r].frame_idx = frame_idx
                active_tracks[r].hits += 1
                active_tracks[r].frames_since_update = 0
                active_tracks[r].history.append((frame_idx, active_tracks[r].bbox.copy()))
                matched_t.add(r)
                matched_d.add(c)

        # New tracks for unmatched detections
        next_id = max((t.track_id for t in self._track_registry.values()), default=0) + 1
        for d_idx, det in enumerate(detections):
            if d_idx not in matched_d and det.confidence >= 0.5:
                new_t = Track(
                    track_id=next_id,
                    state=TrackState.Confirmed,
                    frame_idx=frame_idx,
                    bbox=det.bbox_xyxy.copy(),
                    score=det.confidence,
                )
                self._track_registry[next_id] = new_t
                next_id += 1

        # Remove stale
        stale_ids = [
            tid for tid, t in self._track_registry.items()
            if t.frames_since_update > self._max_age
        ]
        for tid in stale_ids:
            del self._track_registry[tid]

        return [t for t in self._track_registry.values() if t.frames_since_update == 0]

    def reset(self) -> None:
        """Reset tracker state."""
        self._track_registry = {}
        if self._tracker is not None:
            try:
                self._tracker = type(self._tracker)(
                    max_iou_distance=self._max_iou_distance,
                    max_age=self._max_age,
                    n_init=self._n_init,
                    nn_budget=self._nn_budget,
                )
            except Exception:
                pass
