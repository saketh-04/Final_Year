"""
HumanMM — Person Tracker Pipeline Stage.

Thin orchestration wrapper around :class:`~models.bytetrack_tracker.ByteTrackTracker`.

Responsibilities
----------------
* Drive the tracker frame-by-frame, accumulate results.
* Pass the raw BGR frame so appearance matching works.
* Log per-frame and summary tracking statistics.
* Expose tracking stats (ID switches, FPS, latency) for evaluation.

Public API (unchanged)
----------------------
``PersonTracker.track_frame()``        → ``List[Track]``
``PersonTracker.run_on_frames()``      → ``Dict[int, List[Track]]``
``PersonTracker.get_results()``        → ``Dict[int, List[Track]]``
``PersonTracker.get_frame_tracks()``   → ``List[Track]``
``PersonTracker.total_unique_persons()`` → ``int``

Non-breaking additions
----------------------
``PersonTracker.get_tracking_summary()`` → ``Dict[str, Any]``
``PersonTracker.get_latency()``          → ``float``
``PersonTracker.get_mean_latency_ms()``  → ``float``
"""

from __future__ import annotations

import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from models.base_model import BaseModel
from models.bytetrack_tracker import Track, TrackState
from models.yolo_detector import Detection
from utils.frame_utils import Frame
from utils.logger import get_logger

log = get_logger(__name__)


class PersonTracker:
    """Pipeline stage that runs multi-person tracking across all video frames.

    Wraps :class:`~models.bytetrack_tracker.ByteTrackTracker` and adds:

    * Frame-by-frame result accumulation.
    * BGR frame pass-through for appearance-based matching.
    * Per-frame latency tracking.
    * Summary logging on completion.

    Args:
        model: Initialised :class:`~models.bytetrack_tracker.ByteTrackTracker`
               (or any compatible ``BaseModel`` subclass).

    Example:
        >>> tracker = PersonTracker(model=bytetrack_model)
        >>> results = tracker.run_on_frames(detections_per_frame, frames)
        >>> tracks_at_42 = tracker.get_frame_tracks(42)
    """

    def __init__(self, model: BaseModel) -> None:
        self._model = model
        self._results:   Dict[int, List[Track]] = {}
        self._latencies: Dict[int, float]       = {}

    # ------------------------------------------------------------------
    # Single-frame tracking
    # ------------------------------------------------------------------

    def track_frame(
        self,
        detections: List[Detection],
        frame_idx:  int,
        frame:      Optional[Frame] = None,
    ) -> List[Track]:
        """Update the tracker with detections from a single frame.

        Args:
            detections: Person detections for this frame (persons only;
                        non-person detections are silently ignored inside
                        the tracker model).
            frame_idx:  Current frame index.
            frame:      Optional BGR image for appearance-based matching.

        Returns:
            List of confirmed/predicted :class:`Track` objects sorted by
            ``track_id`` ascending.
        """
        try:
            t0 = time.perf_counter()

            if frame is not None and hasattr(self._model, "run"):
                tracks: List[Track] = self._model.run(
                    detections, frame_idx=frame_idx, frame=frame
                )
            else:
                tracks = self._model.run(detections, frame_idx=frame_idx)

            elapsed_ms = (time.perf_counter() - t0) * 1000.0

            self._results[frame_idx]   = tracks
            self._latencies[frame_idx] = elapsed_ms
            return tracks

        except Exception as exc:
            log.warning(
                "PersonTracker: tracking failed at frame {}: {}", frame_idx, exc
            )
            self._results[frame_idx]   = []
            self._latencies[frame_idx] = 0.0
            return []

    # ------------------------------------------------------------------
    # Batch tracking
    # ------------------------------------------------------------------

    def run_on_frames(
        self,
        detections_per_frame: Dict[int, List[Detection]],
        frames:               List[Tuple[Frame, int]],
    ) -> Dict[int, List[Track]]:
        """Run tracking across all frames.

        Args:
            detections_per_frame: Dict ``frame_idx → List[Detection]``.
            frames: Ordered list of ``(BGR frame, frame_idx)`` tuples.

        Returns:
            Dict mapping ``frame_idx → List[Track]`` (confirmed + predicted).
        """
        frame_lookup: Dict[int, Frame] = {fidx: f for f, fidx in frames}
        unique_ids: set = set()
        log_every = max(1, len(frames) // 10)

        for step, (frame, frame_idx) in enumerate(frames):
            dets   = detections_per_frame.get(frame_idx, [])
            tracks = self.track_frame(dets, frame_idx,
                                      frame=frame_lookup.get(frame_idx))

            for t in tracks:
                unique_ids.add(t.track_id)

            if step % log_every == 0 or step == len(frames) - 1:
                confirmed = sum(1 for t in tracks if t.state == TrackState.Confirmed)
                predicted = sum(1 for t in tracks if t.state == TrackState.Predicted)
                lat = self._latencies.get(frame_idx, 0.0)
                log.debug(
                    "Tracker [{:>4}/{:>4}] frame={} | "
                    "confirmed={} predicted={} | lat={:.1f}ms",
                    step + 1, len(frames), frame_idx,
                    confirmed, predicted, lat,
                )

        # Pull summary from the underlying model if available
        tracker_summary = {}
        if hasattr(self._model, "summary"):
            tracker_summary = self._model.summary()

        log.info(
            "Tracking complete: {} frames | {} unique persons | "
            "id_switches={} | mean_lat={:.1f}ms",
            len(frames),
            len(unique_ids),
            tracker_summary.get("total_id_switches", "?"),
            self.get_mean_latency_ms(),
        )

        return self._results

    # ------------------------------------------------------------------
    # Result accessors  (unchanged public API)
    # ------------------------------------------------------------------

    def get_results(self) -> Dict[int, List[Track]]:
        """Return all accumulated tracking results.

        Returns:
            Dict mapping ``frame_idx → List[Track]``.
        """
        return self._results

    def get_frame_tracks(self, frame_idx: int) -> List[Track]:
        """Return tracks for a specific frame.

        Args:
            frame_idx: Frame index to query.

        Returns:
            List of :class:`Track` objects, or ``[]`` if not processed.
        """
        return self._results.get(frame_idx, [])

    def total_unique_persons(self) -> int:
        """Return the number of unique track IDs observed.

        Returns:
            Unique ID count.
        """
        ids: set = set()
        for tracks in self._results.values():
            for t in tracks:
                ids.add(t.track_id)
        return len(ids)

    # ------------------------------------------------------------------
    # Non-breaking additions
    # ------------------------------------------------------------------

    def get_latency(self, frame_idx: int) -> float:
        """Return tracking latency in ms for a specific frame.

        Args:
            frame_idx: Frame index to query.

        Returns:
            Latency in milliseconds, or ``0.0`` if not processed.
        """
        return self._latencies.get(frame_idx, 0.0)

    def get_mean_latency_ms(self) -> float:
        """Return mean tracking latency across all processed frames.

        Returns:
            Mean latency in milliseconds.
        """
        if not self._latencies:
            return 0.0
        return float(sum(self._latencies.values()) / len(self._latencies))

    def get_tracking_summary(self) -> Dict[str, Any]:
        """Return a combined tracking summary dict.

        Merges data from the underlying ByteTrack model summary (if
        available) with pipeline-level counts.

        Returns:
            Dict with keys including ``total_frames``, ``unique_persons``,
            ``total_id_switches``, ``mean_latency_ms``.
        """
        base = {}
        if hasattr(self._model, "summary"):
            base = self._model.summary()

        base.update({
            "pipeline_frames":  len(self._results),
            "unique_persons":   self.total_unique_persons(),
            "mean_latency_ms":  round(self.get_mean_latency_ms(), 2),
        })
        return base
