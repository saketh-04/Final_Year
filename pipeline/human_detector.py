"""
HumanMM — Human Detector Pipeline Stage.

Thin orchestration wrapper around :class:`~models.yolo_detector.YOLODetector`.

Responsibilities
----------------
* Drive the detector frame-by-frame and accumulate results.
* Separate person detections from non-person detections.
* Forward only person detections (``class_id == 0``) to the downstream
  tracking pipeline via :meth:`get_results`.
* Provide non-person detections via :meth:`get_overlay_detections` so the
  visualization layer can draw them in a different colour.
* Track per-frame and rolling inference latency.
* Log per-frame stats at DEBUG level, periodic summaries at INFO.

Public API (unchanged)
----------------------
``HumanDetector.detect_frame()`` → ``List[Detection]``  (persons only)
``HumanDetector.run_on_frames()`` → ``Dict[int, List[Detection]]``  (persons only)
``HumanDetector.get_results()`` → ``Dict[int, List[Detection]]``
``HumanDetector.get_frame_detections()`` → ``List[Detection]``
``HumanDetector.get_latency()`` → ``float``
``HumanDetector.get_mean_latency_ms()`` → ``float``
``HumanDetector.total_detections()`` → ``int``

New helpers (non-breaking additions)
-------------------------------------
``HumanDetector.get_overlay_detections()`` → ``List[Detection]``  (ALL classes, for viz)
``HumanDetector.get_all_frame_detections()`` → ``List[Detection]``  (ALL classes)
"""

from __future__ import annotations

import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from models.base_model import BaseModel
from models.yolo_detector import Detection
from utils.frame_utils import Frame
from utils.logger import get_logger

log = get_logger(__name__)


class HumanDetector:
    """Pipeline stage that runs YOLO human detection across all video frames.

    Wraps :class:`~models.yolo_detector.YOLODetector` and adds:

    * Frame-by-frame result accumulation (persons only) for downstream use.
    * Separate non-person detection storage for the visualization overlay.
    * Per-frame latency tracking.
    * Periodic INFO-level progress logging.

    Args:
        model: Initialised :class:`~models.yolo_detector.YOLODetector`
            (or any compatible ``BaseModel`` subclass).
        max_persons: Maximum person detections to keep per frame.

    Example:
        >>> hd = HumanDetector(model=yolo_model, max_persons=10)
        >>> results = hd.run_on_frames(frame_list)
        >>> persons_at_42 = hd.get_frame_detections(42)
        >>> all_viz = hd.get_all_frame_detections(42)   # includes non-persons
    """

    def __init__(
        self,
        model: BaseModel,
        max_persons: int = 10,
    ) -> None:
        self._model = model
        self._max_persons = max_persons

        # Persons only — forwarded to tracking pipeline
        self._results:    Dict[int, List[Detection]] = {}
        # ALL classes — used by visualization overlay
        self._all_dets:   Dict[int, List[Detection]] = {}
        # Latency per frame
        self._latencies:  Dict[int, float] = {}

    # ------------------------------------------------------------------
    # Single-frame inference
    # ------------------------------------------------------------------

    def detect_frame(
        self,
        frame: Frame,
        frame_idx: int,
    ) -> List[Detection]:
        """Run detection on one frame and return person detections only.

        Non-person detections are stored internally for overlay use but
        are NOT returned here (to keep the downstream API unchanged).

        Args:
            frame: BGR image array ``(H, W, 3)``.
            frame_idx: Zero-based frame index.

        Returns:
            List of person-only :class:`Detection` objects (capped at
            ``max_persons``), sorted by descending confidence.
        """
        try:
            t0 = time.perf_counter()
            all_dets: List[Detection] = self._model.run(
                frame, frame_idx=frame_idx
            )
            elapsed_ms = (time.perf_counter() - t0) * 1000.0

            # Use the model's own latency counter when available
            if hasattr(self._model, "get_last_latency_ms"):
                elapsed_ms = self._model.get_last_latency_ms()

            # Split persons / non-persons
            persons   = [d for d in all_dets if d.is_person][: self._max_persons]
            non_persons = [d for d in all_dets if not d.is_person]

            self._results[frame_idx]  = persons
            self._all_dets[frame_idx] = all_dets      # full set for overlay
            self._latencies[frame_idx] = elapsed_ms

            return persons

        except Exception as exc:
            log.warning("HumanDetector: detection failed at frame {}: {}", frame_idx, exc)
            self._results[frame_idx]   = []
            self._all_dets[frame_idx]  = []
            self._latencies[frame_idx] = 0.0
            return []

    # ------------------------------------------------------------------
    # Batch inference
    # ------------------------------------------------------------------

    def run_on_frames(
        self,
        frames: List[Tuple[Frame, int]],
    ) -> Dict[int, List[Detection]]:
        """Run detection across every frame in ``frames``.

        Args:
            frames: Ordered list of ``(BGR frame, frame_idx)`` tuples.

        Returns:
            Dictionary mapping ``frame_idx → List[Detection]`` (persons only).
        """
        total = len(frames)
        log_every = max(1, total // 10)

        for step, (frame, frame_idx) in enumerate(frames):
            persons = self.detect_frame(frame, frame_idx)

            if step % log_every == 0 or step == total - 1:
                lat  = self._latencies.get(frame_idx, 0.0)
                mean_conf = (
                    sum(d.confidence for d in persons) / len(persons)
                    if persons else 0.0
                )
                log.debug(
                    "HumanDetector [{:>4}/{:>4}] "
                    "frame={} | persons={} | conf={:.3f} | lat={:.1f}ms",
                    step + 1, total, frame_idx, len(persons), mean_conf, lat,
                )

        total_dets = sum(len(v) for v in self._results.values())
        mean_pf    = total_dets / max(total, 1)
        mean_lat   = self.get_mean_latency_ms()

        log.info(
            "Detection complete: {} frames | {:.2f} persons/frame | "
            "mean_lat={:.1f}ms | model={}",
            total, mean_pf, mean_lat,
            getattr(self._model, "get_resolved_model_name", lambda: "?")(),
        )

        return self._results

    # ------------------------------------------------------------------
    # Result accessors — unchanged public API
    # ------------------------------------------------------------------

    def get_results(self) -> Dict[int, List[Detection]]:
        """Return all accumulated person-only detection results.

        Returns:
            Dict mapping ``frame_idx → List[Detection]`` (persons only).
        """
        return self._results

    def get_frame_detections(self, frame_idx: int) -> List[Detection]:
        """Return person detections for a specific frame.

        Args:
            frame_idx: Frame index to query.

        Returns:
            List of person :class:`Detection` objects, or ``[]`` if the
            frame was not processed.
        """
        return self._results.get(frame_idx, [])

    def get_latency(self, frame_idx: int) -> float:
        """Return inference latency in ms for a specific frame.

        Args:
            frame_idx: Frame index to query.

        Returns:
            Latency in milliseconds, or ``0.0`` if not processed.
        """
        return self._latencies.get(frame_idx, 0.0)

    def get_mean_latency_ms(self) -> float:
        """Return mean inference latency across all processed frames.

        Returns:
            Mean latency in milliseconds.
        """
        if not self._latencies:
            return 0.0
        return float(sum(self._latencies.values()) / len(self._latencies))

    def total_detections(self) -> int:
        """Return the total number of person detections across all frames.

        Returns:
            Cumulative detection count.
        """
        return sum(len(v) for v in self._results.values())

    # ------------------------------------------------------------------
    # New helpers for multi-class overlay (non-breaking additions)
    # ------------------------------------------------------------------

    def get_all_frame_detections(self, frame_idx: int) -> List[Detection]:
        """Return ALL class detections for a frame (persons + non-persons).

        Used by the detection overlay renderer to draw non-person objects
        (vehicles, animals …) in a different colour.  Not forwarded to
        tracking/pose.

        Args:
            frame_idx: Frame index to query.

        Returns:
            Full detection list (all COCO classes), or ``[]`` if not processed.
        """
        return self._all_dets.get(frame_idx, [])

    def get_overlay_detections(self) -> Dict[int, List[Detection]]:
        """Return all-class detections for every processed frame.

        Returns:
            Dict mapping ``frame_idx → List[Detection]`` (all classes).
        """
        return self._all_dets
