"""
HumanMM — YOLO11x / YOLOv8x Human Detector.

Research-quality person detection for the HumanMM pipeline.

Key capabilities
----------------
* **Model selection** – Tries YOLO11x first (strongest backbone), then
  YOLO11x-seg (for mask-based box refinement), then YOLOv8x equivalents.
  All fallbacks happen automatically and silently.
* **Tight bounding boxes** – When a segmentation variant is loaded, the
  predicted instance mask is used to compute the exact body silhouette
  bounding box, then padded by ``mask_padding_px`` pixels to guarantee
  hands and feet are never clipped.  A uniform-margin fallback is
  available for detection-only models.
* **Multi-class overlay** – All COCO classes that YOLO detects are
  returned in ``run()`` so the visualizer can draw non-person objects
  (vehicles, animals …) with a different colour.  Only ``class_id == 0``
  (person) detections are forwarded to the tracking pipeline.
* **Temporal EMA smoothing** – A per-detection-id exponential moving
  average damps jitter in box position and size between frames.
* **FP16 on CUDA** – Half-precision inference is enabled automatically
  when ``device == "cuda"``, disabled on CPU to avoid hardware errors.
* **Latency tracking** – Every ``run()`` call records inference time;
  callers can read ``get_last_latency_ms()`` for overlay rendering.

Public API (unchanged)
----------------------
``Detection``         dataclass (x1, y1, x2, y2, confidence, class_id, frame_idx)
``YOLODetector``      BaseModel subclass with ``load()`` / ``run()`` / ``release()``
"""

from __future__ import annotations

import time
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, DefaultDict, Dict, List, Optional, Tuple

import cv2
import numpy as np

from models.base_model import BaseModel
from utils.logger import get_logger

log = get_logger(__name__)

# COCO class index for "person"
_PERSON_CLASS_ID: int = 0

# COCO class names (index → label) for overlay rendering
_COCO_NAMES: Dict[int, str] = {
    0: "person",   1: "bicycle",  2: "car",      3: "motorcycle", 4: "airplane",
    5: "bus",      6: "train",    7: "truck",    8: "boat",       9: "traffic light",
    10: "fire hydrant", 11: "stop sign", 13: "bench", 14: "bird",
    15: "cat",    16: "dog",    17: "horse",   18: "sheep",  19: "cow",
    20: "elephant", 21: "bear", 22: "zebra",  23: "giraffe",
    24: "backpack", 25: "umbrella", 26: "handbag", 27: "tie",
    56: "chair", 57: "couch", 58: "potted plant", 59: "bed", 60: "dining table",
    62: "tv", 63: "laptop", 64: "mouse", 67: "cell phone",
    72: "refrigerator", 74: "clock", 75: "vase",
}


# ---------------------------------------------------------------------------
# Detection dataclass — public API, unchanged
# ---------------------------------------------------------------------------

@dataclass
class Detection:
    """A single YOLO detection result.

    Attributes:
        x1: Left edge of the bounding box in pixels.
        y1: Top edge of the bounding box in pixels.
        x2: Right edge of the bounding box in pixels.
        y2: Bottom edge of the bounding box in pixels.
        confidence: Detection confidence in ``[0, 1]``.
        class_id: COCO class index.  ``0`` = person.
        frame_idx: Source frame index.
    """

    x1: float
    y1: float
    x2: float
    y2: float
    confidence: float
    class_id: int = 0
    frame_idx: int = 0

    # ── Convenience properties (unchanged from original API) ──────────────

    @property
    def bbox(self) -> Tuple[float, float, float, float]:
        """Return box as ``(x1, y1, x2, y2)``."""
        return (self.x1, self.y1, self.x2, self.y2)

    @property
    def bbox_xyxy(self) -> np.ndarray:
        """Return box as ``np.ndarray([x1, y1, x2, y2])`` float32."""
        return np.array([self.x1, self.y1, self.x2, self.y2], dtype=np.float32)

    @property
    def bbox_xywh(self) -> np.ndarray:
        """Return box in centre format ``[cx, cy, w, h]`` float32."""
        cx = (self.x1 + self.x2) / 2.0
        cy = (self.y1 + self.y2) / 2.0
        return np.array([cx, cy, self.x2 - self.x1, self.y2 - self.y1],
                        dtype=np.float32)

    @property
    def area(self) -> float:
        """Bounding box area in pixels²."""
        return float((self.x2 - self.x1) * (self.y2 - self.y1))

    @property
    def class_name(self) -> str:
        """Human-readable COCO class name."""
        return _COCO_NAMES.get(self.class_id, f"cls_{self.class_id}")

    @property
    def is_person(self) -> bool:
        """True when this detection is a person (class 0)."""
        return self.class_id == _PERSON_CLASS_ID

    def to_dict(self) -> Dict[str, Any]:
        """Serialise to a plain dict for JSON export."""
        return {
            "x1": round(self.x1, 2), "y1": round(self.y1, 2),
            "x2": round(self.x2, 2), "y2": round(self.y2, 2),
            "confidence": round(self.confidence, 4),
            "class_id": self.class_id,
            "class_name": self.class_name,
            "frame_idx": self.frame_idx,
        }


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _tighten_with_mask(
    mask: np.ndarray,
    fallback: np.ndarray,
    padding: int,
    min_ratio: float,
) -> np.ndarray:
    """Derive a tight box from a binary instance mask.

    Finds the axis-aligned bounding box of nonzero mask pixels, adds
    ``padding`` on every side (body-safe margin so hands / feet are
    never clipped), then validates the result.

    Args:
        mask: Boolean or float mask of shape ``(H, W)`` in frame coords.
        fallback: Raw YOLO box ``[x1, y1, x2, y2]`` — returned on failure.
        padding: Pixels to add around the mask bounding box.
        min_ratio: Reject the refined box if it is less than this fraction
            of the fallback box's area.

    Returns:
        Refined ``[x1, y1, x2, y2]`` float32 array.
    """
    ys, xs = np.nonzero(mask)
    if ys.size == 0:
        return fallback

    fx1, fy1, fx2, fy2 = fallback
    rx1 = max(fx1, float(xs.min()) - padding)
    ry1 = max(fy1, float(ys.min()) - padding)
    rx2 = min(fx2, float(xs.max()) + padding)
    ry2 = min(fy2, float(ys.max()) + padding)

    if rx2 <= rx1 or ry2 <= ry1:
        return fallback

    refined_area  = (rx2 - rx1) * (ry2 - ry1)
    fallback_area = max((fx2 - fx1) * (fy2 - fy1), 1e-6)
    if refined_area / fallback_area < min_ratio:
        return fallback  # mask was fragmented / failed — keep raw box

    return np.array([rx1, ry1, rx2, ry2], dtype=np.float32)


def _tighten_with_margin(box: np.ndarray, ratio: float) -> np.ndarray:
    """Uniformly trim a fraction off each side of a detection box.

    Args:
        box: ``[x1, y1, x2, y2]`` box.
        ratio: Fraction of width/height to trim from each side.

    Returns:
        Trimmed ``[x1, y1, x2, y2]`` float32 array.
    """
    if ratio <= 0:
        return box
    x1, y1, x2, y2 = box
    dx, dy = (x2 - x1) * ratio, (y2 - y1) * ratio
    return np.array([x1 + dx, y1 + dy, x2 - dx, y2 - dy], dtype=np.float32)


# ---------------------------------------------------------------------------
# Per-detection EMA smoother
# ---------------------------------------------------------------------------

class _BoxSmoother:
    """Per-identity exponential moving-average box smoother.

    Tracks a smoothed box state keyed by ``(frame_idx, class_id, det_rank)``
    — essentially smoothing spatially-matched detections between frames.
    Uses a simple IoU-based nearest-neighbour match rather than a full
    tracker so it stays independent of ByteTrack.

    Args:
        pos_alpha: EMA weight for box centre updates.
        size_alpha: EMA weight for box width/height updates.
        max_disappeared: Frames a box state persists without a new match.
    """

    def __init__(
        self,
        pos_alpha: float = 0.55,
        size_alpha: float = 0.35,
        max_disappeared: int = 3,
    ) -> None:
        self._pos_alpha = pos_alpha
        self._size_alpha = size_alpha
        self._max_disappeared = max_disappeared
        # state: id -> [smooth_box(4), disappeared_count]
        self._states: Dict[int, List] = {}
        self._next_id: int = 0

    def update(self, boxes: np.ndarray) -> np.ndarray:
        """Apply EMA smoothing to a batch of boxes for one frame.

        Args:
            boxes: Array of shape ``(N, 4)`` with ``[x1, y1, x2, y2]`` rows.

        Returns:
            Smoothed array of shape ``(N, 4)``.
        """
        if boxes.shape[0] == 0:
            # Age all existing states
            for sid in list(self._states.keys()):
                self._states[sid][1] += 1
                if self._states[sid][1] > self._max_disappeared:
                    del self._states[sid]
            return boxes

        # Match incoming boxes to existing states by IoU
        state_ids = list(self._states.keys())
        smoothed = boxes.copy()

        if state_ids:
            state_boxes = np.array([self._states[sid][0] for sid in state_ids])
            iou = self._iou_matrix(boxes, state_boxes)  # (N, S)

            matched_new = set()
            matched_state = set()

            # Greedy match: largest IoU first
            flat_order = np.argsort(-iou.ravel())
            for idx in flat_order:
                ni, si = divmod(int(idx), len(state_ids))
                if ni in matched_new or si in matched_state:
                    continue
                if iou[ni, si] < 0.20:
                    break
                sid = state_ids[si]
                old = self._states[sid][0]
                new = boxes[ni]
                # Smooth centre and size separately
                old_cx, old_cy = (old[0]+old[2])/2, (old[1]+old[3])/2
                old_w,  old_h  = old[2]-old[0], old[3]-old[1]
                new_cx, new_cy = (new[0]+new[2])/2, (new[1]+new[3])/2
                new_w,  new_h  = new[2]-new[0], new[3]-new[1]

                s_cx = self._pos_alpha  * new_cx + (1-self._pos_alpha)  * old_cx
                s_cy = self._pos_alpha  * new_cy + (1-self._pos_alpha)  * old_cy
                s_w  = self._size_alpha * new_w  + (1-self._size_alpha) * old_w
                s_h  = self._size_alpha * new_h  + (1-self._size_alpha) * old_h

                s_box = np.array([s_cx-s_w/2, s_cy-s_h/2, s_cx+s_w/2, s_cy+s_h/2],
                                  dtype=np.float32)
                self._states[sid] = [s_box, 0]
                smoothed[ni] = s_box
                matched_new.add(ni)
                matched_state.add(si)

            # Age unmatched states
            for si, sid in enumerate(state_ids):
                if si not in matched_state:
                    self._states[sid][1] += 1
                    if self._states[sid][1] > self._max_disappeared:
                        del self._states[sid]

            # Create new states for unmatched incoming boxes
            for ni in range(len(boxes)):
                if ni not in matched_new:
                    self._states[self._next_id] = [boxes[ni].copy(), 0]
                    self._next_id += 1
        else:
            # No existing states — initialise from all incoming boxes
            for ni in range(len(boxes)):
                self._states[self._next_id] = [boxes[ni].copy(), 0]
                self._next_id += 1

        return smoothed

    @staticmethod
    def _iou_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
        """Compute pairwise IoU between two sets of boxes."""
        ax1, ay1, ax2, ay2 = a[:,0:1], a[:,1:2], a[:,2:3], a[:,3:4]
        bx1, by1, bx2, by2 = b[:,0], b[:,1], b[:,2], b[:,3]
        ix1 = np.maximum(ax1, bx1); iy1 = np.maximum(ay1, by1)
        ix2 = np.minimum(ax2, bx2); iy2 = np.minimum(ay2, by2)
        inter = np.maximum(0, ix2-ix1) * np.maximum(0, iy2-iy1)
        area_a = (ax2-ax1) * (ay2-ay1)
        area_b = (bx2-bx1) * (by2-by1)
        return inter / (area_a + area_b - inter + 1e-8)


# ---------------------------------------------------------------------------
# YOLODetector — public API
# ---------------------------------------------------------------------------

class YOLODetector(BaseModel):
    """YOLO11x / YOLOv8x person detector with tight-box refinement.

    Loads the strongest available YOLO checkpoint, applies optimised
    inference settings, refines bounding boxes to tightly fit the human
    body silhouette, and returns a list of :class:`Detection` objects for
    every frame.

    Non-person COCO objects are also returned (with a flag) so the
    visualization layer can draw them in a different colour without
    forwarding them to the tracking pipeline.

    Args:
        device: Compute device (``"cuda"`` or ``"cpu"``).
        config: Detector config dict (mirrors ``configs/detector.yaml``).

    Example:
        >>> det = YOLODetector(device="cuda", config=cfg.detector)
        >>> det.initialize()
        >>> dets = det.run(bgr_frame, frame_idx=0)
        >>> persons = [d for d in dets if d.is_person]
    """

    def __init__(
        self,
        device: str = "cpu",
        config: Optional[Dict[str, Any]] = None,
    ) -> None:
        super().__init__(name="YOLO11x/YOLOv8x", device=device, config=config)
        self._model = None
        self._resolved_name: str = ""
        self._is_seg_model: bool = False

        # ── Config ──────────────────────────────────────────────────────
        self._model_name: str      = self.config.get("model_name",      "yolo11x.pt")
        self._fallback_name: str   = self.config.get("fallback_model_name", "yolov8x.pt")
        self._use_seg: bool        = self.config.get("use_segmentation", True)
        self._seg_name: str        = self.config.get("segmentation_model_name", "yolo11x-seg.pt")
        self._seg_fallback: str    = self.config.get("segmentation_fallback_model_name", "yolov8x-seg.pt")

        self._conf: float          = self.config.get("confidence_threshold", 0.25)
        self._iou: float           = self.config.get("iou_threshold",        0.45)
        self._imgsz: int           = self.config.get("input_size",           640)
        self._max_det: int         = self.config.get("max_detections",       50)
        self._augment: bool        = self.config.get("augment",              False)

        # FP16 — only on CUDA; CPU half() causes errors on most systems
        _want_half: bool = self.config.get("half_precision", False)
        self._half: bool = _want_half and device == "cuda"

        self._min_area: float      = self.config.get("min_bbox_area",      400.0)
        self._min_ar: float        = self.config.get("min_aspect_ratio",   0.08)
        self._max_ar: float        = self.config.get("max_aspect_ratio",   8.0)

        # Non-person class overlay
        self._show_all_cls: bool   = self.config.get("show_all_classes",   True)
        self._other_conf: float    = self.config.get("non_person_conf_threshold", 0.40)

        # Box refinement
        ref = self.config.get("box_refinement", {})
        self._refine: bool         = ref.get("enabled",             True)
        self._refine_mode: str     = ref.get("mode",                "mask")
        self._refine_pad: int      = ref.get("mask_padding_px",     10)
        self._refine_margin: float = ref.get("margin_trim_ratio",   0.0)
        self._refine_min: float    = ref.get("min_refined_area_ratio", 0.25)

        # Temporal smoothing
        sm = self.config.get("temporal_smoothing", {})
        self._smooth_enabled: bool = sm.get("enabled",          True)
        self._smoother = _BoxSmoother(
            pos_alpha=sm.get("position_alpha", 0.55),
            size_alpha=sm.get("size_alpha",    0.35),
            max_disappeared=sm.get("max_disappeared", 3),
        )
        self._non_person_smoother = _BoxSmoother(
            pos_alpha=sm.get("position_alpha", 0.55),
            size_alpha=sm.get("size_alpha",    0.35),
            max_disappeared=sm.get("max_disappeared", 3),
        )

        # Latency tracking
        self._expose_latency: bool  = self.config.get("expose_latency_metrics", True)
        self._last_latency_ms: float = 0.0
        self._latency_history: List[float] = []

    # ------------------------------------------------------------------
    # BaseModel interface
    # ------------------------------------------------------------------

    def load(self) -> None:
        """Load YOLO weights — tries seg variant first, then detection-only.

        Candidate order:
        1. ``segmentation_model_name``     (e.g. yolo11x-seg.pt)
        2. ``segmentation_fallback_model_name`` (e.g. yolov8x-seg.pt)
        3. ``model_name``                  (e.g. yolo11x.pt)
        4. ``fallback_model_name``         (e.g. yolov8x.pt)

        Raises:
            ImportError: If ``ultralytics`` is not installed.
            RuntimeError: If all candidates fail to load.
        """
        try:
            from ultralytics import YOLO
        except ImportError as exc:
            raise ImportError(
                "ultralytics is required: pip install -U ultralytics"
            ) from exc

        # Build candidate list
        candidates: List[Tuple[str, bool]] = []  # (name, is_seg)
        if self._use_seg and self._refine_mode == "mask":
            candidates += [
                (self._seg_name,     True),
                (self._seg_fallback, True),
            ]
        candidates += [
            (self._model_name,   False),
            (self._fallback_name, False),
        ]

        last_err: Optional[Exception] = None
        for name, is_seg in candidates:
            try:
                log.info("YOLODetector: loading {} …", name)
                model = YOLO(name)
                # Quick sanity — trigger model build
                _ = model.model
                self._model = model
                self._resolved_name = name
                self._is_seg_model  = is_seg
                log.info(
                    "YOLODetector: loaded {} (seg={}) on {}",
                    name, is_seg, self.device,
                )
                break
            except Exception as exc:
                last_err = exc
                log.warning("YOLODetector: could not load '{}' — {}", name, exc)
                self._model = None

        if self._model is None:
            raise RuntimeError(
                f"All YOLO candidates failed: {[c[0] for c in candidates]}"
            ) from last_err

        # Warm-up to avoid first-frame spike
        dummy = np.zeros((self._imgsz, self._imgsz, 3), dtype=np.uint8)
        self._model(dummy, verbose=False)
        log.info("YOLODetector: warm-up complete — ready for inference")

    def run(
        self,
        frame: np.ndarray,
        frame_idx: int = 0,
        **kwargs: Any,
    ) -> List[Detection]:
        """Run inference on a single BGR frame and return all detections.

        Person detections (``class_id == 0``) are refined to tight body
        boxes and temporally smoothed.  Non-person detections are included
        when ``show_all_classes`` is ``True`` so the visualizer can render
        them; they are filtered out by :class:`~pipeline.human_detector.HumanDetector`
        before being forwarded to the tracking pipeline.

        Args:
            frame: BGR image array of shape ``(H, W, 3)``.
            frame_idx: Zero-based frame index (attached to each Detection).
            **kwargs: Forwarded to the YOLO ``predict`` call.

        Returns:
            List of :class:`Detection` objects sorted by descending
            confidence.  All person detections come first, then non-person
            (if ``show_all_classes`` is True).

        Raises:
            RuntimeError: If :meth:`initialize` has not been called.
        """
        self._require_loaded()

        h_frame, w_frame = frame.shape[:2]

        # ── YOLO inference ────────────────────────────────────────────────
        t0 = time.perf_counter()
        results = self._model.predict(
            source=frame,
            conf=self._conf,
            iou=self._iou,
            imgsz=self._imgsz,
            max_det=self._max_det,
            half=self._half,
            augment=self._augment,
            verbose=False,
            device=self.device,
            retina_masks=self._is_seg_model,
        )
        latency_ms = (time.perf_counter() - t0) * 1000.0
        self._last_latency_ms = latency_ms
        if self._expose_latency:
            self._latency_history.append(latency_ms)
            if len(self._latency_history) > 500:
                self._latency_history.pop(0)

        if not results or results[0].boxes is None:
            return []

        result = results[0]
        boxes  = result.boxes
        masks  = result.masks  # None for detection-only models

        # ── Parse & refine ────────────────────────────────────────────────
        person_dets: List[Detection] = []
        other_dets:  List[Detection] = []

        raw_person_boxes:   List[np.ndarray] = []
        raw_other_boxes:    List[np.ndarray] = []

        for i, box in enumerate(boxes):
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().tolist()
            conf = float(box.conf[0].cpu().numpy())
            cls  = int(box.cls[0].cpu().numpy())

            raw = np.array([x1, y1, x2, y2], dtype=np.float32)

            if cls == _PERSON_CLASS_ID:
                # ── Tight-box refinement (person only) ────────────────────
                refined = raw.copy()
                if self._refine:
                    if (
                        self._refine_mode == "mask"
                        and masks is not None
                        and i < len(masks.data)
                    ):
                        mask_np = masks.data[i].cpu().numpy()
                        if mask_np.shape[:2] != (h_frame, w_frame):
                            mask_np = cv2.resize(
                                mask_np, (w_frame, h_frame),
                                interpolation=cv2.INTER_NEAREST,
                            )
                        refined = _tighten_with_mask(
                            mask_np > 0.5, raw,
                            padding=self._refine_pad,
                            min_ratio=self._refine_min,
                        )
                    elif self._refine_mode == "margin" and self._refine_margin > 0:
                        refined = _tighten_with_margin(raw, self._refine_margin)

                # Geometric sanity
                rw = refined[2] - refined[0]
                rh = refined[3] - refined[1]
                if rw * rh < self._min_area:
                    continue
                ar = rh / (rw + 1e-6)
                if not (self._min_ar <= ar <= self._max_ar):
                    continue

                raw_person_boxes.append(refined)
                person_dets.append(
                    Detection(
                        x1=float(refined[0]), y1=float(refined[1]),
                        x2=float(refined[2]), y2=float(refined[3]),
                        confidence=conf, class_id=cls, frame_idx=frame_idx,
                    )
                )

            else:
                # Non-person: include for overlay if enabled and conf OK
                if self._show_all_cls and conf >= self._other_conf:
                    raw_other_boxes.append(raw)
                    other_dets.append(
                        Detection(
                            x1=float(raw[0]), y1=float(raw[1]),
                            x2=float(raw[2]), y2=float(raw[3]),
                            confidence=conf, class_id=cls, frame_idx=frame_idx,
                        )
                    )

        # ── EMA temporal smoothing (per class bucket) ─────────────────────
        if self._smooth_enabled and raw_person_boxes:
            arr = np.array(raw_person_boxes, dtype=np.float32)
            arr = self._smoother.update(arr)
            for i, det in enumerate(person_dets):
                det.x1, det.y1, det.x2, det.y2 = (
                    float(arr[i, 0]), float(arr[i, 1]),
                    float(arr[i, 2]), float(arr[i, 3]),
                )

        if self._smooth_enabled and raw_other_boxes:
            arr = np.array(raw_other_boxes, dtype=np.float32)
            arr = self._non_person_smoother.update(arr)
            for i, det in enumerate(other_dets):
                det.x1, det.y1, det.x2, det.y2 = (
                    float(arr[i, 0]), float(arr[i, 1]),
                    float(arr[i, 2]), float(arr[i, 3]),
                )

        # ── Sort + combine ────────────────────────────────────────────────
        person_dets.sort(key=lambda d: d.confidence, reverse=True)
        other_dets.sort(key=lambda d: d.confidence, reverse=True)

        all_dets = person_dets + other_dets

        # ── Logging ──────────────────────────────────────────────────────
        if person_dets:
            mean_conf = sum(d.confidence for d in person_dets) / len(person_dets)
            log.debug(
                "Frame {:>4} | {:>2} persons detected | "
                "avg_conf={:.3f} | latency={:.1f}ms",
                frame_idx, len(person_dets), mean_conf, latency_ms,
            )
        else:
            log.debug(
                "Frame {:>4} | 0 persons detected | latency={:.1f}ms",
                frame_idx, latency_ms,
            )

        return all_dets

    # ------------------------------------------------------------------
    # Latency accessors (used by HumanDetector and visualization)
    # ------------------------------------------------------------------

    def get_last_latency_ms(self) -> float:
        """Return the inference latency of the most recent ``run()`` call.

        Returns:
            Latency in milliseconds, or ``0.0`` if no inference has run.
        """
        return self._last_latency_ms

    def get_mean_latency_ms(self) -> float:
        """Return the rolling mean inference latency over recent calls.

        Returns:
            Mean latency in milliseconds over up to the last 500 calls.
        """
        if not self._latency_history:
            return 0.0
        return float(sum(self._latency_history) / len(self._latency_history))

    def get_resolved_model_name(self) -> str:
        """Return the checkpoint name that was actually loaded after fallback.

        Returns:
            E.g. ``"yolo11x-seg.pt"`` or ``"yolov8x.pt"``.
        """
        return self._resolved_name

    # ------------------------------------------------------------------
    # Cleanup
    # ------------------------------------------------------------------

    def release(self) -> None:
        """Free CUDA memory held by the YOLO model."""
        if self._model is not None:
            del self._model
            self._model = None
            try:
                import torch
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except ImportError:
                pass
        super().release()
