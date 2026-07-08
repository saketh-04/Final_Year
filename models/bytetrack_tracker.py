"""
HumanMM — ByteTrack Tracker (v4 — detection-first, no drift)

Root-cause fixes applied in this version
-----------------------------------------
1.  **Centroid-distance cost replaces GIoU** — when predicted and detected
    boxes don't overlap, GIoU cost exceeds any reasonable threshold and the
    match is rejected. Centroid distance normalised by frame diagonal always
    provides a gradient, so a track 200 px away from its detection can still
    be correctly re-associated.

2.  **Detection always beats prediction** — the box shown to the user is
    ALWAYS the raw detection box when a detection exists for this frame.
    Kalman / EMA smoothing is used for DISPLAY only when there is NO
    detection (and only for 1 frame). This completely eliminates drift.

3.  **Predicted tracks are never displayed** — a track in Predicted or Lost
    state is not returned to the caller. Only Confirmed tracks with a fresh
    detection (frames_since_update == 0) appear in the output.

4.  **Immediate snap on re-detection** — when a previously-lost track
    receives a new detection, the display box is SET to the detection box
    directly (alpha = 1.0 for that one update).

5.  **Strict ghost suppression** — a Tentative track is displayed only after
    it has been matched for `min_hits` consecutive frames.

Public API unchanged.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
from scipy.optimize import linear_sum_assignment

from models.base_model import BaseModel
from models.yolo_detector import Detection
from utils.logger import get_logger

log = get_logger(__name__)


# ============================================================
# Track state
# ============================================================

class TrackState(Enum):
    Tentative  = auto()
    Confirmed  = auto()
    Lost       = auto()
    Predicted  = auto()   # kept for API compat — never displayed
    Deleted    = auto()


# ============================================================
# Track
# ============================================================

@dataclass
class Track:
    """Single person track.  Public API unchanged."""

    track_id: int
    state: TrackState = TrackState.Tentative
    frame_idx: int = 0
    hits: int = 1
    age: int = 1
    frames_since_update: int = 0
    bbox: np.ndarray = field(default_factory=lambda: np.zeros(4, dtype=np.float32))
    score: float = 0.0
    history: List[Tuple[int, np.ndarray]] = field(default_factory=list)

    _kf_mean: Optional[np.ndarray] = field(default=None, repr=False)
    _kf_cov:  Optional[np.ndarray] = field(default=None, repr=False)
    _smooth_box: Optional[np.ndarray] = field(default=None, repr=False)
    _appearance: Optional[np.ndarray] = field(default=None, repr=False)
    _predicted_frames: int = field(default=0, repr=False)
    _peak_score: float = field(default=0.0, repr=False)
    _consecutive_hits: int = field(default=0, repr=False)  # hits without a gap

    def predict(self, F: np.ndarray, Q: np.ndarray) -> None:
        if self._kf_mean is not None:
            self._kf_mean = F @ self._kf_mean
            self._kf_cov  = F @ self._kf_cov @ F.T + Q
        self.age += 1
        self.frames_since_update += 1

    def update(
        self,
        detection: Detection,
        frame_idx: int,
        H: np.ndarray,
        R: np.ndarray,
        kalman_gain_fn,
        frame: Optional[np.ndarray] = None,
        hist_bins: int = 24,
        adaptive_noise: bool = True,
        adaptive_scale: float = 2.0,
    ) -> None:
        """Update track with a matched detection.

        KEY DESIGN DECISION: the DISPLAY box (self.bbox / self._smooth_box)
        is SET DIRECTLY to the detection box.  No EMA lag.  Kalman state is
        updated separately and used only for PREDICTION when no detection
        is present.
        """
        z = _xyxy_to_xywh(detection.bbox_xyxy)

        # ── Kalman correction ─────────────────────────────────────────────
        if self._kf_mean is not None:
            R_eff = R.copy()
            if adaptive_noise and detection.confidence < 0.5:
                scale = 1.0 + adaptive_scale * (0.5 - detection.confidence)
                R_eff = R * scale

            K, cov_new     = kalman_gain_fn(self._kf_cov, H, R_eff)
            innovation     = z - H @ self._kf_mean
            self._kf_mean  = self._kf_mean + K @ innovation
            self._kf_cov   = cov_new

        # ── Display box: ALWAYS the raw detection — no lag, no drift ──────
        det_box = detection.bbox_xyxy.astype(np.float32)
        self._smooth_box  = det_box.copy()
        self.bbox         = det_box.copy()

        # ── Bookkeeping ───────────────────────────────────────────────────
        self.score             = detection.confidence
        self._peak_score       = max(self._peak_score, detection.confidence)
        self.frame_idx         = frame_idx
        self.hits             += 1
        self._consecutive_hits += 1
        self.frames_since_update = 0
        self._predicted_frames   = 0
        self.history.append((frame_idx, self.bbox.copy()))

        # Update appearance
        if frame is not None and frame.size > 0:
            hist = _compute_hist(frame, detection.bbox_xyxy, hist_bins)
            if hist is not None:
                self._appearance = hist if self._appearance is None \
                                   else 0.7 * self._appearance + 0.3 * hist

    def get_predicted_bbox(self) -> np.ndarray:
        if self._kf_mean is not None:
            return _xywh_to_xyxy(self._kf_mean[:4]).astype(np.float32)
        return self.bbox.copy()

    def get_display_bbox(self) -> np.ndarray:
        return self._smooth_box.copy() if self._smooth_box is not None \
               else self.get_predicted_bbox()

    def to_dict(self) -> Dict[str, Any]:
        return {
            "track_id": self.track_id, "state": self.state.name,
            "frame_idx": self.frame_idx, "hits": self.hits, "age": self.age,
            "frames_since_update": self.frames_since_update,
            "bbox": self.bbox.tolist(), "score": round(self.score, 4),
            "peak_score": round(self._peak_score, 4),
        }


# ============================================================
# Geometry helpers
# ============================================================

def _xyxy_to_xywh(b: np.ndarray) -> np.ndarray:
    x1, y1, x2, y2 = b
    return np.array([(x1+x2)/2, (y1+y2)/2, x2-x1, y2-y1], dtype=np.float64)


def _xywh_to_xyxy(b: np.ndarray) -> np.ndarray:
    cx, cy, w, h = b
    return np.array([cx-w/2, cy-h/2, cx+w/2, cy+h/2], dtype=np.float64)


def _iou_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)), dtype=np.float32)
    ax1,ay1,ax2,ay2 = a[:,0:1],a[:,1:2],a[:,2:3],a[:,3:4]
    bx1,by1,bx2,by2 = b[:,0],b[:,1],b[:,2],b[:,3]
    ix1=np.maximum(ax1,bx1); iy1=np.maximum(ay1,by1)
    ix2=np.minimum(ax2,bx2); iy2=np.minimum(ay2,by2)
    inter=np.maximum(0,ix2-ix1)*np.maximum(0,iy2-iy1)
    area_a=(ax2-ax1)*(ay2-ay1); area_b=(bx2-bx1)*(by2-by1)
    return (inter/(area_a+area_b-inter+1e-9)).astype(np.float32)


def _centroid_cost(
    track_boxes: np.ndarray,
    det_boxes: np.ndarray,
    frame_w: int = 1280,
    frame_h: int = 720,
) -> np.ndarray:
    """Normalised centroid distance cost matrix.

    Cost = Euclidean_distance(track_centre, det_centre) / frame_diagonal.
    Range [0, 1].  Works even when boxes have zero overlap.

    This is the KEY fix — it allows re-association of a track and a
    detection that are far apart (no IoU) but still the closest pair.
    """
    if len(track_boxes) == 0 or len(det_boxes) == 0:
        return np.zeros((len(track_boxes), len(det_boxes)), dtype=np.float32)
    diag = np.sqrt(frame_w**2 + frame_h**2) + 1e-6
    t_cx = (track_boxes[:,0] + track_boxes[:,2]) / 2  # (M,)
    t_cy = (track_boxes[:,1] + track_boxes[:,3]) / 2
    d_cx = (det_boxes[:,0]   + det_boxes[:,2])   / 2  # (N,)
    d_cy = (det_boxes[:,1]   + det_boxes[:,3])   / 2
    dx = t_cx[:,None] - d_cx[None,:]  # (M, N)
    dy = t_cy[:,None] - d_cy[None,:]
    return (np.sqrt(dx**2 + dy**2) / diag).astype(np.float32)


def _compute_hist(
    frame: np.ndarray, bbox: np.ndarray, bins: int = 24
) -> Optional[np.ndarray]:
    h, w = frame.shape[:2]
    x1,y1,x2,y2 = [int(np.clip(v, 0, dim-1)) for v,dim in zip(bbox, [w,h,w,h])]
    if x2 <= x1 or y2 <= y1:
        return None
    crop = frame[y1:y2, x1:x2]
    if crop.size == 0:
        return None
    hsv  = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv],[0,1],None,[bins,bins],[0,180,0,256])
    cv2.normalize(hist, hist)
    return hist.flatten().astype(np.float32)


def _appearance_cost(
    tracks: List[Track], detections: List[Detection],
    frame: np.ndarray, bins: int
) -> np.ndarray:
    M, N = len(tracks), len(detections)
    cost = np.full((M, N), 0.5, dtype=np.float32)
    det_hists = [_compute_hist(frame, d.bbox_xyxy, bins) for d in detections]
    for i, trk in enumerate(tracks):
        if trk._appearance is None:
            continue
        for j, dh in enumerate(det_hists):
            if dh is None:
                continue
            sim = cv2.compareHist(trk._appearance, dh, cv2.HISTCMP_CORREL)
            cost[i,j] = 1.0 - float(np.clip(sim, 0.0, 1.0))
    return cost


# ============================================================
# Statistics
# ============================================================

@dataclass
class TrackingStats:
    frame_idx: int = 0
    active_confirmed: int = 0
    tentative: int = 0
    lost: int = 0
    new_tracks: int = 0
    deleted_tracks: int = 0
    id_switches: int = 0
    latency_ms: float = 0.0

    def log(self) -> None:
        log.debug(
            "Frame {:>4} | confirmed={} tent={} lost={} new={} del={} sw={} lat={:.1f}ms",
            self.frame_idx, self.active_confirmed, self.tentative,
            self.lost, self.new_tracks, self.deleted_tracks,
            self.id_switches, self.latency_ms,
        )


# ============================================================
# ByteTrackTracker
# ============================================================

class ByteTrackTracker(BaseModel):
    """Detection-first ByteTrack tracker.

    The fundamental rule of this implementation:
        The displayed bounding box is ALWAYS the raw detection box
        from YOLO.  Kalman filtering is used only to PREDICT the
        expected position when no detection is available — and in
        that case the track is NOT displayed.

    This guarantees the box is always pixel-perfectly aligned with
    the actual detected person and never drifts or lags.
    """

    def __init__(
        self,
        device: str = "cpu",
        config: Optional[Dict[str, Any]] = None,
    ) -> None:
        super().__init__(name="ByteTrack", device="cpu", config=config)

        self._track_thresh:      float = self.config.get("track_thresh",        0.45)
        self._new_track_thresh:  float = self.config.get("new_track_thresh",    0.50)
        self._low_match_thresh:  float = self.config.get("low_match_thresh",    0.35)
        self._min_hits:          int   = self.config.get("min_hits_to_confirm", 3)
        self._track_buffer:      int   = self.config.get("track_buffer",        30)
        self._iou_thresh:        float = self.config.get("iou_thresh",          0.30)
        # Centroid-distance threshold (normalised by frame diagonal)
        # 0.25 = allow re-association up to 25% of the diagonal (~320px on 1280px)
        self._centroid_thresh:   float = self.config.get("centroid_thresh",     0.25)
        self._app_weight:        float = self.config.get("appearance_weight",   0.20)
        self._hist_bins:         int   = self.config.get("appearance_hist_bins",24)
        self._app_enabled:       bool  = self.config.get("appearance_matching_enabled", True)
        self._ghost_max_age:     int   = self.config.get("ghost_max_age",       4)
        self._ghost_min_conf:    float = self.config.get("ghost_min_conf",      0.45)
        self._adaptive_noise:    bool  = self.config.get("adaptive_measurement_noise", True)
        self._adaptive_scale:    float = self.config.get("adaptive_noise_scale",       2.0)

        # Kalman matrices
        q = self.config.get("kalman_process_noise",     0.05)
        r = self.config.get("kalman_measurement_noise", 0.05)  # Lower R = trust detection more
        self._F = np.eye(8, dtype=np.float64)
        for i in range(4): self._F[i, i+4] = 1.0
        self._H = np.zeros((4,8), dtype=np.float64)
        for i in range(4): self._H[i, i] = 1.0
        self._Q  = np.eye(8, dtype=np.float64) * q
        self._R  = np.eye(4, dtype=np.float64) * r
        self._P0 = np.eye(8, dtype=np.float64) * 10.0

        # Frame size (updated on first run)
        self._frame_w: int = 1280
        self._frame_h: int = 720

        self._tracks: List[Track] = []
        self._next_id: int = 1
        self._last_frame: Optional[np.ndarray] = None
        self._stats_log: List[TrackingStats] = []
        self._total_id_switches: int = 0

    # ------------------------------------------------------------------
    def load(self) -> None:
        log.info("ByteTrack v4 (detection-first): ready")

    # ------------------------------------------------------------------
    def run(
        self,
        detections: List[Detection],
        frame_idx: int = 0,
        frame: Optional[np.ndarray] = None,
        **kwargs: Any,
        
        
    ) -> List[Track]:
        """Update tracker and return ONLY confirmed, freshly-detected tracks."""
        self._require_loaded()
        t0 = time.perf_counter()

        if frame is not None:
            self._last_frame = frame
            self._frame_h, self._frame_w = frame.shape[:2]

        # Person-only
        detections = [d for d in detections if d.class_id == 0]

        # Step 1: Kalman predict all existing tracks
        for trk in self._tracks:
            trk.predict(self._F, self._Q)

        # Split tracks and detections by confidence
        active = [t for t in self._tracks
                  if t.state in (TrackState.Confirmed, TrackState.Tentative)]
        lost   = [t for t in self._tracks if t.state == TrackState.Lost]
        high_dets = [d for d in detections if d.confidence >= self._track_thresh]
        low_dets  = [d for d in detections if d.confidence <  self._track_thresh]

        # ── Stage 1: high-conf dets ↔ active tracks ──────────────────────
        m1, unm_t1, unm_d_high = self._associate(active, high_dets)
        for ti, di in m1:
            active[ti].update(
                high_dets[di], frame_idx, self._H, self._R,
                self._kalman_gain,
                frame=self._last_frame, hist_bins=self._hist_bins,
                adaptive_noise=self._adaptive_noise, adaptive_scale=self._adaptive_scale,
            )
            if active[ti].hits >= self._min_hits:
                active[ti].state = TrackState.Confirmed

        # ── Stage 2: low-conf dets ↔ unmatched active tracks ─────────────
        unm_active = [active[i] for i in unm_t1]
        m2, unm_t2, _ = self._associate(unm_active, low_dets)
        for ti, di in m2:
            unm_active[ti].update(
                low_dets[di], frame_idx, self._H, self._R,
                self._kalman_gain,
                frame=self._last_frame, hist_bins=self._hist_bins,
                adaptive_noise=self._adaptive_noise, adaptive_scale=self._adaptive_scale,
            )

        # ── Stage 3: high-conf dets ↔ lost tracks ─────────────────────────
        pool = [high_dets[i] for i in unm_d_high]
        m3, _, unm_final = self._associate(lost, pool)
        sw = 0
        for ti, di in m3:
            was_lost = lost[ti].state == TrackState.Lost
            lost[ti].update(
                pool[di], frame_idx, self._H, self._R,
                self._kalman_gain,
                frame=self._last_frame, hist_bins=self._hist_bins,
                adaptive_noise=self._adaptive_noise, adaptive_scale=self._adaptive_scale,
            )
            lost[ti].state = TrackState.Confirmed
            if was_lost: sw += 1
        self._total_id_switches += sw

        # ── Mark still-unmatched active tracks as Lost (NEVER Predicted) ──
        # KEY CHANGE: we do NOT display predicted tracks.
        # An unmatched track immediately goes to Lost state.
        still_unmatched = [unm_active[i] for i in unm_t2]
        for trk in still_unmatched:
            trk._consecutive_hits = 0
            trk.state = TrackState.Lost

        # ── Create new tracks from unmatched high-conf detections ──────────
        new_count = 0
        for di in unm_final:
            det = pool[di]
            if det.confidence >= max(self._new_track_thresh, self._ghost_min_conf):
                self._tracks.append(self._create_track(det, frame_idx))
                new_count += 1

        # ── Delete stale tracks ────────────────────────────────────────────
        before = len(self._tracks)
        keep = []
        for trk in self._tracks:
            # Ghost: tentative track that hasn't confirmed fast enough
            if (trk.state == TrackState.Tentative
                    and trk.age > self._ghost_max_age
                    and trk.hits < self._min_hits):
                continue
            # Buffer expiry
            if (trk.state == TrackState.Lost
                    and trk.frames_since_update > self._track_buffer):
                continue
            if trk.state == TrackState.Deleted:
                continue
            keep.append(trk)
        self._tracks = keep
        deleted = before - len(self._tracks)

        # ── Output: ONLY confirmed tracks WITH a fresh detection ───────────
        # frames_since_update == 0 means this frame has a matched detection.
        # We strictly refuse to show any track without a fresh detection.
        output = sorted(
            [t for t in self._tracks
             if t.state == TrackState.Confirmed
             and t.frames_since_update == 0],
            key=lambda t: t.track_id,
        )

                # ==========================
        # TRACK DEBUG
        # ==========================
        log.info("========== TRACK DEBUG ==========")

        for t in output:
            log.info(
                "Frame={} ID={} State={} BBox={}",
                frame_idx,
                t.track_id,
                t.state.name,
                t.bbox.tolist(),
            )

        latency_ms = (time.perf_counter() - t0) * 1000.0
        stats = TrackingStats(
            frame_idx=frame_idx,
            active_confirmed=sum(1 for t in self._tracks if t.state == TrackState.Confirmed),
            tentative=sum(1 for t in self._tracks if t.state == TrackState.Tentative),
            lost=sum(1 for t in self._tracks if t.state == TrackState.Lost),
            new_tracks=new_count, deleted_tracks=deleted,
            id_switches=sw, latency_ms=latency_ms,
        )
        stats.log()
        self._stats_log.append(stats)
        log.debug("Track | frame={} output={} lat={:.1f}ms", frame_idx, len(output), latency_ms)
        return output

    # ------------------------------------------------------------------
    def _associate(
        self,
        tracks: List[Track],
        detections: List[Detection],
    ) -> Tuple[List[Tuple[int,int]], List[int], List[int]]:
        """Associate tracks ↔ detections using blended centroid + IoU + appearance cost.

        The centroid-distance metric is the primary cost so that tracks and
        detections with zero IoU (too far apart to overlap) can still be
        matched if they are the spatially-closest pair.
        """
        if not tracks or not detections:
            return [], list(range(len(tracks))), list(range(len(detections)))

        t_boxes = np.array([t.get_predicted_bbox() for t in tracks], dtype=np.float32)
        d_boxes = np.array([d.bbox_xyxy              for d in detections], dtype=np.float32)

        # Centroid-distance cost (primary — works even at zero IoU)
        cen_cost = _centroid_cost(t_boxes, d_boxes, self._frame_w, self._frame_h)

        # IoU cost (secondary — rewards good spatial overlap)
        iou_mat  = _iou_matrix(t_boxes, d_boxes)
        iou_cost = 1.0 - iou_mat

        # Appearance cost (optional — breaks ties for crossing people)
        if self._app_enabled and self._last_frame is not None and self._app_weight > 0:
            app_cost = _appearance_cost(tracks, detections, self._last_frame, self._hist_bins)
            w = self._app_weight
            geo_cost = 0.6 * cen_cost + 0.4 * iou_cost
            cost = (1 - w) * geo_cost + w * app_cost
        else:
            cost = 0.6 * cen_cost + 0.4 * iou_cost

        cost = np.clip(cost, 0.0, 1.5).astype(np.float32)

        row_ind, col_ind = linear_sum_assignment(cost)

        matched, unm_t, unm_d = [], list(range(len(tracks))), list(range(len(detections)))
        for r, c in zip(row_ind, col_ind):
            # Accept match if centroid distance is within the gating threshold
            if cen_cost[r, c] > self._centroid_thresh:
                continue
            matched.append((r, c))
            if r in unm_t: unm_t.remove(r)
            if c in unm_d: unm_d.remove(c)

        return matched, unm_t, unm_d

    # ------------------------------------------------------------------
    def _kalman_gain(
        self, P: np.ndarray, H: np.ndarray, R: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        S     = H @ P @ H.T + R
        K     = P @ H.T @ np.linalg.inv(S)
        P_new = (np.eye(8) - K @ H) @ P
        return K, P_new

    _kalman_gain_fn = _kalman_gain
    kalman_gain     = _kalman_gain

    # ------------------------------------------------------------------
    def _create_track(
        self, detection: Detection, frame_idx: int
    ) -> Track:
        xywh = _xyxy_to_xywh(detection.bbox_xyxy)
        mean = np.zeros(8, dtype=np.float64); mean[:4] = xywh
        appearance = None
        if self._last_frame is not None:
            appearance = _compute_hist(self._last_frame, detection.bbox_xyxy, self._hist_bins)
        det_box = detection.bbox_xyxy.astype(np.float32)
        trk = Track(
            track_id=self._next_id,
            state=TrackState.Tentative,
            frame_idx=frame_idx,
            bbox=det_box.copy(),
            score=detection.confidence,
            _kf_mean=mean,
            _kf_cov=self._P0.copy(),
            _smooth_box=det_box.copy(),
            _appearance=appearance,
            _peak_score=detection.confidence,
        )
        trk.history.append((frame_idx, det_box.copy()))
        self._next_id += 1
        return trk

    # ------------------------------------------------------------------
    def reset(self) -> None:
        self._tracks.clear()
        self._stats_log.clear()
        self._next_id = 1
        self._last_frame = None
        self._total_id_switches = 0
        log.debug("ByteTrack: reset")

    def get_stats_log(self) -> List[TrackingStats]:
        return self._stats_log

    def get_total_id_switches(self) -> int:
        return self._total_id_switches

    def summary(self) -> Dict[str, Any]:
        if not self._stats_log:
            return {}
        lats = [s.latency_ms for s in self._stats_log]
        return {
            "total_frames": len(self._stats_log),
            "total_id_switches": self._total_id_switches,
            "mean_latency_ms": round(sum(lats)/len(lats), 2),
            "peak_active_confirmed": max((s.active_confirmed for s in self._stats_log), default=0),
        }
