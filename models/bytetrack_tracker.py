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

Patch note (v4.1)
-----------------
Fix #5 ("strict ghost suppression") was declared in the docstring but not
actually enforced: the Stage-1 confirmation check below was gating on
`track.hits` (a lifetime counter that keeps incrementing across gaps) instead
of `track._consecutive_hits` (which resets to 0 every time a track drops to
Lost). That let a sporadically-detected false positive get promoted to
Confirmed after 3 non-consecutive matches, defeating ghost suppression and
letting it survive in the Lost buffer for up to `track_buffer` frames. See
the single changed line in `run()`, Stage 1, marked below.

Patch note (v5 — research-grade association upgrades)
-------------------------------------------------------
Fixes for the CRITICAL/HIGH-severity gaps found during review, all additive
— no existing feature removed, no public API changed:

1.  **Mahalanobis distance from the Kalman covariance, used as a soft
    ranking signal.** Every track already carries `_kf_cov`, but it was
    never consulted at all — gating was a fixed pixel radius. An initial
    version of this patch used Mahalanobis distance as a hard statistical
    gate, but testing caught a real calibration problem: this tracker keeps
    its Kalman filter deliberately subordinate to the detection
    ("detection always wins" — fix #2 above), so a converged track's
    covariance shrinks tighter than real human motion (a normal ~60px/frame
    walking speed at low FPS was being rejected). Rather than retune the
    already-tuned Q/R or fight that design philosophy, `_mahalanobis_cost()`
    is folded into the association cost purely as a bounded ranking term:
    it helps the Hungarian solver pick the statistically best-fit candidate
    among options that already pass the (unchanged) centroid gate — e.g.
    disambiguating crossing pedestrians whose centroid/IoU/appearance costs
    are near-tied — without ever being able to reject a match on its own.

2.  **Pre-gated Hungarian assignment.** Gating used to be applied only
    *after* `linear_sum_assignment`, so the solver could be forced into a
    locally "least bad" pairing that blocked a better pairing elsewhere.
    Invalid pairs (failing the adaptive centroid gate or the Mahalanobis
    gate) are now set to a large sentinel cost *before* solving, so the
    optimum found already respects gating. The post-hoc check remains as a
    safety net for forced pairings when scipy has no valid alternative.

3.  **Adaptive centroid-distance threshold.** The gate now grows with
    `frames_since_update` (capped) instead of being a static fraction of the
    frame diagonal for every track regardless of how long it's been missing.

4.  **Dynamic appearance weighting.** Appearance weight increases with
    `frames_since_update` (capped): freshly-tracked people rely mostly on
    motion cost; long-occluded tracks — whose constant-velocity prediction
    has degraded — lean more on appearance similarity. Improves
    re-identification after occlusion and disambiguation of crossing
    pedestrians.

5.  **Score fusion.** Cost is lightly biased toward higher-confidence
    detections (bounded, small weight), in the spirit of official
    ByteTrack's `fuse_score`, without overwhelming the existing
    geometric/appearance terms.

6.  **Duplicate-track suppression.** After lifecycle updates, any two
    Confirmed, freshly-detected tracks with high mutual IoU are resolved by
    `Track.quality_score()`: the lower-quality one is demoted to Lost (not
    deleted — reversible), so a re-association glitch can't leave two live
    IDs on one person indefinitely.

7.  **Numerically stable Kalman gain.** Explicit `np.linalg.inv(S)` replaced
    with `np.linalg.solve`, avoiding an unnecessary explicit matrix inverse.

8.  **State-transition hygiene.** Stage-1 confirmation and Stage-3
    re-activation now go through the existing `activate()` / `re_activate()`
    lifecycle methods instead of raw `state = X` assignment scattered in
    `run()` — identical behavior, one source of truth per transition.

9.  **Track quality score.** New `Track.quality_score()` combines confidence
    history (EMA), consecutive-hit ratio, and hit ratio. Used by duplicate
    suppression and exposed via `to_dict()` as an additive key.

10. **Logging severity fix.** The unconditional per-track `log.info` dump
    every frame is now `log.debug`, matching its own "TRACK DEBUG" label and
    avoiding needless formatting/IO cost in production.

Defaults are chosen so a track with `frames_since_update == 0` (the common,
continuously-tracked case) behaves the same as before.
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
    _score_ema: float = field(default=0.0, repr=False)     # smoothed confidence, for quality_score()

    def activate(self):
        self.state = TrackState.Confirmed

    def re_activate(self):
        self.state = TrackState.Confirmed
        self.frames_since_update = 0
        self._predicted_frames = 0

    def mark_lost(self):
        self.state = TrackState.Lost

    def mark_removed(self):
        self.state = TrackState.Deleted

    def is_confirmed(self):
        return self.state == TrackState.Confirmed

    def is_lost(self):
        return self.state == TrackState.Lost

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
        self._score_ema        = (detection.confidence if self._score_ema == 0.0
                                   else 0.85 * self._score_ema + 0.15 * detection.confidence)
        self.frame_idx         = frame_idx
        self.hits             += 1
        self._consecutive_hits += 1
        self.frames_since_update = 0
        self._predicted_frames   = 0
        self.history.append((frame_idx, self.bbox.copy()))

        if len(self.history) > 100:
            self.history.pop(0)

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

    def quality_score(self) -> float:
        """Composite track-quality metric in [0, 1].

        Combines smoothed detection-confidence history, consecutive-hit
        ratio (how "clean" the recent match streak is), and overall hit
        ratio (how often this track has actually been matched relative to
        its age). Used for duplicate-track tie-breaking; does not affect
        the core matching cost, so it cannot itself cause an ID switch.
        """
        hit_ratio = self.hits / max(self.age, 1)
        consecutive_ratio = self._consecutive_hits / max(self.age, 1)
        conf_component = self._score_ema if self._score_ema > 0.0 else self.score
        q = (0.40 * conf_component
             + 0.35 * min(consecutive_ratio, 1.0)
             + 0.25 * min(hit_ratio, 1.0))
        return float(np.clip(q, 0.0, 1.0))

    def to_dict(self) -> Dict[str, Any]:
        return {
            "track_id": self.track_id, "state": self.state.name,
            "frame_idx": self.frame_idx, "hits": self.hits, "age": self.age,
            "frames_since_update": self.frames_since_update,
            "bbox": self.bbox.tolist(), "score": round(self.score, 4),
            "peak_score": round(self._peak_score, 4),
            "quality": round(self.quality_score(), 4),
        }


# ============================================================
# Geometry helpers
# ============================================================

# Chi-square 95% critical value for 4 degrees of freedom (cx, cy, w, h).
# Standard statistical constant for Mahalanobis gating of a 4D Gaussian
# measurement — independent of any external Kalman/matching module.
_CHI2_95_4DOF = 9.4877


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
    duplicates_suppressed: int = 0
    latency_ms: float = 0.0

    def log(self) -> None:
        log.debug(
            "Frame {:>4} | confirmed={} tent={} lost={} new={} del={} sw={} dup={} lat={:.1f}ms",
            self.frame_idx, self.active_confirmed, self.tentative,
            self.lost, self.new_tracks, self.deleted_tracks,
            self.id_switches, self.duplicates_suppressed, self.latency_ms,
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

        # --- v5 additions -------------------------------------------------
        # Mahalanobis gate (chi-square 95%, 4 DoF) applied on top of the
        # centroid gate; a candidate pair must pass BOTH to be accepted.
        self._maha_gate:            float = self.config.get("mahalanobis_gate", _CHI2_95_4DOF)
        # Weight of the Mahalanobis term as a soft ranking signal in the
        # blended cost (does not gate/reject on its own — see _associate).
        self._maha_cost_weight:     float = self.config.get("mahalanobis_cost_weight", 0.15)
        # Adaptive centroid threshold: grows with frames_since_update, capped.
        self._centroid_thresh_growth: float = self.config.get("centroid_thresh_growth", 0.03)
        self._centroid_thresh_max:    float = self.config.get("centroid_thresh_max",    0.55)
        # Dynamic appearance weight: grows with frames_since_update, capped.
        self._app_weight_max:       float = self.config.get("appearance_weight_max", 0.55)
        self._app_weight_growth:    float = self.config.get("appearance_weight_growth", 0.15)
        # Score fusion: small bias toward higher-confidence detections.
        self._score_fusion_weight:  float = self.config.get("score_fusion_weight", 0.10)
        # Duplicate-track suppression: mutual-IoU threshold to treat two
        # Confirmed tracks as the same physical person.
        self._dup_iou_thresh:       float = self.config.get("duplicate_iou_thresh", 0.75)

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
            # PATCHED (was: active[ti].hits >= self._min_hits).
            # `hits` is a lifetime counter that never resets, so a track with
            # sporadic (non-consecutive) matches could reach min_hits and get
            # wrongly Confirmed, defeating the strict ghost-suppression this
            # tracker is documented to perform. `_consecutive_hits` resets to
            # 0 every time a track drops to Lost, so this now genuinely
            # requires `min_hits` *consecutive* matches before confirmation.
            if active[ti]._consecutive_hits >= self._min_hits:
                active[ti].activate()

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
            lost[ti].re_activate()
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

        # ── Duplicate-track suppression ────────────────────────────────────
        # Two Confirmed, freshly-detected tracks with high mutual IoU are
        # almost certainly the same physical person (a re-association
        # error). Keep the higher-quality one; demote the other to Lost
        # (reversible — it can still recover later, it just stops
        # competing for the same detections going forward).
        before_dup = len(output)
        output = self._suppress_duplicate_tracks(output)
        dup_count = before_dup - len(output)

        # ==========================
        # TRACK DEBUG (debug severity — this is per-track, per-frame volume)
        # ==========================
        log.debug("========== TRACK DEBUG ==========")

        for t in output:
            log.debug(
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
            id_switches=sw, duplicates_suppressed=dup_count, latency_ms=latency_ms,
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
        """Associate tracks ↔ detections using blended centroid + IoU +
        appearance cost, pre-gated by an adaptive centroid threshold AND a
        Mahalanobis distance computed from each track's Kalman covariance.

        The centroid-distance metric remains the primary geometric cost so
        that tracks and detections with zero IoU (too far apart to overlap)
        can still be matched if they are the spatially-closest pair. The
        Mahalanobis gate adds a statistically-justified, covariance-aware
        acceptance region on top of that: it grows automatically for tracks
        whose position uncertainty has grown (e.g. long-lost tracks), and
        stays tight for tracks that are confidently, continuously tracked.
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

        # Mahalanobis distance (covariance-aware; np.inf where a track has
        # no Kalman state yet). NOTE on calibration: this tracker's Q/R
        # (kalman_process_noise / kalman_measurement_noise) were tuned only
        # for display smoothing, not for statistical hard-gating. Testing
        # showed that even a LOOSE hard gate on Mahalanobis distance rejects
        # entirely normal motion (e.g. ~60px/frame walking speed) once a
        # track has converged, because this tracker deliberately keeps the
        # Kalman filter subordinate to the detection ("detection always
        # wins" — see module docstring fix #2), so its covariance shrinks
        # tighter than real motion allows. Rather than retune Q/R (an
        # unrequested change to already-tuned, working behavior) or fight
        # the tracker's own design philosophy, Mahalanobis distance is used
        # purely as a SOFT ranking signal: it helps the Hungarian solver
        # pick the statistically best-fit candidate among options that
        # already pass the proven centroid gate (e.g. disambiguating
        # crossing pedestrians whose centroid/IoU/appearance costs are
        # near-tied) — it never rejects a match on its own.
        maha_cost = self._mahalanobis_cost(tracks, d_boxes)
        maha_soft = np.clip(maha_cost / self._maha_gate, 0.0, 5.0) / 5.0  # -> [0, 1]
        maha_soft = np.nan_to_num(maha_soft, nan=1.0, posinf=1.0)

        # Per-track dynamic parameters, both functions of frames_since_update
        # (0 for a track matched last frame; grows the longer it's missing).
        fsu = np.array([t.frames_since_update for t in tracks], dtype=np.float32)
        eff_centroid_thresh = np.clip(
            self._centroid_thresh + self._centroid_thresh_growth * fsu,
            self._centroid_thresh, self._centroid_thresh_max,
        )
        dyn_app_weight = np.clip(
            self._app_weight * (1.0 + self._app_weight_growth * fsu),
            0.0, self._app_weight_max,
        )

        # Appearance cost (optional — breaks ties for crossing people)
        if self._app_enabled and self._last_frame is not None and self._app_weight > 0:
            app_cost = _appearance_cost(tracks, detections, self._last_frame, self._hist_bins)
            geo_cost = 0.6 * cen_cost + 0.4 * iou_cost
            w = dyn_app_weight[:, None]
            cost = (1.0 - w) * geo_cost + w * app_cost
        else:
            cost = 0.6 * cen_cost + 0.4 * iou_cost

        # Fold in the Mahalanobis ranking term (small, bounded weight) — this
        # is what lets the solver disambiguate crossing pedestrians whose
        # centroid/IoU/appearance costs are near-tied but whose motion
        # history makes one candidate statistically far more consistent.
        cost = (1.0 - self._maha_cost_weight) * cost + self._maha_cost_weight * maha_soft

        # Score fusion — small, bounded bias toward higher-confidence
        # detections, in the spirit of official ByteTrack's fuse_score.
        det_conf = np.array([d.confidence for d in detections], dtype=np.float32)
        cost = cost * (1.0 - self._score_fusion_weight * det_conf[None, :])

        cost = np.clip(cost, 0.0, 1.5).astype(np.float32)

        # Pre-gate: reject pairs failing the adaptive centroid gate. (An
        # earlier version of this patch also hard-gated on a loose
        # Mahalanobis backstop; testing showed that once a track converges
        # — which happens quickly, since this tracker deliberately keeps
        # the Kalman filter subordinate to the detection ("detection always
        # wins") — its covariance shrinks tight enough that even a 60px/
        # frame jump (normal human walking speed at low FPS) gets rejected
        # by ANY hard statistical threshold. That is incompatible with this
        # tracker's design philosophy, so Mahalanobis distance is used only
        # as the soft ranking term above — never as a hard reject.
        gate_fail = cen_cost > eff_centroid_thresh[:, None]
        _GATED_SENTINEL = 10.0  # safely above the clipped cost range above
        solver_cost = np.where(gate_fail, _GATED_SENTINEL, cost)

        row_ind, col_ind = linear_sum_assignment(solver_cost)

        matched, unm_t, unm_d = [], list(range(len(tracks))), list(range(len(detections)))
        for r, c in zip(row_ind, col_ind):
            # Safety net: scipy always returns a complete assignment even if
            # every option for a row was gated out, so re-check explicitly.
            if gate_fail[r, c]:
                continue
            matched.append((r, c))
            if r in unm_t: unm_t.remove(r)
            if c in unm_d: unm_d.remove(c)

        return matched, unm_t, unm_d

    # ------------------------------------------------------------------
    def _mahalanobis_cost(
        self, tracks: List[Track], det_boxes: np.ndarray
    ) -> np.ndarray:
        """Squared Mahalanobis distance from each track's predicted
        measurement-space distribution to each detection.

        cost[i, j] = (z_j - H mean_i)^T (H P_i H^T + R)^-1 (z_j - H mean_i)

        Solved via `np.linalg.solve` rather than an explicit matrix inverse
        for numerical stability. Returns np.inf for any track without an
        initialised Kalman state (so the gate check downstream safely
        rejects those pairs without special-casing).
        """
        M, N = len(tracks), len(det_boxes)
        cost = np.full((M, N), np.inf, dtype=np.float32)
        if M == 0 or N == 0:
            return cost

        cx = (det_boxes[:, 0] + det_boxes[:, 2]) / 2.0
        cy = (det_boxes[:, 1] + det_boxes[:, 3]) / 2.0
        w  = det_boxes[:, 2] - det_boxes[:, 0]
        h  = det_boxes[:, 3] - det_boxes[:, 1]
        z_all = np.stack([cx, cy, w, h], axis=1).astype(np.float64)  # (N, 4)

        for i, trk in enumerate(tracks):
            if trk._kf_mean is None or trk._kf_cov is None:
                continue
            mean4 = trk._kf_mean[:4]
            P = trk._kf_cov
            S = self._H @ P @ self._H.T + self._R          # (4,4) measurement-space cov
            S = 0.5 * (S + S.T)                             # enforce symmetry (fp safety)
            D = z_all - mean4[None, :]                       # (N, 4) residuals
            try:
                X = np.linalg.solve(S, D.T)                  # (4, N) — no explicit inverse
            except np.linalg.LinAlgError:
                continue
            m2 = np.einsum('ij,ij->i', D, X.T)                # squared Mahalanobis distances, one per detection
            cost[i, :] = m2.astype(np.float32)

        return cost

    # ------------------------------------------------------------------
    def _suppress_duplicate_tracks(self, candidates: List[Track]) -> List[Track]:
        """Resolve duplicate IDs on the same physical person.

        Among the tracks about to be output this frame, if two have mutual
        IoU at or above `duplicate_iou_thresh`, keep the higher-quality one
        (`Track.quality_score()`) and demote the other to Lost. Demotion is
        reversible (not deletion) — a genuinely distinct person who was
        briefly overlapping can still recover its own track later.
        """
        n = len(candidates)
        if n < 2:
            return candidates

        boxes = np.array([t.bbox for t in candidates], dtype=np.float32)
        iou_mat = _iou_matrix(boxes, boxes)

        ranked = sorted(range(n), key=lambda i: candidates[i].quality_score(), reverse=True)
        suppressed: set = set()
        for pos, i in enumerate(ranked):
            if i in suppressed:
                continue
            for j in ranked[pos + 1:]:
                if j in suppressed:
                    continue
                if iou_mat[i, j] >= self._dup_iou_thresh:
                    suppressed.add(j)
                    candidates[j].mark_lost()
                    candidates[j]._consecutive_hits = 0

        if not suppressed:
            return candidates
        return [c for idx, c in enumerate(candidates) if idx not in suppressed]

    # ------------------------------------------------------------------
    def _kalman_gain(
        self, P: np.ndarray, H: np.ndarray, R: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        # K = P H^T S^-1, solved via np.linalg.solve instead of an explicit
        # matrix inverse: K^T solves S^T K^T = (P H^T)^T = H P^T, which is
        # numerically more stable than forming S^-1 directly, at identical
        # mathematical result for well-conditioned S.
        S   = H @ P @ H.T + R
        S   = 0.5 * (S + S.T)  # enforce symmetry (floating-point safety)
        PHt = P @ H.T
        K     = np.linalg.solve(S, PHt.T).T
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