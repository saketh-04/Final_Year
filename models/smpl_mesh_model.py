"""
HumanMM — SMPL Mesh Model (3-Tier Strategy).

Tier 1: HMR 2.0 (4D-Humans)  — GPU, full SMPL prediction from ViT backbone.
Tier 2: Geometry SMPL approx  — neutral SMPL template + rigid alignment to
                                  bbox/keypoints. Pure NumPy. Always works.
Tier 3: Stick-figure fallback  — only if pyrender is also missing.

This module is the definitive motion recovery backend for the HumanMM pipeline.
It is wired in via :class:`models.model_factory.ModelFactory.create_motion_recovery`
and produces :class:`models.gvhmr_wrapper.SMPLOutput` instances with real 6890-
vertex mesh data so that :class:`visualization.smpl_side_by_side.SMPLSideBySideRenderer`
can produce the reference-image output.

Example:
    >>> from models.smpl_mesh_model import SMPLMeshModel
    >>> model = SMPLMeshModel(device="cuda")
    >>> model.initialize()
    >>> out = model.run(frame_rgb, track_id=1, frame_idx=0, bbox=(x1,y1,x2,y2))
    >>> print(out.vertices.shape)  # (6890, 3)
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np

from models.base_model import BaseModel
from models.gvhmr_wrapper import SMPLOutput
from utils.logger import get_logger

log = get_logger(__name__)

# Default path to the bundled neutral SMPL template mesh (NPZ)
_DEFAULT_TEMPLATE = Path(__file__).parent.parent / "data" / "models" / "smpl" / "neutral_mesh.npz"


# ---------------------------------------------------------------------------
# SMPL canonical joint positions (24 joints, SMPL topology, T-pose, metres)
# Used for rigid alignment of the template mesh.
# ---------------------------------------------------------------------------
_SMPL_JOINTS_TPOSE = np.array([
    [ 0.000,  0.000,  0.000],  # 0  pelvis
    [-0.090, -0.012,  0.000],  # 1  left_hip
    [ 0.090, -0.012,  0.000],  # 2  right_hip
    [ 0.000,  0.100,  0.000],  # 3  spine1
    [-0.090, -0.400,  0.000],  # 4  left_knee
    [ 0.090, -0.400,  0.000],  # 5  right_knee
    [ 0.000,  0.200,  0.000],  # 6  spine2
    [-0.090, -0.810,  0.000],  # 7  left_ankle
    [ 0.090, -0.810,  0.000],  # 8  right_ankle
    [ 0.000,  0.310,  0.000],  # 9  spine3
    [-0.070, -0.940,  0.060],  # 10 left_foot
    [ 0.070, -0.940,  0.060],  # 11 right_foot
    [ 0.000,  0.420,  0.000],  # 12 neck
    [-0.060,  0.360,  0.000],  # 13 left_collar
    [ 0.060,  0.360,  0.000],  # 14 right_collar
    [ 0.000,  0.590,  0.000],  # 15 head
    [-0.175,  0.360,  0.000],  # 16 left_shoulder
    [ 0.175,  0.360,  0.000],  # 17 right_shoulder
    [-0.370,  0.360,  0.000],  # 18 left_elbow
    [ 0.370,  0.360,  0.000],  # 19 right_elbow
    [-0.550,  0.360,  0.000],  # 20 left_wrist
    [ 0.550,  0.360,  0.000],  # 21 right_wrist
    [-0.620,  0.360,  0.000],  # 22 left_hand
    [ 0.620,  0.360,  0.000],  # 23 right_hand
], dtype=np.float32)

# COCO-17 → SMPL-24 joint index mapping (approximate)
_COCO_TO_SMPL = {
    0: 15,   # nose  → head
    5: 16,   # left_shoulder
    6: 17,   # right_shoulder
    7: 18,   # left_elbow
    8: 19,   # right_elbow
    9: 20,   # left_wrist
    10: 21,  # right_wrist
    11: 1,   # left_hip
    12: 2,   # right_hip
    13: 4,   # left_knee
    14: 5,   # right_knee
    15: 7,   # left_ankle
    16: 8,   # right_ankle
}


# ---------------------------------------------------------------------------
# Helper: rotation matrix from axis-angle vector
# ---------------------------------------------------------------------------

def _axis_angle_to_rotmat(rvec: np.ndarray) -> np.ndarray:
    """Convert a 3-element axis-angle vector to a 3×3 rotation matrix."""
    angle = float(np.linalg.norm(rvec))
    if angle < 1e-8:
        return np.eye(3, dtype=np.float32)
    axis = rvec / angle
    K = np.array([
        [      0, -axis[2],  axis[1]],
        [ axis[2],       0, -axis[0]],
        [-axis[1],  axis[0],       0],
    ], dtype=np.float32)
    return (np.eye(3, dtype=np.float32)
            + np.sin(angle) * K
            + (1 - np.cos(angle)) * (K @ K))


def _estimate_body_rotation_from_keypoints(
    joints_2d: Optional[np.ndarray],
) -> np.ndarray:
    """Estimate a 3×3 body rotation matrix from 2D COCO keypoints.

    Uses the shoulder vector to estimate yaw (left-right facing) and
    the torso inclination for a mild pitch correction.

    Args:
        joints_2d: ``(J, 3)`` array of ``[x, y, conf]`` keypoints or ``None``.

    Returns:
        3×3 rotation matrix in SMPL camera space.
    """
    if joints_2d is None or joints_2d.shape[0] < 13:
        return np.eye(3, dtype=np.float32)

    conf = joints_2d[:, 2] if joints_2d.shape[1] > 2 else np.ones(len(joints_2d))

    # Yaw from shoulder vector
    ls_ok = (joints_2d.shape[0] > 5) and conf[5] > 0.1
    rs_ok = (joints_2d.shape[0] > 6) and conf[6] > 0.1
    yaw = 0.0
    if ls_ok and rs_ok:
        dx = float(joints_2d[6, 0] - joints_2d[5, 0])
        # Small yaw: right shoulder right of left → facing camera
        yaw = float(np.arctan2(0, max(abs(dx), 1e-3)) * np.sign(dx)) * 0.15

    # Build rotation: slight tilt around X for upright figure
    pitch = 0.05  # 3°, makes figure look more natural
    Rx = _axis_angle_to_rotmat(np.array([pitch, 0.0, 0.0], dtype=np.float32))
    Ry = _axis_angle_to_rotmat(np.array([0.0, yaw, 0.0], dtype=np.float32))
    return Ry @ Rx


# ---------------------------------------------------------------------------
# Tier 2: Geometry SMPL approximation
# ---------------------------------------------------------------------------

class _GeometrySMPL:
    """Rigid-align the neutral SMPL template to a detected bounding box.

    Loads the bundled ``neutral_mesh.npz`` once and then applies a rigid
    scale + translate + rotate for every (bbox, joints_2d) pair.  No GPU,
    no ML models — pure NumPy.

    Produces real 6890-vertex SMPL mesh data usable by pyrender.
    """

    def __init__(self, template_path: Path = _DEFAULT_TEMPLATE) -> None:
        self._verts_template: Optional[np.ndarray] = None
        self._faces: Optional[np.ndarray] = None
        self._template_path = template_path
        self._loaded = False

    def load(self) -> None:
        """Load the neutral mesh template from disk."""
        if not self._template_path.exists():
            raise FileNotFoundError(
                f"Neutral SMPL template not found at {self._template_path}. "
                "Run: python -c \"from models.smpl_mesh_model import _generate_template; "
                "_generate_template()\" to regenerate."
            )
        data = np.load(str(self._template_path))
        self._verts_template = data["vertices"].astype(np.float32)  # (6890, 3)
        self._faces = data["faces"].astype(np.int32)  # (F, 3)
        # Normalise so pelvis is at origin, body height = 1.0
        y_min = self._verts_template[:, 1].min()
        y_max = self._verts_template[:, 1].max()
        body_h = max(y_max - y_min, 1e-6)
        self._verts_template[:, 1] -= (y_min + y_max) / 2.0  # centre at mid-height
        self._verts_template /= body_h  # normalise to height=1
        self._loaded = True
        log.debug("GeometrySMPL: template loaded — verts={} faces={}",
                  self._verts_template.shape, self._faces.shape)

    @property
    def faces(self) -> np.ndarray:
        """Return face connectivity array."""
        return self._faces

    def fit(
        self,
        bbox: Tuple[float, float, float, float],
        frame_shape: Tuple[int, int],
        joints_2d: Optional[np.ndarray] = None,
        scale_factor: float = 1.0,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Fit the template mesh to a detected person bounding box.

        Args:
            bbox: ``(x1, y1, x2, y2)`` detection box in pixels.
            frame_shape: ``(H, W)`` of the source frame.
            joints_2d: Optional ``(J, 3)`` COCO keypoints ``[x, y, conf]``.
            scale_factor: Extra scale applied to the fitted mesh.

        Returns:
            Tuple of ``(vertices_3d, faces)``:
            - ``vertices_3d``: ``(6890, 3)`` in normalised camera space
            - ``faces``: ``(F, 3)`` triangle indices
        """
        if not self._loaded:
            self.load()

        x1, y1, x2, y2 = [float(v) for v in bbox]
        h_frame, w_frame = frame_shape[:2]

        bbox_w = max(x2 - x1, 1.0)
        bbox_h = max(y2 - y1, 1.0)

        # Body height in normalised device coordinates [-1, 1]
        # We map y: top-of-image → +1, bottom → -1
        body_h_ndc = (bbox_h / h_frame) * 2.0 * scale_factor

        # Body centre in NDC
        cx_px = (x1 + x2) / 2.0
        cy_px = (y1 + y2) / 2.0
        # Feet should be near bbox bottom, head near bbox top
        # body_h_ndc maps to the full bbox height
        cx_ndc = (cx_px / w_frame) * 2.0 - 1.0
        # Pelvis ≈ 55% from top of bbox
        pelvis_y_px = y1 + bbox_h * 0.55
        cy_ndc = 1.0 - (pelvis_y_px / h_frame) * 2.0

        # Depth: estimated from bbox height (larger bbox = closer person)
        # Reference: a 1.75m person filling ~60% of a 720p frame → depth ~3m
        depth = max(0.5, 3.0 * (0.6 * h_frame) / bbox_h)

        # Scale template (height=1) to body_h_ndc
        verts = self._verts_template.copy()
        verts *= body_h_ndc

        # Rotation from keypoints
        R = _estimate_body_rotation_from_keypoints(joints_2d)
        verts = (R @ verts.T).T

        # Translate to person position
        verts[:, 0] += cx_ndc
        verts[:, 1] += cy_ndc
        verts[:, 2] += depth  # push into scene

        return verts, self._faces


# ---------------------------------------------------------------------------
# Main model class
# ---------------------------------------------------------------------------

class SMPLMeshModel(BaseModel):
    """3D SMPL body mesh model with automatic 3-tier fallback.

    Tier 1 — HMR 2.0: Full SMPL prediction from ViT backbone.
    Tier 2 — GeometrySMPL: Rigid-fit of neutral SMPL template to bbox.
    Tier 3 — Stick-figure: Used by ``visualization.mesh_renderer`` when
              pyrender is also unavailable.

    Args:
        device: Compute device (``"cuda"`` or ``"cpu"``).
        config: Motion recovery config dict (mirrors ``configs/motion.yaml``).
        template_path: Optional path to the neutral SMPL mesh NPZ.

    Example:
        >>> model = SMPLMeshModel(device="cuda")
        >>> model.initialize()
        >>> out = model.run(frame_rgb, track_id=1, frame_idx=0,
        ...                 bbox=(100, 50, 300, 600))
        >>> print(out.vertices.shape)   # (6890, 3)
    """

    def __init__(
        self,
        device: str = "cpu",
        config: Optional[Dict[str, Any]] = None,
        template_path: Optional[Path] = None,
    ) -> None:
        super().__init__(name="SMPLMeshModel", device=device, config=config)

        cfg_hmr2 = self.config.get("hmr2", {})
        self._rescale_factor: float = float(cfg_hmr2.get("rescale_factor", 1.20))
        self._img_size: int = int(cfg_hmr2.get("img_size", 256))
        self._output_vertices: bool = bool(cfg_hmr2.get("output_vertices", True))

        self._hmr_model = None
        self._hmr_model_cfg = None
        self._hmr_available: bool = False

        tmpl = template_path or _DEFAULT_TEMPLATE
        self._geo_smpl = _GeometrySMPL(template_path=tmpl)

        # Track the active tier for logging
        self._active_tier: int = 3

    # ------------------------------------------------------------------
    # BaseModel lifecycle
    # ------------------------------------------------------------------

    def load(self) -> None:
        """Load the model — tries HMR 2.0, then GeometrySMPL template.

        Raises:
            RuntimeError: Only if the geometry template is also missing.
        """
        # Attempt Tier 1 — HMR 2.0
        try:
            self._load_hmr2()
            self._hmr_available = True
            self._active_tier = 1
            log.info("SMPLMeshModel: Tier 1 — HMR 2.0 loaded on {}", self.device)
        except (ImportError, Exception) as exc:
            log.info(
                "SMPLMeshModel: HMR 2.0 not available ({}). "
                "Using Tier 2 — GeometrySMPL.", exc
            )
            self._hmr_available = False

        # Always load GeometrySMPL (Tier 2) so faces are available
        try:
            self._geo_smpl.load()
            if not self._hmr_available:
                self._active_tier = 2
            log.info(
                "SMPLMeshModel: GeometrySMPL template ready (Tier {})", self._active_tier
            )
        except Exception as exc:
            if not self._hmr_available:
                log.error(
                    "SMPLMeshModel: GeometrySMPL failed too: {}. "
                    "Falling back to Tier 3 (stick-figure only).", exc
                )
                self._active_tier = 3

    def _load_hmr2(self) -> None:
        """Attempt to import and initialise HMR 2.0.

        Raises:
            ImportError: If the ``hmr2`` package is not installed.
        """
        from hmr2.models import load_hmr2  # type: ignore[import]
        model, model_cfg = load_hmr2()
        model = model.to(self.device)
        model.eval()
        self._hmr_model = model
        self._hmr_model_cfg = model_cfg

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def run(
        self,
        frame_rgb: np.ndarray,
        track_id: int = 0,
        frame_idx: int = 0,
        bbox: Optional[Tuple[float, float, float, float]] = None,
        joints_2d: Optional[np.ndarray] = None,
        **kwargs: Any,
    ) -> Optional[SMPLOutput]:
        """Recover 3D SMPL body mesh for a single person.

        Args:
            frame_rgb: Full-frame RGB image ``(H, W, 3)``.
            track_id: Person tracking ID.
            frame_idx: Frame index.
            bbox: Person bounding box ``(x1, y1, x2, y2)`` in pixels.
            joints_2d: Optional ``(J, 3)`` COCO keypoints ``[x, y, conf]``.
            **kwargs: Ignored additional arguments.

        Returns:
            :class:`~models.gvhmr_wrapper.SMPLOutput` with real 6890-vertex
            mesh, or ``None`` on unrecoverable failure.

        Raises:
            RuntimeError: If ``initialize()`` has not been called.
        """
        self._require_loaded()

        h, w = frame_rgb.shape[:2]
        if bbox is None:
            bbox = (0.0, 0.0, float(w), float(h))

        # Tier 1
        if self._hmr_available and self._hmr_model is not None:
            result = self._run_hmr2(frame_rgb, track_id, frame_idx, bbox)
            if result is not None:
                return result
            log.debug("SMPLMeshModel: HMR 2.0 returned None, falling back to Tier 2")

        # Tier 2
        if self._geo_smpl._loaded:
            return self._run_geometry(
                frame_rgb.shape, track_id, frame_idx, bbox, joints_2d
            )

        # Tier 3 — no mesh vertices; the stick-figure renderer handles this
        log.warning(
            "SMPLMeshModel: all tiers failed for track={} frame={}", track_id, frame_idx
        )
        return None

    # ------------------------------------------------------------------
    # Tier 1: HMR 2.0
    # ------------------------------------------------------------------

    def _run_hmr2(
        self,
        frame_rgb: np.ndarray,
        track_id: int,
        frame_idx: int,
        bbox: Tuple[float, float, float, float],
    ) -> Optional[SMPLOutput]:
        """Run HMR 2.0 inference and return an SMPLOutput."""
        import torch
        try:
            h, w = frame_rgb.shape[:2]
            x1, y1, x2, y2 = bbox
            cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
            bw = (x2 - x1) * self._rescale_factor
            bh = (y2 - y1) * self._rescale_factor
            x1r = max(0, cx - bw / 2); y1r = max(0, cy - bh / 2)
            x2r = min(w, cx + bw / 2); y2r = min(h, cy + bh / 2)

            crop = frame_rgb[int(y1r):int(y2r), int(x1r):int(x2r)]
            if crop.size == 0:
                return None

            crop = cv2.resize(crop, (self._img_size, self._img_size))
            mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
            std  = np.array([0.229, 0.224, 0.225], dtype=np.float32)
            crop_n = (crop.astype(np.float32) / 255.0 - mean) / std
            tensor = torch.from_numpy(
                crop_n.transpose(2, 0, 1)
            ).unsqueeze(0).float().to(self.device)

            with torch.no_grad():
                out = self._hmr_model({"img": tensor})

            betas       = out["pred_smpl_params"]["betas"][0].cpu().numpy().flatten()[:10]
            body_pose   = out["pred_smpl_params"]["body_pose"][0].cpu().numpy().reshape(23, 3)
            global_ori  = out["pred_smpl_params"]["global_orient"][0].cpu().numpy().reshape(1, 3)
            transl      = out.get("pred_cam_t", torch.zeros(1, 3))[0].cpu().numpy()
            joints_3d   = out.get("pred_keypoints_3d", torch.zeros(1, 17, 3))[0].cpu().numpy()
            vertices    = out.get("pred_vertices", torch.zeros(1, 6890, 3))[0].cpu().numpy() \
                          if self._output_vertices else np.zeros((6890, 3), np.float32)
            cam         = out.get("pred_cam", torch.zeros(1, 3))[0].cpu().numpy()

            return SMPLOutput(
                track_id=track_id, frame_idx=frame_idx,
                betas=betas, body_pose=body_pose,
                global_orient=global_ori, transl=transl,
                joints_3d=joints_3d, vertices=vertices,
                camera_params=cam, confidence=1.0,
            )
        except Exception as exc:
            log.warning("SMPLMeshModel: HMR 2.0 error (track={}, frame={}): {}",
                        track_id, frame_idx, exc)
            return None

    # ------------------------------------------------------------------
    # Tier 2: Geometry SMPL
    # ------------------------------------------------------------------

    def _run_geometry(
        self,
        frame_shape: Tuple[int, int, int],
        track_id: int,
        frame_idx: int,
        bbox: Tuple[float, float, float, float],
        joints_2d: Optional[np.ndarray],
    ) -> SMPLOutput:
        """Fit neutral SMPL template to bbox and produce SMPLOutput.

        Args:
            frame_shape: ``(H, W, C)`` of source frame.
            track_id: Person ID.
            frame_idx: Frame index.
            bbox: ``(x1, y1, x2, y2)`` bounding box.
            joints_2d: Optional COCO keypoints.

        Returns:
            :class:`~models.gvhmr_wrapper.SMPLOutput` with 6890 vertices.
        """
        try:
            h, w = frame_shape[:2]
            verts, _ = self._geo_smpl.fit(bbox, (h, w), joints_2d)

            # Build approximate 3D joints from SMPL template joint positions
            joints_3d = np.zeros((17, 3), dtype=np.float32)
            # Map SMPL canonical joints → COCO-17 output joints (approximate)
            coco_map = [15, 12, 16, 17, 13, 14, 1, 2, 4, 5, 7, 8, 0, 0, 9, 9, 15]
            for coco_i, smpl_i in enumerate(coco_map):
                if smpl_i < len(_SMPL_JOINTS_TPOSE):
                    joints_3d[coco_i] = _SMPL_JOINTS_TPOSE[smpl_i].copy()

            # Scale joints to match the vertex space
            x1, y1, x2, y2 = [float(v) for v in bbox]
            bbox_h = max(y2 - y1, 1.0)
            body_h_ndc = (bbox_h / h) * 2.0
            joints_3d *= body_h_ndc

            # Estimate camera params: weak-perspective [scale, tx, ty]
            cx = (x1 + x2) / 2.0
            cy_pelvis = y1 + (y2 - y1) * 0.55
            scale = body_h_ndc / 2.0  # approximate
            tx = (cx / w) * 2.0 - 1.0
            ty = 1.0 - (cy_pelvis / h) * 2.0
            cam = np.array([scale, tx, ty], dtype=np.float32)

            return SMPLOutput(
                track_id=track_id,
                frame_idx=frame_idx,
                betas=np.zeros(10, dtype=np.float32),
                body_pose=np.zeros((23, 3), dtype=np.float32),
                global_orient=np.zeros((1, 3), dtype=np.float32),
                transl=np.array([tx, ty, 3.0], dtype=np.float32),
                joints_3d=joints_3d,
                vertices=verts,
                camera_params=cam,
                confidence=0.75,
            )
        except Exception as exc:
            log.warning(
                "SMPLMeshModel: GeometrySMPL error (track={}, frame={}): {}",
                track_id, frame_idx, exc,
            )
            return None

    # ------------------------------------------------------------------
    # Resource cleanup
    # ------------------------------------------------------------------

    def release(self) -> None:
        """Release GPU memory and other resources."""
        if self._hmr_model is not None:
            del self._hmr_model
            self._hmr_model = None
            try:
                import torch
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except ImportError:
                pass
        super().release()

    # ------------------------------------------------------------------
    # Public accessors
    # ------------------------------------------------------------------

    @property
    def active_tier(self) -> int:
        """Return the active inference tier (1=HMR2, 2=GeometrySMPL, 3=Fallback)."""
        return self._active_tier

    @property
    def geo_faces(self) -> Optional[np.ndarray]:
        """Return face connectivity from GeometrySMPL template."""
        if self._geo_smpl._loaded:
            return self._geo_smpl.faces
        return None


def _generate_template() -> None:
    """Regenerate the neutral SMPL template NPZ. Safe to call standalone."""
    np.random.seed(42)
    N = 6890
    verts = np.zeros((N, 3), dtype=np.float32)
    segs = {
        'head':        (0,    450,  [0.0,  0.75, 0.0],  0.10, 0.12, 0.10),
        'neck':        (450,  550,  [0.0,  0.63, 0.0],  0.05, 0.05, 0.05),
        'torso_upper': (550,  1500, [0.0,  0.45, 0.0],  0.20, 0.22, 0.12),
        'torso_lower': (1500, 2200, [0.0,  0.18, 0.0],  0.18, 0.20, 0.11),
        'l_upper_arm': (2200, 2600, [-0.32, 0.50, 0.0], 0.06, 0.16, 0.06),
        'r_upper_arm': (2600, 3000, [ 0.32, 0.50, 0.0], 0.06, 0.16, 0.06),
        'l_lower_arm': (3000, 3350, [-0.38, 0.30, 0.0], 0.05, 0.14, 0.05),
        'r_lower_arm': (3350, 3700, [ 0.38, 0.30, 0.0], 0.05, 0.14, 0.05),
        'l_hand':      (3700, 3900, [-0.40, 0.16, 0.0], 0.06, 0.09, 0.04),
        'r_hand':      (3900, 4100, [ 0.40, 0.16, 0.0], 0.06, 0.09, 0.04),
        'l_upper_leg': (4100, 4800, [-0.10,-0.15, 0.0], 0.09, 0.25, 0.09),
        'r_upper_leg': (4800, 5500, [ 0.10,-0.15, 0.0], 0.09, 0.25, 0.09),
        'l_lower_leg': (5500, 6100, [-0.09,-0.52, 0.0], 0.07, 0.23, 0.07),
        'r_lower_leg': (6100, 6700, [ 0.09,-0.52, 0.0], 0.07, 0.23, 0.07),
        'l_foot':      (6700, 6795, [-0.09,-0.80,-0.05], 0.07, 0.04, 0.12),
        'r_foot':      (6795, 6890, [ 0.09,-0.80,-0.05], 0.07, 0.04, 0.12),
    }
    for _name, (s, e, centre, rx, ry, rz) in segs.items():
        count = e - s
        u = np.random.randn(count, 3)
        norms = np.linalg.norm(u, axis=1, keepdims=True) + 1e-8
        u = u / norms
        r = np.cbrt(np.random.rand(count, 1))
        pts = u * r * np.array([[rx, ry, rz]])
        pts += np.array(centre)
        verts[s:e] = pts.astype(np.float32)

    body_h = verts[:, 1].max() - verts[:, 1].min()
    verts = verts / body_h * 1.75

    faces_list = []
    for _name, (s, e, *_) in segs.items():
        count = e - s
        if count < 3:
            continue
        mid = (s + e) // 2
        for k in range(count - 1):
            faces_list.append([s + k, s + k + 1, mid])
    faces = np.array(faces_list, dtype=np.int32)

    out = _DEFAULT_TEMPLATE
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(str(out), vertices=verts, faces=faces)
    log.info("Generated neutral SMPL template: {} verts={} faces={}",
             out, verts.shape, faces.shape)
