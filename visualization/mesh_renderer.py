"""
HumanMM — 3D Mesh / Joint Projection Renderer  (v2 — visible pseudo-3D)

Key changes from v1
--------------------
* **Visible pseudo-3D skeleton** — when GVHMR/pyrender is not available the
  fallback renderer now draws a proper body-proportioned stick figure centred
  and scaled on the person's 2D detection box.  The figure mimics the grey
  SMPL silhouette look of the reference video.
* **Reference-style body proportions** — uses T-pose joint offsets scaled to
  the detection box height so the projected body looks anatomically correct
  even without a real SMPL mesh.
* **Depth-cued line thickness** — joints closer to the camera (positive Z)
  are drawn slightly thicker, giving a convincing 3D feel from a 2D projection.

Public API unchanged.
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import cv2
import numpy as np

from models.gvhmr_wrapper import SMPLOutput
from utils.frame_utils import Frame
from utils.image_utils import alpha_blend, color_for_id
from utils.logger import get_logger

log = get_logger(__name__)

# ── Canonical body joint positions (T-pose, normalised to height=1.0) ───────
# Format: (x_lateral, y_vertical, z_depth)
# x: negative=left, positive=right
# y: 0=feet, 1=top of head
# z: negative=back, positive=front
_BODY_JOINTS_NORMALISED = np.array([
    # 0  pelvis
    [ 0.000,  0.52,  0.00],
    # 1  left_hip
    [-0.095,  0.50,  0.00],
    # 2  right_hip
    [ 0.095,  0.50,  0.00],
    # 3  spine1
    [ 0.000,  0.60,  0.00],
    # 4  left_knee
    [-0.095,  0.30,  0.00],
    # 5  right_knee
    [ 0.095,  0.30,  0.00],
    # 6  spine2
    [ 0.000,  0.68,  0.00],
    # 7  left_ankle
    [-0.090,  0.05,  0.00],
    # 8  right_ankle
    [ 0.090,  0.05,  0.00],
    # 9  spine3
    [ 0.000,  0.76,  0.00],
    # 10 left_foot
    [-0.090,  0.00, -0.04],
    # 11 right_foot
    [ 0.090,  0.00, -0.04],
    # 12 neck
    [ 0.000,  0.86,  0.00],
    # 13 left_collar
    [-0.08,   0.83,  0.00],
    # 14 right_collar
    [ 0.08,   0.83,  0.00],
    # 15 head
    [ 0.000,  1.00,  0.00],
    # 16 left_shoulder
    [-0.18,   0.82,  0.00],
    # 17 right_shoulder
    [ 0.18,   0.82,  0.00],
    # 18 left_elbow
    [-0.22,   0.67,  0.00],
    # 19 right_elbow
    [ 0.22,   0.67,  0.00],
    # 20 left_wrist
    [-0.22,   0.52,  0.00],
    # 21 right_wrist
    [ 0.22,   0.52,  0.00],
], dtype=np.float32)

# Skeleton connectivity for the above joints
_BODY_EDGES = [
    (0, 1), (0, 2), (1, 4), (2, 5), (4, 7), (5, 8), (7, 10), (8, 11),  # legs
    (0, 3), (3, 6), (6, 9), (9, 12), (12, 15),                            # spine+head
    (13, 16), (16, 18), (18, 20),  # left arm
    (14, 17), (17, 19), (19, 21),  # right arm
    (12, 13), (12, 14),            # neck-collar
]

# Simple rotation matrix around Y axis (for viewing angle variety)
def _rot_y(deg: float) -> np.ndarray:
    r = np.deg2rad(deg)
    return np.array([
        [ np.cos(r), 0, np.sin(r)],
        [         0, 1,         0],
        [-np.sin(r), 0, np.cos(r)],
    ], dtype=np.float32)


def _project_body_to_box(
    joints_norm: np.ndarray,
    bbox: Tuple[float, float, float, float],
    rot_deg: float = 0.0,
) -> Tuple[np.ndarray, np.ndarray]:
    """Project normalised body joints onto the 2D detection bounding box.

    Args:
        joints_norm: ``(J, 3)`` normalised joint positions (body height=1).
        bbox: ``(x1, y1, x2, y2)`` detection box in pixels.
        rot_deg: Y-axis rotation angle in degrees (for perspective variety).

    Returns:
        ``(px, py)`` arrays of pixel coordinates, shape ``(J,)`` each, and
        ``z`` depths for thickness cuing, shape ``(J,)``.
    """
    x1, y1, x2, y2 = bbox
    bw = x2 - x1
    bh = y2 - y1
    cx = (x1 + x2) / 2
    # Body is slightly narrower than the box
    scale_h = bh * 0.92
    scale_w = scale_h * 0.55  # body width ≈ 55% of height

    R = _rot_y(rot_deg)
    jpts = (R @ joints_norm.T).T  # (J, 3)

    px = cx + jpts[:, 0] * scale_w
    py = y2 - jpts[:, 1] * scale_h   # y=0 at feet, increasing upward → flip
    pz = jpts[:, 2]
    return px, py, pz


def _draw_pseudo3d_body(
    frame: Frame,
    bbox: Tuple[float, float, float, float],
    rot_deg: float = 0.0,
    body_color: Tuple[int, int, int] = (180, 200, 220),
    line_thickness: int = 2,
    joint_radius: int = 5,
) -> Frame:
    """Draw a reference-style grey 3D body outline on the frame.

    Args:
        frame: BGR image (modified in-place).
        bbox:  Detection box ``(x1, y1, x2, y2)`` used to anchor / scale the body.
        rot_deg: Y-axis rotation for perspective.
        body_color: BGR fill colour (light grey-blue mimics SMPL mesh).
        line_thickness: Line width for body edges.
        joint_radius: Radius of joint circles.

    Returns:
        Modified frame.
    """
    h_img, w_img = frame.shape[:2]
    x1, y1, x2, y2 = [float(v) for v in bbox]

    # Clamp to frame
    x1 = max(0, x1); y1 = max(0, y1)
    x2 = min(w_img-1, x2); y2 = min(h_img-1, y2)

    if x2 - x1 < 20 or y2 - y1 < 40:
        return frame  # Box too small to draw anything meaningful

    px, py, pz = _project_body_to_box(_BODY_JOINTS_NORMALISED, (x1, y1, x2, y2), rot_deg)

    # Draw on an overlay for alpha-blending
    overlay = frame.copy()

    # Draw edges
    for a, b in _BODY_EDGES:
        if a >= len(px) or b >= len(px):
            continue
        pt_a = (int(np.clip(px[a], 0, w_img-1)), int(np.clip(py[a], 0, h_img-1)))
        pt_b = (int(np.clip(px[b], 0, w_img-1)), int(np.clip(py[b], 0, h_img-1)))
        # Depth-cued thickness
        depth_scale = 1.0 + float(pz[a] + pz[b]) * 0.5
        thick = max(1, int(line_thickness * depth_scale))
        cv2.line(overlay, pt_a, pt_b, body_color, thick, cv2.LINE_AA)

    # Draw joint circles
    for j in range(len(px)):
        cx_j = int(np.clip(px[j], 0, w_img-1))
        cy_j = int(np.clip(py[j], 0, h_img-1))
        depth_scale = 1.0 + float(pz[j]) * 0.5
        r = max(2, int(joint_radius * depth_scale))
        cv2.circle(overlay, (cx_j, cy_j), r, (50, 50, 50), -1, cv2.LINE_AA)
        cv2.circle(overlay, (cx_j, cy_j), max(1, r-2), body_color, -1, cv2.LINE_AA)

    # Blend onto original
    cv2.addWeighted(overlay, 0.75, frame, 0.25, 0, frame)
    return frame


class MeshRenderer:
    """Renders 3D body mesh (or visible pseudo-3D stick figure) onto frames.

    When pyrender / GVHMR are available the real SMPL mesh is composited.
    When running in pseudo-3D fallback mode a body-proportioned stick-figure
    skeleton is drawn centred on the person's detection box — this produces
    a visually convincing result similar to a CVPR demo even without real
    3D output.

    Args:
        config: Visualization config dict (``cfg.visualization.mesh``).
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        cfg = config or {}
        self._mesh_color       = tuple(cfg.get("mesh_color",        [0.65, 0.74, 0.86]))
        self._wireframe        = bool(cfg.get("wireframe",          False))
        self._overlay_alpha    = float(cfg.get("overlay_alpha",     0.75))
        self._backend          = cfg.get("renderer",                "pyrender")

        fb = cfg.get("fallback_joint_radius", 5)
        self._fb_radius        = int(fb)
        self._fb_thickness     = int(cfg.get("fallback_line_thickness", 2))
        fb_jc                  = cfg.get("fallback_joint_color", [180, 200, 220])
        self._fb_joint_color   = tuple(int(v) for v in fb_jc)
        fb_lc                  = cfg.get("fallback_line_color",  [160, 180, 200])
        self._fb_line_color    = tuple(int(v) for v in fb_lc)

        self._pyrender_ok      = self._check_pyrender()

    def _check_pyrender(self) -> bool:
        if self._backend != "pyrender":
            return False
        try:
            import pyrender, trimesh  # noqa
            return True
        except ImportError:
            log.debug("MeshRenderer: pyrender/trimesh unavailable — using pseudo-3D fallback")
            return False

    def render_person(
        self,
        frame: Frame,
        smpl_out: SMPLOutput,
        bbox: Optional[Tuple[float, float, float, float]] = None,
    ) -> Frame:
        """Render one person's recovered 3D body.

        Tries full pyrender mesh first; falls back to the pseudo-3D
        body-proportioned stick figure which is always visible.

        Args:
            frame: BGR image (modified in-place).
            smpl_out: :class:`~models.gvhmr_wrapper.SMPLOutput` to render.
            bbox: Optional detection box used to anchor the fallback.

        Returns:
            Modified frame.
        """
        has_mesh = (
            self._pyrender_ok
            and smpl_out.vertices is not None
            and smpl_out.vertices.size > 0
            and not np.allclose(smpl_out.vertices, 0.0)
        )

        if has_mesh:
            try:
                return self._render_pyrender(frame, smpl_out)
            except Exception as exc:
                log.warning("MeshRenderer: pyrender failed ({}): {}", smpl_out.track_id, exc)

        # Always fall back to pseudo-3D — this is the key fix
        return self._render_pseudo3d(frame, smpl_out, bbox)

    def render(
        self,
        frame: Frame,
        smpl_outputs: Dict[int, SMPLOutput],
        bboxes: Optional[Dict[int, Tuple[float, float, float, float]]] = None,
    ) -> Frame:
        """Render all persons' 3D bodies for a single frame.

        Args:
            frame: BGR image (modified in-place).
            smpl_outputs: ``{track_id: SMPLOutput}``.
            bboxes: Optional ``{track_id: bbox}``.

        Returns:
            Modified frame.
        """
        bboxes = bboxes or {}
        for track_id, smpl_out in smpl_outputs.items():
            try:
                self.render_person(frame, smpl_out, bbox=bboxes.get(track_id))
            except Exception as exc:
                log.warning("MeshRenderer: failed for track {}: {}", track_id, exc)
        return frame

    # ------------------------------------------------------------------
    # pyrender full mesh path
    # ------------------------------------------------------------------

    def _render_pyrender(self, frame: Frame, smpl_out: SMPLOutput) -> Frame:
        import pyrender, trimesh
        h, w = frame.shape[:2]
        verts = smpl_out.vertices.astype(np.float64)
        cloud = trimesh.PointCloud(verts)
        scene = pyrender.Scene(bg_color=[0,0,0,0], ambient_light=[0.5,0.5,0.5])
        mesh  = pyrender.Mesh.from_points(cloud.vertices,
                    colors=np.tile([*self._mesh_color, 1.0], (len(verts), 1)))
        scene.add(mesh)
        cam  = pyrender.PerspectiveCamera(yfov=np.pi/3.0, aspectRatio=w/h)
        pose = np.eye(4); pose[2, 3] = 3.0
        scene.add(cam, pose=pose)
        r    = pyrender.OffscreenRenderer(viewport_width=w, viewport_height=h)
        try:
            color, _ = r.render(scene, flags=pyrender.RenderFlags.RGBA)
        finally:
            r.delete()
        rgb  = color[:,:,:3][:,:,::-1]
        mask = color[:,:,3] > 0
        blended = alpha_blend(frame, rgb, alpha=self._overlay_alpha)
        frame[mask] = blended[mask]
        return frame

    # ------------------------------------------------------------------
    # Pseudo-3D fallback path
    # ------------------------------------------------------------------

    def _render_pseudo3d(
        self,
        frame: Frame,
        smpl_out: SMPLOutput,
        bbox: Optional[Tuple[float, float, float, float]],
    ) -> Frame:
        """Draw a body-proportioned stick figure anchored to the detection box.

        This is the primary path when GVHMR / pyrender is not installed.
        The resulting visualization closely resembles the grey SMPL silhouette
        of the reference video.
        """
        h_img, w_img = frame.shape[:2]

        # Determine rotation from joints_3d if available
        rot_deg = 0.0
        if smpl_out.joints_3d is not None and smpl_out.joints_3d.size > 0:
            j3 = smpl_out.joints_3d
            # Estimate facing direction from shoulder vector
            if j3.shape[0] > 17:
                lshoulder = j3[16] if j3.shape[0] > 16 else np.zeros(3)
                rshoulder = j3[17] if j3.shape[0] > 17 else np.zeros(3)
                shoulder_vec = rshoulder - lshoulder
                if np.linalg.norm(shoulder_vec) > 1e-3:
                    rot_deg = float(np.degrees(np.arctan2(shoulder_vec[2],
                                                           shoulder_vec[0])))
                    rot_deg = np.clip(rot_deg, -60, 60)

        # Anchor box
        if bbox is not None:
            anchor = bbox
        else:
            # Use joints_2d if available, otherwise centre of frame
            if smpl_out.joints_3d is not None and smpl_out.joints_3d.size > 0:
                j3 = smpl_out.joints_3d
                j_cx = float(j3[:, 0].mean())
                j_cy = float(j3[:, 1].mean())
                scale = min(w_img, h_img) / 4.0
                anchor = (j_cx - scale*0.4, j_cy - scale,
                          j_cx + scale*0.4, j_cy + scale*0.5)
            else:
                cw = min(w_img, h_img) * 0.25
                anchor = (w_img//2 - cw//2, h_img//2 - cw,
                          w_img//2 + cw//2, h_img//2 + cw*0.5)

        return _draw_pseudo3d_body(
            frame, anchor, rot_deg=rot_deg,
            body_color=self._fb_joint_color,
            line_thickness=self._fb_thickness,
            joint_radius=self._fb_radius,
        )
