"""
HumanMM — SMPL Side-by-Side Renderer (Reference Image Output).

Produces the exact layout shown in the reference image:

    ┌─────────────────────────┬─────────────────────────┐
    │  Original video frame   │  White canvas +          │
    │  with grey SMPL mesh    │  checkerboard floor +    │
    │  overlaid on person     │  isolated grey SMPL mesh │
    └─────────────────────────┴─────────────────────────┘

The output is written to ``08_smpl_sidebyside.mp4`` without touching any
existing pipeline stage output.

Two rendering backends are supported:
- **pyrender** (recommended): GPU-accelerated Phong-shaded 3D mesh.
- **OpenCV projection** (fallback): orthographic projection of mesh vertices
  onto a 2D canvas — produces a silhouette / depth-shaded look without pyrender.

Public API:
    >>> renderer = SMPLSideBySideRenderer(config=cfg)
    >>> renderer.initialize()
    >>> combined = renderer.render_frame(
    ...     frame, smpl_outputs={1: smpl_out}, bboxes={1: (x1,y1,x2,y2)},
    ...     frame_idx=0
    ... )
    >>> # combined.shape == (H, W*2+2, 3)

Example (batch over all frames):
    >>> out_path = renderer.render_video(
    ...     frames, smpl_results, tracks_per_frame, output_path
    ... )
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import cv2
import numpy as np

from models.gvhmr_wrapper import SMPLOutput
from utils.frame_utils import Frame, resize_frame
from utils.image_utils import draw_header_bar
from utils.logger import get_logger
from utils.video_writer import VideoWriter

log = get_logger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Grey mesh colour (BGR) — matches reference image's grey SMPL mesh
_MESH_BGR: Tuple[int, int, int] = (210, 213, 168)   # warm grey-blue
_MESH_RGB_F: Tuple[float, float, float] = (0.66, 0.74, 0.86)  # float for pyrender

# Checkerboard tile colour pair (grey + white)
_CHECKER_A: Tuple[int, int, int] = (200, 200, 200)
_CHECKER_B: Tuple[int, int, int] = (240, 240, 240)


# ---------------------------------------------------------------------------
# Checkerboard floor helper
# ---------------------------------------------------------------------------

def _draw_checkerboard(
    canvas: Frame,
    tiles_x: int = 8,
    tiles_y: int = 4,
    floor_frac: float = 0.30,
) -> None:
    """Draw a perspective-warped checkerboard floor on the lower portion of canvas.

    Args:
        canvas: BGR image to draw on (modified in-place).
        tiles_x: Number of tiles across.
        tiles_y: Number of tile rows.
        floor_frac: Fraction of canvas height used by the floor region.
    """
    h, w = canvas.shape[:2]
    floor_top = int(h * (1.0 - floor_frac))
    floor_bot = h

    tile_w = w / tiles_x
    tile_h = (floor_bot - floor_top) / tiles_y

    for row in range(tiles_y):
        for col in range(tiles_x):
            x1 = int(col * tile_w)
            y1 = int(floor_top + row * tile_h)
            x2 = int((col + 1) * tile_w)
            y2 = int(floor_top + (row + 1) * tile_h)
            color = _CHECKER_A if (row + col) % 2 == 0 else _CHECKER_B
            cv2.rectangle(canvas, (x1, y1), (x2, y2), color, -1)


# ---------------------------------------------------------------------------
# OpenCV-based orthographic mesh projector (fallback when pyrender missing)
# ---------------------------------------------------------------------------

def _project_verts_ortho(
    verts: np.ndarray,
    canvas_w: int,
    canvas_h: int,
    margin: float = 0.15,
    camera_y_offset: float = 0.0,
) -> Tuple[np.ndarray, np.ndarray]:
    """Project 3D vertices orthographically onto a 2D canvas.

    Args:
        verts: ``(N, 3)`` vertex array in camera coordinates.
        canvas_w: Canvas width in pixels.
        canvas_h: Canvas height in pixels.
        margin: Fractional margin on each side.
        camera_y_offset: Vertical offset to apply to vertices before projection.

    Returns:
        Tuple of ``(px, py)`` integer pixel coordinate arrays of shape ``(N,)``.
    """
    v = verts.copy()
    v[:, 1] += camera_y_offset

    x_range = v[:, 0].max() - v[:, 0].min() + 1e-6
    y_range = v[:, 1].max() - v[:, 1].min() + 1e-6

    usable_w = canvas_w * (1.0 - 2 * margin)
    usable_h = canvas_h * (1.0 - 2 * margin)

    scale = min(usable_w / x_range, usable_h / y_range)

    x_mid = (v[:, 0].max() + v[:, 0].min()) / 2.0
    y_mid = (v[:, 1].max() + v[:, 1].min()) / 2.0

    px = (v[:, 0] - x_mid) * scale + canvas_w / 2.0
    # y axis: 3D y increases upward, image y increases downward → flip
    py = -(v[:, 1] - y_mid) * scale + canvas_h / 2.0

    # Depth for z-buffer (normalised to [0, 1])
    z_norm = (v[:, 2] - v[:, 2].min()) / (v[:, 2].max() - v[:, 2].min() + 1e-6)

    return px.astype(np.float32), py.astype(np.float32), z_norm.astype(np.float32)


def _render_mesh_opencv(
    verts: np.ndarray,
    faces: Optional[np.ndarray],
    canvas: Frame,
    mesh_color: Tuple[int, int, int] = _MESH_BGR,
    rotation_y_deg: float = 0.0,
    camera_y_offset: float = 0.0,
) -> Frame:
    """Render 3D mesh onto canvas via OpenCV (no pyrender dependency).

    Applies a Gouraud-like depth shading: triangles further away are darker.
    Filled triangles are drawn in painter's-algorithm order (back-to-front).

    Args:
        verts: ``(N, 3)`` vertex array.
        faces: ``(F, 3)`` face index array, or ``None`` (point-cloud only).
        canvas: BGR canvas to draw on (modified in-place).
        mesh_color: BGR fill colour for the mesh.
        rotation_y_deg: Y-axis rotation to apply for 3/4 view.
        camera_y_offset: Vertical camera offset.

    Returns:
        Modified canvas.
    """
    if verts is None or verts.size == 0:
        return canvas

    h, w = canvas.shape[:2]

    # Apply Y-axis rotation for 3/4 view
    angle = math.radians(rotation_y_deg)
    Ry = np.array([
        [math.cos(angle), 0, math.sin(angle)],
        [0,               1, 0              ],
        [-math.sin(angle), 0, math.cos(angle)],
    ], dtype=np.float32)
    v = (Ry @ verts.T).T

    px, py, pz = _project_verts_ortho(v, w, h, camera_y_offset=camera_y_offset)

    if faces is not None and len(faces) > 0:
        # Compute per-face depth (centroid z)
        v0 = pz[faces[:, 0]]
        v1 = pz[faces[:, 1]]
        v2 = pz[faces[:, 2]]
        face_z = (v0 + v1 + v2) / 3.0

        # Sort back-to-front (painter's algorithm)
        order = np.argsort(face_z)[::-1]

        br, bg, bb = mesh_color[2], mesh_color[1], mesh_color[0]

        for fi in order:
            tri = faces[fi]
            pts = np.array([
                [px[tri[0]], py[tri[0]]],
                [px[tri[1]], py[tri[1]]],
                [px[tri[2]], py[tri[2]]],
            ], dtype=np.int32)

            # Clamp to canvas
            if (pts[:, 0].max() < 0 or pts[:, 0].min() >= w or
                    pts[:, 1].max() < 0 or pts[:, 1].min() >= h):
                continue

            # Depth-based shading: nearer faces slightly lighter
            depth = float(face_z[fi])
            shade = 0.75 + depth * 0.25
            col = (
                int(np.clip(bb * shade, 0, 255)),
                int(np.clip(bg * shade, 0, 255)),
                int(np.clip(br * shade, 0, 255)),
            )

            cv2.fillPoly(canvas, [pts], col)

        # Draw silhouette edges for crisp outline
        cv2.polylines(
            canvas,
            [np.column_stack([px.astype(np.int32), py.astype(np.int32)])],
            isClosed=False,
            color=(80, 80, 80),
            thickness=1,
        )
    else:
        # Point cloud fallback
        for i in range(len(px)):
            xi, yi = int(px[i]), int(py[i])
            if 0 <= xi < w and 0 <= yi < h:
                cv2.circle(canvas, (xi, yi), 1, mesh_color, -1)

    return canvas


# ---------------------------------------------------------------------------
# pyrender-based mesh renderer
# ---------------------------------------------------------------------------

def _check_pyrender() -> bool:
    """Return True if pyrender + trimesh are importable."""
    try:
        import pyrender  # noqa: F401
        import trimesh   # noqa: F401
        return True
    except ImportError:
        return False


def _render_mesh_pyrender(
    verts: np.ndarray,
    faces: Optional[np.ndarray],
    width: int,
    height: int,
    mesh_color_rgb: Tuple[float, float, float] = _MESH_RGB_F,
    camera_distance: float = 2.5,
    rotation_y_deg: float = 0.0,
    bg_color: Tuple[float, float, float, float] = (1.0, 1.0, 1.0, 0.0),
) -> Optional[np.ndarray]:
    """Render the SMPL mesh using pyrender (Phong shading).

    Args:
        verts: ``(N, 3)`` vertex array.
        faces: ``(F, 3)`` face array.
        width: Render canvas width.
        height: Render canvas height.
        mesh_color_rgb: Float RGB mesh colour.
        camera_distance: Camera distance from origin.
        rotation_y_deg: Y-axis rotation for 3/4 view.
        bg_color: RGBA background colour (transparent = (1,1,1,0)).

    Returns:
        BGR ``(H, W, 3)`` rendered image or ``None`` on failure.
    """
    try:
        import pyrender
        import trimesh

        v = verts.copy()

        # Y-axis rotation for 3/4 view
        angle = math.radians(rotation_y_deg)
        Ry = np.array([
            [math.cos(angle), 0, math.sin(angle), 0],
            [0,               1, 0,               0],
            [-math.sin(angle),0, math.cos(angle), 0],
            [0,               0, 0,               1],
        ])

        # Centre mesh at origin
        v -= v.mean(axis=0)

        body_h = v[:, 1].max() - v[:, 1].min() + 1e-6
        v = v / body_h  # normalise height to 1 unit

        if faces is not None and len(faces) > 0:
            mesh_tm = trimesh.Trimesh(
                vertices=v,
                faces=faces,
                process=False,
            )
            py_mesh = pyrender.Mesh.from_trimesh(
                mesh_tm,
                material=pyrender.MetallicRoughnessMaterial(
                    baseColorFactor=[*mesh_color_rgb, 1.0],
                    metallicFactor=0.0,
                    roughnessFactor=0.8,
                    alphaMode="OPAQUE",
                ),
                smooth=True,
            )
        else:
            # Point cloud
            pts = trimesh.PointCloud(v)
            colors = np.tile(
                [int(c * 255) for c in mesh_color_rgb] + [255],
                (len(v), 1),
            )
            py_mesh = pyrender.Mesh.from_points(v, colors=colors)

        scene = pyrender.Scene(
            bg_color=list(bg_color),
            ambient_light=[0.4, 0.4, 0.4],
        )
        mesh_node = scene.add(py_mesh)

        # Apply Y rotation
        mesh_node.matrix = Ry

        # Camera
        cam = pyrender.PerspectiveCamera(yfov=math.pi / 4.0, aspectRatio=width / height)
        cam_pose = np.eye(4)
        cam_pose[2, 3] = camera_distance
        scene.add(cam, pose=cam_pose)

        # Directional light from above-front-right
        light = pyrender.DirectionalLight(color=[1.0, 1.0, 1.0], intensity=3.0)
        light_pose = np.eye(4)
        light_pose[:3, :3] = np.array([
            [1, 0, 0],
            [0, 0.866, -0.5],
            [0, 0.5, 0.866],
        ])
        scene.add(light, pose=light_pose)

        r = pyrender.OffscreenRenderer(viewport_width=width, viewport_height=height)
        try:
            color, _depth = r.render(scene, flags=pyrender.RenderFlags.RGBA)
        finally:
            r.delete()

        # RGBA → BGR
        bgr = color[:, :, :3][:, :, ::-1].copy()
        alpha = color[:, :, 3]

        return bgr, alpha

    except Exception as exc:
        log.warning("SMPLSideBySide: pyrender failed: {}", exc)
        return None, None


# ---------------------------------------------------------------------------
# Main renderer class
# ---------------------------------------------------------------------------

class SMPLSideBySideRenderer:
    """Renders the reference-image side-by-side SMPL mesh output.

    LEFT panel  = original video frame + grey SMPL mesh overlaid.
    RIGHT panel = white canvas + checkerboard floor + isolated grey mesh.

    Args:
        config: Visualization config dict (``cfg.visualization.smpl_sidebyside``).
        panel_w: Width of each panel in pixels.
        panel_h: Height of each panel in pixels.

    Example:
        >>> r = SMPLSideBySideRenderer(config=cfg)
        >>> r.initialize()
        >>> frame = r.render_frame(orig, smpl_results, bboxes, frame_idx=0)
    """

    def __init__(
        self,
        config: Optional[Dict[str, Any]] = None,
        panel_w: int = 960,
        panel_h: int = 540,
    ) -> None:
        cfg = config or {}
        self._pw: int = int(cfg.get("panel_width",  panel_w))
        self._ph: int = int(cfg.get("panel_height", panel_h))

        self._overlay_alpha: float = float(cfg.get("overlay_alpha", 0.72))
        self._mesh_bgr: Tuple[int, int, int] = tuple(
            int(v) for v in cfg.get("mesh_color_bgr", list(_MESH_BGR))
        )
        self._mesh_rgb_f: Tuple[float, ...] = tuple(
            v / 255.0 for v in self._mesh_bgr[::-1]
        )
        self._camera_distance: float = float(cfg.get("camera_distance", 2.5))
        self._rotation_y_deg: float  = float(cfg.get("rotation_y_deg", 25.0))
        self._floor_frac: float      = float(cfg.get("floor_frac", 0.30))
        self._bg_color: Tuple[int, int, int] = tuple(
            int(v) for v in cfg.get("bg_color", [245, 245, 245])
        )
        self._divider_color: Tuple[int, int, int] = tuple(
            int(v) for v in cfg.get("divider_color", [120, 120, 120])
        )
        self._show_labels: bool = bool(cfg.get("show_labels", True))
        self._label_height: int = int(cfg.get("label_height", 32))

        self._pyrender_ok: bool = False
        self._faces_cache: Dict[int, Optional[np.ndarray]] = {}  # track_id → faces

    def initialize(self) -> None:
        """Check pyrender availability."""
        self._pyrender_ok = _check_pyrender()
        backend = "pyrender" if self._pyrender_ok else "OpenCV projection"
        log.info("SMPLSideBySideRenderer initialised — backend={}", backend)

    # ------------------------------------------------------------------
    # Public frame-level API
    # ------------------------------------------------------------------

    def render_frame(
        self,
        frame: Frame,
        smpl_outputs: Dict[int, SMPLOutput],
        bboxes: Optional[Dict[int, Tuple[float, float, float, float]]] = None,
        frame_idx: int = 0,
        fps: float = 0.0,
    ) -> Frame:
        """Render one combined side-by-side frame.

        Args:
            frame: Original BGR video frame.
            smpl_outputs: ``{track_id: SMPLOutput}`` for this frame.
            bboxes: ``{track_id: (x1,y1,x2,y2)}`` detection boxes.
            frame_idx: Current frame index (for labels).
            fps: Pipeline FPS (unused, kept for API symmetry).

        Returns:
            Combined BGR frame of shape ``(ph, pw*2+2, 3)``.
        """
        bboxes = bboxes or {}

        # ── LEFT panel: original frame + mesh overlay ──────────────────
        left = resize_frame(frame, self._pw, self._ph).copy()
        for track_id, smpl_out in smpl_outputs.items():
            bbox = bboxes.get(track_id)
            left = self._overlay_mesh_on_frame(left, smpl_out, bbox)

        # ── RIGHT panel: white canvas + floor + isolated mesh ──────────
        right = np.full((self._ph, self._pw, 3), self._bg_color, dtype=np.uint8)
        _draw_checkerboard(right, floor_frac=self._floor_frac)

        for track_id, smpl_out in smpl_outputs.items():
            faces = self._get_faces(track_id, smpl_out)
            right = self._render_isolated_mesh(right, smpl_out.vertices, faces)

        # ── Labels ────────────────────────────────────────────────────
        if self._show_labels:
            draw_header_bar(left,  "SMPL Mesh — Original",  height=self._label_height,
                            bg_color=(15, 15, 15), text_color=(220, 220, 220))
            draw_header_bar(right, "SMPL Mesh — Isolated 3D", height=self._label_height,
                            bg_color=(15, 15, 15), text_color=(220, 220, 220))

        # ── Divider + concatenate ─────────────────────────────────────
        div = np.full((self._ph, 2, 3), self._divider_color, dtype=np.uint8)
        return np.concatenate([left, div, right], axis=1)

    # ------------------------------------------------------------------
    # Video-level API
    # ------------------------------------------------------------------

    def render_video(
        self,
        frames: List[Tuple[Frame, int]],
        smpl_results: Dict[Tuple[int, int], SMPLOutput],
        tracks_per_frame: Dict[int, List[Any]],
        output_path: Union[str, Path],
        fps: float = 30.0,
        codec: str = "mp4v",
    ) -> Path:
        """Render the full side-by-side video.

        Args:
            frames: List of ``(BGR frame, frame_idx)`` tuples.
            smpl_results: ``{(frame_idx, track_id): SMPLOutput}``.
            tracks_per_frame: ``{frame_idx: List[Track]}`` for bbox extraction.
            output_path: Destination ``.mp4`` path.
            fps: Output video FPS.
            codec: OpenCV FourCC codec.

        Returns:
            Path to the written video file.
        """
        from tqdm import tqdm

        out_path = Path(output_path)
        out_w = self._pw * 2 + 2
        out_h = self._ph
        total = len(frames)

        log.info(
            "SMPLSideBySide: rendering {} frames → {}", total, out_path
        )

        with VideoWriter(out_path, fps=fps, width=out_w, height=out_h, codec=codec) as vw:
            for frame, frame_idx in tqdm(frames, desc="SMPL side-by-side", unit="frame"):
                smpl_f = {
                    tid: out
                    for (fi, tid), out in smpl_results.items()
                    if fi == frame_idx
                }
                tracks = tracks_per_frame.get(frame_idx, [])
                bboxes = {t.track_id: tuple(t.bbox.tolist()) for t in tracks}

                combined = self.render_frame(
                    frame, smpl_f, bboxes, frame_idx=frame_idx, fps=fps
                )
                vw.write(combined)

        log.info("SMPLSideBySide: video written → {}", out_path)
        return out_path

    # ------------------------------------------------------------------
    # Internal rendering helpers
    # ------------------------------------------------------------------

    def _get_faces(
        self, track_id: int, smpl_out: SMPLOutput
    ) -> Optional[np.ndarray]:
        """Retrieve or infer face connectivity for a track's SMPL output."""
        if track_id in self._faces_cache:
            return self._faces_cache[track_id]

        faces = None

        # Try to get faces from the model via the smpl_out (if attached)
        if hasattr(smpl_out, "_faces") and smpl_out._faces is not None:  # type: ignore[attr-defined]
            faces = smpl_out._faces  # type: ignore[attr-defined]
        else:
            # Reconstruct faces from the template NPZ if available
            tmpl = _DEFAULT_TEMPLATE
            if tmpl.exists():
                try:
                    data = np.load(str(tmpl))
                    faces = data["faces"].astype(np.int32)
                except Exception:
                    faces = None

        self._faces_cache[track_id] = faces
        return faces

    def _overlay_mesh_on_frame(
        self,
        frame: Frame,
        smpl_out: SMPLOutput,
        bbox: Optional[Tuple[float, float, float, float]],
    ) -> Frame:
        """Overlay grey SMPL mesh on original video frame (LEFT panel).

        Args:
            frame: BGR frame (modified in-place).
            smpl_out: SMPL output with vertices.
            bbox: Detection bbox for scale/position reference.

        Returns:
            Modified frame.
        """
        verts = smpl_out.vertices
        if verts is None or verts.size == 0 or np.allclose(verts, 0.0):
            return frame

        faces = self._faces_cache.get(smpl_out.track_id)
        if faces is None:
            # Lazy-load faces
            faces = self._get_faces(smpl_out.track_id, smpl_out)

        h, w = frame.shape[:2]

        # Project vertices to pixel space using camera_params (weak-perspective)
        cam = smpl_out.camera_params  # [scale, tx, ty]
        if cam is not None and cam.size >= 3 and abs(cam[0]) > 1e-6:
            scale = float(cam[0])
            tx    = float(cam[1])
            ty    = float(cam[2])

            v = verts.copy()
            v -= v.mean(axis=0)
            body_h = v[:, 1].max() - v[:, 1].min() + 1e-6
            v /= body_h

            # Weak-perspective projection: px = tx + scale * X
            px_ndc = tx + scale * v[:, 0]
            py_ndc = ty + scale * v[:, 1]

            # NDC → pixel
            px_px = (px_ndc + 1.0) / 2.0 * w
            py_px = (1.0 - (py_ndc + 1.0) / 2.0) * h
        else:
            # Fall back to bbox-based projection
            if bbox is not None:
                x1, y1, x2, y2 = [float(v2) for v2 in bbox]
                bw = max(x2 - x1, 1.0)
                bh = max(y2 - y1, 1.0)
                cx = (x1 + x2) / 2.0
                cy_pel = y1 + bh * 0.55

                v = verts.copy()
                v -= v.mean(axis=0)
                body_h = v[:, 1].max() - v[:, 1].min() + 1e-6
                v /= body_h

                px_px = cx + v[:, 0] * bh * 0.45
                py_px = cy_pel - v[:, 1] * bh * 0.90
            else:
                return frame

        # Build a pixel-space vertex array
        verts_2d = np.column_stack([px_px, py_px, verts[:, 2]])
        overlay = frame.copy()
        self._draw_mesh_on_image(overlay, verts_2d, faces)
        cv2.addWeighted(overlay, self._overlay_alpha, frame, 1.0 - self._overlay_alpha, 0, frame)
        return frame

    def _draw_mesh_on_image(
        self,
        img: Frame,
        verts_2d: np.ndarray,   # (N, 3): px, py, z
        faces: Optional[np.ndarray],
    ) -> None:
        """Draw filled triangles onto img (in-place) using painter's algorithm."""
        h, w = img.shape[:2]
        px = verts_2d[:, 0]
        py = verts_2d[:, 1]
        pz = verts_2d[:, 2] if verts_2d.shape[1] > 2 else np.zeros(len(verts_2d))

        if faces is not None and len(faces) > 0:
            # Per-face depth
            fz = (pz[faces[:, 0]] + pz[faces[:, 1]] + pz[faces[:, 2]]) / 3.0
            order = np.argsort(fz)[::-1]
            br, bg, bb = self._mesh_bgr[2], self._mesh_bgr[1], self._mesh_bgr[0]
            z_min, z_max = fz.min(), fz.max()
            z_span = max(z_max - z_min, 1e-6)

            for fi in order:
                tri = faces[fi]
                pts = np.array([
                    [int(px[tri[0]]), int(py[tri[0]])],
                    [int(px[tri[1]]), int(py[tri[1]])],
                    [int(px[tri[2]]), int(py[tri[2]])],
                ], dtype=np.int32)

                if (pts[:, 0].max() < 0 or pts[:, 0].min() >= w or
                        pts[:, 1].max() < 0 or pts[:, 1].min() >= h):
                    continue

                depth_t = (fz[fi] - z_min) / z_span
                shade = 0.70 + depth_t * 0.30
                col = (
                    int(np.clip(bb * shade, 0, 255)),
                    int(np.clip(bg * shade, 0, 255)),
                    int(np.clip(br * shade, 0, 255)),
                )
                cv2.fillPoly(img, [pts], col)
        else:
            # Point cloud
            for i in range(len(px)):
                xi, yi = int(px[i]), int(py[i])
                if 0 <= xi < w and 0 <= yi < h:
                    cv2.circle(img, (xi, yi), 2, self._mesh_bgr, -1)

    def _render_isolated_mesh(
        self,
        canvas: Frame,
        verts: np.ndarray,
        faces: Optional[np.ndarray],
    ) -> Frame:
        """Render isolated 3D mesh on the right-panel canvas.

        Tries pyrender first; falls back to OpenCV projection.

        Args:
            canvas: BGR canvas (white + checkerboard already drawn).
            verts: ``(N, 3)`` vertex array.
            faces: ``(F, 3)`` face array or ``None``.

        Returns:
            Modified canvas.
        """
        if verts is None or verts.size == 0 or np.allclose(verts, 0.0):
            return canvas

        h, w = canvas.shape[:2]

        if self._pyrender_ok:
            bgr_render, alpha = _render_mesh_pyrender(
                verts, faces,
                width=w, height=h,
                mesh_color_rgb=self._mesh_rgb_f,
                camera_distance=self._camera_distance,
                rotation_y_deg=self._rotation_y_deg,
                bg_color=(1.0, 1.0, 1.0, 0.0),
            )
            if bgr_render is not None and alpha is not None:
                # Composite over canvas: only where alpha > 0
                mask = alpha > 10
                canvas[mask] = (
                    bgr_render[mask] * self._overlay_alpha
                    + canvas[mask] * (1.0 - self._overlay_alpha)
                ).astype(np.uint8)
                return canvas
            # pyrender failed silently → fall through

        # OpenCV fallback
        # Reserve bottom floor_frac for checkerboard — shift mesh up
        cam_y_offset = self._floor_frac * 0.8
        _render_mesh_opencv(
            verts, faces, canvas,
            mesh_color=self._mesh_bgr,
            rotation_y_deg=self._rotation_y_deg,
            camera_y_offset=cam_y_offset,
        )
        return canvas


# ---------------------------------------------------------------------------
# Module-level path reference for face loading
# ---------------------------------------------------------------------------
_DEFAULT_TEMPLATE = Path(__file__).parent.parent / "data" / "models" / "smpl" / "neutral_mesh.npz"
