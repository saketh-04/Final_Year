"""
HumanMM — Visualizer (CVPR Research Demo Layout).

Produces stage videos matching the reference research-demo format:

  02_detection.mp4   : Tight YOLO boxes on original video
  03_tracking.mp4    : ByteTrack boxes + trails on original video
  04_pose.mp4        : Skeleton overlay on original video
  05_mesh.mp4        : 3D mesh / joint projection on original video
  06_trajectory.mp4  : Animated 3D trajectory plot
  07_final.mp4       : Research-demo side-by-side
                         LEFT  = tracking boxes on colour video
                         RIGHT = skeleton on white canvas
  comparison.mp4     : Full 4-panel grid (Original | Detection | Pose | Mesh)

All public APIs unchanged.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

from models.bytetrack_tracker import Track
from models.gvhmr_wrapper import SMPLOutput
from models.mediapipe_pose import PersonPose
from models.yolo_detector import Detection
from pipeline.shot_detector import ShotSegment
from pipeline.trajectory_aligner import AlignedTrackResult
from utils.frame_utils import Frame
from utils.logger import get_logger
from utils.timer import FPSCounter
from utils.video_writer import VideoWriter
from visualization.mesh_renderer import MeshRenderer
from visualization.overlay_renderer import (
    ComparisonRenderer, OverlayRenderer, ResearchDemoRenderer,
)
from visualization.skeleton_renderer import SkeletonRenderer
from visualization.smpl_side_by_side import SMPLSideBySideRenderer
from visualization.trajectory_renderer import TrajectoryRenderer
from visualization.video_renderer import DetectionRenderer, TrackingRenderer

log = get_logger(__name__)


class Visualizer:
    """Produces every visualization stage video for a full HumanMM run.

    Args:
        config: Visualization config dict (mirrors ``cfg.visualization``).
        output_dir: Root output directory.
        pose_skeleton_pairs: Joint connectivity for skeleton rendering.

    Example:
        >>> viz = Visualizer(config=cfg.visualization, output_dir="outputs")
        >>> viz.render_all(frames, detections, tracks, poses, smpl, aligned, shots)
    """

    def __init__(
        self,
        config: Optional[Dict[str, Any]] = None,
        output_dir: Union[str, Path] = "outputs",
        pose_skeleton_pairs: Optional[List[Tuple[int, int]]] = None,
    ) -> None:
        self._cfg: Dict[str, Any] = config or {}
        self._root = Path(output_dir)
        self._root.mkdir(parents=True, exist_ok=True)

        global_cfg = self._cfg.get("global", {})
        self._codec: str   = global_cfg.get("video_codec", "mp4v")
        self._out_fps: float = global_cfg.get("output_fps", 30.0)
        self._show_fps: bool = global_cfg.get("show_fps", True)

        det_cfg   = self._cfg.get("detection", {})
        track_cfg = self._cfg.get("tracking", {})
        pose_cfg  = self._cfg.get("pose", {})
        mesh_cfg  = self._cfg.get("mesh", {})
        traj_cfg  = self._cfg.get("trajectory", {})
        comp_cfg  = self._cfg.get("comparison", {})

        smpl_sbs_cfg = self._cfg.get("smpl_sidebyside", {})

        self._det_renderer    = DetectionRenderer(det_cfg)
        self._track_renderer  = TrackingRenderer(track_cfg)
        self._skeleton_renderer = SkeletonRenderer(pose_cfg, skeleton_pairs=pose_skeleton_pairs)
        self._mesh_renderer   = MeshRenderer(mesh_cfg)
        self._traj_renderer   = TrajectoryRenderer(traj_cfg)
        self._overlay         = OverlayRenderer(self._cfg)
        self._comp_renderer   = ComparisonRenderer(comp_cfg)

        # Research demo side-by-side renderer (07_final)
        self._demo_renderer = ResearchDemoRenderer(comp_cfg)

        # SMPL side-by-side renderer (08_smpl_sidebyside) — reference image output
        self._smpl_sbs_renderer = SMPLSideBySideRenderer(
            config=smpl_sbs_cfg,
            panel_w=int(smpl_sbs_cfg.get("panel_width",  self._cfg.get("global", {}).get("output_width",  1280) // 2)),
            panel_h=int(smpl_sbs_cfg.get("panel_height", self._cfg.get("global", {}).get("output_height", 720))),
        )
        self._smpl_sbs_renderer.initialize()
        self._smpl_sbs_enabled: bool = smpl_sbs_cfg.get("enabled", True)

        self._detection_enabled:    bool = det_cfg.get("enabled", True)
        self._tracking_enabled:     bool = track_cfg.get("enabled", True)
        self._pose_enabled:         bool = pose_cfg.get("enabled", True)
        self._mesh_enabled:         bool = mesh_cfg.get("enabled", True)
        self._trajectory_enabled:   bool = traj_cfg.get("enabled", True)
        self._comparison_enabled:   bool = comp_cfg.get("enabled", True)

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------

    def render_all(
        self,
        frames: List[Tuple[Frame, int]],
        detections_per_frame: Dict[int, List[Detection]],
        tracks_per_frame: Dict[int, List[Track]],
        poses_per_frame: Dict[int, Dict[int, PersonPose]],
        smpl_results: Dict[Tuple[int, int], SMPLOutput],
        aligned_results: Dict[int, AlignedTrackResult],
        shot_segments: List[ShotSegment],
    ) -> Dict[str, Path]:
        """Render every visualization stage video.

        Args:
            frames: List of ``(BGR frame, frame_idx)`` tuples.
            detections_per_frame: ``frame_idx → List[Detection]``.
            tracks_per_frame: ``frame_idx → List[Track]``.
            poses_per_frame: ``frame_idx → {track_id → PersonPose}``.
            smpl_results: ``(frame_idx, track_id) → SMPLOutput``.
            aligned_results: ``track_id → AlignedTrackResult``.
            shot_segments: Shot boundary list.

        Returns:
            Dict mapping stage name → written video path.
        """
        if not frames:
            log.warning("Visualizer.render_all: no frames — skipping")
            return {}

        h, w = frames[0][0].shape[:2]
        written: Dict[str, Path] = {}

        boundary_frames = [s.start_frame for s in shot_segments[1:]] if shot_segments else []
        self._overlay.set_shot_boundaries(boundary_frames)

        writers: Dict[str, VideoWriter] = {}

        def _writer(name: str, out_w: int = w, out_h: int = h) -> VideoWriter:
            if name not in writers:
                vw = VideoWriter(
                    self._root / f"{name}.mp4",
                    fps=self._out_fps, width=out_w, height=out_h,
                    codec=self._codec,
                )
                vw.open()
                writers[name] = vw
            return writers[name]

        fps_counter = FPSCounter(window_size=30)

        # Determine demo panel size from first frame
        panel_w = w
        panel_h = h
        self._demo_renderer._panel_w = panel_w
        self._demo_renderer._panel_h = panel_h

        for frame, frame_idx in frames:
            fps_counter.tick()
            fps = fps_counter.fps or self._out_fps

            dets    = detections_per_frame.get(frame_idx, [])
            tracks  = tracks_per_frame.get(frame_idx, [])
            poses   = poses_per_frame.get(frame_idx, {})
            smpl_f  = {tid: out for (fi, tid), out in smpl_results.items() if fi == frame_idx}
            bboxes  = {t.track_id: tuple(t.bbox.tolist()) for t in tracks}

            det_frame  = None
            pose_frame = None
            mesh_frame = None

            # Stage 02 — Detection
            if self._detection_enabled:
                det_frame = frame.copy()
                self._det_renderer.render(det_frame, dets, frame_idx=frame_idx, fps=fps)
                _writer("02_detection").write(det_frame)

            # Stage 03 — Tracking
            if self._tracking_enabled:
                track_frame = frame.copy()
                self._track_renderer.render(
                    track_frame, tracks, frame_idx=frame_idx, fps=fps
                )
                _writer("03_tracking").write(track_frame)

            # Stage 04 — Pose
            if self._pose_enabled:
                pose_frame = frame.copy()
                self._skeleton_renderer.render(pose_frame, poses)
                if self._show_fps:
                    self._overlay.apply_fps(pose_frame, fps)
                _writer("04_pose").write(pose_frame)

            # Stage 05 — Mesh
            if self._mesh_enabled:
                mesh_frame = frame.copy()
                self._mesh_renderer.render(mesh_frame, smpl_f, bboxes)
                if self._show_fps:
                    self._overlay.apply_fps(mesh_frame, fps)
                _writer("05_mesh").write(mesh_frame)

            # Stage 07 — Research demo side-by-side
            # LEFT: tracking boxes on original video
            # RIGHT: skeleton stick-figure on white canvas
            track_frame_07 = frame.copy()
            self._track_renderer.render(
                track_frame_07, tracks, frame_idx=frame_idx, fps=fps
            )
            demo_frame = self._demo_renderer.compose(
                track_frame_07, poses=poses, frame_idx=frame_idx, fps=fps
            )
            dw, dh = demo_frame.shape[1], demo_frame.shape[0]
            _writer("07_final", out_w=dw, out_h=dh).write(demo_frame)

            # Stage 08 — SMPL Side-by-Side (reference image output)
            # LEFT: original frame + grey SMPL mesh overlay
            # RIGHT: white canvas + checkerboard floor + isolated 3D mesh
            if self._smpl_sbs_enabled and smpl_f:
                sbs_frame = self._smpl_sbs_renderer.render_frame(
                    frame, smpl_f, bboxes, frame_idx=frame_idx, fps=fps
                )
                sw, sh = sbs_frame.shape[1], sbs_frame.shape[0]
                _writer("08_smpl_sidebyside", out_w=sw, out_h=sh).write(sbs_frame)

            # Comparison 4-panel
            if self._comparison_enabled:
                grid = self._comp_renderer.compose(
                    original=frame,
                    detection=det_frame if det_frame is not None else frame,
                    pose=pose_frame if pose_frame is not None else frame,
                    mesh=mesh_frame if mesh_frame is not None else frame,
                )
                gw, gh = grid.shape[1], grid.shape[0]
                _writer("comparison", out_w=gw, out_h=gh).write(grid)

        # Close all writers
        for name, vw in writers.items():
            vw.release()
            written[name] = vw.path

        # Stage 06 — Trajectory video (animated)
        if self._trajectory_enabled and aligned_results:
            try:
                traj_path = self._render_trajectory_video(
                    aligned_results, shot_segments, frames, w, h
                )
                written["06_trajectory"] = traj_path
            except Exception as exc:
                log.warning("Trajectory video render failed: {}", exc)

        log.info(
            "Visualizer: rendered {} stage videos → {}", len(written), self._root
        )
        return written

    # ------------------------------------------------------------------
    # Trajectory video (animated, subsampled for performance)
    # ------------------------------------------------------------------

    def _render_trajectory_video(
        self,
        aligned_results: Dict[int, AlignedTrackResult],
        shot_segments: List[ShotSegment],
        frames: List[Tuple[Frame, int]],
        width: int,
        height: int,
    ) -> Path:
        path = self._root / "06_trajectory.mp4"
        step = max(1, len(frames) // 150)
        last_plot: Optional[np.ndarray] = None

        with VideoWriter(
            path, fps=self._out_fps, width=width, height=height, codec=self._codec
        ) as vw:
            for i, (_, frame_idx) in enumerate(frames):
                if i % step == 0 or last_plot is None:
                    last_plot = self._traj_renderer.render_frame(
                        aligned_results, shot_segments,
                        up_to_frame=frame_idx, width=width, height=height,
                    )
                vw.write(last_plot)

        return path
