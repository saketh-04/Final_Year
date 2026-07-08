"""
HumanMM — Trajectory Renderer.

Renders 3D (and 2D-projected) trajectory plots from
:class:`~pipeline.trajectory_aligner.AlignedTrackResult` data using
Matplotlib.  Plots can be saved as standalone static figures or converted
to BGR frames suitable for writing into a video via
:class:`utils.video_writer.VideoWriter`.

Example:
    >>> from visualization.trajectory_renderer import TrajectoryRenderer
    >>> renderer = TrajectoryRenderer(config=cfg.visualization.trajectory)
    >>> renderer.save_static_plot(aligned_results, shots, "outputs/trajectory.png")
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import numpy as np

from pipeline.shot_detector import ShotSegment
from pipeline.trajectory_aligner import AlignedTrackResult
from utils.frame_utils import Frame, resize_frame
from utils.image_utils import color_for_id
from utils.logger import get_logger

log = get_logger(__name__)


def _bgr_to_mpl(bgr: tuple) -> tuple:
    """Convert a ``(B, G, R)`` 0-255 tuple to a Matplotlib RGB 0-1 tuple."""
    b, g, r = bgr
    return (r / 255.0, g / 255.0, b / 255.0)


class TrajectoryRenderer:
    """Renders 3D/2D trajectory visualisations using Matplotlib.

    Args:
        config: Visualization config dict (mirrors
            ``cfg.visualization.trajectory``).

    Example:
        >>> renderer = TrajectoryRenderer(config=cfg.visualization.trajectory)
        >>> frame = renderer.render_frame(aligned_results, shots, up_to_frame=120)
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        self._cfg: Dict[str, Any] = config or {}

        self._plot_3d: bool = self._cfg.get("plot_3d", True)
        self._plot_2d_projections: bool = self._cfg.get("plot_2d_projections", True)
        self._show_shot_boundaries: bool = self._cfg.get("show_shot_boundaries", True)
        self._color_by_person: bool = self._cfg.get("color_by_person", True)
        self._line_width: float = self._cfg.get("line_width", 2)
        self._marker_size: float = self._cfg.get("marker_size", 4)
        self._dpi: int = self._cfg.get("figure_dpi", 150)

    # ------------------------------------------------------------------
    # Static export
    # ------------------------------------------------------------------

    def save_static_plot(
        self,
        aligned_results: Dict[int, AlignedTrackResult],
        shot_segments: Optional[List[ShotSegment]] = None,
        output_path: Union[str, Path] = "outputs/trajectory.png",
    ) -> None:
        """Render and save a complete (final) trajectory figure to disk.

        Produces a 3D trajectory panel and, optionally, three 2D projection
        panels (XY, XZ, YZ).

        Args:
            aligned_results: Dict mapping ``track_id → AlignedTrackResult``.
            shot_segments: Optional shot list, used to annotate cut points.
            output_path: Destination PNG path.

        Raises:
            OSError: If the parent directory cannot be created.
        """
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        n_panels = 1 + (3 if self._plot_2d_projections else 0)
        fig = plt.figure(figsize=(6 * n_panels, 5), dpi=self._dpi)

        panel_idx = 1
        if self._plot_3d:
            ax3d = fig.add_subplot(1, n_panels, panel_idx, projection="3d")
            self._draw_3d(ax3d, aligned_results, shot_segments)
            panel_idx += 1

        if self._plot_2d_projections:
            for (dims, title) in [((0, 1), "XY"), ((0, 2), "XZ"), ((1, 2), "YZ")]:
                ax2d = fig.add_subplot(1, n_panels, panel_idx)
                self._draw_2d_projection(ax2d, aligned_results, dims, title)
                panel_idx += 1

        fig.tight_layout()
        fig.savefig(str(output_path))
        plt.close(fig)
        log.info("Trajectory plot saved → {}", output_path)

    def save_per_person_plots(
        self,
        aligned_results: Dict[int, AlignedTrackResult],
        output_dir: Union[str, Path] = "outputs/trajectories",
    ) -> None:
        """Save one standalone 3D trajectory plot per tracked person.

        Args:
            aligned_results: Dict mapping ``track_id → AlignedTrackResult``.
            output_dir: Directory in which to save ``person_<id>.png`` files.
        """
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        for track_id, result in aligned_results.items():
            fig = plt.figure(figsize=(6, 5), dpi=self._dpi)
            ax = fig.add_subplot(111, projection="3d")
            self._draw_3d(ax, {track_id: result}, None)
            fig.tight_layout()
            out_path = output_dir / f"person_{track_id}.png"
            fig.savefig(str(out_path))
            plt.close(fig)
            log.debug("Per-person trajectory plot saved → {}", out_path)

    # ------------------------------------------------------------------
    # Frame export (for video rendering)
    # ------------------------------------------------------------------

    def render_frame(
        self,
        aligned_results: Dict[int, AlignedTrackResult],
        shot_segments: Optional[List[ShotSegment]],
        up_to_frame: Optional[int] = None,
        width: int = 960,
        height: int = 720,
    ) -> Frame:
        """Render the trajectory plot as a single BGR frame (for video export).

        Args:
            aligned_results: Dict mapping ``track_id → AlignedTrackResult``.
            shot_segments: Optional shot list for boundary annotation.
            up_to_frame: If given, only draws trajectory points with
                ``frame_idx <= up_to_frame`` — enables animated playback.
            width: Output frame width in pixels.
            height: Output frame height in pixels.

        Returns:
            BGR frame array of shape ``(height, width, 3)``.
        """
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig = plt.figure(figsize=(width / 100.0, height / 100.0), dpi=100)
        ax = fig.add_subplot(111, projection="3d")

        clipped = aligned_results
        if up_to_frame is not None:
            clipped = {
                tid: self._clip_result(res, up_to_frame)
                for tid, res in aligned_results.items()
            }

        self._draw_3d(ax, clipped, shot_segments)
        fig.canvas.draw()

        buf = np.asarray(fig.canvas.buffer_rgba())
        rgb = buf[:, :, :3]
        frame = rgb[:, :, ::-1].copy()  # RGB -> BGR
        plt.close(fig)

        return resize_frame(frame, width, height)

    # ------------------------------------------------------------------
    # Internal drawing helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _clip_result(result: AlignedTrackResult, up_to_frame: int) -> AlignedTrackResult:
        """Return a copy of ``result`` truncated to frames ``<= up_to_frame``."""
        frame_arr = np.array(result.frame_indices)
        mask = frame_arr <= up_to_frame
        if not mask.any():
            mask[:1] = True

        return AlignedTrackResult(
            track_id=result.track_id,
            frame_indices=list(frame_arr[mask]),
            joints_3d_aligned=result.joints_3d_aligned[mask],
            translations_aligned=result.translations_aligned[mask],
            rotations_aligned=result.rotations_aligned[mask],
            shot_boundaries=result.shot_boundaries,
            smoothing_applied=result.smoothing_applied,
        )

    def _draw_3d(
        self,
        ax: Any,
        aligned_results: Dict[int, AlignedTrackResult],
        shot_segments: Optional[List[ShotSegment]],
    ) -> None:
        """Draw all tracks' 3D root trajectories onto a 3D Matplotlib axis."""
        for track_id, result in aligned_results.items():
            trans = result.translations_aligned
            if trans.shape[0] == 0:
                continue
            color = _bgr_to_mpl(color_for_id(track_id)) if self._color_by_person else "tab:blue"
            ax.plot(
                trans[:, 0], trans[:, 1], trans[:, 2],
                color=color, linewidth=self._line_width, label=f"Person {track_id}",
            )
            ax.scatter(
                trans[-1, 0], trans[-1, 1], trans[-1, 2],
                color=color, s=self._marker_size * 10,
            )

            if self._show_shot_boundaries and shot_segments:
                frame_arr = np.array(result.frame_indices)
                for sb in result.shot_boundaries:
                    idx = int(np.argmin(np.abs(frame_arr - sb)))
                    ax.scatter(
                        trans[idx, 0], trans[idx, 1], trans[idx, 2],
                        color="red", marker="x", s=40,
                    )

        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Z")
        ax.set_title("3D Root Trajectories")
        if aligned_results:
            ax.legend(loc="upper right", fontsize=8)

    def _draw_2d_projection(
        self,
        ax: Any,
        aligned_results: Dict[int, AlignedTrackResult],
        dims: tuple,
        title: str,
    ) -> None:
        """Draw a 2D projection of all tracks' trajectories onto ``dims``."""
        d0, d1 = dims
        for track_id, result in aligned_results.items():
            trans = result.translations_aligned
            if trans.shape[0] == 0:
                continue
            color = _bgr_to_mpl(color_for_id(track_id)) if self._color_by_person else "tab:blue"
            ax.plot(trans[:, d0], trans[:, d1], color=color, linewidth=self._line_width)

        labels = ["X", "Y", "Z"]
        ax.set_xlabel(labels[d0])
        ax.set_ylabel(labels[d1])
        ax.set_title(f"{title} Projection")
        ax.set_aspect("equal", adjustable="datalim")
