"""
HumanMM — Video / GIF Exporter.

Thin orchestration layer over :class:`utils.video_writer.VideoWriter` and
:func:`utils.io_utils.save_gif`, used by :class:`pipeline.output_writer.OutputWriter`
to materialise each visualization stage's frame list (or a streaming
generator) into a final video/GIF file under the output directory.

Example:
    >>> from exporters.video_exporter import VideoExporter
    >>> exporter = VideoExporter(output_dir="outputs", fps=30, codec="mp4v")
    >>> exporter.export_frames("07_final.mp4", frame_list, width=1280, height=720)
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, Iterator, List, Optional, Union

import numpy as np

from utils.frame_utils import Frame
from utils.io_utils import save_gif
from utils.logger import get_logger
from utils.video_writer import VideoWriter

log = get_logger(__name__)


class VideoExporter:
    """Writes rendered frame sequences to MP4 (and optionally GIF) files.

    Args:
        output_dir: Root output directory (mirrors ``cfg.output.root_dir``).
        fps: Output video frame rate.
        codec: OpenCV FourCC codec string (mirrors ``cfg.output.video_codec``).

    Example:
        >>> exporter = VideoExporter(output_dir="outputs", fps=30)
        >>> path = exporter.export_frames("02_detection.mp4", frames, 1280, 720)
    """

    def __init__(
        self,
        output_dir: Union[str, Path] = "outputs",
        fps: float = 30.0,
        codec: str = "mp4v",
    ) -> None:
        self._root = Path(output_dir)
        self._root.mkdir(parents=True, exist_ok=True)
        self._fps = fps
        self._codec = codec

    def export_frames(
        self,
        filename: str,
        frames: Iterable[Frame],
        width: int,
        height: int,
    ) -> Path:
        """Write a sequence (list or generator) of BGR frames to an MP4 file.

        Accepts either a materialised list or a streaming generator, so
        visualization stages can write frames incrementally without
        holding the entire rendered video in memory.

        Args:
            filename: Output filename (relative to ``output_dir``).
            frames: Iterable of BGR frame arrays.
            width: Output video width in pixels.
            height: Output video height in pixels.

        Returns:
            Path to the written video file.

        Raises:
            RuntimeError: If the video writer cannot be opened.

        Example:
            >>> exporter.export_frames("03_tracking.mp4", tracking_frames, 1280, 720)
        """
        path = self._root / filename
        count = 0

        with VideoWriter(path, fps=self._fps, width=width, height=height, codec=self._codec) as vw:
            for frame in frames:
                vw.write(frame)
                count += 1

        log.info("VideoExporter: wrote {} frames → {}", count, path)
        return path

    def export_gif(
        self,
        filename: str,
        frames: List[Frame],
        gif_fps: int = 15,
        max_frames: int = 100,
    ) -> Path:
        """Write a (subsampled) frame sequence to an animated GIF.

        Args:
            filename: Output filename (relative to ``output_dir``).
            frames: List of BGR frame arrays.
            gif_fps: GIF playback frame rate.
            max_frames: Maximum number of frames to include (uniformly
                subsampled if ``frames`` is longer).

        Returns:
            Path to the written GIF file.

        Raises:
            ValueError: If ``frames`` is empty.
        """
        if not frames:
            raise ValueError("frames list must not be empty for GIF export")

        if len(frames) > max_frames:
            indices = np.linspace(0, len(frames) - 1, max_frames).astype(int)
            sampled = [frames[i] for i in indices]
        else:
            sampled = frames

        path = self._root / filename
        save_gif(sampled, path, fps=gif_fps)
        return path

    def export_thumbnail_strip(
        self,
        filename: str,
        frames: List[Frame],
        num_thumbnails: int = 6,
        thumb_width: int = 200,
    ) -> Path:
        """Export a horizontal contact-sheet thumbnail strip as a JPEG.

        Useful as a quick visual summary of an entire run without opening
        any of the full-length output videos.

        Args:
            filename: Output filename (relative to ``output_dir``).
            frames: Full list of rendered frames to sample from.
            num_thumbnails: Number of evenly-spaced thumbnails to include.
            thumb_width: Width of each thumbnail panel in pixels.

        Returns:
            Path to the written JPEG file.

        Raises:
            ValueError: If ``frames`` is empty.
        """
        if not frames:
            raise ValueError("frames list must not be empty for thumbnail strip export")

        from utils.frame_utils import resize_frame, save_frame, stack_frames_side_by_side

        n = min(num_thumbnails, len(frames))
        indices = np.linspace(0, len(frames) - 1, n).astype(int)

        thumbs = []
        for i in indices:
            f = frames[i]
            h, w = f.shape[:2]
            scale = thumb_width / w
            thumbs.append(resize_frame(f, thumb_width, int(h * scale)))

        strip = stack_frames_side_by_side(thumbs, gap=4, gap_color=20)
        path = self._root / filename
        save_frame(strip, path, quality=90)
        log.info("VideoExporter: thumbnail strip ({} frames) → {}", n, path)
        return path
