"""
HumanMM — High-Level Video Utility Module.

Convenience helpers that sit one layer above :mod:`utils.video_writer` and
:class:`pipeline.video_loader.VideoLoader`: probing, thumbnail extraction,
side-by-side / grid concatenation of already-rendered stage videos, and
audio-preserving re-muxing.  Pipeline stages that only need raw frame I/O
should use ``VideoLoader``/``VideoWriter`` directly; this module is for
post-hoc operations on whole video *files*.

Example:
    >>> from utils.video_utils import probe_video, concat_videos_horizontally
    >>> info = probe_video("outputs/07_final.mp4")
    >>> concat_videos_horizontally(["a.mp4", "b.mp4"], "side_by_side.mp4")
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Union

import cv2
import numpy as np

from utils.frame_utils import Frame, resize_frame, stack_frames_side_by_side, stack_frames_2x2
from utils.logger import get_logger
from utils.video_writer import VideoWriter, get_video_properties

log = get_logger(__name__)


def probe_video(path: Union[str, Path]) -> Dict[str, object]:
    """Read basic metadata from a video file (thin wrapper for readability).

    Args:
        path: Path to the video file.

    Returns:
        Dict with ``width``, ``height``, ``fps``, ``frame_count``,
        ``duration_sec``, and ``codec`` keys.

    Raises:
        FileNotFoundError: If the video does not exist.
        RuntimeError: If OpenCV cannot open the file.
    """
    return get_video_properties(path)


def extract_thumbnail(
    video_path: Union[str, Path],
    frame_idx: int = 0,
    output_path: Optional[Union[str, Path]] = None,
) -> Frame:
    """Extract a single frame from a video as a thumbnail image.

    Args:
        video_path: Path to the source video.
        frame_idx: Zero-based index of the frame to extract.
        output_path: If provided, the thumbnail is also saved to disk.

    Returns:
        BGR frame array of the extracted thumbnail.

    Raises:
        FileNotFoundError: If the video does not exist.
        RuntimeError: If the frame cannot be read.
    """
    video_path = Path(video_path)
    if not video_path.exists():
        raise FileNotFoundError(f"Video not found: {video_path}")

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"OpenCV cannot open video: {video_path}")

    try:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        if not ret:
            raise RuntimeError(f"Failed to read frame {frame_idx} from {video_path}")
    finally:
        cap.release()

    if output_path:
        from utils.frame_utils import save_frame

        save_frame(frame, output_path)
        log.debug("Thumbnail saved → {}", output_path)

    return frame


def video_to_frames(
    video_path: Union[str, Path],
    max_frames: int = -1,
) -> List[Frame]:
    """Load all (or up to ``max_frames``) frames of a video into memory.

    Prefer streaming via ``VideoLoader.frame_iterator`` for long videos —
    this helper materialises the whole list at once and is intended for
    short clips or already-rendered stage videos (e.g. building a
    comparison grid from finished outputs).

    Args:
        video_path: Path to the source video.
        max_frames: Maximum number of frames to read (``-1`` = all).

    Returns:
        List of BGR frame arrays.

    Raises:
        FileNotFoundError: If the video does not exist.
    """
    video_path = Path(video_path)
    if not video_path.exists():
        raise FileNotFoundError(f"Video not found: {video_path}")

    cap = cv2.VideoCapture(str(video_path))
    frames: List[Frame] = []
    try:
        while True:
            if max_frames > 0 and len(frames) >= max_frames:
                break
            ret, frame = cap.read()
            if not ret:
                break
            frames.append(frame)
    finally:
        cap.release()

    log.debug("Loaded {} frames from {}", len(frames), video_path.name)
    return frames


def concat_videos_horizontally(
    video_paths: List[Union[str, Path]],
    output_path: Union[str, Path],
    fps: Optional[float] = None,
    panel_height: int = 480,
) -> None:
    """Concatenate multiple videos side-by-side into a single output video.

    All input videos are resized to a common ``panel_height`` (preserving
    aspect ratio is not enforced — width is taken from each source frame
    scaled to match height) and stacked horizontally frame-by-frame.  Videos
    of differing lengths are padded by freezing on their final frame.

    Args:
        video_paths: List of input video file paths (≥ 2).
        output_path: Destination path for the combined video.
        fps: Output frame rate.  If ``None``, uses the first video's FPS.
        panel_height: Height (pixels) each panel is resized to.

    Raises:
        ValueError: If fewer than two videos are supplied.
        FileNotFoundError: If any input video does not exist.
    """
    if len(video_paths) < 2:
        raise ValueError("Need at least two videos to concatenate horizontally")

    caps = []
    for p in video_paths:
        p = Path(p)
        if not p.exists():
            raise FileNotFoundError(f"Video not found: {p}")
        caps.append(cv2.VideoCapture(str(p)))

    if fps is None:
        fps = caps[0].get(cv2.CAP_PROP_FPS) or 30.0

    last_frames: List[Optional[Frame]] = [None] * len(caps)
    writer: Optional[VideoWriter] = None

    try:
        while True:
            panels: List[Frame] = []
            any_alive = False

            for i, cap in enumerate(caps):
                ret, frame = cap.read()
                if ret:
                    last_frames[i] = frame
                    any_alive = True
                elif last_frames[i] is not None:
                    frame = last_frames[i]
                else:
                    frame = np.zeros((panel_height, 640, 3), dtype=np.uint8)

                h, w = frame.shape[:2]
                scale = panel_height / h
                resized = resize_frame(frame, int(w * scale), panel_height)
                panels.append(resized)

            if not any_alive:
                break

            combined = stack_frames_side_by_side(panels, gap=2, gap_color=30)

            if writer is None:
                h, w = combined.shape[:2]
                writer = VideoWriter(output_path, fps=fps, width=w, height=h)
                writer.open()

            writer.write(combined)

    finally:
        for cap in caps:
            cap.release()
        if writer is not None:
            writer.release()

    log.info("Concatenated {} videos → {}", len(video_paths), output_path)


def concat_videos_grid(
    video_paths: List[Union[str, Path]],
    output_path: Union[str, Path],
    fps: Optional[float] = None,
    panel_width: int = 640,
    panel_height: int = 360,
) -> None:
    """Arrange exactly four videos in a 2×2 grid into a single output video.

    Args:
        video_paths: Exactly four input video paths (top-left, top-right,
            bottom-left, bottom-right order).
        output_path: Destination path for the combined video.
        fps: Output frame rate.  If ``None``, uses the first video's FPS.
        panel_width: Width each panel is resized to.
        panel_height: Height each panel is resized to.

    Raises:
        ValueError: If exactly four videos are not supplied.
        FileNotFoundError: If any input video does not exist.
    """
    if len(video_paths) != 4:
        raise ValueError("concat_videos_grid requires exactly 4 video paths")

    caps = []
    for p in video_paths:
        p = Path(p)
        if not p.exists():
            raise FileNotFoundError(f"Video not found: {p}")
        caps.append(cv2.VideoCapture(str(p)))

    if fps is None:
        fps = caps[0].get(cv2.CAP_PROP_FPS) or 30.0

    last_frames: List[Optional[Frame]] = [None] * 4
    writer: Optional[VideoWriter] = None

    try:
        while True:
            frames: List[Frame] = []
            any_alive = False

            for i, cap in enumerate(caps):
                ret, frame = cap.read()
                if ret:
                    last_frames[i] = frame
                    any_alive = True
                elif last_frames[i] is not None:
                    frame = last_frames[i]
                else:
                    frame = np.zeros((panel_height, panel_width, 3), dtype=np.uint8)
                frames.append(frame)

            if not any_alive:
                break

            combined = stack_frames_2x2(
                frames[0], frames[1], frames[2], frames[3],
                target_w=panel_width, target_h=panel_height,
            )

            if writer is None:
                h, w = combined.shape[:2]
                writer = VideoWriter(output_path, fps=fps, width=w, height=h)
                writer.open()

            writer.write(combined)

    finally:
        for cap in caps:
            cap.release()
        if writer is not None:
            writer.release()

    log.info("Built 2×2 comparison grid → {}", output_path)


def mux_audio_from_source(
    silent_video_path: Union[str, Path],
    audio_source_path: Union[str, Path],
    output_path: Union[str, Path],
) -> bool:
    """Copy the audio track from a source video onto a (silent) rendered video.

    Uses ``imageio-ffmpeg``'s bundled ``ffmpeg`` binary if available.  This
    is best-effort: failures are logged and ``False`` is returned rather
    than raising, since audio muxing is a cosmetic enhancement and should
    never break the pipeline.

    Args:
        silent_video_path: Path to the rendered (audio-less) video.
        audio_source_path: Path to the original video containing audio.
        output_path: Destination path for the muxed result.

    Returns:
        ``True`` on success, ``False`` if muxing failed or ffmpeg is
        unavailable.
    """
    try:
        import imageio_ffmpeg
        import subprocess

        ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()
        cmd = [
            ffmpeg_exe, "-y",
            "-i", str(silent_video_path),
            "-i", str(audio_source_path),
            "-c:v", "copy",
            "-map", "0:v:0",
            "-map", "1:a:0?",
            "-shortest",
            str(output_path),
        ]
        subprocess.run(cmd, check=True, capture_output=True)
        log.info("Audio muxed → {}", output_path)
        return True

    except Exception as exc:  # pylint: disable=broad-except
        log.warning("Audio muxing skipped ({}): {}", type(exc).__name__, exc)
        return False
