"""
HumanMM — OpenCV Video Writer Helper Module.

Wraps ``cv2.VideoWriter`` with a clean context-manager API, automatic codec
selection, and metadata logging.  All pipeline stages that write output videos
use this module exclusively.

Example:
    >>> from utils.video_writer import VideoWriter
    >>> with VideoWriter("outputs/02_detection.mp4", fps=30, width=1280, height=720) as vw:
    ...     for frame in processed_frames:
    ...         vw.write(frame)
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Optional, Tuple, Union

import cv2
import numpy as np

from utils.logger import get_logger

log = get_logger(__name__)


class VideoWriter:
    """Context-manager wrapper around ``cv2.VideoWriter``.

    Provides a clean, exception-safe interface for writing MP4 video files
    with automatic directory creation, codec negotiation, and FPS tracking.

    Args:
        path: Output file path (e.g. ``"outputs/02_detection.mp4"``).
        fps: Output frame rate.
        width: Frame width in pixels.
        height: Frame height in pixels.
        codec: FourCC codec string (``"mp4v"``, ``"XVID"``, ``"avc1"``).

    Example:
        >>> with VideoWriter("out.mp4", fps=30, width=1280, height=720) as vw:
        ...     for frame in frames:
        ...         vw.write(frame)
        >>> print(vw.frame_count)
    """

    def __init__(
        self,
        path: Union[str, Path],
        fps: float,
        width: int,
        height: int,
        codec: str = "mp4v",
    ) -> None:
        self._path = Path(path)
        self._fps = fps
        self._width = width
        self._height = height
        self._codec = codec

        self._writer: Optional[cv2.VideoWriter] = None
        self._frame_count: int = 0
        self._start_time: float = 0.0

    # ------------------------------------------------------------------
    # Context Manager
    # ------------------------------------------------------------------

    def __enter__(self) -> "VideoWriter":
        self.open()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> bool:
        self.release()
        return False  # Do not suppress exceptions

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def open(self) -> None:
        """Open the video file for writing.

        Creates the parent directory if it does not exist.

        Raises:
            RuntimeError: If ``cv2.VideoWriter`` fails to initialise.
        """
        self._path.parent.mkdir(parents=True, exist_ok=True)
        fourcc = cv2.VideoWriter_fourcc(*self._codec)
        self._writer = cv2.VideoWriter(
            str(self._path), fourcc, self._fps, (self._width, self._height)
        )
        if not self._writer.isOpened():
            raise RuntimeError(
                f"cv2.VideoWriter failed to open: {self._path} "
                f"(codec={self._codec}, {self._width}×{self._height} @ {self._fps}fps)"
            )
        self._start_time = time.perf_counter()
        self._frame_count = 0
        log.debug("VideoWriter opened: {} ({}×{} @ {}fps)", self._path, self._width, self._height, self._fps)

    def release(self) -> None:
        """Flush and close the video file."""
        if self._writer and self._writer.isOpened():
            self._writer.release()
            elapsed = time.perf_counter() - self._start_time
            actual_fps = self._frame_count / elapsed if elapsed > 0 else 0.0
            log.info(
                "VideoWriter closed: {} | {} frames | {:.1f}s | {:.1f} fps",
                self._path.name,
                self._frame_count,
                elapsed,
                actual_fps,
            )

    # ------------------------------------------------------------------
    # Frame Writing
    # ------------------------------------------------------------------

    def write(self, frame: np.ndarray) -> None:
        """Write a single BGR frame to the video file.

        Automatically resizes the frame if its dimensions do not match the
        writer's configured resolution.

        Args:
            frame: BGR NumPy array of shape ``(H, W, 3)``.

        Raises:
            RuntimeError: If the writer is not open.
            ValueError: If ``frame`` is not a 3-channel uint8 array.

        Example:
            >>> vw.write(detection_frame)
        """
        if self._writer is None or not self._writer.isOpened():
            raise RuntimeError("VideoWriter is not open. Call open() or use as context manager.")

        if frame.ndim != 3 or frame.shape[2] != 3:
            raise ValueError(
                f"Frame must be (H, W, 3) BGR uint8, got shape {frame.shape}"
            )

        h, w = frame.shape[:2]
        if w != self._width or h != self._height:
            frame = cv2.resize(frame, (self._width, self._height), interpolation=cv2.INTER_LINEAR)

        if frame.dtype != np.uint8:
            frame = np.clip(frame, 0, 255).astype(np.uint8)

        self._writer.write(frame)
        self._frame_count += 1

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def frame_count(self) -> int:
        """Number of frames written so far."""
        return self._frame_count

    @property
    def path(self) -> Path:
        """Output file path."""
        return self._path

    @property
    def is_open(self) -> bool:
        """Whether the writer is currently open."""
        return self._writer is not None and self._writer.isOpened()


# ---------------------------------------------------------------------------
# Standalone helper
# ---------------------------------------------------------------------------

def frames_to_video(
    frames: list[np.ndarray],
    path: Union[str, Path],
    fps: float = 30.0,
    codec: str = "mp4v",
) -> None:
    """Write a list of BGR frames to an MP4 video file.

    Convenience wrapper around :class:`VideoWriter` for batch writing.

    Args:
        frames: List of BGR NumPy arrays (must all have the same shape).
        path: Output video file path.
        fps: Output frame rate.
        codec: FourCC codec string.

    Raises:
        ValueError: If ``frames`` is empty.

    Example:
        >>> frames_to_video(frame_list, "outputs/04_pose.mp4", fps=30)
    """
    if not frames:
        raise ValueError("frames list must not be empty")

    h, w = frames[0].shape[:2]

    with VideoWriter(path, fps=fps, width=w, height=h, codec=codec) as vw:
        for frame in frames:
            vw.write(frame)

    log.info("Video saved: {} ({} frames)", path, len(frames))


def get_video_properties(path: Union[str, Path]) -> dict:
    """Read basic properties from a video file without decoding all frames.

    Args:
        path: Path to the video file.

    Returns:
        Dictionary with ``width``, ``height``, ``fps``, ``frame_count``,
        ``duration_sec``, and ``codec`` keys.

    Raises:
        FileNotFoundError: If the video file does not exist.
        RuntimeError: If OpenCV cannot open the file.

    Example:
        >>> props = get_video_properties("data/input/demo.mp4")
        >>> print(props["fps"])
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Video not found: {path}")

    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise RuntimeError(f"OpenCV cannot open video: {path}")

    try:
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fourcc_int = int(cap.get(cv2.CAP_PROP_FOURCC))
        codec = "".join([chr((fourcc_int >> (i * 8)) & 0xFF) for i in range(4)])
        duration_sec = frame_count / fps if fps > 0 else 0.0

        return {
            "width": width,
            "height": height,
            "fps": fps,
            "frame_count": frame_count,
            "duration_sec": round(duration_sec, 3),
            "codec": codec.strip(),
        }
    finally:
        cap.release()
