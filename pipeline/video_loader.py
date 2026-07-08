"""
HumanMM — Video Loader Pipeline Module.

Responsible for all video I/O operations:
- Reading video metadata (FPS, resolution, frame count, duration, codec)
- Extracting frames to disk or in-memory buffers
- Serving frames on demand to downstream pipeline stages

Design Pattern: Repository Pattern — abstracts all video I/O behind a
clean interface; the rest of the pipeline never touches ``cv2.VideoCapture``
directly.

Example:
    >>> from pipeline.video_loader import VideoLoader
    >>> loader = VideoLoader(video_path="data/input/demo.mp4", config=cfg.video)
    >>> loader.load()
    >>> for frame, idx in loader.frame_iterator():
    ...     process(frame, idx)
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Generator, Iterator, List, Optional, Tuple, Union

import cv2
import numpy as np

from utils.logger import get_logger
from utils.frame_utils import Frame, save_frame

log = get_logger(__name__)


# ---------------------------------------------------------------------------
# Metadata dataclass
# ---------------------------------------------------------------------------

@dataclass
class VideoMetadata:
    """Complete metadata record for an input video.

    Attributes:
        path: Absolute path to the video file.
        width: Frame width in pixels.
        height: Frame height in pixels.
        fps: Nominal frame rate (frames per second).
        frame_count: Total number of frames.
        duration_sec: Total duration in seconds.
        codec: FourCC codec string (e.g. ``"avc1"``).
        file_size_mb: File size in megabytes.
    """

    path: Path
    width: int
    height: int
    fps: float
    frame_count: int
    duration_sec: float
    codec: str
    file_size_mb: float

    def to_dict(self) -> Dict[str, Any]:
        """Serialise to a plain dictionary for JSON export."""
        return {
            "path": str(self.path),
            "width": self.width,
            "height": self.height,
            "fps": round(self.fps, 3),
            "frame_count": self.frame_count,
            "duration_sec": round(self.duration_sec, 3),
            "codec": self.codec,
            "file_size_mb": round(self.file_size_mb, 2),
        }

    @property
    def resolution(self) -> str:
        """Resolution string e.g. ``"1280x720"``."""
        return f"{self.width}x{self.height}"


# ---------------------------------------------------------------------------
# Video Loader
# ---------------------------------------------------------------------------

class VideoLoader:
    """Repository-pattern video reader for the HumanMM pipeline.

    Opens a video file with OpenCV, reads metadata, and provides a
    generator-based frame iterator with configurable start/end frames,
    frame step, and optional on-disk frame saving.

    Args:
        video_path: Path to the input video file.
        config: Video config dict (mirrors ``cfg.video``).

    Attributes:
        metadata: :class:`VideoMetadata` populated after :meth:`load` is called.

    Example:
        >>> loader = VideoLoader("data/input/demo.mp4", config={"frame_step": 1})
        >>> loader.load()
        >>> print(loader.metadata.fps)
        >>> for frame, idx in loader.frame_iterator():
        ...     process(frame)
    """

    def __init__(
        self,
        video_path: Union[str, Path],
        config: Optional[Dict[str, Any]] = None,
    ) -> None:
        self._path = Path(video_path)
        self._config: Dict[str, Any] = config or {}
        self._cap: Optional[cv2.VideoCapture] = None
        self.metadata: Optional[VideoMetadata] = None

        # Config with defaults
        self._max_frames: int = self._config.get("max_frames", -1)
        self._start_frame: int = self._config.get("start_frame", 0)
        self._end_frame: int = self._config.get("end_frame", -1)
        self._frame_step: int = max(1, self._config.get("frame_step", 1))
        self._resize_w: int = self._config.get("resize_width", 0)
        self._resize_h: int = self._config.get("resize_height", 0)

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def load(self) -> VideoMetadata:
        """Open the video file and read all metadata.

        Returns:
            :class:`VideoMetadata` populated from the video file headers.

        Raises:
            FileNotFoundError: If the video file does not exist.
            RuntimeError: If OpenCV cannot open the file.
        """
        if not self._path.exists():
            raise FileNotFoundError(f"Input video not found: {self._path}")

        self._cap = cv2.VideoCapture(str(self._path))
        if not self._cap.isOpened():
            raise RuntimeError(f"OpenCV cannot open video: {self._path}")

        width = int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = float(self._cap.get(cv2.CAP_PROP_FPS))
        frame_count = int(self._cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fourcc_int = int(self._cap.get(cv2.CAP_PROP_FOURCC))
        codec = "".join([chr((fourcc_int >> (8 * i)) & 0xFF) for i in range(4)]).strip()
        file_size_mb = self._path.stat().st_size / (1024 ** 2)
        duration_sec = frame_count / fps if fps > 0 else 0.0

        self.metadata = VideoMetadata(
            path=self._path.resolve(),
            width=width,
            height=height,
            fps=fps,
            frame_count=frame_count,
            duration_sec=duration_sec,
            codec=codec,
            file_size_mb=file_size_mb,
        )

        log.info(
            "Video loaded: {} | {}×{} | {:.2f} fps | {} frames | {:.1f}s | {:.1f} MB",
            self._path.name,
            width,
            height,
            fps,
            frame_count,
            duration_sec,
            file_size_mb,
        )
        return self.metadata

    def release(self) -> None:
        """Release the OpenCV VideoCapture handle."""
        if self._cap and self._cap.isOpened():
            self._cap.release()
            log.debug("VideoCapture released: {}", self._path.name)

    # ------------------------------------------------------------------
    # Frame iterator
    # ------------------------------------------------------------------

    def frame_iterator(
        self,
        save_dir: Optional[Union[str, Path]] = None,
    ) -> Generator[Tuple[Frame, int], None, None]:
        """Yield ``(frame, frame_index)`` tuples for every selected frame.

        Handles seek to ``start_frame``, stops at ``end_frame`` or
        ``max_frames``, and applies ``frame_step`` subsampling.

        Args:
            save_dir: If provided, each extracted frame is saved as a JPEG
                in this directory.

        Yields:
            Tuple of:
                - ``frame``: BGR NumPy array ``(H, W, 3)``
                - ``frame_idx``: Zero-based frame index within the video

        Raises:
            RuntimeError: If :meth:`load` has not been called first.

        Example:
            >>> for frame, idx in loader.frame_iterator(save_dir="outputs/frames"):
            ...     detections = detector.run(frame, frame_idx=idx)
        """
        if self._cap is None or not self._cap.isOpened():
            raise RuntimeError("VideoLoader.load() must be called before iterating frames.")

        # Rewind to start
        self._cap.set(cv2.CAP_PROP_POS_FRAMES, self._start_frame)

        end_frame = self._end_frame
        if end_frame < 0:
            end_frame = self.metadata.frame_count - 1 if self.metadata else 999999

        save_path = Path(save_dir) if save_dir else None
        if save_path:
            save_path.mkdir(parents=True, exist_ok=True)

        frames_yielded = 0
        frame_idx = self._start_frame

        t0 = time.perf_counter()

        while frame_idx <= end_frame:
            if self._max_frames > 0 and frames_yielded >= self._max_frames:
                break

            ret, frame = self._cap.read()
            if not ret:
                log.debug("VideoCapture.read() returned False at frame {}", frame_idx)
                break

            # Apply optional resize
            if self._resize_w > 0 and self._resize_h > 0:
                frame = cv2.resize(frame, (self._resize_w, self._resize_h), interpolation=cv2.INTER_LINEAR)

            # Save to disk if requested
            if save_path:
                frame_path = save_path / f"{frame_idx:06d}.jpg"
                save_frame(frame, frame_path, quality=85)

            yield frame, frame_idx
            frames_yielded += 1

            # Skip frames according to frame_step
            next_idx = frame_idx + self._frame_step
            if self._frame_step > 1:
                self._cap.set(cv2.CAP_PROP_POS_FRAMES, next_idx)
            frame_idx = next_idx

        elapsed = time.perf_counter() - t0
        fps_achieved = frames_yielded / elapsed if elapsed > 0 else 0.0
        log.info(
            "Frame extraction complete: {} frames | {:.1f}s | {:.1f} fps",
            frames_yielded,
            elapsed,
            fps_achieved,
        )

    def read_all_frames(
        self,
        save_dir: Optional[Union[str, Path]] = None,
    ) -> List[Tuple[Frame, int]]:
        """Read all selected frames into a list (memory-intensive for long videos).

        Prefer :meth:`frame_iterator` for large videos.

        Args:
            save_dir: Optional directory to save frames to disk.

        Returns:
            List of ``(frame, frame_idx)`` tuples.
        """
        return list(self.frame_iterator(save_dir=save_dir))

    def read_frame_at(self, frame_idx: int) -> Optional[Frame]:
        """Read a single frame by index using ``CAP_PROP_POS_FRAMES`` seek.

        Args:
            frame_idx: Zero-based frame index to seek to.

        Returns:
            BGR frame array, or ``None`` if seek/read fails.

        Raises:
            RuntimeError: If :meth:`load` has not been called.
        """
        if self._cap is None or not self._cap.isOpened():
            raise RuntimeError("VideoLoader.load() must be called first.")

        self._cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = self._cap.read()
        if not ret:
            log.warning("Failed to read frame {} from {}", frame_idx, self._path.name)
            return None

        if self._resize_w > 0 and self._resize_h > 0:
            frame = cv2.resize(frame, (self._resize_w, self._resize_h))

        return frame

    def __enter__(self) -> "VideoLoader":
        self.load()
        return self

    def __exit__(self, *args: Any) -> bool:
        self.release()
        return False
