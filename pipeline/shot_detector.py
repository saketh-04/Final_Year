"""
HumanMM — Shot Boundary Detector Pipeline Module.

Detects scene/shot cuts in the video using PySceneDetect with a
content-aware detector.  Returns a structured list of ``ShotSegment``
objects that the rest of the pipeline uses to handle multi-shot alignment.

Design Pattern: Facade — wraps the PySceneDetect API behind a clean,
pipeline-compatible interface.

Example:
    >>> from pipeline.shot_detector import ShotDetector, ShotSegment
    >>> detector = ShotDetector(config=cfg.pipeline)
    >>> shots = detector.detect(video_path="data/input/demo.mp4")
    >>> print(f"{len(shots)} shots detected")
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from utils.logger import get_logger

log = get_logger(__name__)


# ---------------------------------------------------------------------------
# Data structure
# ---------------------------------------------------------------------------

@dataclass
class ShotSegment:
    """A single detected shot (scene cut) in the video.

    Attributes:
        shot_id: Zero-based shot index.
        start_frame: First frame of this shot.
        end_frame: Last frame of this shot (inclusive).
        start_time_sec: Start timestamp in seconds.
        end_time_sec: End timestamp in seconds.
        frame_count: Number of frames in this shot.
    """

    shot_id: int
    start_frame: int
    end_frame: int
    start_time_sec: float
    end_time_sec: float

    @property
    def frame_count(self) -> int:
        """Number of frames in this shot."""
        return self.end_frame - self.start_frame + 1

    @property
    def duration_sec(self) -> float:
        """Duration of this shot in seconds."""
        return self.end_time_sec - self.start_time_sec

    def contains_frame(self, frame_idx: int) -> bool:
        """Return ``True`` if ``frame_idx`` falls within this shot.

        Args:
            frame_idx: Frame index to test.

        Returns:
            Boolean membership result.
        """
        return self.start_frame <= frame_idx <= self.end_frame

    def to_dict(self) -> Dict[str, Any]:
        """Serialise to a plain dictionary for JSON export."""
        return {
            "shot_id": self.shot_id,
            "start_frame": self.start_frame,
            "end_frame": self.end_frame,
            "start_time_sec": round(self.start_time_sec, 3),
            "end_time_sec": round(self.end_time_sec, 3),
            "frame_count": self.frame_count,
            "duration_sec": round(self.duration_sec, 3),
        }


# ---------------------------------------------------------------------------
# Detector
# ---------------------------------------------------------------------------

class ShotDetector:
    """PySceneDetect-based shot boundary detector.

    Supports ``ContentDetector`` (default) and ``ThresholdDetector`` via
    config.  If PySceneDetect is not installed, falls back to treating the
    entire video as a single shot.

    Args:
        config: Pipeline config dict (mirrors ``cfg.pipeline``).

    Example:
        >>> sd = ShotDetector()
        >>> shots = sd.detect("data/input/demo.mp4", fps=30.0, total_frames=900)
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        self._config: Dict[str, Any] = config or {}
        self._threshold: float = self._config.get("shot_detection_threshold", 27.0)
        self._detector_type: str = self._config.get("shot_detector_type", "content")
        self._min_scene_len: int = self._config.get("min_scene_len", 15)

    def detect(
        self,
        video_path: Union[str, Path],
        fps: float = 30.0,
        total_frames: int = 0,
    ) -> List[ShotSegment]:
        """Detect shot boundaries in a video file.

        Args:
            video_path: Path to the input video.
            fps: Video frame rate (used for time computation in fallback mode).
            total_frames: Total frame count (used for fallback single-shot).

        Returns:
            List of :class:`ShotSegment` objects ordered by ``shot_id``.

        Raises:
            FileNotFoundError: If the video file does not exist.
        """
        video_path = Path(video_path)
        if not video_path.exists():
            raise FileNotFoundError(f"Video not found: {video_path}")

        try:
            shots = self._run_scenedetect(video_path, fps, total_frames)
        except ImportError:
            log.warning(
                "PySceneDetect not installed — treating video as single shot. "
                "Install via: pip install scenedetect[opencv]"
            )
            shots = self._single_shot_fallback(fps, total_frames)
        except Exception as exc:
            log.error("Shot detection error: {} — falling back to single shot", exc)
            shots = self._single_shot_fallback(fps, total_frames)

        log.info(
            "Shot detection complete: {} shots detected in {}",
            len(shots),
            video_path.name,
        )
        for shot in shots:
            log.debug(
                "  Shot {:03d}: frames [{} – {}] | {:.2f}s",
                shot.shot_id,
                shot.start_frame,
                shot.end_frame,
                shot.duration_sec,
            )

        return shots

    def detect_from_frames(
        self,
        frame_indices: List[int],
        fps: float = 30.0,
    ) -> List[ShotSegment]:
        """Build shot segments from a list of known cut frame indices.

        Useful when shot boundaries have been determined externally (e.g., via
        a custom detector or manual annotation).

        Args:
            frame_indices: Sorted list of cut frame indices (first frame of
                each new shot, including 0).
            fps: Video frame rate for time computation.

        Returns:
            List of :class:`ShotSegment` objects.
        """
        if not frame_indices:
            return []

        # Ensure 0 is the first index
        if frame_indices[0] != 0:
            frame_indices = [0] + frame_indices

        shots: List[ShotSegment] = []
        for i, start in enumerate(frame_indices):
            end = frame_indices[i + 1] - 1 if i + 1 < len(frame_indices) else start + 1000

            shots.append(
                ShotSegment(
                    shot_id=i,
                    start_frame=start,
                    end_frame=end,
                    start_time_sec=start / fps,
                    end_time_sec=end / fps,
                )
            )

        return shots

    def get_shot_for_frame(
        self, shots: List[ShotSegment], frame_idx: int
    ) -> Optional[ShotSegment]:
        """Return the shot that contains ``frame_idx``, or ``None``.

        Args:
            shots: List of detected shot segments.
            frame_idx: Frame index to look up.

        Returns:
            Matching :class:`ShotSegment` or ``None``.
        """
        for shot in shots:
            if shot.contains_frame(frame_idx):
                return shot
        return None

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _run_scenedetect(
        self,
        video_path: Path,
        fps: float,
        total_frames: int,
    ) -> List[ShotSegment]:
        """Run PySceneDetect on the video file.

        Args:
            video_path: Input video path.
            fps: Frame rate.
            total_frames: Total frames in video.

        Returns:
            List of :class:`ShotSegment` objects.

        Raises:
            ImportError: If PySceneDetect is not installed.
        """
        from scenedetect import (  # type: ignore
            open_video,
            SceneManager,
        )
        from scenedetect.detectors import ContentDetector, ThresholdDetector  # type: ignore

        log.debug(
            "Running PySceneDetect ({}, threshold={}) on {}",
            self._detector_type,
            self._threshold,
            video_path.name,
        )

        video = open_video(str(video_path))
        scene_manager = SceneManager()

        if self._detector_type == "threshold":
            scene_manager.add_detector(
                ThresholdDetector(threshold=self._threshold, min_scene_len=self._min_scene_len)
            )
        else:
            scene_manager.add_detector(
                ContentDetector(threshold=self._threshold, min_scene_len=self._min_scene_len)
            )

        scene_manager.detect_scenes(video, show_progress=False)
        scene_list = scene_manager.get_scene_list()

        if not scene_list:
            # Entire video is one shot
            return self._single_shot_fallback(fps, total_frames)

        shots: List[ShotSegment] = []
        for i, (start_tc, end_tc) in enumerate(scene_list):
            shots.append(
                ShotSegment(
                    shot_id=i,
                    start_frame=start_tc.get_frames(),
                    end_frame=end_tc.get_frames() - 1,
                    start_time_sec=start_tc.get_seconds(),
                    end_time_sec=end_tc.get_seconds(),
                )
            )

        return shots

    def _single_shot_fallback(
        self, fps: float, total_frames: int
    ) -> List[ShotSegment]:
        """Return a single shot covering the entire video.

        Args:
            fps: Frame rate for time computation.
            total_frames: Total frame count.

        Returns:
            Single-element list with one :class:`ShotSegment`.
        """
        end_frame = max(0, total_frames - 1)
        duration = end_frame / fps if fps > 0 else 0.0
        return [
            ShotSegment(
                shot_id=0,
                start_frame=0,
                end_frame=end_frame,
                start_time_sec=0.0,
                end_time_sec=duration,
            )
        ]
