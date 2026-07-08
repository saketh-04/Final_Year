"""
HumanMM — Performance Metrics Module.

Tracks and reports CPU, GPU, RAM usage, FPS, and per-stage inference times
throughout the pipeline.  All monitoring is non-blocking; resource samples are
collected in a background thread and aggregated on demand.

Example:
    >>> from utils.metrics import MetricsTracker
    >>> tracker = MetricsTracker(enabled=True, track_gpu=True)
    >>> tracker.start()
    >>> with tracker.stage("detection"):
    ...     run_detection(frame)
    >>> report = tracker.get_report()
"""

from __future__ import annotations

import threading
import time
from collections import defaultdict
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Dict, Generator, List, Optional

import psutil

from utils.logger import get_logger, log_metrics

log = get_logger(__name__)


@dataclass
class StageMetrics:
    """Stores timing and resource samples for a single pipeline stage.

    Attributes:
        name: Stage identifier (e.g. ``"detection"``).
        elapsed_ms_list: List of per-call elapsed times in milliseconds.
        frame_counts: List of frame counts processed per call.
    """

    name: str
    elapsed_ms_list: List[float] = field(default_factory=list)
    frame_counts: List[int] = field(default_factory=list)

    @property
    def total_ms(self) -> float:
        """Total elapsed time in milliseconds across all calls."""
        return sum(self.elapsed_ms_list)

    @property
    def mean_ms(self) -> float:
        """Mean elapsed time per call in milliseconds."""
        return self.total_ms / len(self.elapsed_ms_list) if self.elapsed_ms_list else 0.0

    @property
    def total_frames(self) -> int:
        """Total frames processed by this stage."""
        return sum(self.frame_counts)

    @property
    def fps(self) -> float:
        """Average frames per second for this stage."""
        if self.total_ms < 1e-6:
            return 0.0
        return self.total_frames / (self.total_ms / 1000.0)


@dataclass
class ResourceSnapshot:
    """A single sample of system resource utilisation.

    Attributes:
        timestamp: Unix timestamp of the sample.
        cpu_percent: CPU usage percentage (0–100).
        ram_used_mb: RAM used in megabytes.
        gpu_used_mb: GPU memory used in megabytes (0 if no GPU).
        gpu_utilization: GPU compute utilization percentage (0–100).
    """

    timestamp: float
    cpu_percent: float
    ram_used_mb: float
    gpu_used_mb: float = 0.0
    gpu_utilization: float = 0.0


class MetricsTracker:
    """Thread-safe pipeline performance tracker.

    Collects per-stage timing data and background system resource samples.
    Call :meth:`start` before the pipeline begins and :meth:`stop` after it
    ends.  Use :meth:`stage` as a context manager around each pipeline step.

    Args:
        enabled: If ``False``, all methods are no-ops (zero overhead).
        track_gpu: If ``True``, attempt to sample GPU memory and utilization.
        track_cpu: If ``True``, sample CPU and RAM usage.
        sample_interval_sec: Seconds between background resource samples.

    Example:
        >>> tracker = MetricsTracker(enabled=True, track_gpu=True)
        >>> tracker.start()
        >>> with tracker.stage("detection", frames=1):
        ...     detections = detector.run(frame)
        >>> tracker.stop()
        >>> print(tracker.get_report())
    """

    def __init__(
        self,
        enabled: bool = True,
        track_gpu: bool = True,
        track_cpu: bool = True,
        sample_interval_sec: float = 1.0,
    ) -> None:
        self._enabled = enabled
        self._track_gpu = track_gpu
        self._track_cpu = track_cpu
        self._interval = sample_interval_sec

        self._stages: Dict[str, StageMetrics] = defaultdict(lambda: StageMetrics(name="unknown"))
        self._resources: List[ResourceSnapshot] = []
        self._pipeline_start: float = 0.0
        self._pipeline_end: float = 0.0

        self._monitor_thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()

        self._gpu_available = False
        if track_gpu:
            self._gpu_available = self._check_gpu()

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def start(self) -> None:
        """Start the metrics tracker and background resource monitor.

        Should be called once before the pipeline begins processing.
        """
        if not self._enabled:
            return
        self._pipeline_start = time.perf_counter()
        self._stop_event.clear()

        if self._track_cpu or self._track_gpu:
            self._monitor_thread = threading.Thread(
                target=self._monitor_loop,
                daemon=True,
                name="MetricsMonitor",
            )
            self._monitor_thread.start()

        log.debug("MetricsTracker started")

    def stop(self) -> None:
        """Stop the background monitor and finalise all metrics."""
        if not self._enabled:
            return
        self._pipeline_end = time.perf_counter()
        self._stop_event.set()
        if self._monitor_thread and self._monitor_thread.is_alive():
            self._monitor_thread.join(timeout=3.0)
        log.debug("MetricsTracker stopped")

    # ------------------------------------------------------------------
    # Stage context manager
    # ------------------------------------------------------------------

    @contextmanager
    def stage(self, stage_name: str, frames: int = 1) -> Generator[None, None, None]:
        """Context manager to time a single pipeline stage call.

        Args:
            stage_name: Identifier for the stage (e.g. ``"detection"``).
            frames: Number of frames processed in this call.

        Yields:
            Nothing.  Times the body of the ``with`` block.

        Example:
            >>> with tracker.stage("pose", frames=1):
            ...     poses = pose_estimator.run(frame)
        """
        if not self._enabled:
            yield
            return

        if stage_name not in self._stages:
            self._stages[stage_name] = StageMetrics(name=stage_name)

        t0 = time.perf_counter()
        try:
            yield
        finally:
            elapsed_ms = (time.perf_counter() - t0) * 1000.0
            self._stages[stage_name].elapsed_ms_list.append(elapsed_ms)
            self._stages[stage_name].frame_counts.append(frames)

    # ------------------------------------------------------------------
    # Reporting
    # ------------------------------------------------------------------

    def get_report(self) -> dict:
        """Build a structured metrics report dictionary.

        Returns:
            Dictionary with pipeline summary, per-stage stats, and resource
            utilisation averages.

        Example:
            >>> report = tracker.get_report()
            >>> print(report["summary"]["total_pipeline_fps"])
        """
        total_elapsed = self._pipeline_end - self._pipeline_start
        total_frames = max(
            (m.total_frames for m in self._stages.values()), default=0
        )

        stages_report = {}
        for name, sm in self._stages.items():
            stages_report[name] = {
                "fps": round(sm.fps, 2),
                "mean_ms": round(sm.mean_ms, 2),
                "total_ms": round(sm.total_ms, 2),
                "calls": len(sm.elapsed_ms_list),
                "total_frames": sm.total_frames,
            }
            log_metrics(name, fps=sm.fps, elapsed_ms=sm.total_ms)

        resources_report: dict = {}
        if self._resources:
            cpus = [r.cpu_percent for r in self._resources]
            rams = [r.ram_used_mb for r in self._resources]
            resources_report["cpu_mean_percent"] = round(float(np.mean(cpus)), 2)
            resources_report["cpu_max_percent"] = round(float(np.max(cpus)), 2)
            resources_report["ram_mean_mb"] = round(float(np.mean(rams)), 2)
            resources_report["ram_max_mb"] = round(float(np.max(rams)), 2)
            if self._gpu_available:
                gpus_mb = [r.gpu_used_mb for r in self._resources]
                gpus_ut = [r.gpu_utilization for r in self._resources]
                resources_report["gpu_mean_mb"] = round(float(np.mean(gpus_mb)), 2)
                resources_report["gpu_max_mb"] = round(float(np.max(gpus_mb)), 2)
                resources_report["gpu_mean_util_pct"] = round(float(np.mean(gpus_ut)), 2)

        return {
            "summary": {
                "total_pipeline_sec": round(total_elapsed, 3),
                "total_pipeline_fps": round(total_frames / total_elapsed, 2) if total_elapsed > 0 else 0.0,
                "total_frames_processed": total_frames,
            },
            "stages": stages_report,
            "resources": resources_report,
        }

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _monitor_loop(self) -> None:
        """Background thread: periodically sample resource utilisation."""
        while not self._stop_event.is_set():
            try:
                cpu_pct = psutil.cpu_percent(interval=None)
                ram_mb = psutil.virtual_memory().used / (1024 ** 2)
                gpu_mb, gpu_util = 0.0, 0.0

                if self._gpu_available:
                    gpu_mb, gpu_util = self._sample_gpu()

                self._resources.append(
                    ResourceSnapshot(
                        timestamp=time.time(),
                        cpu_percent=cpu_pct,
                        ram_used_mb=ram_mb,
                        gpu_used_mb=gpu_mb,
                        gpu_utilization=gpu_util,
                    )
                )
            except Exception as exc:  # pylint: disable=broad-except
                log.warning("Resource monitor error: {}", exc)

            self._stop_event.wait(timeout=self._interval)

    def _check_gpu(self) -> bool:
        """Check whether a GPU monitoring library is available."""
        try:
            import GPUtil  # noqa: F401
            return True
        except ImportError:
            log.debug("GPUtil not found — GPU monitoring disabled")
            return False

    def _sample_gpu(self) -> tuple[float, float]:
        """Sample GPU memory usage and utilization.

        Returns:
            Tuple of ``(used_mb, utilization_pct)``.
        """
        try:
            import GPUtil
            gpus = GPUtil.getGPUs()
            if not gpus:
                return 0.0, 0.0
            gpu = gpus[0]
            return float(gpu.memoryUsed), float(gpu.load * 100)
        except Exception:  # pylint: disable=broad-except
            return 0.0, 0.0


# ---------------------------------------------------------------------------
# Lazy numpy import (needed in get_report)
# ---------------------------------------------------------------------------
import numpy as np  # noqa: E402
