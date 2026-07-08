"""
HumanMM — Profiling Utility Module.

Provides deeper, opt-in profiling helpers that complement
:class:`utils.metrics.MetricsTracker`.  While ``MetricsTracker`` is meant to
run continuously during a production pipeline run, ``Profiler`` here is
intended for development-time deep dives: function-level ``cProfile``
snapshots and simple memory-delta tracking.

Example:
    >>> from utils.profiler import Profiler
    >>> profiler = Profiler(enabled=True)
    >>> with profiler.profile("motion_recovery"):
    ...     mr.run_on_frames(tracks, poses, frames)
    >>> profiler.dump_report("outputs/logs/profile_report.txt")
"""

from __future__ import annotations

import cProfile
import io
import pstats
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Callable, Dict, Generator, Optional, Union

import psutil

from utils.logger import get_logger

log = get_logger(__name__)


class Profiler:
    """Development-time function and memory profiler.

    Wraps Python's standard ``cProfile`` to capture per-call statistics for
    named code regions, plus simple RSS memory-delta measurement using
    ``psutil``.  Disabled by default to avoid overhead in production runs.

    Args:
        enabled: If ``False``, all profiling calls are no-ops.
        top_n: Number of top functions (by cumulative time) to report
            per profiled region.

    Example:
        >>> profiler = Profiler(enabled=True)
        >>> with profiler.profile("pose_estimation"):
        ...     estimator.run_on_frames(tracks, frames)
        >>> print(profiler.get_text_report("pose_estimation"))
    """

    def __init__(self, enabled: bool = False, top_n: int = 20) -> None:
        self._enabled = enabled
        self._top_n = top_n
        self._profiles: Dict[str, cProfile.Profile] = {}
        self._mem_deltas: Dict[str, float] = {}
        self._process = psutil.Process()

    @contextmanager
    def profile(self, name: str) -> Generator[None, None, None]:
        """Context manager that profiles CPU time and RSS memory delta.

        Args:
            name: Identifier for the profiled region.

        Yields:
            Nothing.  Profiles the body of the ``with`` block.

        Example:
            >>> with profiler.profile("detection"):
            ...     detector.run_on_frames(frames)
        """
        if not self._enabled:
            yield
            return

        mem_before = self._rss_mb()
        prof = cProfile.Profile()
        prof.enable()
        try:
            yield
        finally:
            prof.disable()
            mem_after = self._rss_mb()
            self._profiles[name] = prof
            self._mem_deltas[name] = mem_after - mem_before
            log.debug(
                "Profiler[{}]: RSS delta {:.1f} MB", name, self._mem_deltas[name]
            )

    def get_text_report(self, name: str) -> str:
        """Return a formatted ``pstats`` report for a profiled region.

        Args:
            name: The region name passed to :meth:`profile`.

        Returns:
            Human-readable text report, or an empty string if ``name`` was
            never profiled (or profiling is disabled).
        """
        if name not in self._profiles:
            return ""

        buf = io.StringIO()
        stats = pstats.Stats(self._profiles[name], stream=buf)
        stats.sort_stats("cumulative")
        stats.print_stats(self._top_n)

        header = (
            f"=== Profile: {name} "
            f"(RSS delta: {self._mem_deltas.get(name, 0.0):+.1f} MB) ===\n"
        )
        return header + buf.getvalue()

    def dump_report(self, path: Union[str, Path]) -> None:
        """Write text reports for all profiled regions to a single file.

        Args:
            path: Destination file path.

        Raises:
            OSError: If the parent directory cannot be created.
        """
        if not self._enabled or not self._profiles:
            log.debug("Profiler: nothing to dump (enabled={})", self._enabled)
            return

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        sections = [self.get_text_report(name) for name in self._profiles]
        with path.open("w", encoding="utf-8") as fh:
            fh.write("\n\n".join(sections))

        log.info("Profiler report written → {}", path)

    def get_memory_report(self) -> Dict[str, float]:
        """Return RSS memory delta (MB) for every profiled region.

        Returns:
            Dict mapping region name → memory delta in megabytes.
        """
        return dict(self._mem_deltas)

    def _rss_mb(self) -> float:
        """Return the current process RSS memory usage in megabytes."""
        try:
            return self._process.memory_info().rss / (1024 ** 2)
        except Exception as exc:  # pylint: disable=broad-except
            log.warning("Profiler: failed to read RSS memory: {}", exc)
            return 0.0


def profile_function(profiler: Optional[Profiler] = None, name: Optional[str] = None) -> Callable:
    """Decorator factory binding a function to a shared :class:`Profiler`.

    Args:
        profiler: Existing :class:`Profiler` instance to record into.  If
            ``None``, a disabled (no-op) profiler is used.
        name: Optional override for the profiled region name; defaults to
            the wrapped function's ``__name__``.

    Returns:
        Decorator that wraps the function with profiling.

    Example:
        >>> shared_profiler = Profiler(enabled=True)
        >>> @profile_function(shared_profiler)
        ... def run_stage():
        ...     ...
    """
    active_profiler = profiler or Profiler(enabled=False)

    def decorator(func: Callable) -> Callable:
        label = name or func.__name__

        def wrapper(*args: Any, **kwargs: Any) -> Any:
            with active_profiler.profile(label):
                return func(*args, **kwargs)

        wrapper.__name__ = getattr(func, "__name__", label)
        wrapper.__doc__ = func.__doc__
        return wrapper

    return decorator
