"""
HumanMM — Timing Utility Module.

Lightweight stopwatch utilities used throughout the pipeline for ad-hoc
timing of individual operations.  For full per-stage performance tracking
(FPS, GPU/CPU sampling) see :class:`utils.metrics.MetricsTracker` — ``Timer``
is intentionally simpler and dependency-free, suitable for quick profiling
inside loops or one-off CLI diagnostics.

Example:
    >>> from utils.timer import Timer
    >>> with Timer("detection") as t:
    ...     run_detection(frame)
    >>> print(t.elapsed_ms)
"""

from __future__ import annotations

import time
from contextlib import ContextDecorator
from typing import Callable, Dict, List, Optional

from utils.logger import get_logger

log = get_logger(__name__)


class Timer(ContextDecorator):
    """Context-manager / decorator stopwatch for timing a code block.

    Can be used either as a ``with`` block or as a function decorator.

    Args:
        name: Identifier used in log messages.
        log_on_exit: If ``True``, automatically logs elapsed time on exit.
        silent: If ``True``, suppresses the log message even if
            ``log_on_exit`` is ``True`` (useful for toggling via config).

    Example:
        >>> with Timer("pose_estimation") as t:
        ...     poses = estimator.run(frame)
        >>> print(f"{t.elapsed_ms:.2f} ms")

        >>> @Timer("my_function")
        ... def my_function():
        ...     ...
    """

    def __init__(
        self,
        name: str = "block",
        log_on_exit: bool = True,
        silent: bool = False,
    ) -> None:
        self.name = name
        self.log_on_exit = log_on_exit
        self.silent = silent
        self._t0: float = 0.0
        self._t1: float = 0.0
        self.elapsed_sec: float = 0.0

    def __enter__(self) -> "Timer":
        self._t0 = time.perf_counter()
        return self

    def __exit__(self, *exc: object) -> bool:
        self._t1 = time.perf_counter()
        self.elapsed_sec = self._t1 - self._t0
        if self.log_on_exit and not self.silent:
            log.debug("Timer[{}]: {:.2f} ms", self.name, self.elapsed_ms)
        return False

    @property
    def elapsed_ms(self) -> float:
        """Elapsed time in milliseconds."""
        return self.elapsed_sec * 1000.0

    def reset(self) -> None:
        """Reset the timer to start a fresh measurement."""
        self._t0 = time.perf_counter()
        self.elapsed_sec = 0.0


class FPSCounter:
    """Rolling frames-per-second counter.

    Maintains a sliding window of recent frame timestamps and computes
    the instantaneous FPS.  Intended for live overlay (e.g. via
    :mod:`utils.image_utils`) rather than aggregate reporting.

    Args:
        window_size: Number of recent frames to average over.

    Example:
        >>> fps_counter = FPSCounter(window_size=30)
        >>> for frame in frames:
        ...     fps_counter.tick()
        ...     fps = fps_counter.fps
    """

    def __init__(self, window_size: int = 30) -> None:
        self._window_size = max(1, window_size)
        self._timestamps: List[float] = []

    def tick(self) -> float:
        """Register a new frame and return the current FPS estimate.

        Returns:
            Instantaneous FPS computed over the rolling window.
        """
        now = time.perf_counter()
        self._timestamps.append(now)
        if len(self._timestamps) > self._window_size:
            self._timestamps.pop(0)
        return self.fps

    @property
    def fps(self) -> float:
        """Current rolling-window FPS estimate."""
        if len(self._timestamps) < 2:
            return 0.0
        span = self._timestamps[-1] - self._timestamps[0]
        if span <= 1e-9:
            return 0.0
        return (len(self._timestamps) - 1) / span

    def reset(self) -> None:
        """Clear the timestamp history."""
        self._timestamps.clear()


class MultiStageTimer:
    """Accumulates named timing measurements across many calls.

    Lighter-weight alternative to :class:`utils.metrics.MetricsTracker`
    when only wall-clock timing (no resource sampling) is required.

    Example:
        >>> mt = MultiStageTimer()
        >>> with mt.time("detection"):
        ...     run_detection(frame)
        >>> print(mt.summary())
    """

    def __init__(self) -> None:
        self._totals: Dict[str, float] = {}
        self._counts: Dict[str, int] = {}

    def time(self, name: str) -> Timer:
        """Return a :class:`Timer` bound to this tracker for ``name``.

        Args:
            name: Stage identifier.

        Returns:
            A :class:`Timer` instance; on exit its elapsed time is recorded.
        """
        tracker = self

        class _BoundTimer(Timer):
            def __exit__(self, *exc: object) -> bool:
                result = super().__exit__(*exc)
                tracker._totals[name] = tracker._totals.get(name, 0.0) + self.elapsed_sec
                tracker._counts[name] = tracker._counts.get(name, 0) + 1
                return result

        return _BoundTimer(name=name, log_on_exit=False)

    def summary(self) -> Dict[str, Dict[str, float]]:
        """Return per-stage total/mean timing statistics.

        Returns:
            Dict mapping stage name to ``{"total_ms", "mean_ms", "calls"}``.
        """
        out: Dict[str, Dict[str, float]] = {}
        for name, total in self._totals.items():
            calls = self._counts.get(name, 1)
            out[name] = {
                "total_ms": round(total * 1000.0, 2),
                "mean_ms": round((total / calls) * 1000.0, 2),
                "calls": calls,
            }
        return out


def time_function(name: Optional[str] = None) -> Callable:
    """Decorator factory that logs a function's execution time.

    Args:
        name: Optional override for the logged identifier; defaults to the
            wrapped function's ``__name__``.

    Returns:
        A decorator that wraps the target function with timing + logging.

    Example:
        >>> @time_function()
        ... def slow_op():
        ...     ...
    """

    def decorator(func: Callable) -> Callable:
        label = name or func.__name__

        def wrapper(*args: object, **kwargs: object) -> object:
            with Timer(label, log_on_exit=True):
                return func(*args, **kwargs)

        wrapper.__name__ = getattr(func, "__name__", label)
        wrapper.__doc__ = func.__doc__
        return wrapper

    return decorator
