"""
HumanMM — Structured Logging Module.

Provides a centralized, configurable logging interface using the ``loguru``
library. Every pipeline module imports ``get_logger`` from this module instead
of calling ``logging`` or ``loguru`` directly, ensuring consistent formatting,
rotation, and output targets across the entire project.

Example:
    >>> from utils.logger import get_logger
    >>> log = get_logger(__name__)
    >>> log.info("Pipeline started")
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional, Union

from loguru import logger


# ---------------------------------------------------------------------------
# Module-level sentinel so we never add duplicate sinks on re-import
# ---------------------------------------------------------------------------
_configured: bool = False


def configure_logging(
    log_dir: Union[str, Path] = "outputs/logs",
    log_file: str = "pipeline.log",
    metrics_file: str = "metrics.log",
    level: str = "INFO",
    colorize: bool = True,
    rotation: str = "10 MB",
    retention: str = "7 days",
    fmt: Optional[str] = None,
) -> None:
    """Configure the global loguru logger with console and file sinks.

    This function is idempotent — calling it multiple times has no effect after
    the first successful configuration.  It should be called once from
    ``main.py`` after Hydra has loaded the configuration.

    Args:
        log_dir: Directory where log files will be written.
        log_file: Name of the main pipeline log file.
        metrics_file: Name of the performance metrics log file.
        level: Minimum log level (``DEBUG``, ``INFO``, ``WARNING``, ``ERROR``).
        colorize: Enable ANSI colour codes in console output.
        rotation: Loguru rotation policy (e.g. ``"10 MB"``, ``"1 day"``).
        retention: Loguru retention policy (e.g. ``"7 days"``, ``"5 files"``).
        fmt: Optional custom format string.  If ``None``, a sensible default
            is used.

    Raises:
        OSError: If the log directory cannot be created.
    """
    global _configured  # noqa: PLW0603
    if _configured:
        return

    log_path = Path(log_dir)
    log_path.mkdir(parents=True, exist_ok=True)

    default_fmt = (
        "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
        "<level>{level:<8}</level> | "
        "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> | "
        "<level>{message}</level>"
    )
    chosen_fmt = fmt or default_fmt

    # Remove the default loguru sink
    logger.remove()

    # --- Console sink -------------------------------------------------------
    logger.add(
        sys.stderr,
        format=chosen_fmt,
        level=level,
        colorize=colorize,
        backtrace=True,
        diagnose=True,
    )

    # --- Main pipeline log file ---------------------------------------------
    logger.add(
        log_path / log_file,
        format=chosen_fmt,
        level=level,
        rotation=rotation,
        retention=retention,
        encoding="utf-8",
        backtrace=True,
        diagnose=True,
    )

    # --- Metrics-only log file (INFO level, less verbose) -------------------
    logger.add(
        log_path / metrics_file,
        format="{time:YYYY-MM-DD HH:mm:ss} | {level:<8} | {message}",
        level="INFO",
        filter=lambda record: "METRICS" in record["extra"],
        rotation=rotation,
        retention=retention,
        encoding="utf-8",
    )

    _configured = True
    logger.info("Logging configured — level={} log_dir={}", level, log_path)


def get_logger(name: str) -> "logger":
    """Return a loguru logger bound to the given module name.

    Calls ``configure_logging`` with defaults if the logger has not yet been
    configured by ``main.py``.  This ensures that modules imported standalone
    (e.g. during unit tests) still produce readable output.

    Args:
        name: Typically ``__name__`` of the calling module.

    Returns:
        A ``loguru`` logger instance with the ``name`` bound as extra context.

    Example:
        >>> log = get_logger(__name__)
        >>> log.debug("Processing frame {}", frame_id)
    """
    if not _configured:
        configure_logging()

    return logger.bind(name=name)


def log_metrics(
    stage: str,
    fps: float,
    elapsed_ms: float,
    extra: Optional[dict] = None,
) -> None:
    """Emit a structured metrics log entry tagged for the metrics sink.

    Args:
        stage: Pipeline stage name (e.g. ``"detection"``, ``"pose"``).
        fps: Frames processed per second at this stage.
        elapsed_ms: Total elapsed time in milliseconds for this stage.
        extra: Optional dictionary of additional key-value pairs to log.

    Example:
        >>> log_metrics("detection", fps=28.4, elapsed_ms=352.1,
        ...             extra={"gpu_mb": 412})
    """
    extra_str = ""
    if extra:
        extra_str = " | " + " | ".join(f"{k}={v}" for k, v in extra.items())

    logger.bind(METRICS=True).info(
        "STAGE={} FPS={:.2f} elapsed_ms={:.1f}{}",
        stage,
        fps,
        elapsed_ms,
        extra_str,
    )
