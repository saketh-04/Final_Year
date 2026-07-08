#!/usr/bin/env python
"""
HumanMM — CLI Entry Point.

Loads the master Hydra configuration, applies any CLI overrides, configures
logging, and runs the full pipeline via :class:`pipeline.pipeline_manager.PipelineManager`.

Usage:
    python main.py --video data/input/demo.mp4
    python main.py --video demo.mp4 --device cuda --pose-backend vitpose --debug

See ``README.md`` for the full list of supported arguments.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import List, Optional

from utils.config_loader import load_config, save_config
from utils.logger import configure_logging, get_logger


def build_arg_parser() -> argparse.ArgumentParser:
    """Construct the CLI argument parser.

    Returns:
        Configured :class:`argparse.ArgumentParser`.
    """
    parser = argparse.ArgumentParser(
        prog="main.py",
        description="HumanMM: Global Human Motion Recovery from Multi-Shot Videos",
    )

    parser.add_argument("--video", required=True, help="Path to input video file")
    parser.add_argument(
        "--config", default="configs/config.yaml", help="Path to config YAML"
    )
    parser.add_argument("--output-dir", default=None, help="Output directory (default: outputs/)")
    parser.add_argument(
        "--device", choices=["cuda", "cpu", "auto"], default=None, help="Compute device"
    )
    parser.add_argument(
        "--pose-backend", choices=["mediapipe", "vitpose"], default=None,
        help="Pose estimator backend",
    )
    parser.add_argument(
        "--tracker", choices=["bytetrack", "deepsort"], default=None,
        help="Tracking algorithm",
    )
    parser.add_argument("--visualize", action="store_true", help="Generate all visualization videos")
    parser.add_argument("--save-json", action="store_true", help="Export results as JSON")
    parser.add_argument("--save-mesh", action="store_true", help="Export 3D meshes/vertices")
    parser.add_argument("--save-video", action="store_true", help="Save all pipeline stage videos")
    parser.add_argument("--show-fps", action="store_true", help="Display FPS counter on output videos")
    parser.add_argument("--max-persons", type=int, default=None, help="Maximum persons to track")
    parser.add_argument(
        "--conf-threshold", type=float, default=None, help="YOLO confidence threshold"
    )
    parser.add_argument("--debug", action="store_true", help="Enable debug logging")

    return parser


def build_overrides(args: argparse.Namespace) -> List[str]:
    """Translate parsed CLI args into dot-notation Hydra/OmegaConf overrides.

    Args:
        args: Parsed CLI arguments from :func:`build_arg_parser`.

    Returns:
        List of ``key=value`` override strings consumable by
        :func:`utils.config_loader.load_config`.
    """
    overrides: List[str] = [f"video.path={args.video}"]

    if args.output_dir:
        overrides.append(f"output.root_dir={args.output_dir}")
    if args.device:
        overrides.append(f"device.backend={args.device}")
    if args.pose_backend:
        overrides.append(f"pose.backend={args.pose_backend}")
    if args.tracker:
        overrides.append(f"tracker.backend={args.tracker}")
    if args.max_persons is not None:
        overrides.append(f"pipeline.max_persons={args.max_persons}")
    if args.conf_threshold is not None:
        overrides.append(f"detector.confidence_threshold={args.conf_threshold}")

    if args.visualize:
        overrides.append("pipeline.run_visualization=true")
    if args.save_json:
        overrides.append("output.save_json=true")
    if args.save_mesh:
        overrides.append("output.save_mesh=true")
    if args.save_video:
        overrides.append("output.save_video=true")
    if args.show_fps:
        overrides.append("output.show_fps=true")
    if args.debug:
        overrides.append("logging.level=DEBUG")

    return overrides


def main(argv: Optional[List[str]] = None) -> int:
    """Parse arguments, build the config, and run the HumanMM pipeline.

    Args:
        argv: Optional explicit argument list (primarily for testing).
            Defaults to ``sys.argv[1:]``.

    Returns:
        Process exit code (``0`` on success, ``1`` on failure).
    """
    parser = build_arg_parser()
    args = parser.parse_args(argv)

    video_path = Path(args.video)
    if not video_path.exists():
        print(f"Error: input video not found: {video_path}", file=sys.stderr)
        return 1

    config_path = Path(args.config)
    if not config_path.exists():
        print(f"Error: config file not found: {config_path}", file=sys.stderr)
        return 1

    overrides = build_overrides(args)
    cfg = load_config(config_path, overrides=overrides)

    log_cfg = cfg.get("logging", {})
    configure_logging(
        log_dir=log_cfg.get("log_dir", "outputs/logs"),
        log_file=log_cfg.get("log_file", "pipeline.log"),
        metrics_file=log_cfg.get("metrics_file", "metrics.log"),
        level="DEBUG" if args.debug else log_cfg.get("level", "INFO"),
        colorize=log_cfg.get("colorize", True),
        rotation=log_cfg.get("rotation", "10 MB"),
        retention=log_cfg.get("retention", "7 days"),
    )
    log = get_logger(__name__)

    log.info("HumanMM pipeline starting — video={}", video_path)
    log.debug("Resolved overrides: {}", overrides)

    # Import after logging is configured so all module-level loggers in the
    # dependency graph inherit the chosen sinks/level.
    from pipeline.pipeline_manager import PipelineManager

    try:
        manager = PipelineManager(cfg)
        report = manager.run()
    except Exception as exc:  # pylint: disable=broad-except
        log.exception("Pipeline run failed: {}", exc)
        return 1

    output_root = Path(cfg.get("output", {}).get("root_dir", "outputs"))
    try:
        from exporters.json_exporter import JSONExporter

        JSONExporter(output_dir=output_root).export_metrics(report)
    except Exception as exc:  # pylint: disable=broad-except
        log.warning("Could not export final metrics report: {}", exc)

    log.info(
        "Pipeline finished successfully. FPS={:.2f} | total={:.1f}s | outputs → {}",
        report.get("summary", {}).get("total_pipeline_fps", 0.0),
        report.get("summary", {}).get("total_wall_clock_sec", 0.0),
        output_root,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
