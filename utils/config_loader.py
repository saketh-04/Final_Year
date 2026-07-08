"""
HumanMM — Configuration Loader Module.

Provides utilities for loading, merging, and validating Hydra / OmegaConf
configuration objects throughout the pipeline.  All pipeline modules receive
a ``DictConfig`` object; this module centralises all config-access helpers to
avoid scattered ``OmegaConf.to_container()`` calls.

Example:
    >>> from utils.config_loader import load_config, get_nested
    >>> cfg = load_config("configs/config.yaml")
    >>> thresh = get_nested(cfg, "detector.confidence_threshold", default=0.5)
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Optional, Union

import yaml
from omegaconf import DictConfig, OmegaConf

from utils.logger import get_logger

log = get_logger(__name__)


def load_config(
    config_path: Union[str, Path] = "configs/config.yaml",
    overrides: Optional[list[str]] = None,
) -> DictConfig:
    """Load and merge a YAML configuration file into an OmegaConf DictConfig.

    This function is used in non-Hydra contexts (e.g. unit tests, scripts)
    where ``@hydra.main`` is not available.  For production runs, the
    ``PipelineRunner`` receives the Hydra-injected ``DictConfig`` directly.

    Args:
        config_path: Path to the master ``config.yaml`` file.
        overrides: Optional list of dot-notation override strings, e.g.
            ``["detector.confidence_threshold=0.6", "device.backend=cpu"]``.

    Returns:
        Merged ``DictConfig`` with all sub-config files resolved.

    Raises:
        FileNotFoundError: If ``config_path`` does not exist.
        yaml.YAMLError: If any config file contains invalid YAML.

    Example:
        >>> cfg = load_config("configs/config.yaml", overrides=["device.backend=cpu"])
        >>> print(cfg.device.backend)
        'cpu'
    """
    config_path = Path(config_path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    log.debug("Loading master config from {}", config_path)

    # Load master config
    with config_path.open("r", encoding="utf-8") as fh:
        raw: dict = yaml.safe_load(fh)

    # Strip Hydra-specific keys that cannot be merged in non-Hydra mode
    raw.pop("defaults", None)
    raw.pop("hydra", None)

    cfg: DictConfig = OmegaConf.create(raw)

    # Merge sub-configs located in same directory
    config_dir = config_path.parent
    sub_configs = ["detector", "tracker", "pose", "motion", "visualization"]
    for sub_name in sub_configs:
        sub_path = config_dir / f"{sub_name}.yaml"
        if sub_path.exists():
            log.debug("Merging sub-config: {}", sub_path)
            with sub_path.open("r", encoding="utf-8") as fh:
                sub_raw: dict = yaml.safe_load(fh) or {}
            sub_cfg = OmegaConf.create({sub_name: sub_raw})
            cfg = OmegaConf.merge(cfg, sub_cfg)

    # Apply CLI-style overrides
    if overrides:
        for override in overrides:
            key, _, val = override.partition("=")
            OmegaConf.update(cfg, key, val, merge=True)
            log.debug("Applied override: {}={}", key, val)

    log.info("Configuration loaded successfully ({} top-level keys)", len(cfg))
    return cfg


def get_nested(cfg: DictConfig, key: str, default: Any = None) -> Any:
    """Safely retrieve a nested config value by dot-notation key.

    Args:
        cfg: The ``DictConfig`` object to query.
        key: Dot-separated key path, e.g. ``"detector.confidence_threshold"``.
        default: Value to return if the key does not exist.

    Returns:
        The value at the specified key, or ``default`` if missing.

    Example:
        >>> val = get_nested(cfg, "alignment.savgol_window", default=11)
    """
    try:
        parts = key.split(".")
        node: Any = cfg
        for part in parts:
            node = node[part]
        return node
    except (KeyError, AttributeError):
        return default


def resolve_device(cfg: DictConfig) -> str:
    """Resolve the compute device string from the configuration.

    Handles ``auto`` mode by checking CUDA availability at runtime.

    Args:
        cfg: The master ``DictConfig`` object.

    Returns:
        One of ``"cuda"`` or ``"cpu"``.

    Example:
        >>> device = resolve_device(cfg)
        >>> print(device)  # 'cuda' or 'cpu'
    """
    import torch  # local import to avoid top-level torch dep in config utils

    requested = get_nested(cfg, "device.backend", default="auto")

    if requested == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    elif requested == "cuda":
        if not torch.cuda.is_available():
            log.warning("CUDA requested but not available — falling back to CPU")
            device = "cpu"
        else:
            device = "cuda"
    else:
        device = "cpu"

    log.info("Compute device resolved: {}", device)
    return device


def cfg_to_dict(cfg: DictConfig) -> dict:
    """Convert an OmegaConf DictConfig to a plain Python dict.

    Useful for serialising the config to JSON/YAML for reproducibility.

    Args:
        cfg: OmegaConf configuration object.

    Returns:
        Plain Python dictionary representation.
    """
    return OmegaConf.to_container(cfg, resolve=True, throw_on_missing=False)  # type: ignore[return-value]


def save_config(cfg: DictConfig, output_path: Union[str, Path]) -> None:
    """Persist the resolved configuration to a YAML file for reproducibility.

    Args:
        cfg: OmegaConf configuration object.
        output_path: File path where the YAML will be written.

    Raises:
        OSError: If the output directory cannot be created.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", encoding="utf-8") as fh:
        OmegaConf.save(cfg, fh, resolve=True)

    log.info("Config saved to {}", output_path)


def validate_paths(cfg: DictConfig) -> None:
    """Validate that all required paths in the config exist on disk.

    Logs a warning for each missing optional path and raises for required ones.

    Args:
        cfg: The master ``DictConfig`` object.

    Raises:
        FileNotFoundError: If the input video path does not exist.
    """
    video_path = get_nested(cfg, "video.path")
    if video_path and not Path(video_path).exists():
        raise FileNotFoundError(f"Input video not found: {video_path}")

    # Optional paths — warn but do not fail
    optional_paths = [
        ("motion.hmr2.smpl_dir", "SMPL model directory"),
        ("motion.hmr2.checkpoint", "HMR2 checkpoint"),
        ("tracker.deepsort.reid_weights", "ReID model weights"),
    ]
    for key, description in optional_paths:
        path_val = get_nested(cfg, key)
        if path_val and not Path(str(path_val)).exists():
            log.warning("{} not found at '{}' — will attempt auto-download", description, path_val)
