"""
HumanMM — I/O Utility Module.

Provides helpers for reading and writing structured data in all output
formats required by the pipeline: JSON, CSV, NPY, PLY, and OBJ.
All functions handle directory creation, error logging, and type coercion
automatically.

Example:
    >>> from utils.io_utils import save_json, save_csv, save_npy, save_ply
    >>> save_json({"fps": 28.4}, "outputs/metrics.json")
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Union

import numpy as np
import pandas as pd

from utils.logger import get_logger

log = get_logger(__name__)


# ---------------------------------------------------------------------------
# JSON
# ---------------------------------------------------------------------------

class _NumpyEncoder(json.JSONEncoder):
    """Custom JSON encoder that converts NumPy types to native Python types."""

    def default(self, obj: Any) -> Any:
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)


def save_json(
    data: Any,
    path: Union[str, Path],
    indent: int = 2,
    ensure_ascii: bool = False,
) -> None:
    """Serialise ``data`` to a JSON file.

    NumPy arrays and scalars are automatically converted to JSON-compatible
    Python types.

    Args:
        data: Any JSON-serialisable Python object (dict, list, etc.).
        path: Destination file path.
        indent: Number of spaces for pretty-printing.  Use ``0`` for compact.
        ensure_ascii: If ``True``, escape non-ASCII characters.

    Raises:
        TypeError: If ``data`` contains types that cannot be serialised.
        OSError: If the parent directory cannot be created.

    Example:
        >>> save_json({"stage": "detection", "fps": 28.4}, "outputs/metrics.json")
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    try:
        with path.open("w", encoding="utf-8") as fh:
            json.dump(data, fh, cls=_NumpyEncoder, indent=indent or None, ensure_ascii=ensure_ascii)
        log.debug("JSON saved → {}", path)
    except TypeError as exc:
        log.error("Failed to serialise JSON: {}", exc)
        raise


def load_json(path: Union[str, Path]) -> Any:
    """Load a JSON file and return the parsed Python object.

    Args:
        path: Path to the JSON file.

    Returns:
        Parsed Python object.

    Raises:
        FileNotFoundError: If the file does not exist.
        json.JSONDecodeError: If the file is not valid JSON.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"JSON file not found: {path}")

    with path.open("r", encoding="utf-8") as fh:
        return json.load(fh)


# ---------------------------------------------------------------------------
# CSV
# ---------------------------------------------------------------------------

def save_csv(
    data: Union[pd.DataFrame, List[Dict[str, Any]]],
    path: Union[str, Path],
    index: bool = False,
) -> None:
    """Save tabular data to a CSV file.

    Args:
        data: A Pandas ``DataFrame`` or a list of row dicts.
        path: Destination file path.
        index: Whether to write the DataFrame index.

    Raises:
        OSError: If the parent directory cannot be created.

    Example:
        >>> rows = [{"frame": 0, "person_id": 1, "x": 0.5, "y": 0.3}]
        >>> save_csv(rows, "outputs/trajectory.csv")
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    if isinstance(data, list):
        df = pd.DataFrame(data)
    else:
        df = data

    df.to_csv(path, index=index)
    log.debug("CSV saved ({} rows, {} cols) → {}", len(df), len(df.columns), path)


def load_csv(path: Union[str, Path]) -> pd.DataFrame:
    """Load a CSV file into a Pandas DataFrame.

    Args:
        path: Path to the CSV file.

    Returns:
        Pandas ``DataFrame``.

    Raises:
        FileNotFoundError: If the file does not exist.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"CSV file not found: {path}")
    return pd.read_csv(path)


# ---------------------------------------------------------------------------
# NumPy NPY
# ---------------------------------------------------------------------------

def save_npy(
    array: np.ndarray,
    path: Union[str, Path],
    allow_pickle: bool = False,
) -> None:
    """Save a NumPy array to a ``.npy`` binary file.

    Args:
        array: NumPy array to save.
        path: Destination file path (will add ``.npy`` extension if missing).
        allow_pickle: Whether to allow pickling for object arrays.

    Raises:
        OSError: If the parent directory cannot be created.

    Example:
        >>> save_npy(joints_array, "outputs/joints.npy")
    """
    path = Path(path)
    if path.suffix != ".npy":
        path = path.with_suffix(".npy")
    path.parent.mkdir(parents=True, exist_ok=True)

    np.save(str(path), array, allow_pickle=allow_pickle)
    log.debug("NPY saved shape={} dtype={} → {}", array.shape, array.dtype, path)


def load_npy(path: Union[str, Path], allow_pickle: bool = False) -> np.ndarray:
    """Load a NumPy ``.npy`` file.

    Args:
        path: Path to the ``.npy`` file.
        allow_pickle: Whether to allow pickle loading.

    Returns:
        Loaded NumPy array.

    Raises:
        FileNotFoundError: If the file does not exist.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"NPY file not found: {path}")
    return np.load(str(path), allow_pickle=allow_pickle)


# ---------------------------------------------------------------------------
# PLY
# ---------------------------------------------------------------------------

def save_ply(
    vertices: np.ndarray,
    faces: Optional[np.ndarray],
    path: Union[str, Path],
    vertex_colors: Optional[np.ndarray] = None,
) -> None:
    """Save a 3D mesh to a PLY file using trimesh.

    Args:
        vertices: Vertex positions of shape ``(V, 3)``, dtype float.
        faces: Triangle indices of shape ``(F, 3)``, dtype int.  If ``None``,
            saves a point cloud.
        path: Destination file path.
        vertex_colors: Optional per-vertex colours of shape ``(V, 3)`` or
            ``(V, 4)`` in uint8 [0, 255].

    Raises:
        ImportError: If ``trimesh`` is not installed.
        OSError: If the parent directory cannot be created.

    Example:
        >>> save_ply(verts, faces, "outputs/meshes/person_0.ply")
    """
    try:
        import trimesh  # local import — optional dependency
    except ImportError as exc:
        raise ImportError("trimesh is required for PLY export: pip install trimesh") from exc

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    if faces is not None:
        mesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)
        if vertex_colors is not None:
            mesh.visual.vertex_colors = vertex_colors
    else:
        mesh = trimesh.PointCloud(vertices=vertices)
        if vertex_colors is not None:
            mesh.colors = vertex_colors

    mesh.export(str(path))
    log.debug("PLY saved (V={}, F={}) → {}", len(vertices), len(faces) if faces is not None else 0, path)


# ---------------------------------------------------------------------------
# OBJ
# ---------------------------------------------------------------------------

def save_obj(
    vertices: np.ndarray,
    faces: np.ndarray,
    path: Union[str, Path],
) -> None:
    """Save a 3D mesh to a Wavefront OBJ file.

    Writes vertices and face indices in standard OBJ text format.  For
    textured meshes, use trimesh or a dedicated library.

    Args:
        vertices: Vertex positions of shape ``(V, 3)``.
        faces: Zero-indexed triangle indices of shape ``(F, 3)``.
        path: Destination file path.

    Raises:
        OSError: If the parent directory cannot be created.

    Example:
        >>> save_obj(verts, faces, "outputs/meshes/person_0.obj")
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    lines = ["# HumanMM exported mesh\n"]
    for v in vertices:
        lines.append(f"v {v[0]:.6f} {v[1]:.6f} {v[2]:.6f}\n")
    for f in faces:
        # OBJ uses 1-indexed faces
        lines.append(f"f {f[0]+1} {f[1]+1} {f[2]+1}\n")

    with path.open("w", encoding="utf-8") as fh:
        fh.writelines(lines)

    log.debug("OBJ saved (V={}, F={}) → {}", len(vertices), len(faces), path)


# ---------------------------------------------------------------------------
# GIF
# ---------------------------------------------------------------------------

def save_gif(
    frames: Sequence[np.ndarray],
    path: Union[str, Path],
    fps: int = 15,
    loop: int = 0,
) -> None:
    """Save a sequence of BGR frames as an animated GIF.

    Args:
        frames: Sequence of BGR NumPy arrays (uint8).
        path: Destination ``.gif`` file path.
        fps: Frames per second for the GIF playback.
        loop: Number of times the GIF loops; ``0`` = infinite.

    Raises:
        ValueError: If ``frames`` is empty.
        ImportError: If ``imageio`` is not installed.

    Example:
        >>> save_gif(frame_list, "outputs/result.gif", fps=15)
    """
    try:
        import imageio
    except ImportError as exc:
        raise ImportError("imageio is required for GIF export: pip install imageio") from exc

    if not frames:
        raise ValueError("frames sequence must not be empty")

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    duration = 1000 // fps  # milliseconds per frame for imageio

    # imageio expects RGB
    rgb_frames = [frame[:, :, ::-1] for frame in frames]  # BGR → RGB

    imageio.mimsave(
        str(path),
        rgb_frames,
        format="GIF",
        duration=duration,
        loop=loop,
    )
    log.debug("GIF saved ({} frames, {} fps) → {}", len(frames), fps, path)
