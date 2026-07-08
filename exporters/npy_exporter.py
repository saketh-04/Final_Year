"""
HumanMM — NumPy (.npy / .npz) Exporter.

Persists large numeric arrays (joint trajectories, SMPL parameters, mesh
vertices) produced by the motion-recovery and trajectory-alignment stages.
JSON/CSV exporters intentionally omit these arrays for size and
readability reasons — this module is the canonical binary export path.

Example:
    >>> from exporters.npy_exporter import NPYExporter
    >>> exporter = NPYExporter(output_dir="outputs")
    >>> exporter.export_aligned_joints(aligned_results)
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Tuple, Union

import numpy as np

from models.gvhmr_wrapper import SMPLOutput
from pipeline.trajectory_aligner import AlignedTrackResult
from utils.io_utils import save_npy
from utils.logger import get_logger

log = get_logger(__name__)


class NPYExporter:
    """Writes joint/SMPL arrays to ``.npy``/``.npz`` binary files.

    Args:
        output_dir: Root output directory (mirrors ``cfg.output.root_dir``).

    Example:
        >>> exporter = NPYExporter(output_dir="outputs")
        >>> exporter.export_aligned_joints(aligned_results)
    """

    def __init__(self, output_dir: Union[str, Path] = "outputs") -> None:
        self._root = Path(output_dir)
        self._root.mkdir(parents=True, exist_ok=True)

    def export_aligned_joints(
        self, aligned_results: Dict[int, AlignedTrackResult]
    ) -> Path:
        """Export aligned joint trajectories for all tracks to ``joints.npz``.

        Each track is stored under its own key (``track_<id>_joints``,
        ``track_<id>_frames``) since track sequences may have different
        lengths and cannot be stacked into a single rectangular array.

        Args:
            aligned_results: Dict mapping ``track_id → AlignedTrackResult``.

        Returns:
            Path to the written ``.npz`` file.
        """
        path = self._root / "joints.npz"
        arrays = {}
        for track_id, result in aligned_results.items():
            arrays[f"track_{track_id}_joints"] = result.joints_3d_aligned
            arrays[f"track_{track_id}_translations"] = result.translations_aligned
            arrays[f"track_{track_id}_rotations"] = result.rotations_aligned
            arrays[f"track_{track_id}_frames"] = np.array(result.frame_indices, dtype=np.int64)

        np.savez_compressed(str(path), **arrays)
        log.info("NPYExporter: joints saved → {}", path)
        return path

    def export_single_track_joints(self, track_id: int, joints_3d: np.ndarray) -> Path:
        """Export a single track's ``(T, J, 3)`` joint array to its own ``.npy`` file.

        Args:
            track_id: Track identifier (used in the filename).
            joints_3d: Joint array of shape ``(T, J, 3)``.

        Returns:
            Path to the written ``.npy`` file.
        """
        path = self._root / f"joints_track_{track_id}.npy"
        save_npy(joints_3d, path)
        return path

    def export_smpl_params(
        self, smpl_results: Dict[Tuple[int, int], SMPLOutput]
    ) -> Path:
        """Export full SMPL parameters (betas, pose, vertices) to ``smpl_params.npz``.

        Stored as parallel arrays keyed by ``frame_idx``/``track_id`` for
        easy reconstruction, since SMPL outputs are sparse across the
        ``(frame, track)`` grid.

        Args:
            smpl_results: Dict mapping ``(frame_idx, track_id) → SMPLOutput``.

        Returns:
            Path to the written ``.npz`` file.
        """
        path = self._root / "smpl_params.npz"
        if not smpl_results:
            np.savez_compressed(str(path))
            return path

        keys = sorted(smpl_results.keys())
        frame_indices = np.array([k[0] for k in keys], dtype=np.int64)
        track_ids = np.array([k[1] for k in keys], dtype=np.int64)
        betas = np.stack([smpl_results[k].betas for k in keys], axis=0)
        body_pose = np.stack([smpl_results[k].body_pose for k in keys], axis=0)
        global_orient = np.stack([smpl_results[k].global_orient for k in keys], axis=0)
        transl = np.stack([smpl_results[k].transl for k in keys], axis=0)
        joints_3d = np.stack([smpl_results[k].joints_3d for k in keys], axis=0)
        camera_params = np.stack([smpl_results[k].camera_params for k in keys], axis=0)
        confidence = np.array([smpl_results[k].confidence for k in keys], dtype=np.float32)

        np.savez_compressed(
            str(path),
            frame_indices=frame_indices,
            track_ids=track_ids,
            betas=betas,
            body_pose=body_pose,
            global_orient=global_orient,
            transl=transl,
            joints_3d=joints_3d,
            camera_params=camera_params,
            confidence=confidence,
        )
        log.info("NPYExporter: SMPL params saved ({} entries) → {}", len(keys), path)
        return path

    def export_mesh_vertices(
        self, smpl_results: Dict[Tuple[int, int], SMPLOutput], min_confidence: float = 0.0
    ) -> Path:
        """Export mesh vertex arrays separately (large, kept out of the main archive).

        Args:
            smpl_results: Dict mapping ``(frame_idx, track_id) → SMPLOutput``.
            min_confidence: Skip outputs below this confidence (saves space
                when only the fallback pseudo-3D estimates are available,
                since those carry no real mesh data).

        Returns:
            Path to the written ``.npz`` file.
        """
        path = self._root / "mesh_vertices.npz"
        filtered = {
            k: v for k, v in smpl_results.items()
            if v.confidence >= min_confidence and v.vertices is not None and v.vertices.size > 0
        }

        if not filtered:
            np.savez_compressed(str(path))
            log.debug("NPYExporter: no qualifying mesh vertices to export")
            return path

        keys = sorted(filtered.keys())
        frame_indices = np.array([k[0] for k in keys], dtype=np.int64)
        track_ids = np.array([k[1] for k in keys], dtype=np.int64)
        vertices = np.stack([filtered[k].vertices for k in keys], axis=0)

        np.savez_compressed(
            str(path),
            frame_indices=frame_indices,
            track_ids=track_ids,
            vertices=vertices,
        )
        log.info("NPYExporter: mesh vertices saved ({} entries) → {}", len(keys), path)
        return path
