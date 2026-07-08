"""
HumanMM — 3D Geometry & Rotation Utility Module.

Provides all mathematical primitives required for 3D human motion processing:
rotation representations (rotation matrix, axis-angle, quaternion, Euler),
spherical linear interpolation, Procrustes alignment, and coordinate transforms.

All functions operate on NumPy arrays.  No PyTorch tensors are used here to
keep this module importable without a CUDA environment.

Example:
    >>> from utils.geometry import quaternion_slerp, procrustes_align
    >>> q0 = np.array([1., 0., 0., 0.])
    >>> q1 = np.array([0., 1., 0., 0.])
    >>> q_mid = quaternion_slerp(q0, q1, t=0.5)
"""

from __future__ import annotations

import math
from typing import Tuple

import numpy as np
from scipy.spatial.transform import Rotation, Slerp

from utils.logger import get_logger

log = get_logger(__name__)


# ---------------------------------------------------------------------------
# Rotation conversions
# ---------------------------------------------------------------------------

def axis_angle_to_rotmat(axis_angle: np.ndarray) -> np.ndarray:
    """Convert axis-angle rotation to a 3×3 rotation matrix.

    Args:
        axis_angle: Array of shape ``(3,)`` representing the rotation vector
            (axis × angle in radians).

    Returns:
        Rotation matrix of shape ``(3, 3)``.

    Example:
        >>> R = axis_angle_to_rotmat(np.array([0., 0., np.pi / 4]))
    """
    return Rotation.from_rotvec(axis_angle).as_matrix()


def rotmat_to_axis_angle(rotmat: np.ndarray) -> np.ndarray:
    """Convert a 3×3 rotation matrix to axis-angle (rotation vector).

    Args:
        rotmat: Rotation matrix of shape ``(3, 3)``.

    Returns:
        Axis-angle array of shape ``(3,)``.
    """
    return Rotation.from_matrix(rotmat).as_rotvec()


def rotmat_to_quaternion(rotmat: np.ndarray) -> np.ndarray:
    """Convert a 3×3 rotation matrix to a unit quaternion [w, x, y, z].

    Args:
        rotmat: Rotation matrix of shape ``(3, 3)``.

    Returns:
        Quaternion of shape ``(4,)`` in ``[w, x, y, z]`` order.
    """
    # scipy returns [x, y, z, w]; convert to [w, x, y, z]
    xyzw = Rotation.from_matrix(rotmat).as_quat()
    return np.array([xyzw[3], xyzw[0], xyzw[1], xyzw[2]], dtype=np.float64)


def quaternion_to_rotmat(q: np.ndarray) -> np.ndarray:
    """Convert a unit quaternion [w, x, y, z] to a 3×3 rotation matrix.

    Args:
        q: Quaternion of shape ``(4,)`` in ``[w, x, y, z]`` order.

    Returns:
        Rotation matrix of shape ``(3, 3)``.

    Raises:
        ValueError: If the quaternion norm is near zero.
    """
    norm = np.linalg.norm(q)
    if norm < 1e-8:
        raise ValueError(f"Quaternion norm is near zero: {q}")
    q_normed = q / norm
    # Reorder to scipy [x, y, z, w]
    xyzw = np.array([q_normed[1], q_normed[2], q_normed[3], q_normed[0]])
    return Rotation.from_quat(xyzw).as_matrix()


def euler_to_rotmat(angles: np.ndarray, order: str = "xyz", degrees: bool = False) -> np.ndarray:
    """Convert Euler angles to a rotation matrix.

    Args:
        angles: Array of shape ``(3,)`` with Euler angles.
        order: Axis order string (e.g. ``"xyz"``, ``"zyx"``).
        degrees: If ``True``, ``angles`` are in degrees; otherwise radians.

    Returns:
        Rotation matrix of shape ``(3, 3)``.
    """
    return Rotation.from_euler(order, angles, degrees=degrees).as_matrix()


# ---------------------------------------------------------------------------
# Quaternion SLERP
# ---------------------------------------------------------------------------

def quaternion_slerp(q0: np.ndarray, q1: np.ndarray, t: float) -> np.ndarray:
    """Spherical linear interpolation between two unit quaternions.

    Args:
        q0: Start quaternion ``[w, x, y, z]`` of shape ``(4,)``.
        q1: End quaternion ``[w, x, y, z]`` of shape ``(4,)``.
        t: Interpolation parameter in ``[0.0, 1.0]``.

    Returns:
        Interpolated quaternion ``[w, x, y, z]`` of shape ``(4,)``.

    Raises:
        ValueError: If ``t`` is outside ``[0.0, 1.0]``.

    Example:
        >>> q_mid = quaternion_slerp(q0, q1, t=0.5)
    """
    if not 0.0 <= t <= 1.0:
        raise ValueError(f"Interpolation parameter t must be in [0,1], got {t}")

    # Convert [w,x,y,z] → scipy [x,y,z,w]
    def _to_scipy(q: np.ndarray) -> np.ndarray:
        return np.array([q[1], q[2], q[3], q[0]])

    r0 = Rotation.from_quat(_to_scipy(q0 / np.linalg.norm(q0)))
    r1 = Rotation.from_quat(_to_scipy(q1 / np.linalg.norm(q1)))

    slerp = Slerp([0.0, 1.0], Rotation.concatenate([r0, r1]))
    r_interp = slerp([t])[0]

    xyzw = r_interp.as_quat()
    return np.array([xyzw[3], xyzw[0], xyzw[1], xyzw[2]], dtype=np.float64)


def slerp_sequence(
    rotations: np.ndarray,
    timestamps: np.ndarray,
    query_times: np.ndarray,
) -> np.ndarray:
    """Interpolate a sequence of rotation matrices at arbitrary time steps.

    Args:
        rotations: Rotation matrices of shape ``(N, 3, 3)``.
        timestamps: Known time steps of shape ``(N,)``.
        query_times: Query time steps of shape ``(M,)``.

    Returns:
        Interpolated rotation matrices of shape ``(M, 3, 3)``.
    """
    rots = Rotation.from_matrix(rotations)
    slerp_fn = Slerp(timestamps, rots)
    return slerp_fn(query_times).as_matrix()


# ---------------------------------------------------------------------------
# Procrustes Alignment
# ---------------------------------------------------------------------------

def procrustes_align(
    source: np.ndarray,
    target: np.ndarray,
    allow_scale: bool = False,
) -> Tuple[np.ndarray, np.ndarray, float, np.ndarray]:
    """Compute the optimal rigid (or similarity) transform from source to target.

    Uses the orthogonal Procrustes method (SVD-based) to find the rotation R,
    translation t, and optionally scale s such that ``target ≈ s * R @ source + t``.

    Args:
        source: Point set of shape ``(N, 3)`` — the data to be aligned.
        target: Point set of shape ``(N, 3)`` — the reference.
        allow_scale: If ``True``, also estimate isotropic scale.

    Returns:
        Tuple of:
            - ``R``: Rotation matrix ``(3, 3)``
            - ``t``: Translation vector ``(3,)``
            - ``scale``: Scale factor (1.0 if ``allow_scale=False``)
            - ``aligned``: Aligned source points ``(N, 3)``

    Raises:
        ValueError: If source and target have different shapes.

    Example:
        >>> R, t, s, aligned = procrustes_align(src_joints, tgt_joints)
    """
    if source.shape != target.shape:
        raise ValueError(
            f"Source {source.shape} and target {target.shape} must have the same shape"
        )

    # Centre both point clouds
    src_mean = source.mean(axis=0)
    tgt_mean = target.mean(axis=0)

    src_c = source - src_mean
    tgt_c = target - tgt_mean

    # Frobenius norm (size normalization)
    src_norm = np.linalg.norm(src_c)
    tgt_norm = np.linalg.norm(tgt_c)

    if src_norm < 1e-8 or tgt_norm < 1e-8:
        log.warning("Procrustes: near-zero point cloud — returning identity transform")
        return np.eye(3), tgt_mean - src_mean, 1.0, source + (tgt_mean - src_mean)

    src_c /= src_norm
    tgt_c_norm = tgt_c / tgt_norm

    # SVD
    M = tgt_c_norm.T @ src_c
    U, sigma, Vt = np.linalg.svd(M)

    # Ensure proper rotation (det = +1)
    d = np.linalg.det(U @ Vt)
    D = np.diag([1.0, 1.0, d])

    R = U @ D @ Vt

    if allow_scale:
        scale = sigma @ np.array([1.0, 1.0, d]) * (tgt_norm / src_norm)
    else:
        scale = tgt_norm / src_norm

    t = tgt_mean - scale * R @ src_mean
    aligned = scale * (source @ R.T) + t

    return R, t, float(scale), aligned


# ---------------------------------------------------------------------------
# Coordinate transforms
# ---------------------------------------------------------------------------

def project_3d_to_2d(
    points_3d: np.ndarray,
    K: np.ndarray,
    R: Optional[np.ndarray] = None,
    t: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Project 3D world points to 2D image coordinates using a pinhole model.

    Args:
        points_3d: 3D points of shape ``(N, 3)`` in world / camera coordinates.
        K: Camera intrinsic matrix of shape ``(3, 3)``.
        R: Optional rotation matrix ``(3, 3)`` (world → camera).  Identity if
            ``None``.
        t: Optional translation vector ``(3,)`` (world → camera).  Zero if
            ``None``.

    Returns:
        2D projected points of shape ``(N, 2)`` in pixel coordinates.
    """
    N = points_3d.shape[0]
    if R is None:
        R = np.eye(3)
    if t is None:
        t = np.zeros(3)

    # Transform to camera space
    pts_cam = (R @ points_3d.T + t[:, None]).T  # (N, 3)

    # Project
    pts_h = (K @ pts_cam.T).T  # (N, 3)
    pts_2d = pts_h[:, :2] / pts_h[:, 2:3]

    return pts_2d


def compute_bbox_from_joints(
    joints_2d: np.ndarray,
    margin: float = 0.15,
    frame_w: int = 1280,
    frame_h: int = 720,
) -> Tuple[int, int, int, int]:
    """Compute a bounding box from 2D joint positions.

    Args:
        joints_2d: Joint coordinates of shape ``(J, 2)``.
        margin: Fractional margin to add around the tight box.
        frame_w: Frame width for clamping.
        frame_h: Frame height for clamping.

    Returns:
        Tuple ``(x1, y1, x2, y2)`` in pixel coordinates.
    """
    valid = joints_2d[~np.any(np.isnan(joints_2d), axis=1)]
    if len(valid) == 0:
        return 0, 0, frame_w, frame_h

    x1, y1 = valid.min(axis=0)
    x2, y2 = valid.max(axis=0)

    bw, bh = x2 - x1, y2 - y1
    x1 = int(max(0, x1 - margin * bw))
    y1 = int(max(0, y1 - margin * bh))
    x2 = int(min(frame_w, x2 + margin * bw))
    y2 = int(min(frame_h, y2 + margin * bh))

    return x1, y1, x2, y2


# ---------------------------------------------------------------------------
# Missing import fix
# ---------------------------------------------------------------------------
from typing import Optional  # noqa: E402 — must be after TYPE_CHECKING block


# ---------------------------------------------------------------------------
# Smooth / interpolation helpers
# ---------------------------------------------------------------------------

def compute_geodesic_distance(R1: np.ndarray, R2: np.ndarray) -> float:
    """Compute the geodesic distance (angle) between two rotation matrices.

    Args:
        R1: Rotation matrix ``(3, 3)``.
        R2: Rotation matrix ``(3, 3)``.

    Returns:
        Geodesic angle in radians ``[0, π]``.
    """
    R_rel = R1.T @ R2
    trace = np.clip(np.trace(R_rel), -1.0, 3.0)
    return float(math.acos((trace - 1.0) / 2.0))


def angular_velocity(
    rotations: np.ndarray, fps: float = 30.0
) -> np.ndarray:
    """Estimate angular velocity from a sequence of rotation matrices.

    Args:
        rotations: Rotation matrices of shape ``(N, 3, 3)``.
        fps: Frames per second (used for time normalisation).

    Returns:
        Angular velocity vectors of shape ``(N-1, 3)`` in rad/s.
    """
    n = len(rotations)
    if n < 2:
        return np.zeros((0, 3))

    vels = []
    for i in range(n - 1):
        R_rel = rotations[i].T @ rotations[i + 1]
        rotvec = Rotation.from_matrix(R_rel).as_rotvec()
        vels.append(rotvec * fps)
    return np.array(vels)
