from __future__ import annotations

import numpy as np
import scipy.linalg
from typing import Tuple, Dict

class KalmanFilter:
    """
    A Kalman filter for tracking bounding boxes in image space.

    This implementation is specialized for the ByteTrack algorithm, using an 
    8-dimensional state space: (x, y, a, h, vx, vy, va, vh), where (x, y) is the 
    bounding box center, 'a' is the aspect ratio (width/height), 'h' is the 
    height, and (vx, vy, va, vh) are their respective velocities.
    """

    # Chi-square distributions for 95% confidence interval
    chi2inv95: Dict[int, float] = {
        1: 3.8415,
        2: 5.9915,
        3: 7.8147,
        4: 9.4877,
        5: 11.070,
        6: 12.592,
        7: 14.067,
        8: 15.507,
        9: 16.919,
    }

    def __init__(self) -> None:
        """
        Initializes the Kalman Filter with fixed state transition and 
        observation matrices.
        """
        ndim, dt = 4, 1.0

        # Create state transition matrix (F)
        self._motion_mat = np.eye(2 * ndim, 2 * ndim)
        for i in range(ndim):
            self._motion_mat[i, ndim + i] = dt

        # Create observation matrix (H)
        self._update_mat = np.eye(ndim, 2 * ndim)

        # Motion and observation noise hyperparameters
        self._std_weight_position = 1.0 / 20
        self._std_weight_velocity = 1.0 / 160

    def initiate(self, measurement: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Create track from unassociated measurement.
        """
        mean_pos = measurement
        mean_vel = np.zeros_like(mean_pos)
        mean = np.r_[mean_pos, mean_vel]

        std = [
            2 * self._std_weight_position * measurement[3],
            2 * self._std_weight_position * measurement[3],
            1e-2,
            2 * self._std_weight_position * measurement[3],
            10 * self._std_weight_velocity * measurement[3],
            10 * self._std_weight_velocity * measurement[3],
            1e-5,
            10 * self._std_weight_velocity * measurement[3],
        ]
        covariance = np.diag(np.square(std))
        return mean, covariance

    def predict(self, mean: np.ndarray, covariance: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Run Kalman filter prediction step.
        """
        std_pos = [
            self._std_weight_position * mean[3],
            self._std_weight_position * mean[3],
            1e-2,
            self._std_weight_position * mean[3],
        ]
        std_vel = [
            self._std_weight_velocity * mean[3],
            self._std_weight_velocity * mean[3],
            1e-5,
            self._std_weight_velocity * mean[3],
        ]
        motion_cov = np.diag(np.square(np.r_[std_pos, std_vel]))

        mean = np.dot(mean, self._motion_mat.T)
        covariance = np.linalg.multi_dot((
            self._motion_mat, covariance, self._motion_mat.T
        )) + motion_cov

        return mean, covariance

    def multi_predict(self, mean: np.ndarray, covariance: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Vectorized version of the prediction step for multiple tracks.
        Optimized to avoid explicit looping for covariance noise construction.
        """
        std_pos = [
            self._std_weight_position * mean[:, 3],
            self._std_weight_position * mean[:, 3],
            1e-2 * np.ones_like(mean[:, 3]),
            self._std_weight_position * mean[:, 3],
        ]
        std_vel = [
            self._std_weight_velocity * mean[:, 3],
            self._std_weight_velocity * mean[:, 3],
            1e-5 * np.ones_like(mean[:, 3]),
            self._std_weight_velocity * mean[:, 3],
        ]
        
        sq_std = np.square(np.stack(std_pos + std_vel, axis=1))
        
        # Efficiently create batch diagonal matrices
        motion_cov = np.zeros((len(mean), 8, 8))
        for i in range(8):
            motion_cov[:, i, i] = sq_std[:, i]

        # Vectorized matrix multiplication: F * Cov * F.T
        mean = np.dot(mean, self._motion_mat.T)
        left = np.matmul(self._motion_mat, covariance)
        covariance = np.matmul(left, self._motion_mat.T) + motion_cov

        return mean, covariance

    def project(self, mean: np.ndarray, covariance: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Project state distribution to measurement space.
        """
        std = [
            self._std_weight_position * mean[3],
            self._std_weight_position * mean[3],
            1e-1,
            self._std_weight_position * mean[3],
        ]
        innovation_cov = np.diag(np.square(std))

        mean = np.dot(self._update_mat, mean)
        covariance = np.linalg.multi_dot((
            self._update_mat, covariance, self._update_mat.T
        ))
        return mean, covariance + innovation_cov

    def update(
        self, mean: np.ndarray, covariance: np.ndarray, measurement: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Run Kalman filter correction step using Cholesky decomposition.
        """
        projected_mean, projected_cov = self.project(mean, covariance)

        chol_factor, lower = scipy.linalg.cho_factor(
            projected_cov, lower=True, check_finite=False
        )
        kalman_gain = scipy.linalg.cho_solve(
            (chol_factor, lower), 
            np.dot(covariance, self._update_mat.T).T, 
            check_finite=False
        ).T
        
        innovation = measurement - projected_mean

        new_mean = mean + np.dot(innovation, kalman_gain.T)
        new_covariance = covariance - np.linalg.multi_dot((
            kalman_gain, projected_cov, kalman_gain.T
        ))
        return new_mean, new_covariance

    def gating_distance(
        self,
        mean: np.ndarray,
        covariance: np.ndarray,
        measurements: np.ndarray,
        only_position: bool = False,
        metric: str = "maha"
    ) -> np.ndarray:
        """
        Compute gating distance between state distribution and measurements.

        Args:
            mean (np.ndarray): The predicted state mean.
            covariance (np.ndarray): The predicted state covariance.
            measurements (np.ndarray): (N, 4) matrix of measurements.
            only_position (bool): If True, computes distance for (x, y) only.
            metric (str): Either "maha" (Mahalanobis) or "gaussian".

        Returns:
            np.ndarray: Array of distances. Returns np.inf for values outside 
                the chi-square 95% confidence interval.
        """
        mean, covariance = self.project(mean, covariance)
        if only_position:
            mean, covariance = mean[:2], covariance[:2, :2]
            measurements = measurements[:, :2]

        cholesky_factor = scipy.linalg.cho_factor(
            covariance, lower=True, check_finite=False
        )
        d = measurements - mean
        z = scipy.linalg.cho_solve(cholesky_factor, d.T, check_finite=False).T
        squared_maha = np.sum(d * z, axis=1)

        if metric == "gaussian":
            return np.sum(d * d, axis=1)
        elif metric == "maha":
            limit = self.chi2inv95[2 if only_position else 4]
            squared_maha[squared_maha > limit] = np.inf
            return squared_maha
        else:
            raise ValueError(f"Unknown metric: {metric}")   

        # Use chi-square threshold based on degrees of freedom
        limit = self.chi2inv95[2 if only_position else 4]
        squared_maha[squared_maha > limit] = np.inf
        
        return squared_maha