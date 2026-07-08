"""
HumanMM — Temporal Filters for smooth tracking and pose estimation.
"""
import math
import numpy as np

def smoothing_factor(t_e: float, cutoff: float) -> float:
    r = 2 * math.pi * cutoff * t_e
    return r / (r + 1.0)

def exponential_smoothing(a: np.ndarray, x: np.ndarray, x_prev: np.ndarray) -> np.ndarray:
    return a * x + (1.0 - a) * x_prev

class OneEuroFilter:
    """
    One Euro Filter for noisy coordinate signals (e.g., human pose keypoints).
    Based on the paper "1€ Filter: A Simple Speed-based Low-pass Filter for Noisy Input in Interactive Systems"
    """
    def __init__(self, t0: float, x0: np.ndarray, dx0: float = 0.0, min_cutoff: float = 1.0, beta: float = 0.0, d_cutoff: float = 1.0):
        """
        Initialize the One Euro Filter.
        """
        self.min_cutoff = float(min_cutoff)
        self.beta = float(beta)
        self.d_cutoff = float(d_cutoff)
        
        self.x_prev = x0.astype(np.float32)
        self.dx_prev = np.zeros_like(x0, dtype=np.float32) + dx0
        self.t_prev = float(t0)

    def __call__(self, t: float, x: np.ndarray) -> np.ndarray:
        """
        Compute the filtered signal.
        """
        t_e = t - self.t_prev
        
        if t_e <= 0.0:
            return self.x_prev
            
        # The noisy derivative
        dx = (x - self.x_prev) / t_e
            
        # Exponential smoothing for the derivative
        a_d = smoothing_factor(t_e, self.d_cutoff)
        dx_hat = exponential_smoothing(np.full_like(dx, a_d), dx, self.dx_prev)
        
        # The cutoff frequency based on the magnitude of the smoothed derivative
        # For multi-dimensional signals (like (J, 2) keypoints), we calculate the norm for each joint
        speed = np.linalg.norm(dx_hat, axis=-1, keepdims=True)
        cutoff = self.min_cutoff + self.beta * speed
        
        # Exponential smoothing for the signal
        a = smoothing_factor(t_e, cutoff)
        x_hat = exponential_smoothing(a, x, self.x_prev)
        
        # Update history
        self.x_prev = x_hat
        self.dx_prev = dx_hat
        self.t_prev = t
        
        return x_hat
