"""Kalman Filter for Odometry-based Prediction and Sensor-based Correction.

SCE-672: Advanced Robotics, KFUPM
Implements state estimation for RoboMaster EP tag approach:
- State vector: x = [d, v]^T (distance to tag, forward velocity)
- Prediction: Driven by Mecanum wheel odometry displacement
- Correction: Driven by visual tag distance measurement from camera
- Baseline: Simultaneously tracks pure dead-reckoning odometry to quantify drift.
"""

from dataclasses import dataclass
from typing import Optional, Tuple, Dict, Any
import numpy as np


@dataclass
class FilterState:
    """Current snapshot of filter estimation."""
    distance: float                 # Estimated distance to tag from robot reference point (m)
    velocity: float                 # Estimated velocity towards tag (m/s)
    pure_odometry_distance: float   # Pure dead-reckoning distance without correction (m)
    variance_pos: float             # P[0,0] position error variance (m^2)
    variance_vel: float             # P[1,1] velocity error variance ((m/s)^2)
    kalman_gain: float              # K[0] position Kalman gain
    innovation: Optional[float]     # Residual z - H x_pred (m), None if no measurement
    has_measurement: bool           # True if visual correction occurred this step


class OdometryVisionKalmanFilter:
    """Discrete-time Kalman Filter fusing chassis odometry and visual tag measurements."""

    def __init__(
        self,
        initial_distance: float = 1.50,
        camera_to_front_offset: float = 0.05,
        q_pos: float = 0.0009,
        q_vel: float = 0.0025,
        r_vis: float = 0.0016,
        p0_pos: float = 0.0025,
        p0_vel: float = 0.0100,
    ):
        """Initialize Kalman Filter.

        Args:
            initial_distance: Starting physical distance from reference point to tag (m).
            camera_to_front_offset: Distance from camera optical center to front bumper (m).
            q_pos: Process noise variance for distance (m^2/s).
            q_vel: Process noise variance for velocity ((m/s)^2/s).
            r_vis: Measurement noise variance for vision sensor (m^2).
            p0_pos: Initial position uncertainty variance (m^2).
            p0_vel: Initial velocity uncertainty variance ((m/s)^2).
        """
        self.initial_distance = float(initial_distance)
        self.camera_to_front_offset = float(camera_to_front_offset)

        # State vector: x = [d, v]^T (distance, velocity towards tag)
        self.x = np.array([[self.initial_distance], [0.0]], dtype=np.float64)

        # Pure odometry baseline for comparison (accumulates uncorrected wheel dead-reckoning)
        self.pure_odo_dist = float(self.initial_distance)
        self.initial_odo_pos: Optional[float] = None
        self.last_odo_pos: Optional[float] = None

        # Covariance matrix P
        self.P = np.array([
            [p0_pos, 0.0],
            [0.0, p0_vel]
        ], dtype=np.float64)

        # Base process noise Q
        self.q_pos_base = float(q_pos)
        self.q_vel_base = float(q_vel)

        # Measurement noise R
        self.R = float(r_vis)

        # Measurement matrix H: z = d + v_meas -> H = [1, 0]
        self.H = np.array([[1.0, 0.0]], dtype=np.float64)

        # Last step statistics
        self.last_innovation: Optional[float] = None
        self.last_K = np.zeros((2, 1), dtype=np.float64)
        self.last_has_measurement = False

    def reset(self, initial_distance: Optional[float] = None):
        """Reset filter states for a new experimental trial."""
        if initial_distance is not None:
            self.initial_distance = float(initial_distance)
        self.x = np.array([[self.initial_distance], [0.0]], dtype=np.float64)
        self.pure_odo_dist = float(self.initial_distance)
        self.initial_odo_pos = None
        self.last_odo_pos = None
        self.P = np.array([
            [0.0025, 0.0],
            [0.0, 0.0100]
        ], dtype=np.float64)
        self.last_innovation = None
        self.last_K = np.zeros((2, 1), dtype=np.float64)
        self.last_has_measurement = False

    def predict(self, odo_pos: float, dt: float = 0.02) -> np.ndarray:
        """Prediction step using wheel odometry displacement.

        Args:
            odo_pos: Cumulative chassis forward position reported by robot (meters).
            dt: Time step duration since last prediction (seconds).

        Returns:
            A priori state estimate x_{k|k-1}.
        """
        dt = max(dt, 0.001)

        # Establish odometry baseline on first tick
        if self.last_odo_pos is None:
            self.initial_odo_pos = float(odo_pos)
            self.last_odo_pos = float(odo_pos)
            delta_odo = 0.0
        else:
            delta_odo = float(odo_pos - self.last_odo_pos)
            self.last_odo_pos = float(odo_pos)

        # Pure odometry baseline propagation (distance decreases as robot moves forward)
        self.pure_odo_dist = max(0.0, self.pure_odo_dist - delta_odo)

        # State transition:
        # d_k = d_{k-1} - delta_odo
        # v_k = delta_odo / dt
        # Or using continuous kinematics F = [[1, -dt], [0, 1]]
        F = np.array([
            [1.0, -dt],
            [0.0,  1.0]
        ], dtype=np.float64)

        # We incorporate delta_odo directly into the state prediction
        # moving towards tag -> distance decreases by delta_odo
        d_pred = self.x[0, 0] - delta_odo
        v_pred = delta_odo / dt if dt > 0 else self.x[1, 0]

        # Process noise covariance scaled by dt and wheel movement
        # Mecanum wheels slip more during acceleration and travel
        slip_scaling = 1.0 + 3.0 * abs(delta_odo)
        Q = np.array([
            [self.q_pos_base * dt * slip_scaling, 0.0],
            [0.0, self.q_vel_base * dt * slip_scaling]
        ], dtype=np.float64)

        # Update a priori state & covariance
        self.x = np.array([[d_pred], [v_pred]], dtype=np.float64)
        self.P = F @ self.P @ F.T + Q

        # Enforce covariance symmetry
        self.P = 0.5 * (self.P + self.P.T)

        self.last_has_measurement = False
        self.last_innovation = None
        return self.x.copy()

    def correct(self, visual_distance_raw: float, r_custom: Optional[float] = None) -> np.ndarray:
        """Correction step using visual tag distance measurement.

        Args:
            visual_distance_raw: Camera-to-tag distance along optical Z axis (meters).
            r_custom: Optional custom measurement variance for this update.

        Returns:
            A posteriori state estimate x_{k|k}.
        """
        # Convert camera-to-tag distance to robot reference point (front bumper)
        z = float(visual_distance_raw) - self.camera_to_front_offset

        # Measurement noise variance
        # Visual distance noise typically grows quadratically with distance (pinhole optics)
        r_meas = self.R if r_custom is None else float(r_custom)
        # Scaled by distance squared: z_std ~ 0.02 * z
        distance_factor = max(1.0, (z / 1.0) ** 2)
        R_effective = r_meas * distance_factor

        # Innovation (residual)
        y = z - (self.H @ self.x)[0, 0]
        self.last_innovation = float(y)

        # Innovation covariance S = H P H^T + R
        S = (self.H @ self.P @ self.H.T)[0, 0] + R_effective

        # Innovation gating (Chi-square gate: y^2 / S <= 9.0 or abs(y) <= 0.25)
        # Prevents corrupted visual frames or outlier glitches from disrupting state
        if (y ** 2) / S > 9.0 and abs(y) > 0.25:
            self.last_has_measurement = False
            return self.x.copy()

        # Kalman gain K = P H^T / S
        K = (self.P @ self.H.T) / S
        self.last_K = K.copy()

        # State update: x = x + K y
        self.x = self.x + K * y

        # Joseph form covariance update for numerical stability:
        # P = (I - K H) P (I - K H)^T + K R K^T
        I = np.eye(2, dtype=np.float64)
        IKH = I - K @ self.H
        self.P = IKH @ self.P @ IKH.T + K * R_effective @ K.T
        self.P = 0.5 * (self.P + self.P.T)

        self.last_has_measurement = True
        return self.x.copy()

    def get_state(self) -> FilterState:
        """Return the current filter state estimate."""
        return FilterState(
            distance=float(self.x[0, 0]),
            velocity=float(self.x[1, 0]),
            pure_odometry_distance=float(self.pure_odo_dist),
            variance_pos=float(self.P[0, 0]),
            variance_vel=float(self.P[1, 1]),
            kalman_gain=float(self.last_K[0, 0]),
            innovation=self.last_innovation,
            has_measurement=self.last_has_measurement
        )
