"""Motion Controller for Straight-Line Approach and Controlled Stopping.

SCE-672: Advanced Robotics, KFUPM
Implements:
- Closed-loop visual yaw heading alignment to ensure strict straight-line motion
- Smooth deceleration profile as the robot approaches the target stopping distance
- Automatic stop trigger and hold command upon reaching stopping distance.
"""

from typing import Tuple, Optional
import numpy as np


class StraightLineMotionController:
    """Controller guiding RoboMaster EP straight towards the detected tag."""

    def __init__(
        self,
        stop_distance: float = 0.40,
        v_cruise: float = 0.18,
        v_min: float = 0.04,
        decel_distance: float = 0.40,
        kp_yaw: float = 0.70,
        kd_yaw: float = 0.05,
        max_yaw_rate: float = 25.0,
    ):
        """Initialize motion controller.

        Args:
            stop_distance: Target stopping distance from robot reference point to tag (m).
            v_cruise: Maximum forward cruising speed (m/s).
            v_min: Minimum crawl speed during final approach (m/s).
            decel_distance: Distance before stop_distance where braking begins (m).
            kp_yaw: Proportional gain for yaw centering (deg/s per deg error).
            kd_yaw: Derivative gain for yaw centering.
            max_yaw_rate: Yaw rate saturation limit (deg/s).
        """
        self.stop_distance = float(stop_distance)
        self.v_cruise = float(v_cruise)
        self.v_min = float(v_min)
        self.decel_distance = float(decel_distance)
        self.kp_yaw = float(kp_yaw)
        self.kd_yaw = float(kd_yaw)
        self.max_yaw_rate = float(max_yaw_rate)

        self.last_bearing_deg: Optional[float] = None
        self.last_time: Optional[float] = None
        self.target_reached = False

    def reset(self, stop_distance: Optional[float] = None):
        """Reset controller state for a new trial."""
        if stop_distance is not None:
            self.stop_distance = float(stop_distance)
        self.last_bearing_deg = None
        self.last_time = None
        self.target_reached = False

    def compute_velocity(
        self,
        current_distance: float,
        tag_bearing_deg: Optional[float],
        current_time: float,
    ) -> Tuple[float, float, float, bool]:
        """Compute chassis velocities (vx, vy, wz).

        Args:
            current_distance: Estimated remaining distance to tag (m) from Kalman Filter.
            tag_bearing_deg: Visual bearing to tag center in degrees (None if not visible).
            current_time: Current timestamp (seconds).

        Returns:
            Tuple of (vx [m/s], vy [m/s], wz [deg/s], target_reached [bool]).
        """
        # Check stopping condition
        if current_distance <= self.stop_distance or self.target_reached:
            self.target_reached = True
            return 0.0, 0.0, 0.0, True

        # Distance remaining until stopping point
        dist_to_stop = current_distance - self.stop_distance

        # Longitudinal velocity profile:
        # High cruise speed when far, smooth ramp down when approaching decel zone
        if dist_to_stop >= self.decel_distance:
            vx = self.v_cruise
        else:
            # Linear deceleration ramp down to v_min
            ratio = max(0.0, min(1.0, dist_to_stop / self.decel_distance))
            vx = self.v_min + (self.v_cruise - self.v_min) * (ratio ** 1.2)

        # Lateral velocity is 0 (pure forward motion)
        vy = 0.0

        # Yaw steering alignment for straight-line tracking
        if tag_bearing_deg is not None:
            # Positive bearing means tag is to the right of camera center -> rotate right (+wz in robomaster)
            # In RoboMaster SDK: positive wz is counter-clockwise (left), negative is clockwise (right).
            # Therefore: wz = -kp * bearing
            bearing_error = float(tag_bearing_deg)

            # Derivative of bearing
            bearing_rate = 0.0
            if self.last_bearing_deg is not None and self.last_time is not None:
                dt = current_time - self.last_time
                if dt > 0.001:
                    bearing_rate = (bearing_error - self.last_bearing_deg) / dt

            wz = -(self.kp_yaw * bearing_error + self.kd_yaw * bearing_rate)

            # Saturate yaw rate
            wz = float(np.clip(wz, -self.max_yaw_rate, self.max_yaw_rate))

            # Deadband to eliminate jitter when nearly aligned
            if abs(bearing_error) < 0.6:
                wz = 0.0

            self.last_bearing_deg = bearing_error
        else:
            # If tag momentarily lost, do not rotate; maintain straight forward heading
            wz = 0.0

        self.last_time = current_time
        return float(vx), float(vy), float(wz), False
