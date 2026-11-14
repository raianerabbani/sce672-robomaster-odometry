"""Smooth Motion Controller for Straight-Line Approach and Controlled Stopping.

SCE-672: Advanced Robotics, KFUPM
Redesigned with:
- Slew-rate acceleration & braking shaping (prevents launch/brake wheel slip & jerk)
- Above-stiction minimum approach velocity (avoids M3508I ESC low-RPM cogging)
- Low-pass EMA filtering on visual bearing + deadband (eliminates steering wiggle)
- Automatic stop trigger upon reaching stopping distance.
"""

from typing import Tuple, Optional
import numpy as np


class StraightLineMotionController:
    """Smooth 50 Hz controller guiding RoboMaster EP straight towards the detected tag."""

    def __init__(
        self,
        stop_distance: float = 0.25,
        v_cruise: float = 0.28,
        v_min: float = 0.14,
        decel_distance: float = 0.45,
        max_accel: float = 0.60,
        kp_yaw: float = 0.65,
        kd_yaw: float = 0.04,
        max_yaw_rate: float = 20.0,
        yaw_deadband_deg: float = 1.2,
    ):
        """Initialize smooth motion controller.

        Args:
            stop_distance: Target stopping distance from robot reference point to tag (m).
            v_cruise: Nominal forward cruising speed (m/s, ~80 RPM).
            v_min: Minimum smooth crawl speed before final stop (m/s, ~40 RPM, above ESC stiction).
            decel_distance: Distance before stop_distance where smooth braking begins (m).
            max_accel: Longitudinal slew-rate acceleration limit (m/s^2).
            kp_yaw: Proportional gain for visual yaw centering (deg/s per deg error).
            kd_yaw: Derivative gain for visual yaw damping.
            max_yaw_rate: Yaw rate saturation limit (deg/s).
            yaw_deadband_deg: Bearing deadband (deg) within which wheels run identically for straight travel.
        """
        self.stop_distance = float(stop_distance)
        self.v_cruise = float(v_cruise)
        self.v_min = float(v_min)
        self.decel_distance = float(decel_distance)
        self.max_accel = float(max_accel)
        self.kp_yaw = float(kp_yaw)
        self.kd_yaw = float(kd_yaw)
        self.max_yaw_rate = float(max_yaw_rate)
        self.yaw_deadband_deg = float(yaw_deadband_deg)

        self.vx_shaped: float = 0.0
        self.wz_shaped: float = 0.0
        self.bearing_filt: Optional[float] = None
        self.last_bearing_filt: Optional[float] = None
        self._outlier_count: int = 0
        self.last_time: Optional[float] = None
        self.target_reached: bool = False

    def reset(self, stop_distance: Optional[float] = None):
        """Reset controller state for a new trial."""
        if stop_distance is not None:
            self.stop_distance = float(stop_distance)
        self.vx_shaped = 0.0
        self.wz_shaped = 0.0
        self.bearing_filt = None
        self.last_bearing_filt = None
        self._outlier_count = 0
        self.last_time = None
        self.target_reached = False

    def compute_velocity(
        self,
        current_distance: float,
        tag_bearing_deg: Optional[float],
        current_time: float,
    ) -> Tuple[float, float, float, bool]:
        """Compute smooth slew-rate-limited chassis velocities (vx, vy, wz)."""
        if self.last_time is None:
            dt = 0.020
        else:
            dt = float(np.clip(current_time - self.last_time, 0.005, 0.10))
        self.last_time = current_time

        # Stopping condition
        if current_distance <= self.stop_distance or self.target_reached:
            self.target_reached = True
            self.vx_shaped = 0.0
            self.wz_shaped = 0.0
            return 0.0, 0.0, 0.0, True

        # 1. Target longitudinal speed profile
        dist_to_stop = current_distance - self.stop_distance
        if dist_to_stop >= self.decel_distance:
            vx_target = self.v_cruise
        else:
            ratio = float(np.clip(dist_to_stop / max(self.decel_distance, 0.05), 0.0, 1.0))
            # Smooth cosine-like ramp down to v_min (kept above ESC stiction deadband)
            vx_target = self.v_min + (self.v_cruise - self.v_min) * (0.5 * (1.0 - np.cos(np.pi * ratio)))

        # Slew-rate limit longitudinal acceleration/deceleration
        max_dv = self.max_accel * dt
        self.vx_shaped = float(self.vx_shaped + np.clip(vx_target - self.vx_shaped, -max_dv, max_dv))

        # 2. Pure forward motion (no lateral strafe)
        vy = 0.0

        # 3. Filtered visual yaw alignment (clockwise positive = turn right)
        if tag_bearing_deg is not None:
            raw_b = float(tag_bearing_deg)
            if self.bearing_filt is None:
                self.bearing_filt = raw_b
                self._outlier_count = 0
            elif abs(raw_b - self.bearing_filt) > 20.0 and self._outlier_count < 2:
                # Reject sudden single/double-frame bearing jump spikes (e.g. false tag glitch)
                self._outlier_count += 1
            else:
                self._outlier_count = 0
                self.bearing_filt = 0.25 * raw_b + 0.75 * self.bearing_filt

            bearing_rate = 0.0
            if self.last_bearing_filt is not None and dt > 0.001:
                bearing_rate = float(np.clip(
                    (self.bearing_filt - self.last_bearing_filt) / dt,
                    -60.0,
                    60.0
                ))
            self.last_bearing_filt = self.bearing_filt

            if abs(self.bearing_filt) < self.yaw_deadband_deg:
                wz_target = 0.0
            else:
                # Subtract deadband smoothly so transition is continuous
                err_eff = self.bearing_filt - np.sign(self.bearing_filt) * self.yaw_deadband_deg
                wz_target = float(np.clip(
                    self.kp_yaw * err_eff + self.kd_yaw * bearing_rate,
                    -self.max_yaw_rate,
                    self.max_yaw_rate
                ))
        else:
            # Tag momentarily unseen: clear derivative history to avoid kick on re-acquisition,
            # gently decay filtered bearing, and command 0 yaw rate to hold straight heading.
            self.last_bearing_filt = None
            self._outlier_count = 0
            if self.bearing_filt is not None:
                self.bearing_filt *= max(0.0, 1.0 - 2.0 * dt)
            wz_target = 0.0

        max_dwz = 90.0 * dt
        self.wz_shaped = float(self.wz_shaped + np.clip(wz_target - self.wz_shaped, -max_dwz, max_dwz))

        return float(self.vx_shaped), float(vy), float(self.wz_shaped), False
