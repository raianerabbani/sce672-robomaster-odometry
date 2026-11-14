"""Straight-Line Motion & Yaw-Alignment Controller (50 Hz).

SCE-672: Advanced Robotics, KFUPM — Project Phase 3

Control pipeline (runs every 20 ms):
1. Stopping check   : Halts early by a small speed-dependent lead to cancel camera/braking lag.
2. Longitudinal vx  : Smooth cosine deceleration from v_cruise to v_min, slew-rate limited.
3. Visual yaw wz    : Low-pass filtered PD controller on tag bearing with deadband & spike guard.
4. Wheel RPM output : Maps (vx, 0, wz) -> (w1, w2, w3, w4) via Mecanum inverse kinematics.
"""

from typing import Optional, Tuple
import numpy as np

from src.robot_interface import mecanum_inverse_kinematics


class StraightLineMotionController:
    """Smooth 50 Hz controller guiding RoboMaster EP straight towards the detected tag."""

    def __init__(
        self,
        stop_distance: float = 0.25,
        v_cruise: float = 0.28,
        v_min: float = 0.105,
        decel_distance: float = 0.55,
        max_accel: float = 0.60,
        kp_yaw: float = 0.65,
        kd_yaw: float = 0.04,
        max_yaw_rate: float = 20.0,
        yaw_deadband_deg: float = 1.2,
        stop_lead_time: float = 0.25,
        stop_lead_min: float = 0.025,
    ):
        self.stop_distance = float(stop_distance)
        self.v_cruise = float(v_cruise)
        self.v_min = float(v_min)
        self.decel_distance = float(decel_distance)
        self.max_accel = float(max_accel)
        self.kp_yaw = float(kp_yaw)
        self.kd_yaw = float(kd_yaw)
        self.max_yaw_rate = float(max_yaw_rate)
        self.yaw_deadband_deg = float(yaw_deadband_deg)
        self.stop_lead_time = float(stop_lead_time)
        self.stop_lead_min = float(stop_lead_min)

        # Internal state
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

    def _compute_dt(self, current_time: float) -> float:
        """Compute bounded loop step dt (seconds)."""
        if self.last_time is None:
            dt = 0.020
        else:
            dt = float(np.clip(current_time - self.last_time, 0.005, 0.10))
        self.last_time = current_time
        return dt

    def _should_stop(self, current_distance: float) -> bool:
        """Check target distance with predictive lead for camera latency and braking inertia."""
        lead_dist = self.stop_lead_min + self.stop_lead_time * max(self.vx_shaped, 0.0)
        return self.target_reached or (current_distance <= self.stop_distance + lead_dist)

    def _compute_forward_speed(self, current_distance: float, dt: float) -> float:
        """Cosine deceleration profile with acceleration slew-rate limiting."""
        dist_to_stop = current_distance - self.stop_distance
        if dist_to_stop >= self.decel_distance:
            vx_target = self.v_cruise
        else:
            ratio = float(np.clip(dist_to_stop / max(self.decel_distance, 0.05), 0.0, 1.0))
            vx_target = self.v_min + (self.v_cruise - self.v_min) * (0.5 * (1.0 - np.cos(np.pi * ratio)))

        max_dv = self.max_accel * dt
        self.vx_shaped = float(self.vx_shaped + np.clip(vx_target - self.vx_shaped, -max_dv, max_dv))
        return self.vx_shaped

    def _compute_yaw_rate(self, tag_bearing_deg: Optional[float], dt: float) -> float:
        """Filtered PD yaw alignment with spike rejection and deadband."""
        if tag_bearing_deg is not None:
            raw_b = float(tag_bearing_deg)
            if self.bearing_filt is None:
                self.bearing_filt = raw_b
                self._outlier_count = 0
            elif abs(raw_b - self.bearing_filt) > 20.0 and self._outlier_count < 2:
                self._outlier_count += 1
            else:
                self._outlier_count = 0
                self.bearing_filt = 0.25 * raw_b + 0.75 * self.bearing_filt

            bearing_rate = 0.0
            if self.last_bearing_filt is not None and dt > 0.001:
                bearing_rate = float(np.clip((self.bearing_filt - self.last_bearing_filt) / dt, -60.0, 60.0))
            self.last_bearing_filt = self.bearing_filt

            if abs(self.bearing_filt) < self.yaw_deadband_deg:
                wz_target = 0.0
            else:
                err_eff = self.bearing_filt - np.sign(self.bearing_filt) * self.yaw_deadband_deg
                wz_target = float(np.clip(
                    self.kp_yaw * err_eff + self.kd_yaw * bearing_rate,
                    -self.max_yaw_rate,
                    self.max_yaw_rate,
                ))
        else:
            # Tag temporarily unseen: hold straight heading and avoid derivative kick on re-acquisition
            self.last_bearing_filt = None
            self._outlier_count = 0
            if self.bearing_filt is not None:
                self.bearing_filt *= max(0.0, 1.0 - 2.0 * dt)
            wz_target = 0.0

        max_dwz = 90.0 * dt
        self.wz_shaped = float(self.wz_shaped + np.clip(wz_target - self.wz_shaped, -max_dwz, max_dwz))
        return self.wz_shaped

    def compute_velocity(
        self,
        current_distance: float,
        tag_bearing_deg: Optional[float],
        current_time: float,
    ) -> Tuple[float, float, float, bool]:
        """Compute smooth chassis velocities (vx, vy, wz, target_reached)."""
        dt = self._compute_dt(current_time)

        if self._should_stop(current_distance):
            self.target_reached = True
            self.vx_shaped = 0.0
            self.wz_shaped = 0.0
            return 0.0, 0.0, 0.0, True

        vx = self._compute_forward_speed(current_distance, dt)
        vy = 0.0
        wz = self._compute_yaw_rate(tag_bearing_deg, dt)
        return vx, vy, wz, False

    def compute_wheel_rpms(
        self,
        current_distance: float,
        tag_bearing_deg: Optional[float],
        current_time: float,
    ) -> Tuple[Tuple[int, int, int, int], float, float, bool]:
        """Compute individual Mecanum wheel RPMs (FR, FL, RL, RR) for drive_wheels()."""
        vx, vy, wz, reached = self.compute_velocity(current_distance, tag_bearing_deg, current_time)
        wheels = (0, 0, 0, 0) if reached else mecanum_inverse_kinematics(vx, vy, wz)
        return wheels, vx, wz, reached
