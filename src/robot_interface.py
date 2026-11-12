"""Robot Interface Module for RoboMaster EP and High-Fidelity Simulation.

SCE-672: Advanced Robotics, KFUPM
Provides:
- BaseRobot: Unified interface for physical hardware and simulator
- RoboMasterRobot: Direct communication with DJI RoboMaster EP via official SDK
- MockRobot: Hardware-in-the-loop simulation with Mecanum slip, noise, and perspective rendering.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, Tuple
import time
import math
import numpy as np
import cv2

try:
    from robomaster import robot, camera, led
    ROBOMASTER_AVAILABLE = True
except ImportError:
    ROBOMASTER_AVAILABLE = False


# RoboMaster EP Core Mecanum geometry constants
# RoboMaster EP Core M3508I ESC velocity-to-RPM scaling (3.50 m/s = 1000 RPM)
MAX_CHASSIS_SPEED_MPS = 3.50
MAX_WHEEL_RPM = 1000.0
YAW_SCALE_DEGS = 250.0
YAW_SCALE_RPM = 600.0


def mecanum_inverse_kinematics(
    vx: float,
    vy: float,
    wz_deg: float,
) -> Tuple[int, int, int, int]:
    """Convert chassis velocities (vx m/s, vy m/s, wz_deg deg/s CW) to wheel RPMs (w1, w2, w3, w4).

    Uses the calibrated DJI RoboMaster EP M3508I ESC mapping (3.50 m/s -> 1000 RPM).
    Wheel layout (positive = forward wheel rotation):
        w1: Front-Right, w2: Front-Left, w3: Rear-Left, w4: Rear-Right
    """
    rpm_x = (float(vx) / MAX_CHASSIS_SPEED_MPS) * MAX_WHEEL_RPM
    rpm_y = (float(vy) / MAX_CHASSIS_SPEED_MPS) * MAX_WHEEL_RPM
    rpm_z = (float(wz_deg) / YAW_SCALE_DEGS) * YAW_SCALE_RPM

    w1 = rpm_x - rpm_y - rpm_z  # FR
    w2 = rpm_x + rpm_y + rpm_z  # FL
    w3 = rpm_x - rpm_y + rpm_z  # RL
    w4 = rpm_x + rpm_y - rpm_z  # RR

    max_w = max(abs(w1), abs(w2), abs(w3), abs(w4))
    if max_w > MAX_WHEEL_RPM:
        scale = MAX_WHEEL_RPM / max_w
        w1, w2, w3, w4 = w1 * scale, w2 * scale, w3 * scale, w4 * scale

    return (
        int(np.clip(round(w1), -1000, 1000)),
        int(np.clip(round(w2), -1000, 1000)),
        int(np.clip(round(w3), -1000, 1000)),
        int(np.clip(round(w4), -1000, 1000)),
    )


def mecanum_forward_kinematics(
    w1: float,
    w2: float,
    w3: float,
    w4: float,
) -> Tuple[float, float, float]:
    """Convert wheel RPMs (w1, w2, w3, w4) to chassis velocities (vx m/s, vy m/s, wz_deg deg/s CW)."""
    rpm_x = 0.25 * (w1 + w2 + w3 + w4)
    rpm_y = 0.25 * (-w1 + w2 - w3 + w4)
    rpm_z = 0.25 * (-w1 + w2 + w3 - w4)

    vx = (rpm_x / MAX_WHEEL_RPM) * MAX_CHASSIS_SPEED_MPS
    vy = (rpm_y / MAX_WHEEL_RPM) * MAX_CHASSIS_SPEED_MPS
    wz_deg = (rpm_z / YAW_SCALE_RPM) * YAW_SCALE_DEGS
    return float(vx), float(vy), float(wz_deg)


class BaseRobot(ABC):
    """Abstract base class for robot interactions."""

    @abstractmethod
    def connect(self) -> bool:
        """Establish connection with robot hardware or simulator."""
        pass

    @abstractmethod
    def disconnect(self):
        """Safely stop motors and close connection."""
        pass

    @abstractmethod
    def get_telemetry(self) -> Dict[str, Any]:
        """Return latest telemetry snapshot: odometry x, y, yaw, velocity, battery."""
        pass

    @abstractmethod
    def get_frame(self) -> Optional[np.ndarray]:
        """Return latest camera frame (BGR format) or None."""
        pass

    @abstractmethod
    def drive_wheels(self, w1: int, w2: int, w3: int, w4: int):
        """Command individual wheel speeds in RPM (FR, FL, RL, RR)."""
        pass

    @abstractmethod
    def drive_speed(self, vx: float, vy: float, wz: float):
        """Command chassis velocities via inverse kinematics to individual wheels."""
        pass

    @abstractmethod
    def stop(self):
        """Immediately halt chassis motion."""
        pass

    @abstractmethod
    def set_led(self, r: int, g: int, b: int):
        """Set chassis LED indicators."""
        pass

    @abstractmethod
    def reset_odometry(self):
        """Zero the local odometry reference accumulator."""
        pass


class RoboMasterRobot(BaseRobot):
    """Physical DJI RoboMaster EP Core hardware interface."""

    def __init__(self, conn_type: str = "ap", resolution: str = "720p"):
        if not ROBOMASTER_AVAILABLE:
            raise RuntimeError("RoboMaster SDK is not installed or available.")

        self.conn_type = conn_type
        self.resolution = resolution
        self.ep_robot: Optional[robot.Robot] = None

        self._connected = False
        self._odo_x = 0.0
        self._odo_y = 0.0
        self._odo_yaw = 0.0
        self._odo_x_offset = 0.0

        self._vel_x = 0.0
        self._vel_y = 0.0
        self._vel_wz = 0.0

        self._imu_yaw = 0.0
        self._imu_pitch = 0.0
        self._imu_roll = 0.0
        self._battery = 100
        self._cmd_wheels = (0, 0, 0, 0)
        self._esc_wheels = (0, 0, 0, 0)

        self._last_frame = None

    def connect(self) -> bool:
        print(f"[Hardware] Connecting to DJI RoboMaster EP via '{self.conn_type}'...")
        try:
            self.ep_robot = robot.Robot()
            self.ep_robot.initialize(conn_type=self.conn_type)

            # Subscriptions at 50 Hz
            self.ep_robot.chassis.sub_position(
                cs=0, freq=50,
                callback=self._on_sub_position
            )
            self.ep_robot.chassis.sub_velocity(
                freq=50,
                callback=self._on_sub_velocity
            )
            self.ep_robot.chassis.sub_attitude(
                freq=50,
                callback=self._on_sub_attitude
            )
            self.ep_robot.chassis.sub_esc(
                freq=50,
                callback=self._on_sub_esc
            )
            self.ep_robot.battery.sub_battery_info(
                50,
                callback=lambda b: setattr(self, "_battery", int(b))
            )

            # Camera stream init
            res_const = camera.STREAM_720P if self.resolution == "720p" else camera.STREAM_360P
            self.ep_robot.camera.start_video_stream(display=False, resolution=res_const)

            # Set LED to solid cyan (connected)
            self.set_led(0, 220, 150)
            self._connected = True
            time.sleep(0.5)
            self.reset_odometry()
            print("[Hardware] Connected successfully! Telemetry & Camera active.")
            return True
        except Exception as e:
            print(f"[Hardware Error] Failed to connect: {e}")
            self._connected = False
            return False

    def _on_sub_position(self, pos_info):
        # pos_info: (x, y, z/yaw)
        try:
            self._odo_x = float(pos_info[0])
            self._odo_y = float(pos_info[1])
            if len(pos_info) > 2:
                self._odo_yaw = float(pos_info[2])
        except Exception:
            pass

    def _on_sub_velocity(self, vel_info):
        try:
            # vel_info: (vx, vy, vz, v_body_x)
            self._vel_x = float(vel_info[0])
            self._vel_y = float(vel_info[1])
            self._vel_wz = float(vel_info[2])
        except Exception:
            pass

    def _on_sub_attitude(self, att_info):
        try:
            self._imu_yaw = float(att_info[0])
            self._imu_pitch = float(att_info[1])
            self._imu_roll = float(att_info[2])
        except Exception:
            pass

    def _on_sub_esc(self, esc_info):
        try:
            spd = esc_info[0]
            self._esc_wheels = (int(spd[0]), int(spd[1]), int(spd[2]), int(spd[3]))
        except Exception:
            pass

    def get_telemetry(self) -> Dict[str, Any]:
        return {
            "odo_x": float(self._odo_x - self._odo_x_offset),
            "odo_y": float(self._odo_y),
            "odo_yaw": float(self._odo_yaw),
            "vel_x": float(self._vel_x),
            "vel_wz": float(self._vel_wz),
            "imu_yaw": float(self._imu_yaw),
            "wheel_rpms": self._cmd_wheels,
            "esc_rpms": self._esc_wheels,
            "battery": int(self._battery),
            "timestamp": time.time(),
        }

    def get_frame(self) -> Optional[np.ndarray]:
        if not self._connected or self.ep_robot is None:
            return None
        try:
            frame = self.ep_robot.camera.read_cv2_image(strategy="newest", timeout=0.5)
            if frame is not None:
                self._last_frame = frame
            return self._last_frame
        except Exception:
            return self._last_frame

    def drive_wheels(self, w1: int, w2: int, w3: int, w4: int):
        """Directly command individual wheel RPMs (FR, FL, RL, RR) at 50 Hz without timeout timers."""
        self._cmd_wheels = (int(w1), int(w2), int(w3), int(w4))
        if self._connected and self.ep_robot is not None:
            try:
                self.ep_robot.chassis.drive_wheels(
                    w1=int(w1), w2=int(w2), w3=int(w3), w4=int(w4)
                )
            except Exception as e:
                print(f"[Drive Wheels Error]: {e}")

    def drive_speed(self, vx: float, vy: float, wz: float):
        """Convert (vx, vy, wz) to individual wheel RPMs via Mecanum inverse kinematics."""
        w1, w2, w3, w4 = mecanum_inverse_kinematics(vx, vy, wz)
        self.drive_wheels(w1, w2, w3, w4)

    def stop(self):
        self._cmd_wheels = (0, 0, 0, 0)
        if self._connected and self.ep_robot is not None:
            try:
                for _ in range(3):
                    self.ep_robot.chassis.drive_wheels(w1=0, w2=0, w3=0, w4=0)
                    time.sleep(0.01)
            except Exception:
                pass

    def set_led(self, r: int, g: int, b: int):
        if self._connected and self.ep_robot is not None:
            try:
                self.ep_robot.led.set_led(
                    comp=led.COMP_ALL,
                    r=int(np.clip(r, 0, 255)),
                    g=int(np.clip(g, 0, 255)),
                    b=int(np.clip(b, 0, 255)),
                    effect=led.EFFECT_ON
                )
            except Exception:
                pass

    def reset_odometry(self):
        self._odo_x_offset = float(self._odo_x)

    def disconnect(self):
        if self.ep_robot is not None:
            print("[Hardware] Stopping and closing connection...")
            try:
                self.stop()
                self.ep_robot.camera.stop_video_stream()
                self.ep_robot.close()
            except Exception:
                pass
            self._connected = False
            self.ep_robot = None
            print("[Hardware] Disconnected.")


class MockRobot(BaseRobot):
    """High-fidelity simulated RoboMaster EP with realistic kinematics, Mecanum slip & vision."""

    def __init__(
        self,
        start_distance: float = 1.50,
        tag_size: float = 0.15,
        tag_type: str = "apriltag",
        tag_id: int = 1,
        qr_payload: str = "SCE672_ROBOMASTER_TARGET",
        slip_ratio: float = 0.05,
        fps: int = 30
    ):
        self.start_distance = float(start_distance)
        self.tag_size = float(tag_size)
        self.tag_type = tag_type.lower()
        self.tag_id = int(tag_id)
        self.qr_payload = qr_payload
        self.slip_ratio = float(slip_ratio)  # 5% Mecanum slip baseline
        self.fps = fps

        # Robot kinematic state
        self.true_distance = float(start_distance)  # True distance to tag
        self.true_x = 0.0                           # Ground truth forward displacement
        self.true_y = 0.0                           # Ground truth lateral displacement
        self.true_yaw = 0.0                         # Ground truth heading (degrees)

        self.odo_x = 0.0                            # Odometry reported forward displacement (with slip)
        self.odo_y = 0.0
        self.odo_yaw = 0.0

        self.cmd_vx = 0.0
        self.cmd_vy = 0.0
        self.cmd_wz = 0.0

        self.last_update_time = time.time()
        self._connected = False

        # Pre-render high-res tag image (AprilTag or QR) for synthetic camera projection
        if self.tag_type in ("apriltag", "auto"):
            at_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_APRILTAG_36h11)
            tag_scaled = cv2.aruco.generateImageMarker(at_dict, self.tag_id, 300)
        else:
            enc = cv2.QRCodeEncoder.create()
            qr_matrix = enc.encode(self.qr_payload)
            tag_scaled = cv2.resize(qr_matrix, (300, 300), interpolation=cv2.INTER_NEAREST)
        tag_gray = cv2.copyMakeBorder(tag_scaled, 45, 45, 45, 45, cv2.BORDER_CONSTANT, value=255)
        self.qr_img = cv2.cvtColor(tag_gray, cv2.COLOR_GRAY2BGR)

        # Camera intrinsics (720p: 1280x720)
        self.width = 1280
        self.height = 720
        self.K = np.array([
            [580.0, 0.0, 640.0],
            [0.0, 580.0, 360.0],
            [0.0, 0.0, 1.0]
        ], dtype=np.float32)
        self.dist_coeffs = np.zeros((4, 1), dtype=np.float32)

        # Camera coordinates: X right, Y down, Z forward
        # Object points centered:
        # Top-Left (-half, -half), Top-Right (+half, -half),
        # Bottom-Right (+half, +half), Bottom-Left (-half, +half)
        half = self.tag_size / 2.0
        self.obj_pts = np.array([
            [-half, -half, 0.0],
            [ half, -half, 0.0],
            [ half,  half, 0.0],
            [-half,  half, 0.0]
        ], dtype=np.float32)

    def connect(self) -> bool:
        self._connected = True
        self.last_update_time = time.time()
        self.reset_odometry()
        print(f"[Simulator] Initialized at initial distance {self.start_distance:.2f}m from QR tag.")
        return True

    def disconnect(self):
        self.stop()
        self._connected = False
        print("[Simulator] Disconnected.")

    def reset_odometry(self):
        self.odo_x = 0.0
        self.odo_y = 0.0
        self.odo_yaw = 0.0
        self.true_x = 0.0
        self.true_y = 0.0
        self.true_yaw = 0.0
        self.true_distance = float(self.start_distance)
        self.last_update_time = time.time()

    def set_start_distance(self, dist: float):
        self.start_distance = float(dist)
        self.reset_odometry()

    def _update_physics(self):
        now = time.time()
        dt = now - self.last_update_time
        if dt <= 0:
            return
        self.last_update_time = now

        # Update true position based on commanded velocities
        # True forward motion includes slight Mecanum roller slip:
        # Wheel spins at vx, but actual displacement is vx * (1 - slip)
        true_vx = self.cmd_vx * (1.0 - self.slip_ratio)
        true_wz = self.cmd_wz

        dx = true_vx * dt
        dy = self.cmd_vy * dt
        dyaw = true_wz * dt

        self.true_x += dx
        self.true_y += dy
        self.true_yaw += dyaw
        self.true_distance = max(0.01, self.start_distance - self.true_x)

        # Odometry reading:
        # Wheel encoders measure wheel rotations directly without sensing slip:
        # Therefore, odometry registers full commanded vx (plus encoder quantization noise)
        encoder_noise = np.random.normal(0, 0.0005)
        self.odo_x += self.cmd_vx * dt + encoder_noise
        self.odo_y += self.cmd_vy * dt
        self.odo_yaw += self.cmd_wz * dt + np.random.normal(0, 0.01)

    def drive_wheels(self, w1: int, w2: int, w3: int, w4: int):
        """Simulate individual wheel speed commands via Mecanum forward kinematics."""
        self._update_physics()
        self._cmd_wheels = (int(w1), int(w2), int(w3), int(w4))
        vx, vy, wz_deg = mecanum_forward_kinematics(w1, w2, w3, w4)
        self.cmd_vx = float(vx)
        self.cmd_vy = float(vy)
        self.cmd_wz = float(wz_deg)

    def drive_speed(self, vx: float, vy: float, wz: float):
        w1, w2, w3, w4 = mecanum_inverse_kinematics(vx, vy, wz)
        self.drive_wheels(w1, w2, w3, w4)

    def stop(self):
        self._cmd_wheels = (0, 0, 0, 0)
        self.cmd_vx = 0.0
        self.cmd_vy = 0.0
        self.cmd_wz = 0.0
        self._update_physics()

    def set_led(self, r: int, g: int, b: int):
        pass

    def get_telemetry(self) -> Dict[str, Any]:
        self._update_physics()
        return {
            "odo_x": float(self.odo_x),
            "odo_y": float(self.odo_y),
            "odo_yaw": float(self.odo_yaw),
            "vel_x": float(self.cmd_vx),
            "vel_wz": float(self.cmd_wz),
            "imu_yaw": float(self.true_yaw),
            "wheel_rpms": getattr(self, "_cmd_wheels", (0, 0, 0, 0)),
            "true_distance": float(self.true_distance),
            "true_x": float(self.true_x),
            "battery": 98,
            "timestamp": time.time(),
        }

    def get_frame(self) -> np.ndarray:
        self._update_physics()

        # Render synthetic scene (1280x720)
        # Background: Realistic laboratory floor and wall gradient
        frame = np.full((self.height, self.width, 3), 235, dtype=np.uint8)

        # Floor horizon line at y=480
        cv2.rectangle(frame, (0, 480), (self.width, self.height), (205, 205, 200), -1)
        cv2.line(frame, (0, 480), (self.width, 480), (160, 160, 155), 2)

        # Grid lines on floor for spatial realism
        for gx in range(0, self.width, 160):
            cv2.line(frame, (gx, 480), (int((gx - 640) * 2.2 + 640), self.height), (180, 180, 175), 1)

        # Compute tag 3D location in camera coordinate frame
        # Camera is facing forward (+Z), X is right, Y is down
        tz = float(self.true_distance)
        tx = float(self.true_y) + float(tz * math.tan(math.radians(-self.true_yaw)))
        ty = 0.05  # tag mounted slightly below camera horizon

        # Project 3D tag corners to image plane
        tvec = np.array([[tx], [ty], [tz]], dtype=np.float32)
        # Orientation of tag relative to camera
        rvec = np.array([[0.0], [math.radians(self.true_yaw)], [0.0]], dtype=np.float32)

        img_pts, _ = cv2.projectPoints(self.obj_pts, rvec, tvec, self.K, self.dist_coeffs)
        img_pts = img_pts.reshape(4, 2)

        # Check if tag is inside image frame
        margin = 10
        in_bounds = (
            np.all(img_pts[:, 0] > margin) and
            np.all(img_pts[:, 0] < self.width - margin) and
            np.all(img_pts[:, 1] > margin) and
            np.all(img_pts[:, 1] < self.height - margin) and
            tz > 0.15
        )

        if in_bounds:
            b = 45
            src_pts = np.array([
                [b, b],
                [b + 300, b],
                [b + 300, b + 300],
                [b, b + 300]
            ], dtype=np.float32)
            M = cv2.getPerspectiveTransform(src_pts, img_pts)
            # Warp QR image onto frame with high quality area interpolation
            warped = cv2.warpPerspective(
                self.qr_img, M, (self.width, self.height),
                flags=cv2.INTER_AREA,
                borderMode=cv2.BORDER_CONSTANT, borderValue=(235, 235, 235)
            )
            mask = cv2.warpPerspective(
                np.ones(self.qr_img.shape[:2], dtype=np.uint8) * 255, M, (self.width, self.height),
                flags=cv2.INTER_NEAREST,
                borderMode=cv2.BORDER_CONSTANT, borderValue=0
            )
            # Mask composite
            mask_bool = mask > 128
            frame[mask_bool] = warped[mask_bool]

        # Add slight camera sensor Gaussian noise
        noise = np.random.normal(0, 1.5, frame.shape).astype(np.int16)
        noisy_frame = np.clip(frame.astype(np.int16) + noise, 0, 255).astype(np.uint8)

        return noisy_frame
