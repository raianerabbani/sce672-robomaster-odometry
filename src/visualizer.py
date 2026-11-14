"""Visualizer and Heads-Up Display (HUD) Module.

SCE-672: Advanced Robotics, KFUPM
Provides:
- Real-time OpenCV HUD with telemetry overlays, distance bars, and detection markers
- Video recording pipeline to produce submission-ready demonstration video
- Headless support if running in terminal without an X11 display.
"""

from typing import Optional, Tuple
import os
import time
import numpy as np
import cv2

from src.tag_detector import TagDetection
from src.kalman_filter import FilterState


class VisualizerHUD:
    """Overlays telemetry, state indicators, and Kalman metrics on video frames."""

    def __init__(
        self,
        window_name: str = "RoboMaster EP Odometry - SCE 672",
        start_distance: float = 1.50,
        stop_distance: float = 0.25,
        enable_display: bool = True,
        record_video: bool = True,
        video_output_path: str = "results/demo_video.mp4",
        fps: int = 30,
        frame_size: Tuple[int, int] = (1280, 720)
    ):
        self.window_name = window_name
        self.start_distance = float(start_distance)
        self.stop_distance = float(stop_distance)
        self.enable_display = enable_display
        self.record_video = record_video
        self.video_output_path = video_output_path
        self.fps = fps
        self.frame_size = frame_size

        self.video_writer: Optional[cv2.VideoWriter] = None
        self.display_active = False

        # Initialize video recorder if requested
        if self.record_video:
            os.makedirs(os.path.dirname(self.video_output_path) or ".", exist_ok=True)
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            self.video_writer = cv2.VideoWriter(
                self.video_output_path, fourcc, self.fps, self.frame_size
            )

        # Check if X11 display is available
        if self.enable_display and "DISPLAY" in os.environ:
            try:
                cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
                cv2.resizeWindow(self.window_name, 1024, 576)
                self.display_active = True
            except Exception:
                self.display_active = False

    def render_hud(
        self,
        frame: np.ndarray,
        det: TagDetection,
        kf_state: FilterState,
        trial_id: int,
        total_trials: int,
        state_label: str,
        vx_cmd: float,
        wz_cmd: float,
        battery: int = 95
    ) -> np.ndarray:
        """Render complete telemetry HUD onto the camera frame."""
        if frame is None or frame.size == 0:
            frame = np.full((self.frame_size[1], self.frame_size[0], 3), 30, dtype=np.uint8)

        hud = frame.copy()
        h, w = hud.shape[:2]

        # 1. Draw Tag Detection Overlay if detected
        if det.detected and det.corners is not None:
            pts = det.corners.astype(np.int32).reshape((-1, 1, 2))
            cv2.polylines(hud, [pts], isClosed=True, color=(0, 255, 100), thickness=2)
            cx, cy = int(det.center[0]), int(det.center[1])
            cv2.circle(hud, (cx, cy), 5, (0, 255, 255), -1)
            cv2.line(hud, (cx - 15, cy), (cx + 15, cy), (0, 255, 255), 1)
            cv2.line(hud, (cx, cy - 15), (cx, cy + 15), (0, 255, 255), 1)

            # Optical center target crosshair
            cv2.line(hud, (w // 2 - 20, h // 2), (w // 2 + 20, h // 2), (180, 180, 180), 1)
            cv2.line(hud, (w // 2, h // 2 - 20), (w // 2, h // 2 + 20), (180, 180, 180), 1)

            # Connect tag center to optical center
            cv2.line(hud, (w // 2, h // 2), (cx, cy), (0, 180, 255), 1, cv2.LINE_AA)

        # 2. Top Header Bar (Dark translucent gradient)
        cv2.rectangle(hud, (0, 0), (w, 54), (15, 15, 20), -1)
        cv2.line(hud, (0, 54), (w, 54), (60, 60, 75), 1)

        # Course & Project Title
        cv2.putText(hud, "SCE-672 ADVANCED ROBOTICS | PHASE 3: ODOMETRY & KALMAN ESTIMATION",
                    (20, 24), cv2.FONT_HERSHEY_DUPLEX, 0.55, (220, 220, 230), 1, cv2.LINE_AA)
        cv2.putText(hud, "King Fahd University of Petroleum & Minerals (KFUPM)",
                    (20, 44), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (150, 160, 180), 1, cv2.LINE_AA)

        # Trial badge on top right
        trial_text = f"TRIAL {trial_id:02d} / {total_trials:02d}"
        cv2.putText(hud, trial_text, (w - 180, 26), cv2.FONT_HERSHEY_DUPLEX, 0.65, (0, 230, 255), 1, cv2.LINE_AA)

        # Battery & Time
        cv2.putText(hud, f"BAT: {battery}%", (w - 180, 44), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 255, 150), 1)

        # 3. Left Telemetry Dashboard Panel (Dark translucent box)
        panel_w = 340
        panel_h = 240
        panel_x = 20
        panel_y = 70
        sub = hud[panel_y:panel_y + panel_h, panel_x:panel_x + panel_w]
        black_rect = np.zeros_like(sub)
        cv2.addWeighted(sub, 0.25, black_rect, 0.75, 0, sub)
        cv2.rectangle(hud, (panel_x, panel_y), (panel_x + panel_w, panel_y + panel_h), (80, 90, 110), 1)

        # State banner
        state_color = (0, 255, 120) if "STOPPED" in state_label else (0, 210, 255)
        cv2.putText(hud, f"STATE: {state_label}", (panel_x + 12, panel_y + 24),
                    cv2.FONT_HERSHEY_DUPLEX, 0.48, state_color, 1, cv2.LINE_AA)

        # Large Primary Distance Estimate (Kalman Filter)
        cv2.putText(hud, "ESTIMATED DISTANCE (KF)", (panel_x + 12, panel_y + 48),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.40, (180, 190, 205), 1)
        dist_str = f"{kf_state.distance:.3f} m"
        cv2.putText(hud, dist_str, (panel_x + 12, panel_y + 88),
                    cv2.FONT_HERSHEY_DUPLEX, 1.25, (0, 255, 230), 2, cv2.LINE_AA)

        # Baseline & Measurement Comparisons
        odo_str = f"Pure Odometry (Odo):  {kf_state.pure_odometry_distance:.3f} m"
        vis_str = f"Visual Raw (Cam):     {det.distance:.3f} m" if det.detected else "Visual Raw (Cam):     [TAG LOST]"
        target_str = f"Target Stopping Dist: {self.stop_distance:.3f} m"

        cv2.putText(hud, odo_str, (panel_x + 12, panel_y + 118), cv2.FONT_HERSHEY_SIMPLEX, 0.44, (200, 200, 210), 1)
        vis_color = (0, 255, 120) if det.detected else (0, 100, 255)
        cv2.putText(hud, vis_str, (panel_x + 12, panel_y + 140), cv2.FONT_HERSHEY_SIMPLEX, 0.44, vis_color, 1)
        cv2.putText(hud, target_str, (panel_x + 12, panel_y + 162), cv2.FONT_HERSHEY_SIMPLEX, 0.44, (255, 180, 60), 1)

        # Motion & Velocity
        vel_str = f"Velocity: vx={vx_cmd:.2f} m/s | wz={wz_cmd:+.1f} deg/s"
        cv2.putText(hud, vel_str, (panel_x + 12, panel_y + 192), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (170, 180, 200), 1)

        # Filter Covariance
        cov_str = f"Pos Variance P: {kf_state.variance_pos:.2e} m^2"
        cv2.putText(hud, cov_str, (panel_x + 12, panel_y + 214), cv2.FONT_HERSHEY_SIMPLEX, 0.40, (140, 150, 170), 1)

        # 4. Vertical Progress Bar on Right Border
        bar_x = w - 40
        bar_y_top = 80
        bar_y_bot = h - 60
        bar_h = bar_y_bot - bar_y_top

        cv2.line(hud, (bar_x, bar_y_top), (bar_x, bar_y_bot), (80, 80, 95), 4)

        # Mark Start (top) and Stop (bottom)
        cv2.circle(hud, (bar_x, bar_y_top), 6, (120, 120, 140), -1)
        cv2.putText(hud, f"{self.start_distance:.1f}m", (bar_x - 52, bar_y_top + 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38, (170, 170, 185), 1)

        # Stopping threshold mark
        stop_ratio = (self.stop_distance) / max(self.start_distance, 0.01)
        stop_y = int(bar_y_top + (1.0 - stop_ratio) * bar_h)
        cv2.circle(hud, (bar_x, stop_y), 6, (0, 0, 255), -1)
        cv2.putText(hud, f"STOP {self.stop_distance:.1f}m", (bar_x - 72, stop_y + 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38, (0, 150, 255), 1)

        # Current robot position marker
        curr_ratio = max(0.0, min(1.0, kf_state.distance / max(self.start_distance, 0.01)))
        curr_y = int(bar_y_top + (1.0 - curr_ratio) * bar_h)
        cv2.circle(hud, (bar_x, curr_y), 9, (0, 255, 230), -1)
        cv2.circle(hud, (bar_x, curr_y), 9, (0, 0, 0), 2)

        # Write frame to video file
        if self.video_writer is not None:
            self.video_writer.write(hud)

        # Display on screen if enabled
        if self.display_active:
            cv2.imshow(self.window_name, hud)
            cv2.waitKey(1)

        return hud

    def close(self):
        """Release resources, finish video encoding, and close windows."""
        if self.video_writer is not None:
            self.video_writer.release()
            self.video_writer = None
            print(f"[Visualizer] Demonstration video finalized: {self.video_output_path}")

        if self.display_active:
            try:
                cv2.destroyAllWindows()
            except Exception:
                pass
            self.display_active = False
