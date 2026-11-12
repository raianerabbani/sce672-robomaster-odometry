#!/usr/bin/env python3
"""Interactive Dashboard GUI for RoboMaster EP Odometry & Teleop Control.

SCE-672: Advanced Robotics, KFUPM
Features:
- Live camera feed with AprilTag / QR detection overlay & optical crosshair
- Easy-to-read "What's Happening Now" status & Kalman prediction/correction details
- Live telemetry: KF estimate, pure odometry, raw vision, covariance, & 4 wheel RPMs
- Full Teleop Takeover (on-screen D-pad + W/A/S/D/Q/E/Space keyboard control)
- Trial manager table, tape measurement entry, video recording, plots & PDF report.
"""

import os
import sys
import time
import threading
import yaml
import numpy as np
import cv2
import tkinter as tk
from tkinter import ttk, messagebox
from PIL import Image, ImageTk

WORKSPACE_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if WORKSPACE_ROOT not in sys.path:
    sys.path.insert(0, WORKSPACE_ROOT)

from src.tag_detector import TagDetector, TagDetection
from src.kalman_filter import OdometryVisionKalmanFilter, FilterState
from src.motion_controller import StraightLineMotionController
from src.robot_interface import (
    RoboMasterRobot,
    MockRobot,
    BaseRobot,
    mecanum_inverse_kinematics,
)
from src.trial_manager import TrialManager
from scripts.analyze_results import generate_plots
from scripts.generate_report import generate_report


# Color Palette (Clean Dark Slate)
BG_MAIN = "#12161f"
BG_CARD = "#1b2230"
BG_PANEL = "#232c3d"
FG_TEXT = "#e6edf3"
FG_MUTED = "#8b949e"
ACCENT_CYAN = "#00e5ff"
ACCENT_GREEN = "#2ea043"
ACCENT_RED = "#da3633"
ACCENT_AMBER = "#d29922"
ACCENT_BLUE = "#388bfd"


class RoboMasterDashboard:
    """Tkinter GUI Dashboard for Odometry Trials, Live Telemetry & Teleop."""

    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("SCE-672 RoboMaster EP Odometry & Teleop Dashboard")
        self.root.geometry("1420x860")
        self.root.configure(bg=BG_MAIN)

        self.cfg = self._load_config()
        exp_cfg = self.cfg.get("experiment", {})
        tag_cfg = self.cfg.get("tag", {})
        cam_cfg = self.cfg.get("camera", {})
        kf_cfg = self.cfg.get("kalman_filter", {})
        mot_cfg = self.cfg.get("motion", {})

        # Shared state protected by lock
        self.lock = threading.Lock()
        self.robot: BaseRobot = None
        self.connected = False
        self.running = True

        # Control mode: "IDLE", "AUTO", "AWAITING_TAPE", "TELEOP"
        self.control_mode = "IDLE"
        self.teleop_active = False
        self.teleop_vx = 0.0
        self.teleop_vy = 0.0
        self.teleop_wz = 0.0

        # Video recording
        self.recording = False
        self.video_writer = None

        # Subsystems
        self.K = np.array([
            [cam_cfg.get("fx", 580.0), 0.0, cam_cfg.get("cx", 640.0)],
            [0.0, cam_cfg.get("fy", 580.0), cam_cfg.get("cy", 360.0)],
            [0.0, 0.0, 1.0]
        ], dtype=np.float64)

        self.detector = TagDetector(
            tag_type=tag_cfg.get("tag_type", "apriltag"),
            tag_size=tag_cfg.get("physical_size", 0.15),
            camera_matrix=self.K,
            apriltag_family=tag_cfg.get("apriltag_family", "auto"),
            target_ids=tag_cfg.get("target_ids", [1, 2, 3]),
        )
        self.kf = OdometryVisionKalmanFilter(
            initial_distance=exp_cfg.get("start_distance", 1.50),
            camera_to_front_offset=exp_cfg.get("camera_to_front_offset", 0.05),
            q_pos=kf_cfg.get("q_pos", 0.0009),
            q_vel=kf_cfg.get("q_vel", 0.0025),
            r_vis=kf_cfg.get("r_vis", 0.0016),
        )
        self.controller = StraightLineMotionController(
            stop_distance=exp_cfg.get("stop_distance", 0.40),
            v_cruise=mot_cfg.get("v_cruise", 0.28),
            v_min=mot_cfg.get("v_min", 0.14),
            decel_distance=mot_cfg.get("decel_distance", 0.45),
            max_accel=mot_cfg.get("max_accel", 0.60),
            kp_yaw=mot_cfg.get("kp_yaw", 0.65),
            kd_yaw=mot_cfg.get("kd_yaw", 0.04),
            max_yaw_rate=mot_cfg.get("max_yaw_rate", 20.0),
        )
        self.trial_mgr = TrialManager(
            output_dir=os.path.join(WORKSPACE_ROOT, "results"),
            start_distance=exp_cfg.get("start_distance", 1.50),
            stop_distance=exp_cfg.get("stop_distance", 0.40),
        )
        self.next_trial_id = 1
        self.trial_start_time = 0.0

        # Async camera/vision state (decoupled from 50 Hz wheel control loop)
        self.frame_lock = threading.Lock()
        self.det_seq = 0
        self.last_det_seq = 0
        self.zero_stop_burst = 0
        self.teleop_vx_shaped = 0.0
        self.teleop_vy_shaped = 0.0
        self.teleop_wz_shaped = 0.0

        # Latest UI snapshot
        self.latest_frame = None
        self.latest_det = TagDetection()
        self.latest_kf = self.kf.get_state()
        self.latest_wheels = (0, 0, 0, 0)
        self.latest_battery = 0
        self.status_headline = "DISCONNECTED — Connect to RoboMaster EP or Simulator"
        self.status_detail = "Waiting for connection. Choose 'Hardware (AP)' or 'Simulator' above and click Connect."

        # Build UI
        self._build_ui(exp_cfg, tag_cfg)
        self._bind_keyboard()

        # Start decoupled 50 Hz control thread & async camera/vision thread
        self.ctrl_thread = threading.Thread(target=self._control_loop, daemon=True)
        self.cam_thread = threading.Thread(target=self._camera_loop, daemon=True)
        self.ctrl_thread.start()
        self.cam_thread.start()
        self.root.after(33, self._update_ui)

    def _load_config(self) -> dict:
        cfg_path = os.path.join(WORKSPACE_ROOT, "config", "settings.yaml")
        if os.path.exists(cfg_path):
            with open(cfg_path, "r") as f:
                return yaml.safe_load(f) or {}
        return {}

    def _build_ui(self, exp_cfg: dict, tag_cfg: dict):
        # Style ttk Treeview
        style = ttk.Style()
        style.theme_use("clam")
        style.configure("Treeview", background=BG_CARD, foreground=FG_TEXT, fieldbackground=BG_CARD, rowheight=24, font=("Sans", 9))
        style.configure("Treeview.Heading", background=BG_PANEL, foreground=ACCENT_CYAN, font=("Sans", 9, "bold"))

        # ---------------- TOP HEADER ----------------
        header = tk.Frame(self.root, bg=BG_CARD, height=56, padx=14, pady=8)
        header.pack(fill=tk.X, side=tk.TOP)

        tk.Label(
            header, text="SCE-672 RoboMaster EP Odometry & Teleop Dashboard",
            bg=BG_CARD, fg=FG_TEXT, font=("Sans", 14, "bold")
        ).pack(side=tk.LEFT)

        # Connection controls on right of header
        conn_frame = tk.Frame(header, bg=BG_CARD)
        conn_frame.pack(side=tk.RIGHT)

        self.mode_var = tk.StringVar(value="Hardware (AP)")
        mode_menu = ttk.Combobox(conn_frame, textvariable=self.mode_var, values=["Hardware (AP)", "Simulator"], width=14, state="readonly")
        mode_menu.pack(side=tk.LEFT, padx=6)

        self.btn_connect = tk.Button(
            conn_frame, text="Connect", bg=ACCENT_BLUE, fg="white", font=("Sans", 10, "bold"),
            relief=tk.FLAT, padx=12, pady=4, command=self._toggle_connect
        )
        self.btn_connect.pack(side=tk.LEFT, padx=6)

        self.lbl_conn_badge = tk.Label(
            conn_frame, text="● OFFLINE", bg=BG_PANEL, fg=ACCENT_RED,
            font=("Sans", 10, "bold"), padx=10, pady=4
        )
        self.lbl_conn_badge.pack(side=tk.LEFT, padx=6)

        self.lbl_battery = tk.Label(
            conn_frame, text="BAT: --%", bg=BG_PANEL, fg=FG_TEXT,
            font=("Sans", 10, "bold"), padx=10, pady=4
        )
        self.lbl_battery.pack(side=tk.LEFT, padx=4)

        # ---------------- MAIN BODY (2 COLUMNS) ----------------
        body = tk.Frame(self.root, bg=BG_MAIN, padx=12, pady=10)
        body.pack(fill=tk.BOTH, expand=True)

        left_col = tk.Frame(body, bg=BG_MAIN)
        left_col.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 10))

        right_col = tk.Frame(body, bg=BG_MAIN, width=520)
        right_col.pack(side=tk.RIGHT, fill=tk.Y)
        right_col.pack_propagate(False)

        # === LEFT COLUMN: CAMERA + LIVE STATUS + TELEMETRY ===
        # Camera Feed Card
        cam_card = tk.Frame(left_col, bg=BG_CARD, bd=1, relief=tk.SOLID)
        cam_card.pack(fill=tk.BOTH, expand=True)

        self.cam_label = tk.Label(cam_card, bg="#0b0e14", text="Camera Feed Inactive", fg=FG_MUTED, font=("Sans", 12))
        self.cam_label.pack(fill=tk.BOTH, expand=True, padx=4, pady=4)

        # "What's Happening Right Now" Status Banner
        status_card = tk.Frame(left_col, bg=BG_CARD, padx=12, pady=10)
        status_card.pack(fill=tk.X, pady=(8, 0))

        tk.Label(status_card, text="WHAT'S HAPPENING RIGHT NOW", bg=BG_CARD, fg=FG_MUTED, font=("Sans", 8, "bold")).pack(anchor=tk.W)
        self.lbl_status_headline = tk.Label(
            status_card, text=self.status_headline, bg=BG_CARD, fg=ACCENT_CYAN,
            font=("Sans", 12, "bold"), anchor=tk.W
        )
        self.lbl_status_headline.pack(fill=tk.X, pady=(2, 2))
        self.lbl_status_detail = tk.Label(
            status_card, text=self.status_detail, bg=BG_CARD, fg=FG_TEXT,
            font=("Sans", 9), anchor=tk.W, wraplength=820, justify=tk.LEFT
        )
        self.lbl_status_detail.pack(fill=tk.X)

        # Live Telemetry Cards Row
        telem_row = tk.Frame(left_col, bg=BG_MAIN)
        telem_row.pack(fill=tk.X, pady=(8, 0))

        # Card 1: Kalman Filter Estimate
        c1 = tk.Frame(telem_row, bg=BG_CARD, padx=12, pady=8)
        c1.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 6))
        tk.Label(c1, text="KALMAN FILTER ESTIMATE", bg=BG_CARD, fg=FG_MUTED, font=("Sans", 8, "bold")).pack(anchor=tk.W)
        self.lbl_kf_dist = tk.Label(c1, text="1.500 m", bg=BG_CARD, fg=ACCENT_CYAN, font=("Sans", 20, "bold"))
        self.lbl_kf_dist.pack(anchor=tk.W)
        self.lbl_kf_sub = tk.Label(c1, text="Vel: 0.00 m/s | Uncertainty: ±5.0 cm", bg=BG_CARD, fg=FG_TEXT, font=("Sans", 9))
        self.lbl_kf_sub.pack(anchor=tk.W)

        # Card 2: Sensors Comparison
        c2 = tk.Frame(telem_row, bg=BG_CARD, padx=12, pady=8)
        c2.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=6)
        tk.Label(c2, text="PREDICTION VS CORRECTION", bg=BG_CARD, fg=FG_MUTED, font=("Sans", 8, "bold")).pack(anchor=tk.W)
        self.lbl_odo_dist = tk.Label(c2, text="Pure Odometry: 1.500 m", bg=BG_CARD, fg=ACCENT_AMBER, font=("Sans", 11, "bold"))
        self.lbl_odo_dist.pack(anchor=tk.W, pady=(4, 2))
        self.lbl_vis_dist = tk.Label(c2, text="Vision Raw: -- (No Tag)", bg=BG_CARD, fg=ACCENT_GREEN, font=("Sans", 11, "bold"))
        self.lbl_vis_dist.pack(anchor=tk.W)

        # Card 3: Individual Wheel RPMs
        c3 = tk.Frame(telem_row, bg=BG_CARD, padx=12, pady=8)
        c3.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(6, 0))
        tk.Label(c3, text="INDIVIDUAL WHEEL RPMs", bg=BG_CARD, fg=FG_MUTED, font=("Sans", 8, "bold")).pack(anchor=tk.W)
        self.lbl_wheels = tk.Label(
            c3, text="FR (w1):   0   |  FL (w2):   0\nRL (w3):   0   |  RR (w4):   0",
            bg=BG_CARD, fg=FG_TEXT, font=("Courier", 10, "bold"), justify=tk.LEFT
        )
        self.lbl_wheels.pack(anchor=tk.W, pady=(6, 0))

        # === RIGHT COLUMN: CONTROLS + TELEOP + COLLECTED DATA TABLE ===
        # 1. Experiment Settings & Run Controls
        ctrl_card = tk.Frame(right_col, bg=BG_CARD, padx=12, pady=10)
        ctrl_card.pack(fill=tk.X)

        tk.Label(ctrl_card, text="EXPERIMENT SETTINGS & TRIAL CONTROL", bg=BG_CARD, fg=ACCENT_CYAN, font=("Sans", 9, "bold")).pack(anchor=tk.W, pady=(0, 6))

        grid_f = tk.Frame(ctrl_card, bg=BG_CARD)
        grid_f.pack(fill=tk.X)

        self.var_start = tk.StringVar(value=str(exp_cfg.get("start_distance", 1.50)))
        self.var_stop = tk.StringVar(value=str(exp_cfg.get("stop_distance", 0.40)))
        self.var_tag_size = tk.StringVar(value=str(tag_cfg.get("physical_size", 0.15)))
        self.var_tag_type = tk.StringVar(value=str(tag_cfg.get("tag_type", "apriltag")))

        for col, (lbl, var) in enumerate([("Start (m):", self.var_start), ("Stop (m):", self.var_stop), ("Tag Size (m):", self.var_tag_size)]):
            tk.Label(grid_f, text=lbl, bg=BG_CARD, fg=FG_MUTED, font=("Sans", 8)).grid(row=0, column=col*2, sticky=tk.W, padx=(0, 2))
            tk.Entry(grid_f, textvariable=var, width=6, bg=BG_PANEL, fg=FG_TEXT, insertbackground=FG_TEXT, relief=tk.FLAT).grid(row=0, column=col*2+1, padx=(0, 8))

        tk.Label(grid_f, text="Tag:", bg=BG_CARD, fg=FG_MUTED, font=("Sans", 8)).grid(row=0, column=6, sticky=tk.W)
        ttk.Combobox(grid_f, textvariable=self.var_tag_type, values=["apriltag", "qr", "aruco", "auto"], width=8, state="readonly").grid(row=0, column=7)

        btn_row = tk.Frame(ctrl_card, bg=BG_CARD)
        btn_row.pack(fill=tk.X, pady=(8, 4))

        self.btn_start_trial = tk.Button(
            btn_row, text="▶ Start Auto Trial", bg=ACCENT_GREEN, fg="white", font=("Sans", 10, "bold"),
            relief=tk.FLAT, padx=10, pady=6, command=self._start_auto_trial
        )
        self.btn_start_trial.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 4))

        self.btn_estop = tk.Button(
            btn_row, text="⏹ STOP / ABORT", bg=ACCENT_RED, fg="white", font=("Sans", 10, "bold"),
            relief=tk.FLAT, padx=10, pady=6, command=self._emergency_stop
        )
        self.btn_estop.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=4)

        self.btn_rec = tk.Button(
            btn_row, text="● Rec Video", bg=BG_PANEL, fg=FG_TEXT, font=("Sans", 9, "bold"),
            relief=tk.FLAT, padx=8, pady=6, command=self._toggle_recording
        )
        self.btn_rec.pack(side=tk.LEFT, padx=(4, 0))

        # Ground-truth tape input row
        gt_row = tk.Frame(ctrl_card, bg=BG_PANEL, padx=8, pady=6)
        gt_row.pack(fill=tk.X, pady=(4, 0))
        tk.Label(gt_row, text="Measured Tape Dist (m):", bg=BG_PANEL, fg=FG_TEXT, font=("Sans", 9, "bold")).pack(side=tk.LEFT)
        self.var_tape = tk.StringVar(value="")
        self.entry_tape = tk.Entry(gt_row, textvariable=self.var_tape, width=8, bg=BG_CARD, fg=ACCENT_CYAN, insertbackground=FG_TEXT, font=("Sans", 10, "bold"))
        self.entry_tape.pack(side=tk.LEFT, padx=6)
        self.btn_save_trial = tk.Button(
            gt_row, text="Save Trial", bg=ACCENT_BLUE, fg="white", font=("Sans", 9, "bold"),
            relief=tk.FLAT, padx=10, pady=2, command=self._save_trial_measurement
        )
        self.btn_save_trial.pack(side=tk.RIGHT)

        # 2. Teleop Takeover Panel
        teleop_card = tk.Frame(right_col, bg=BG_CARD, padx=12, pady=10)
        teleop_card.pack(fill=tk.X, pady=(8, 0))

        t_hdr = tk.Frame(teleop_card, bg=BG_CARD)
        t_hdr.pack(fill=tk.X)
        tk.Label(t_hdr, text="MANUAL TELEOP TAKEOVER (W/A/S/D + Q/E + Space)", bg=BG_CARD, fg=ACCENT_AMBER, font=("Sans", 9, "bold")).pack(side=tk.LEFT)

        self.btn_teleop = tk.Button(
            t_hdr, text="🎮 Take Control", bg=ACCENT_AMBER, fg="black", font=("Sans", 9, "bold"),
            relief=tk.FLAT, padx=8, pady=2, command=self._toggle_teleop
        )
        self.btn_teleop.pack(side=tk.RIGHT)

        t_body = tk.Frame(teleop_card, bg=BG_CARD)
        t_body.pack(fill=tk.X, pady=(6, 0))

        # D-pad grid
        dpad = tk.Frame(t_body, bg=BG_CARD)
        dpad.pack(side=tk.LEFT)

        pad_specs = [
            ("↺ Q", 0, 0, 0.0, 0.0, -35.0),
            ("▲ W", 0, 1, 1.0, 0.0, 0.0),
            ("↻ E", 0, 2, 0.0, 0.0, 35.0),
            ("◀ A", 1, 0, 0.0, -1.0, 0.0),
            ("■ STOP", 1, 1, 0.0, 0.0, 0.0),
            ("▶ D", 1, 2, 0.0, 1.0, 0.0),
            ("▼ S", 2, 1, -1.0, 0.0, 0.0),
        ]
        for label, r, c, vx_m, vy_m, wz_v in pad_specs:
            bg_b = ACCENT_RED if "STOP" in label else BG_PANEL
            btn = tk.Button(dpad, text=label, width=7, bg=bg_b, fg=FG_TEXT, font=("Sans", 9, "bold"), relief=tk.FLAT)
            btn.grid(row=r, column=c, padx=2, pady=2)
            if "STOP" in label:
                btn.configure(command=self._teleop_stop)
            else:
                btn.bind("<ButtonPress-1>", lambda e, x=vx_m, y=vy_m, z=wz_v: self._set_teleop_cmd(x, y, z))
                btn.bind("<ButtonRelease-1>", lambda e: self._teleop_stop())

        # Speed slider
        spd_frame = tk.Frame(t_body, bg=BG_CARD, padx=14)
        spd_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        tk.Label(spd_frame, text="Teleop Speed (m/s)", bg=BG_CARD, fg=FG_MUTED, font=("Sans", 8)).pack(anchor=tk.W)
        self.var_teleop_speed = tk.DoubleVar(value=0.20)
        tk.Scale(
            spd_frame, from_=0.05, to=0.45, resolution=0.05, orient=tk.HORIZONTAL,
            variable=self.var_teleop_speed, bg=BG_CARD, fg=FG_TEXT, highlightthickness=0, troughcolor=BG_PANEL
        ).pack(fill=tk.X)

        # 3. Collected Data Table & Summary
        data_card = tk.Frame(right_col, bg=BG_CARD, padx=12, pady=10)
        data_card.pack(fill=tk.BOTH, expand=True, pady=(8, 0))

        tk.Label(data_card, text="COLLECTED TRIALS DATA", bg=BG_CARD, fg=ACCENT_CYAN, font=("Sans", 9, "bold")).pack(anchor=tk.W, pady=(0, 4))

        cols = ("trial", "actual", "kf", "kf_err", "odo", "odo_err")
        self.tree = ttk.Treeview(data_card, columns=cols, show="headings", height=7)
        for col, title, w in [
            ("trial", "#", 32), ("actual", "Actual(m)", 72), ("kf", "KF Est(m)", 72),
            ("kf_err", "KF Err(cm)", 78), ("odo", "Odo(m)", 68), ("odo_err", "Odo Err(cm)", 82)
        ]:
            self.tree.heading(col, text=title)
            self.tree.column(col, width=w, anchor=tk.CENTER)
        self.tree.pack(fill=tk.BOTH, expand=True)

        self.lbl_stats = tk.Label(
            data_card, text="Trials: 0 | KF MAE: -- | Range: -- | Odo MAE: --",
            bg=BG_PANEL, fg=FG_TEXT, font=("Sans", 9, "bold"), pady=5
        )
        self.lbl_stats.pack(fill=tk.X, pady=(6, 6))

        export_row = tk.Frame(data_card, bg=BG_CARD)
        export_row.pack(fill=tk.X)
        tk.Button(
            export_row, text="📊 Export CSV & Plots", bg=BG_PANEL, fg=FG_TEXT, font=("Sans", 9, "bold"),
            relief=tk.FLAT, pady=5, command=self._export_and_plot
        ).pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 4))
        tk.Button(
            export_row, text="📄 Generate PDF Report", bg=ACCENT_BLUE, fg="white", font=("Sans", 9, "bold"),
            relief=tk.FLAT, pady=5, command=self._compile_report
        ).pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(4, 0))

    def _bind_keyboard(self):
        key_map = {
            "w": (1.0, 0.0, 0.0),
            "s": (-1.0, 0.0, 0.0),
            "a": (0.0, -1.0, 0.0),
            "d": (0.0, 1.0, 0.0),
            "q": (0.0, 0.0, -35.0),
            "e": (0.0, 0.0, 35.0),
        }
        for k, (vx, vy, wz) in key_map.items():
            self.root.bind(f"<KeyPress-{k}>", lambda e, x=vx, y=vy, z=wz: self._on_key_teleop(x, y, z))
            self.root.bind(f"<KeyRelease-{k}>", lambda e: self._on_key_release())
        self.root.bind("<space>", lambda e: self._emergency_stop())

    def _on_key_teleop(self, vx_mult, vy_mult, wz_val):
        # Ignore keyboard teleop if user is typing in an Entry widget
        if isinstance(self.root.focus_get(), tk.Entry):
            return
        if self.teleop_active:
            self._set_teleop_cmd(vx_mult, vy_mult, wz_val)

    def _on_key_release(self):
        if isinstance(self.root.focus_get(), tk.Entry):
            return
        if self.teleop_active:
            self._teleop_stop()

    def _toggle_connect(self):
        if not self.connected:
            mode_str = self.mode_var.get()
            start_d = float(self.var_start.get() or 1.50)
            tag_sz = float(self.var_tag_size.get() or 0.15)
            tag_tp = self.var_tag_type.get()

            if mode_str == "Simulator":
                r = MockRobot(start_distance=start_d, tag_size=tag_sz, tag_type=tag_tp, tag_id=1)
            else:
                r = RoboMasterRobot(conn_type=self.cfg.get("hardware", {}).get("conn_type", "ap"))

            if r.connect():
                with self.lock:
                    self.robot = r
                    self.connected = True
                    self.control_mode = "IDLE"
                    self.status_headline = f"CONNECTED ({mode_str}) — Ready for Trial {self.next_trial_id}"
                    self.status_detail = "Camera stream & 50 Hz wheel odometry active. Click 'Start Auto Trial' or 'Take Control' for Teleop."
                self.btn_connect.configure(text="Disconnect", bg=ACCENT_RED)
                self.lbl_conn_badge.configure(text="● ONLINE", fg=ACCENT_GREEN)
            else:
                messagebox.showerror("Connection Error", "Could not connect to RoboMaster EP. Verify Wi-Fi AP connection (192.168.2.1).")
        else:
            with self.lock:
                self.connected = False
                self.control_mode = "IDLE"
                if self.robot:
                    self.robot.stop()
                    self.robot.disconnect()
                    self.robot = None
                self.status_headline = "DISCONNECTED"
                self.status_detail = "Robot safely stopped and disconnected."
            self.btn_connect.configure(text="Connect", bg=ACCENT_BLUE)
            self.lbl_conn_badge.configure(text="● OFFLINE", fg=ACCENT_RED)

    def _start_auto_trial(self):
        if not self.connected or self.robot is None:
            messagebox.showwarning("Not Connected", "Connect to RoboMaster EP or Simulator first.")
            return
        try:
            start_d = float(self.var_start.get())
            stop_d = float(self.var_stop.get())
            tag_sz = float(self.var_tag_size.get())
            tag_tp = self.var_tag_type.get()
        except ValueError:
            messagebox.showerror("Invalid Settings", "Start, Stop, and Tag Size must be valid numbers.")
            return

        with self.lock:
            self.teleop_active = False
            self.btn_teleop.configure(text="🎮 Take Control", bg=ACCENT_AMBER, fg="black")

            self.detector.tag_type = tag_tp
            self.detector.tag_size = tag_sz
            if isinstance(self.robot, MockRobot):
                self.robot.tag_type = tag_tp
                self.robot.tag_size = tag_sz
                self.robot.set_start_distance(start_d)
            else:
                self.robot.reset_odometry()

            self.kf.reset(initial_distance=start_d)
            self.controller.reset(stop_distance=stop_d)
            self.trial_mgr.start_distance = start_d
            self.trial_mgr.stop_distance = stop_d
            self.trial_mgr.start_trial(self.next_trial_id)
            self.trial_start_time = time.time()
            self.control_mode = "AUTO"
            self.status_headline = f"AUTO TRIAL {self.next_trial_id}: APPROACHING TAG"
            self.status_detail = "Running 50 Hz wheel odometry prediction + AprilTag/QR visual correction."

    def _emergency_stop(self):
        with self.lock:
            self.control_mode = "IDLE"
            self.teleop_vx = 0.0
            self.teleop_vy = 0.0
            self.teleop_wz = 0.0
            if self.robot:
                self.robot.stop()
            self.status_headline = "STOPPED (Manual Halt / E-Stop)"
            self.status_detail = "All 4 wheels commanded to 0 RPM. Select 'Start Auto Trial' or 'Take Control' to resume."

    def _toggle_teleop(self):
        if not self.connected:
            messagebox.showwarning("Not Connected", "Connect to RoboMaster EP or Simulator first.")
            return
        with self.lock:
            self.teleop_active = not self.teleop_active
            self.teleop_vx = 0.0
            self.teleop_vy = 0.0
            self.teleop_wz = 0.0
            if self.robot:
                self.robot.stop()
            if self.teleop_active:
                self.control_mode = "TELEOP"
                self.btn_teleop.configure(text="🤖 Release Teleop", bg=ACCENT_RED, fg="white")
                self.status_headline = "TELEOP CONTROL ACTIVE"
                self.status_detail = "Manual takeover active. Use W/A/S/D/Q/E keys or on-screen D-pad. Space = Stop."
            else:
                self.control_mode = "IDLE"
                self.btn_teleop.configure(text="🎮 Take Control", bg=ACCENT_AMBER, fg="black")
                self.status_headline = "IDLE — Teleop Released"
                self.status_detail = "All wheels stopped. Ready for autonomous trial."

    def _set_teleop_cmd(self, vx_mult: float, vy_mult: float, wz_val: float):
        if not self.connected or not self.teleop_active:
            return
        spd = float(self.var_teleop_speed.get())
        with self.lock:
            self.teleop_vx = vx_mult * spd
            self.teleop_vy = vy_mult * spd
            self.teleop_wz = wz_val

    def _teleop_stop(self):
        with self.lock:
            self.teleop_vx = 0.0
            self.teleop_vy = 0.0
            self.teleop_wz = 0.0
            if self.robot:
                self.robot.stop()

    def _toggle_recording(self):
        with self.lock:
            if not self.recording:
                out_path = os.path.join(WORKSPACE_ROOT, "results", "demo_video.mp4")
                os.makedirs(os.path.dirname(out_path), exist_ok=True)
                fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                self.video_writer = cv2.VideoWriter(out_path, fourcc, 30, (1280, 720))
                self.recording = True
                self.btn_rec.configure(text="⏹ Stop Rec", bg=ACCENT_RED)
            else:
                self.recording = False
                if self.video_writer:
                    self.video_writer.release()
                    self.video_writer = None
                self.btn_rec.configure(text="● Rec Video", bg=BG_PANEL)

    def _save_trial_measurement(self):
        val_str = self.var_tape.get().strip()
        if not val_str:
            messagebox.showwarning("Missing Measurement", "Enter the actual tape-measured distance in meters (e.g. 0.405).")
            return
        try:
            measured = float(val_str)
        except ValueError:
            messagebox.showerror("Invalid Number", "Tape measurement must be a numeric value in meters.")
            return

        with self.lock:
            if self.trial_mgr.current_trial_id is None:
                messagebox.showinfo("No Active Trial", "Run an autonomous trial first before saving.")
                return
            rec = self.trial_mgr.finish_trial(measured)
            self.next_trial_id += 1
            self.control_mode = "IDLE"
            self.var_tape.set("")
            self.status_headline = f"TRIAL {rec.trial_id} SAVED (Err: {rec.kf_error*100:+.2f} cm)"
            self.status_detail = f"Ready for Trial {self.next_trial_id}. Reposition robot at start distance and click 'Start Auto Trial'."

        # Update Treeview & summary
        self.tree.insert("", tk.END, values=(
            rec.trial_id,
            f"{rec.measured_actual_distance:.4f}",
            f"{rec.estimated_final_distance:.4f}",
            f"{rec.kf_error*100:+.2f}",
            f"{rec.odometry_final_distance:.4f}",
            f"{rec.odo_error*100:+.2f}",
        ))
        stats = self.trial_mgr.compute_statistics()
        self.lbl_stats.configure(
            text=f"Trials: {stats.num_trials} | KF MAE: {stats.kf_mae*1000:.1f} mm | Range: {stats.kf_error_range*1000:.1f} mm | Odo MAE: {stats.odo_mae*1000:.1f} mm"
        )

    def _export_and_plot(self):
        if not self.trial_mgr.trials:
            messagebox.showwarning("No Data", "Record at least one completed trial first.")
            return
        self.trial_mgr.export_summary("experiment_summary")
        generate_plots(
            results_dir=os.path.join(WORKSPACE_ROOT, "results"),
            output_dir=os.path.join(WORKSPACE_ROOT, "results", "plots")
        )
        messagebox.showinfo("Exported", "Saved CSV/JSON summary and generated plots in results/plots/.")

    def _compile_report(self):
        if not self.trial_mgr.trials:
            messagebox.showwarning("No Data", "Record at least one completed trial first.")
            return
        self.trial_mgr.export_summary("experiment_summary")
        generate_plots(
            results_dir=os.path.join(WORKSPACE_ROOT, "results"),
            output_dir=os.path.join(WORKSPACE_ROOT, "results", "plots")
        )
        generate_report()
        messagebox.showinfo("Report Compiled", "Generated PDF report at report/RoboMaster_Odometry_Report.pdf.")

    def _camera_loop(self):
        """Asynchronous camera acquisition and tag detection thread (never blocks 50 Hz wheels)."""
        while self.running:
            robot_ref = self.robot if self.connected else None
            if robot_ref is None:
                time.sleep(0.05)
                continue

            frame = robot_ref.get_frame()
            if frame is None:
                time.sleep(0.02)
                continue

            det = self.detector.detect(frame)
            vis_frame = self.detector.draw_detection(frame, det)
            h, w = vis_frame.shape[:2]
            cv2.line(vis_frame, (w // 2 - 15, h // 2), (w // 2 + 15, h // 2), (200, 200, 200), 1)
            cv2.line(vis_frame, (w // 2, h // 2 - 15), (w // 2, h // 2 + 15), (200, 200, 200), 1)

            with self.frame_lock:
                self.latest_frame = vis_frame
                self.latest_det = det
                self.det_seq += 1
                if self.recording and self.video_writer is not None:
                    self.video_writer.write(cv2.resize(vis_frame, (1280, 720)))

            time.sleep(0.01)

    def _control_loop(self):
        """50 Hz (dt=0.020 s) real-time wheel control & Kalman prediction/correction loop."""
        dt_target = 0.020
        last_time = time.time()

        while self.running:
            loop_start = time.time()
            dt = float(np.clip(loop_start - last_time, 0.005, 0.08))
            last_time = loop_start

            with self.lock:
                if not self.connected or self.robot is None:
                    time.sleep(0.04)
                    continue

                telemetry = self.robot.get_telemetry()
                with self.frame_lock:
                    det = self.latest_det
                    new_vision = (self.det_seq != self.last_det_seq)
                    self.last_det_seq = self.det_seq

                mode = self.control_mode

                if mode == "AUTO":
                    # 1. Prediction at 50 Hz via wheel odometry
                    self.kf.predict(odo_pos=telemetry["odo_x"], dt=dt)

                    # 2. Sensor correction whenever a new camera tag detection arrives
                    if new_vision and det.detected:
                        self.kf.correct(visual_distance_raw=det.distance)
                    bearing = det.bearing_deg if det.detected else None

                    kf_state = self.kf.get_state()
                    vx, vy, wz, reached = self.controller.compute_velocity(
                        current_distance=kf_state.distance,
                        tag_bearing_deg=bearing,
                        current_time=loop_start
                    )

                    cam_offset = self.kf.camera_to_front_offset
                    true_d = (telemetry["true_distance"] - cam_offset) if "true_distance" in telemetry else None
                    self.trial_mgr.log_sample(
                        pure_odo_dist=kf_state.pure_odometry_distance,
                        visual_dist=det.distance if (new_vision and det.detected) else None,
                        kf_dist=kf_state.distance,
                        kf_vel=kf_state.velocity,
                        cov_pos=kf_state.variance_pos,
                        cov_vel=kf_state.variance_vel,
                        kalman_gain=kf_state.kalman_gain,
                        tag_detected=det.detected,
                        vx_cmd=vx,
                        wz_cmd=wz,
                        true_distance=true_d
                    )

                    if reached:
                        self.zero_stop_burst = 6
                        self.robot.drive_wheels(0, 0, 0, 0)
                        self.control_mode = "AWAITING_TAPE"
                        self.status_headline = f"TRIAL {self.next_trial_id} COMPLETE — ENTER TAPE MEASUREMENT"
                        self.status_detail = f"Robot stopped at KF estimate {kf_state.distance:.4f} m. Measure actual distance with tape and click 'Save Trial'."
                    else:
                        self.zero_stop_burst = 6
                        w1, w2, w3, w4 = mecanum_inverse_kinematics(vx, vy, wz)
                        self.robot.drive_wheels(w1, w2, w3, w4)
                        if det.detected:
                            self.status_headline = f"AUTO: TRACKING {det.tag_type} {det.tag_id}"
                            self.status_detail = f"50 Hz odometry prediction + {det.tag_type} {det.tag_id} correction (z={det.distance:.3f}m, bearing={det.bearing_deg:+.1f}°)."
                        else:
                            self.status_headline = "AUTO: ODOMETRY DEAD-RECKONING (Tag Not Visible)"
                            self.status_detail = "Tag temporarily out of view — propagating position via Mecanum wheel encoders."

                elif mode == "TELEOP":
                    # Smooth slew-rate shaping for teleop (no sudden jerks)
                    max_dv = 2.0 * dt
                    max_dwz = 240.0 * dt
                    self.teleop_vx_shaped += float(np.clip(self.teleop_vx - self.teleop_vx_shaped, -max_dv, max_dv))
                    self.teleop_vy_shaped += float(np.clip(self.teleop_vy - self.teleop_vy_shaped, -max_dv, max_dv))
                    self.teleop_wz_shaped += float(np.clip(self.teleop_wz - self.teleop_wz_shaped, -max_dwz, max_dwz))

                    is_moving = (
                        abs(self.teleop_vx_shaped) > 0.015
                        or abs(self.teleop_vy_shaped) > 0.015
                        or abs(self.teleop_wz_shaped) > 0.5
                    )
                    if is_moving:
                        self.zero_stop_burst = 6
                        w1, w2, w3, w4 = mecanum_inverse_kinematics(
                            self.teleop_vx_shaped, self.teleop_vy_shaped, self.teleop_wz_shaped
                        )
                        self.robot.drive_wheels(w1, w2, w3, w4)
                        self.status_detail = f"Manual Teleop wheels: FR={w1}, FL={w2}, RL={w3}, RR={w4} RPM."
                    elif self.zero_stop_burst > 0:
                        self.zero_stop_burst -= 1
                        self.robot.drive_wheels(0, 0, 0, 0)
                        self.status_detail = "Manual Teleop: Stopped (0 RPM)."
                    kf_state = self.kf.get_state()

                else:
                    # IDLE or AWAITING_TAPE: finish any pending zero-stop burst without flooding bus
                    if self.zero_stop_burst > 0:
                        self.zero_stop_burst -= 1
                        self.robot.drive_wheels(0, 0, 0, 0)
                    kf_state = self.kf.get_state()

                self.latest_kf = kf_state
                self.latest_wheels = telemetry.get("wheel_rpms", (0, 0, 0, 0))
                self.latest_battery = telemetry.get("battery", 0)

            elapsed = time.time() - loop_start
            time.sleep(max(0.001, dt_target - elapsed))

    def _update_ui(self):
        with self.frame_lock:
            frame = self.latest_frame.copy() if self.latest_frame is not None else None
            det = self.latest_det
        with self.lock:
            kf_s = self.latest_kf
            wheels = self.latest_wheels
            bat = self.latest_battery
            headline = self.status_headline
            detail = self.status_detail
            mode = self.control_mode

        if frame is not None:
            cw = max(320, self.cam_label.winfo_width())
            ch = max(180, self.cam_label.winfo_height())
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(rgb)
            img.thumbnail((cw, ch), Image.Resampling.BILINEAR)
            imgtk = ImageTk.PhotoImage(image=img)
            self.cam_label.imgtk = imgtk
            self.cam_label.configure(image=imgtk, text="")

        self.lbl_battery.configure(text=f"BAT: {bat}%")
        self.lbl_status_headline.configure(text=headline)
        self.lbl_status_detail.configure(text=detail)

        std_cm = np.sqrt(max(0.0, kf_s.variance_pos)) * 100.0
        self.lbl_kf_dist.configure(text=f"{kf_s.distance:.3f} m")
        self.lbl_kf_sub.configure(text=f"Vel: {kf_s.velocity:.2f} m/s | Uncertainty: ±{std_cm:.1f} cm")
        self.lbl_odo_dist.configure(text=f"Pure Odometry: {kf_s.pure_odometry_distance:.3f} m")

        if det.detected:
            self.lbl_vis_dist.configure(text=f"Vision ({det.tag_id}): {det.distance:.3f} m ({det.bearing_deg:+.1f}°)", fg=ACCENT_GREEN)
        else:
            self.lbl_vis_dist.configure(text="Vision Raw: [No Tag Detected]", fg=ACCENT_AMBER)

        self.lbl_wheels.configure(
            text=f"FR (w1): {wheels[0]:+4d}  |  FL (w2): {wheels[1]:+4d}\nRL (w3): {wheels[2]:+4d}  |  RR (w4): {wheels[3]:+4d}"
        )

        # Auto-suggest tape distance when trial just finished
        if mode == "AWAITING_TAPE" and not self.var_tape.get():
            self.var_tape.set(f"{kf_s.distance:.4f}")

        if self.running:
            self.root.after(33, self._update_ui)

    def on_close(self):
        self.running = False
        with self.lock:
            if self.video_writer:
                self.video_writer.release()
            if self.robot:
                self.robot.stop()
                self.robot.disconnect()
        self.root.destroy()


if __name__ == "__main__":
    root = tk.Tk()
    app = RoboMasterDashboard(root)
    root.protocol("WM_DELETE_WINDOW", app.on_close)
    root.mainloop()
