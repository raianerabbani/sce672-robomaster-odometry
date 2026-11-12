#!/usr/bin/env python3
"""Interactive Tag and Camera Calibration Utility for RoboMaster EP.

SCE-672: Advanced Robotics, KFUPM
Allows the user to place the robot at an exact measured tape distance (e.g. 1.00m or 1.50m)
and automatically calibrate camera focal length and tag scale factors.
Updates config/settings.yaml automatically.
"""

import os
import sys
import time
import argparse
import yaml
import numpy as np
import cv2

WORKSPACE_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if WORKSPACE_ROOT not in sys.path:
    sys.path.insert(0, WORKSPACE_ROOT)

from src.tag_detector import TagDetector
from src.robot_interface import RoboMasterRobot, MockRobot


def calibrate(
    mode: str = "auto",
    true_distance: float = 1.00,
    tag_size: float = 0.15,
    tag_type: str = "qr",
    config_path: str = "config/settings.yaml"
):
    """Interactively calibrate camera and tag distance scaling."""
    config_file = os.path.join(WORKSPACE_ROOT, config_path)
    with open(config_file, "r") as f:
        cfg = yaml.safe_load(f)

    print("\n==================================================================")
    print("      ROBOMASTER EP CAMERA & TAG CALIBRATION UTILITY")
    print("         SCE-672: Advanced Robotics - Project Phase 3")
    print("==================================================================")
    print(f" Target Tag Type           : {tag_type.upper()}")
    print(f" Physical Tag Size         : {tag_size*100:.1f} cm")
    print(f" Reference Distance (Tape) : {true_distance:.3f} m")
    print("==================================================================")

    # Connect to robot
    robot = None
    if mode in ("robot", "auto"):
        try:
            robot = RoboMasterRobot(conn_type=cfg.get("hardware", {}).get("conn_type", "ap"))
            if not robot.connect():
                raise ConnectionError("Robot not reachable.")
        except Exception as e:
            if mode == "robot":
                print(f"[Error] Failed to connect to robot: {e}")
                return
            print("[Info] Robot not connected, using MockRobot for calibration demo.")
            robot = MockRobot(start_distance=true_distance, tag_size=tag_size)
            robot.connect()
    else:
        robot = MockRobot(start_distance=true_distance, tag_size=tag_size)
        robot.connect()

    detector = TagDetector(tag_type=tag_type, tag_size=tag_size)
    print("\n[Calibration] Streaming camera frames... Align tag in camera view.")
    print("Collecting 30 consecutive valid frames for calibration...")

    measured_widths = []
    collected_frames = 0

    try:
        while collected_frames < 30:
            frame = robot.get_frame()
            if frame is None:
                time.sleep(0.05)
                continue

            det = detector.detect(frame)
            if det.detected:
                measured_widths.append(det.width_px)
                collected_frames += 1
                sys.stdout.write(f"\r  Frame {collected_frames:02d}/30: Detected {det.tag_type} | Apparent Width = {det.width_px:.1f} px | Raw Dist = {det.distance:.3f} m")
                sys.stdout.flush()

            time.sleep(0.05)

        print("\n\n[Calibration] Frame collection complete.")
        mean_w_px = float(np.mean(measured_widths))
        std_w_px = float(np.std(measured_widths))

        # Pinhole relationship: d = (fx * W) / w_px  =>  fx = (d * w_px) / W
        # For QR: effective size is W * qr_scale_factor
        effective_w = tag_size * (detector.qr_scale_factor if tag_type == "qr" else 1.0)
        calibrated_fx = float((true_distance * mean_w_px) / effective_w)

        print(f"  Mean Apparent Width  : {mean_w_px:.2f} px (std: {std_w_px:.2f} px)")
        print(f"  Current Focal Length : {cfg['camera']['fx']:.1f} px")
        print(f"  Calibrated fx        : {calibrated_fx:.1f} px")

        # Update configuration
        cfg["camera"]["fx"] = round(calibrated_fx, 1)
        cfg["camera"]["fy"] = round(calibrated_fx, 1)
        cfg["tag"]["physical_size"] = float(tag_size)
        cfg["tag"]["tag_type"] = str(tag_type)

        with open(config_file, "w") as f:
            yaml.safe_dump(cfg, f, default_flow_style=False, sort_keys=False)

        print(f"\n[Success] Updated {config_path} with calibrated parameters.")

    finally:
        robot.stop()
        robot.disconnect()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Calibrate RoboMaster EP Camera Tag Distance")
    parser.add_argument("--mode", choices=["robot", "sim", "auto"], default="auto")
    parser.add_argument("--dist", type=float, default=1.00, help="Tape-measured distance to tag in meters")
    parser.add_argument("--tag-size", type=float, default=0.15, help="Physical tag side length in meters")
    parser.add_argument("--tag-type", choices=["qr", "aruco"], default="qr")
    args = parser.parse_args()

    calibrate(
        mode=args.mode,
        true_distance=args.dist,
        tag_size=args.tag_size,
        tag_type=args.tag_type
    )
