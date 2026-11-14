#!/usr/bin/env python3
"""Minimalistic Report Generation Script (LaTeX PDF and Markdown).

SCE-672: Advanced Robotics, KFUPM — Project Phase 3
Generates a concise, straightforward report with the required sections:
approach, sensor measurements, experimental setup, results table,
error calculations, scatterplot, and discussion.
"""

import json
import os
import subprocess
import yaml

WORKSPACE_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _load_cfg() -> dict:
    cfg_path = os.path.join(WORKSPACE_ROOT, "config", "settings.yaml")
    if os.path.exists(cfg_path):
        with open(cfg_path, "r") as f:
            return yaml.safe_load(f) or {}
    return {}


def generate_report(results_dir: str = "results", output_dir: str = "report"):
    """Generate concise Markdown and compiled LaTeX PDF reports."""
    results_path = os.path.join(WORKSPACE_ROOT, results_dir)
    report_path = os.path.join(WORKSPACE_ROOT, output_dir)
    os.makedirs(report_path, exist_ok=True)

    summary_json = os.path.join(results_path, "experiment_summary.json")
    if not os.path.exists(summary_json):
        print(f"[Error] Results file {summary_json} not found. Run experiment first.")
        return

    with open(summary_json, "r") as f:
        data = json.load(f)

    cfg = _load_cfg()
    exp_cfg = cfg.get("experiment", {})
    tag_cfg = cfg.get("tag", {})
    cam_cfg = cfg.get("camera", {})
    mot_cfg = cfg.get("motion", {})

    stats = data["statistics"]
    trials = data["trials"]

    start_d = float(exp_cfg.get("start_distance", 1.50))
    stop_d = float(stats.get("target_stop_distance", exp_cfg.get("stop_distance", 0.25)))
    cam_offset = float(exp_cfg.get("camera_to_front_offset", 0.1876))
    tag_cm = float(tag_cfg.get("physical_size", 0.078)) * 100.0
    fx_val = float(cam_cfg.get("fx", 688.63))
    v_cruise = float(mot_cfg.get("v_cruise", 0.28))
    v_min = float(mot_cfg.get("v_min", 0.105))

    # 1. Markdown Report (REPORT.md)
    md_content = f"""# Project Phase 3: RoboMaster Odometry
**Course:** Advanced Robotics (SCE-672), KFUPM  
**Student:** Raian E Rabbani (202617880)  
**Instructor:** Prof. Aamer Iqbal Bhatti

## 1. Approach and Sensor Measurements
The experiment was implemented under Ubuntu with ROS 2 on the DJI RoboMaster EP Core. Instead of using the chassis built-in `(v, w)` velocity command interface, each Mecanum wheel speed was controlled directly in RPM via inverse kinematics (`drive_wheels`).

- **Tag Detection:** A fixed {tag_cm:.1f} cm AprilTag (`36h11`) is detected from the onboard 720p camera. Using calibrated pinhole geometry ($f_x = {fx_val:.1f}$ px) and the measured camera-to-bumper offset ($d_\\text{{off}} = {cam_offset:.4f}$ m), the visual distance to the tag is $z_k = (f_x W_\\text{{tag}} / w_\\text{{px}}) - d_\\text{{off}}$.
- **Straight-Line Motion:** Visual bearing error drives a PD yaw controller to keep the robot centered on the tag, while forward speed ramps from {v_cruise:.2f} m/s down to {v_min:.3f} m/s before stopping at {stop_d:.2f} m.
- **Prediction and Correction Sensors:**
  - *Prediction (50 Hz):* Wheel encoder odometry provides chassis displacement to propagate the distance estimate between camera updates.
  - *Correction:* Camera-based tag distance measurements update a two-state Kalman filter $[d_k, v_k]^T$ whenever a new tag observation is available.

## 2. Experimental Setup
- **Reference Point:** Front bumper of the RoboMaster EP Core for both estimated and measured distances.
- **Distances:** Starting distance fixed at {start_d:.2f} m; stopping distance set to {stop_d:.2f} m.
- **Trials:** {stats['num_trials']} repeat trials under the same starting conditions.

## 3. Results and Error Calculations
| Trial | Actual $d_m$ (m) | Estimated $\\hat{{d}}$ (m) | Error $\\hat{{d}}-d_m$ (m) | $|\\text{{Error}}|$ (m) | Pure Odo (m) | Odo Error (m) |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
"""
    for t in trials:
        md_content += (
            f"| {t['trial_id']:02d} | {t['measured_actual_distance']:.4f} | "
            f"{t['estimated_final_distance']:.4f} | {t['kf_error']:+.4f} | "
            f"{t['kf_abs_error']:.4f} | {t['odometry_final_distance']:.4f} | "
            f"{t['odo_error']:+.4f} |\n"
        )

    md_content += f"""
- **Kalman Filter MAE:** {stats['kf_mae']*100:.2f} cm ({stats['kf_mae']:.4f} m)
- **Kalman Filter Error Range:** {stats['kf_error_range']*100:.2f} cm (`[{stats['kf_min_error']:+.4f}, {stats['kf_max_error']:+.4f}]` m)
- **Pure Odometry MAE:** {stats['odo_mae']*100:.2f} cm ({stats['odo_mae']:.4f} m)

## 4. Discussion
Across the {stats['num_trials']} trials, the Kalman filter stayed consistent with an MAE of {stats['kf_mae']*100:.2f} cm and a total error range of {stats['kf_error_range']*100:.2f} cm, while pure wheel odometry drifted to an MAE of {stats['odo_mae']*100:.2f} cm.

The main sources of error were Mecanum roller slip during acceleration and braking, slight floor surface variations, stopping inertia, and small pixel-quantization noise on the {tag_cm:.1f} cm tag. Wheel odometry prediction keeps the 50 Hz estimate smooth between frames, while camera correction removes accumulated wheel-slip drift before the robot reaches the target distance.
"""

    md_path = os.path.join(report_path, "REPORT.md")
    with open(md_path, "w") as f:
        f.write(md_content)
    print(f"[Report] Generated Markdown report: {md_path}")

    # 2. Minimalistic LaTeX Report
    err_plot = os.path.join(WORKSPACE_ROOT, "results", "plots", "trial_error_comparison.pdf")

    tex_content = (
        r"""\documentclass[10pt,a4paper]{article}
\usepackage{amsmath}
\usepackage{graphicx}
\usepackage{booktabs}
\usepackage{float}
\usepackage[margin=0.75in]{geometry}

\title{\textbf{Project Phase 3: RoboMaster Odometry}\\[0.2em]
\normalsize Advanced Robotics (SCE-672) -- King Fahd University of Petroleum \& Minerals}
\author{Raian E Rabbani (202617880) \\
\small Instructor: Prof. Aamer Iqbal Bhatti}
\date{November 2026}

\begin{document}
\maketitle

\section{Approach and Sensor Measurements}
The system was implemented on Ubuntu using ROS~2 with the DJI RoboMaster EP Core. Rather than sending high-level $(v, \omega)$ chassis velocity commands, individual wheel speeds were commanded directly in RPM using Mecanum inverse kinematics (\texttt{drive\_wheels}). A two-state Kalman filter $\mathbf{x}_k = [d_k, v_k]^T$ tracks the bumper-to-tag distance and forward velocity:
\begin{itemize}
    \item \textbf{Tag Detection:} Frames from the 720p camera detect a fixed """
        + f"{tag_cm:.1f}"
        + r"""~cm AprilTag (\texttt{36h11}). Distance is obtained from calibrated pinhole geometry ($f_x = """
        + f"{fx_val:.1f}"
        + r"""$~px) minus the camera-to-bumper offset ($d_{\text{off}} = """
        + f"{cam_offset:.4f}"
        + r"""$~m): $z_k = \frac{f_x W_{\text{tag}}}{w_{\text{px}}} - d_{\text{off}}$.
    \item \textbf{Straight-Line Motion:} A PD controller on the visual bearing angle keeps the robot aligned with the tag while forward speed ramps down from $"""
        + f"{v_cruise:.2f}"
        + r"""$~m/s to $"""
        + f"{v_min:.3f}"
        + r"""$~m/s before stopping.
    \item \textbf{Prediction Sensor (50~Hz):} Chassis wheel encoder odometry supplies incremental displacement $\Delta x_{\text{odo}}$ to predict $\hat{d}_{k|k-1} = \hat{d}_{k-1|k-1} - \Delta x_{\text{odo}}$ between camera updates.
    \item \textbf{Correction Sensor:} Each camera tag distance $z_k$ corrects the predicted state and covariance to compensate for wheel slip.
\end{itemize}

\section{Experimental Setup}
All """
        + str(stats["num_trials"])
        + r""" repeat trials were run from a starting distance of """
        + f"{start_d:.2f}"
        + r"""~m with a target stopping distance of """
        + f"{stop_d:.2f}"
        + r"""~m. Both the estimated and measured distances were referenced to the robot's front bumper.

\section{Results and Error Calculations}
Estimation error is calculated as $e = \hat{d} - d_m$ (estimated distance minus measured distance). Table~\ref{tab:results} summarizes all """
        + str(stats["num_trials"])
        + r""" trials alongside pure wheel odometry for comparison.

\begin{table}[H]
\centering
\caption{Final Distance Estimates and Errors Across """
        + str(stats["num_trials"])
        + r""" Repeat Trials ($d_{\text{stop}} = """
        + f"{stop_d:.2f}"
        + r"""$~m)}
\label{tab:results}
\small
\begin{tabular}{ccccccc}
\toprule
\textbf{Trial} & \textbf{Actual $d_m$ (m)} & \textbf{KF Est $\hat{d}$ (m)} & \textbf{KF Error $e$ (m)} & \textbf{$|e|$ (m)} & \textbf{Pure Odo (m)} & \textbf{Odo Error (m)} \\
\midrule
"""
    )

    for t in trials:
        tex_content += (
            f"{t['trial_id']} & {t['measured_actual_distance']:.4f} & "
            f"{t['estimated_final_distance']:.4f} & {t['kf_error']:+.4f} & "
            f"{t['kf_abs_error']:.4f} & {t['odometry_final_distance']:.4f} & "
            f"{t['odo_error']:+.4f} \\\\\n"
        )

    tex_content += (
        r"""\midrule
\textbf{MAE} & -- & -- & -- & \textbf{"""
        + f"{stats['kf_mae']:.4f}"
        + r"""} & -- & """
        + f"{stats['odo_mae']:.4f}"
        + r""" \\
\textbf{Range} & -- & -- & ["""
        + f"{stats['kf_min_error']:+.4f}, {stats['kf_max_error']:+.4f}"
        + r"""] & \textbf{"""
        + f"{stats['kf_error_range']:.4f}"
        + r"""} & -- & """
        + f"{stats['odo_error_range']:.4f}"
        + r""" \\
\bottomrule
\end{tabular}
\end{table}

\begin{figure}[H]
\centering
\includegraphics[width=0.84\textwidth]{"""
        + err_plot
        + r"""}
\caption{Scatterplot of final distance estimation errors across """
        + str(stats["num_trials"])
        + r""" trials.}
\end{figure}

\section{Discussion}
Across the """
        + str(stats["num_trials"])
        + r""" trials, the Kalman filter produced consistent stops near """
        + f"{stop_d:.2f}"
        + r"""~m with a Mean Absolute Error of """
        + f"{stats['kf_mae']*100:.2f}"
        + r"""~cm and an error range of """
        + f"{stats['kf_error_range']*100:.2f}"
        + r"""~cm. By comparison, pure wheel odometry had an MAE of """
        + f"{stats['odo_mae']*100:.2f}"
        + r"""~cm and consistently under-counted the traveled distance.

The primary sources of error were Mecanum roller slip during acceleration and stopping, minor floor unevenness, motor braking inertia, and pixel-level corner quantization on the """
        + f"{tag_cm:.1f}"
        + r"""~cm tag. Odometry prediction provided smooth 50~Hz position tracking between camera frames, while visual correction removed the accumulated slip error so the robot stopped reliably at the target distance.

\end{document}
"""
    )

    tex_path = os.path.join(report_path, "RoboMaster_Odometry_Report.tex")
    with open(tex_path, "w") as f:
        f.write(tex_content)
    print(f"[Report] Generated LaTeX source: {tex_path}")

    # Compile PDF via pdflatex
    try:
        for _ in range(2):
            res = subprocess.run(
                ["pdflatex", "-interaction=nonstopmode", "-output-directory", report_path, tex_path],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                timeout=30,
            )
        pdf_path = os.path.join(report_path, "RoboMaster_Odometry_Report.pdf")
        if os.path.exists(pdf_path) and res.returncode == 0:
            print(f"[Report Success] Compiled PDF: {pdf_path}")
            for ext in (".aux", ".log", ".out"):
                aux_file = os.path.join(report_path, "RoboMaster_Odometry_Report" + ext)
                if os.path.exists(aux_file):
                    os.remove(aux_file)
        else:
            print("[Warning] pdflatex returned non-zero code.")
    except FileNotFoundError:
        print("[Info] pdflatex not found; Markdown and .tex generated.")
    except Exception as e:
        print(f"[Warning] Failed to compile PDF: {e}")


if __name__ == "__main__":
    generate_report()
