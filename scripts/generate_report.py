#!/usr/bin/env python3
"""Minimalistic Report Generation Script (LaTeX PDF and Markdown).

SCE-672: Advanced Robotics, KFUPM — Project Phase 3
Generates a concise report containing strictly what is requested:
approach, sensor measurements, experimental setup, results table,
error calculations (MAE & range), key figures, and brief discussion.
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

    start_d = float(stats.get("start_distance", exp_cfg.get("start_distance", 1.50)))
    stop_d = float(stats.get("target_stop_distance", exp_cfg.get("stop_distance", 0.25)))
    cam_offset = float(exp_cfg.get("camera_to_front_offset", 0.1876))
    tag_cm = float(tag_cfg.get("physical_size", 0.078)) * 100.0
    fx_val = float(cam_cfg.get("fx", 688.63))
    v_cruise = float(mot_cfg.get("v_cruise", 0.28))
    v_min = float(mot_cfg.get("v_min", 0.105))

    # 1. Concise Markdown Report (REPORT.md)
    md_content = f"""# Project Phase 3: RoboMaster Odometry
**Course:** Advanced Robotics (SCE-672), KFUPM | **Instructor:** Prof. Aamer Iqbal Bhatti

## 1. Approach & Sensor Measurements
- **Tag Detection:** Fixed planar visual tag ({tag_cm:.1f} cm black square, AprilTag `36h11`) detected from the onboard 720p camera using calibrated pinhole geometry ($f_x = {fx_val:.1f}$ px, camera-to-bumper offset $d_\\text{{off}} = {cam_offset:.4f}$ m).
- **Straight-Line Motion:** Visual bearing error drives closed-loop PD yaw alignment ($\\omega_z$) with cosine deceleration ($v_\\text{{cruise}} = {v_cruise:.2f}$ m/s to $v_\\text{{min}} = {v_min:.3f}$ m/s), converted to individual wheel RPMs via Mecanum inverse kinematics (`drive_wheels`).
- **Prediction & Correction Sensors:**
  - *Prediction (50 Hz):* 4-wheel chassis encoder odometry (`sub_position`) projected onto the trial heading propagates distance $d_k = d_{{k-1}} - \\Delta x_\\text{{odo}}$.
  - *Correction:* Monocular camera tag distance $z_k = z_\\text{{cam}} - d_\\text{{off}}$ updates the 2-state Kalman Filter $[d_k, v_k]^T$ whenever a new frame arrives.

## 2. Experimental Setup
- **Platform:** DJI RoboMaster EP Core under Ubuntu.
- **Reference Point:** Front chassis bumper for both estimated and actual distances.
- **Distances:** Starting distance $D_0 \\approx {start_d:.2f}$ m; target stopping distance $d_\\text{{stop}} = {stop_d:.2f}$ m; **{stats['num_trials']} repeat trials**.

## 3. Results & Error Calculations
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
- **Kalman Filter MAE:** **{stats['kf_mae']*100:.2f} cm** (`{stats['kf_mae']:.4f}` m) | **Error Range:** **{stats['kf_error_range']*100:.2f} cm** (`[{stats['kf_min_error']:+.4f}, {stats['kf_max_error']:+.4f}]` m)
- **Pure Odometry MAE:** **{stats['odo_mae']*100:.2f} cm** (`{stats['odo_mae']:.4f}` m) | **Error Range:** **{stats['odo_error_range']*100:.2f} cm**

## 4. Discussion
- **Trial Consistency:** Across all {stats['num_trials']} trials, the Kalman filter stayed within a `{stats['kf_error_range']*100:.2f}` cm spread (MAE `{stats['kf_mae']*100:.2f}` cm) and stopped reliably near `{stop_d:.2f}` m.
- **Sources of Error:** Mecanum roller slip (causing pure odometry to under-count travel by up to `{max(abs(stats['odo_min_error']), abs(stats['odo_max_error']))*100:.1f}` cm), camera H.264 stream latency during braking, and small pixel-quantization noise on the {tag_cm:.1f} cm tag.
- **Role of Prediction & Correction:** Wheel odometry provides smooth 50 Hz state propagation between camera frames, while periodic visual tag measurements eliminate accumulated wheel-slip drift.
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
\normalsize Advanced Robotics (SCE-672) --- King Fahd University of Petroleum \& Minerals}
\author{Instructor: Prof. Aamer Iqbal Bhatti}
\date{November 2026}

\begin{document}
\maketitle

\section{Approach and Sensor Measurements}
The RoboMaster EP Core approaches a fixed visual tag along a straight path using a two-state discrete Kalman filter $\mathbf{x}_k = [d_k, v_k]^T$ (bumper-to-tag distance and forward speed):
\begin{itemize}
    \item \textbf{Tag Detection:} Monocular 720p camera frames detect an AprilTag (\texttt{36h11}, """
        + f"{tag_cm:.1f}"
        + r"""~cm black width). Distance is computed via calibrated pinhole optics ($f_x = """
        + f"{fx_val:.1f}"
        + r"""$~px, bumper offset $d_{\text{off}} = """
        + f"{cam_offset:.4f}"
        + r"""$~m): $z_k = \frac{f_x W_{\text{tag}}}{w_{\text{px}}} - d_{\text{off}}$.
    \item \textbf{Straight-Line Motion:} Closed-loop visual bearing control ($\omega_z = K_p \theta + K_d \dot{\theta}$) keeps the tag centered while velocity ramps smoothly from $"""
        + f"{v_cruise:.2f}"
        + r"""$~m/s to $"""
        + f"{v_min:.3f}"
        + r"""$~m/s via individual Mecanum wheel RPM commands (\texttt{drive\_wheels}).
    \item \textbf{Prediction (50~Hz):} Chassis wheel encoder odometry (\texttt{sub\_position}) provides displacement $\Delta x_{\text{odo}}$ to predict $\hat{d}_{k|k-1} = \hat{d}_{k-1|k-1} - \Delta x_{\text{odo}}$.
    \item \textbf{Correction:} Each camera tag measurement $z_k$ corrects the predicted state and covariance, removing wheel-slip drift.
\end{itemize}

\section{Experimental Setup}
All """
        + str(stats["num_trials"])
        + r""" repeat trials were conducted on Ubuntu with the same starting distance ($D_0 \approx """
        + f"{start_d:.2f}"
        + r"""$~m) and target stopping distance ($d_{\text{stop}} = """
        + f"{stop_d:.2f}"
        + r"""$~m), using the robot's \textbf{front bumper} as the common reference point.

\section{Results and Error Calculations}
Error is defined as $e = \hat{d} - d_m$ (estimated minus actual distance). Table~\ref{tab:results} lists all """
        + str(stats["num_trials"])
        + r""" trials alongside pure odometry (prediction-only) for comparison.

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
\includegraphics[width=0.85\textwidth]{"""
        + err_plot
        + r"""}
\caption{Scatterplot of final distance estimation errors across """
        + str(stats["num_trials"])
        + r""" repeat trials: Kalman filter vs.\ pure wheel odometry.}
\end{figure}

\section{Discussion}
\begin{itemize}
    \item \textbf{Consistency:} Across """
        + str(stats["num_trials"])
        + r""" trials, the Kalman filter achieved a Mean Absolute Error of \textbf{"""
        + f"{stats['kf_mae']*100:.2f}"
        + r"""~cm} and an error range of \textbf{"""
        + f"{stats['kf_error_range']*100:.2f}"
        + r"""~cm}, stopping consistently near """
        + f"{stop_d:.2f}"
        + r"""~m.
    \item \textbf{Sources of Error:} Mecanum roller slip caused pure odometry to drift by """
        + f"{stats['odo_mae']*100:.2f}"
        + r"""~cm MAE (up to """
        + f"{max(abs(stats['odo_min_error']), abs(stats['odo_max_error']))*100:.2f}"
        + r"""~cm). Residual Kalman error stems from H.264 video stream latency during deceleration and sub-pixel corner quantization.
    \item \textbf{Role of Prediction and Correction:} Wheel odometry supplies high-rate (50~Hz) motion prediction between camera frames, while visual tag measurements correct accumulated wheel-slip drift, reducing MAE by """
        + f"{stats['mae_improvement_pct']:.1f}"
        + r"""\%.
\end{itemize}

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
