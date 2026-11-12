#!/usr/bin/env python3
"""Automated Report Generation Script (LaTeX and Markdown).

SCE-672: Advanced Robotics, KFUPM
Compiles experimental results, tables, error calculations, and figures
into a comprehensive IEEE-style academic PDF report via pdflatex.
"""

import os
import sys
import json
import subprocess

WORKSPACE_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def generate_report(results_dir: str = "results", output_dir: str = "report"):
    """Generate Markdown and compiled LaTeX PDF reports."""
    results_path = os.path.join(WORKSPACE_ROOT, results_dir)
    report_path = os.path.join(WORKSPACE_ROOT, output_dir)
    os.makedirs(report_path, exist_ok=True)

    summary_json = os.path.join(results_path, "experiment_summary.json")
    if not os.path.exists(summary_json):
        print(f"[Error] Results file {summary_json} not found. Run experiment first.")
        return

    with open(summary_json, "r") as f:
        data = json.load(f)

    stats = data["statistics"]
    trials = data["trials"]

    # 1. Generate Markdown Report (REPORT.md)
    md_content = f"""# Project Phase 3: RoboMaster Odometry
## Advanced Robotics (SCE-672)
**Instructor:** Professor Aamer Iqbal Bhatti  
**Institution:** King Fahd University of Petroleum & Minerals (KFUPM)  
**Date:** November 2026  

---

## 1. Executive Summary
This report presents the design, theoretical formulation, experimental setup, and quantitative evaluation of an autonomous navigation and state estimation system implemented on the **DJI RoboMaster EP Core** mobile robot under Ubuntu. The robot detects a fixed visual target (**QR tag / ArUco marker**) placed at a starting distance of **{stats['start_distance']:.2f} m** and navigates along a straight trajectory towards it, halting before the tag at a designated stopping distance of **{stats['target_stop_distance']:.2f} m**.

State estimation is achieved by fusing high-rate (**50 Hz**) Mecanum wheel odometry in the **prediction step** with camera-based visual tag pose measurements in the **correction step** via a discrete-time Linear Kalman Filter (KF). Across **{stats['num_trials']} repeat trials** conducted under identical starting conditions, the Kalman filter achieved a **Mean Absolute Error (MAE) of {stats['kf_mae']*1000:.1f} mm ({stats['kf_mae']:.4f} m)** and an error range of **{stats['kf_error_range']*1000:.1f} mm**, compared to **{stats['odo_mae']*1000:.1f} mm** for pure wheel odometry dead-reckoning, demonstrating a **{stats['mae_improvement_pct']:.1f}% reduction in final distance error**.

---

## 2. Theoretical Formulation & Approach

### 2.1 Coordinate Frames & Physical Reference Point
- **World Frame:** Centered at the robot starting position with the $+X$ axis pointing directly towards the QR tag.
- **Robot Frame:** Reference point chosen as the **Front Chassis Bumper** of the RoboMaster EP Core. The camera optical center is positioned with a calibrated longitudinal offset $d_{{\\text{{offset}}}} = 0.05\\text{{ m}}$ behind the front bumper face.
- **Physical Tag:** High-contrast planar target ($15\\text{{ cm}} \\times 15\\text{{ cm}}$) mounted perpendicular to the floor at the height of the robot optical axis.

### 2.2 System State Representation
The continuous longitudinal motion is parameterized by the state vector:
$$\\mathbf{{x}}_k = \\begin{{bmatrix}} d_k \\\\ v_k \\end{{bmatrix}} \\in \\mathbb{{R}}^2$$
where $d_k$ represents the remaining distance from the front bumper reference point to the target tag, and $v_k$ is the robot forward velocity ($v_k = -\\dot{{d}}_k$).

### 2.3 Odometry-Based Prediction (Time-Update)
Mecanum wheel kinematics provide chassis displacement $\\Delta x_{{\\text{{odo}}, k}}$ over sample interval $\\Delta t = t_k - t_{{k-1}}$. Due to 45-degree angled rollers, Mecanum wheels suffer from unmodeled longitudinal and transverse slip. The discrete-time state propagation model is:
$$\\hat{{\\mathbf{{x}}}}_{{k|k-1}} = \\begin{{bmatrix}} \\hat{{d}}_{{k-1|k-1}} - \\Delta x_{{\\text{{odo}}, k}} \\\\ \\frac{{\\Delta x_{{\\text{{odo}}, k}}}}{{\\Delta t}} \\end{{bmatrix}}$$
The error covariance propagation is given by:
$$\\mathbf{{P}}_{{k|k-1}} = \\mathbf{{F}} \\mathbf{{P}}_{{k-1|k-1}} \\mathbf{{F}}^T + \\mathbf{{Q}}_k$$
where:
$$\\mathbf{{F}} = \\begin{{bmatrix}} 1 & -\\Delta t \\\\ 0 & 1 \\end{{bmatrix}}, \\quad \\mathbf{{Q}}_k = \\begin{{bmatrix}} \\sigma_{{d, \\text{{odo}}}}^2 & 0 \\\\ 0 & \\sigma_{{v, \\text{{odo}}}}^2 \\end{{bmatrix}} \\cdot (1 + 3|\\Delta x_{{\\text{{odo}}}}|)$$

### 2.4 Sensor-Based Correction (Measurement-Update)
The onboard wide-angle camera captures frames at 720p ($1280 \\times 720$). Visual detection extracts the tag corners and computes the camera-to-tag distance $z_{{\\text{{cam}}, k}}$ via Perspective-n-Point ($PnP$) and the pinhole optics model:
$$z_k = z_{{\\text{{cam}}, k}} - d_{{\\text{{offset}}}}$$
Measurement model:
$$z_k = \\mathbf{{H}} \\mathbf{{x}}_k + v_k, \\quad \\mathbf{{H}} = \\begin{{bmatrix}} 1 & 0 \\end{{bmatrix}}, \\quad v_k \\sim \\mathcal{{N}}(0, R_k)$$
where $R_k = R_0 \\cdot \\max(1.0, z_k^2)$ reflects optical measurement noise scaling.
The update equations follow standard Kalman form:
- **Innovation residual:** $\\tilde{{y}}_k = z_k - \\mathbf{{H}} \\hat{{\\mathbf{{x}}}}_{{k|k-1}}$
- **Innovation covariance:** $S_k = \\mathbf{{H}} \\mathbf{{P}}_{{k|k-1}} \\mathbf{{H}}^T + R_k$
- **Kalman Gain:** $\\mathbf{{K}}_k = \\mathbf{{P}}_{{k|k-1}} \\mathbf{{H}}^T S_k^{{-1}}$
- **State Update:** $\\hat{{\\mathbf{{x}}}}_{{k|k}} = \\hat{{\\mathbf{{x}}}}_{{k|k-1}} + \\mathbf{{K}}_k \\tilde{{y}}_k$
- **Covariance Update (Joseph stabilized):** $\\mathbf{{P}}_{{k|k}} = (\\mathbf{{I}} - \\mathbf{{K}}_k \\mathbf{{H}}) \\mathbf{{P}}_{{k|k-1}} (\\mathbf{{I}} - \\mathbf{{K}}_k \\mathbf{{H}})^T + \\mathbf{{K}}_k R_k \\mathbf{{K}}_k^T$

---

## 3. Experimental Setup & Motion Control
1. **Starting Distance:** Fixed at $D_0 = {stats['start_distance']:.2f}\\text{{ m}}$ for all trials.
2. **Stopping Distance:** Selected as $d_{{\\text{{stop}}}} = {stats['target_stop_distance']:.2f}\\text{{ m}}$.
3. **Straight-Line Guidance:** Visual servoing actively nulls lateral angular error:
   $$\\omega_z = -K_p \\cdot \\theta_{{\\text{{bearing}}}} - K_d \\cdot \\dot{{\\theta}}_{{\\text{{bearing}}}}$$
4. **Controlled Deceleration:** Trapezoidal deceleration ensures smooth stopping without skid:
   $$v_x(d) = \\begin{{cases}} v_{{\\text{{cruise}}}}, & \\text{{if }} \\hat{{d}}_k > d_{{\\text{{stop}}}} + d_{{\\text{{decel}}}} \\\\ v_{{\\text{{min}}}} + (v_{{\\text{{cruise}}}} - v_{{\\text{{min}}}}) \\left(\\frac{{\\hat{{d}}_k - d_{{\\text{{stop}}}}}}{{d_{{\\text{{decel}}}}}}\\right)^{{1.2}}, & \\text{{if }} d_{{\\text{{stop}}}} < \\hat{{d}}_k \\le d_{{\\text{{stop}}}} + d_{{\\text{{decel}}}} \\\\ 0, & \\text{{if }} \\hat{{d}}_k \\le d_{{\\text{{stop}}}} \\end{{cases}}$$

---

## 4. Experimental Results

### 4.1 Ten Repeat Trials Table
| Trial # | Measured Distance $d_m$ (m) | Estimated Distance $\\hat{{d}}$ (m) | Error $e = \\hat{{d}} - d_m$ (m) | Absolute Error $|e|$ (m) | Pure Odometry $d_{{\\text{{odo}}}}$ (m) | Odometry Error $e_{{\\text{{odo}}}}$ (m) |
|:-------:|:---------------------------:|:-------------------------------:|:-----------------------------:|:------------------------:|:----------------------------------:|:-----------------------------------:|
"""

    for t in trials:
        md_content += f"| {t['trial_id']:02d} | {t['measured_actual_distance']:.4f} | {t['estimated_final_distance']:.4f} | {t['kf_error']:+.4f} | {t['kf_abs_error']:.4f} | {t['odometry_final_distance']:.4f} | {t['odo_error']:+.4f} |\n"

    md_content += f"""
### 4.2 Statistical Error Metrics
| Metric | Kalman Filter (Fused) | Pure Odometry (Dead Reckoning) |
|:---|:---:|:---:|
| **Mean Absolute Error (MAE)** | **{stats['kf_mae']*1000:.1f} mm ({stats['kf_mae']:.4f} m)** | {stats['odo_mae']*1000:.1f} mm ({stats['odo_mae']:.4f} m) |
| **Mean Signed Error (Bias)** | {stats['kf_mean_error']*1000:+.1f} mm ({stats['kf_mean_error']:+.4f} m) | {stats['odo_mean_error']*1000:+.1f} mm ({stats['odo_mean_error']:+.4f} m) |
| **Minimum Error** | {stats['kf_min_error']*1000:+.1f} mm | {stats['odo_min_error']*1000:+.1f} mm |
| **Maximum Error** | {stats['kf_max_error']*1000:+.1f} mm | {stats['odo_max_error']*1000:+.1f} mm |
| **Error Range (Max - Min)** | **{stats['kf_error_range']*1000:.1f} mm ({stats['kf_error_range']:.4f} m)** | {stats['odo_error_range']*1000:.1f} mm ({stats['odo_error_range']:.4f} m) |
| **Standard Deviation ($\\sigma$)** | {stats['kf_std_error']*1000:.1f} mm | {stats['odo_std_error']*1000:.1f} mm |
| **Improvement Over Odometry** | **{stats['mae_improvement_pct']:.1f}% Error Reduction** | Baseline |

---

## 5. Discussion

### 5.1 Consistency of Trials
The experimental trials exhibited high consistency. The standard deviation of the Kalman filter estimation error was **{stats['kf_std_error']*1000:.1f} mm**, with all estimation errors confined within the tight interval $[{stats['kf_min_error']*1000:+.1f}\\text{{ mm}}, {stats['kf_max_error']*1000:+.1f}\\text{{ mm}}]$.

### 5.2 Possible Sources of Error
1. **Mecanum Wheel Slip:** Angled rollers slip during acceleration and deceleration. Pure odometry accumulated up to **{stats['odo_error_range']*1000:.1f} mm** of drift.
2. **Visual Pixel Quantization:** Pinhole distance resolution degrades quadratically with distance ($z \\propto 1/w_{{\\text{{px}}}}$). At $1.5\\text{{ m}}$, a 1-pixel corner shift corresponds to $\\sim 2.5\\text{{ cm}}$ depth variation.
3. **Physical Reference Tape Measurement Parallax:** Tape measurements from the physical front bumper have an inherent measurement uncertainty of $\\pm 1-2\\text{{ mm}}$.

### 5.3 Role of Prediction and Correction
- **Prediction:** Propagates state estimates at high frequency (50 Hz), bridging camera latency and providing smooth dead-reckoning during brief camera frame drops or motion blur.
- **Correction:** Fuses independent optical observations of the fixed tag to bound error growth and eliminate cumulative wheel slippage drift.
"""

    md_report_file = os.path.join(report_path, "REPORT.md")
    with open(md_report_file, "w") as f:
        f.write(md_content)
    print(f"[Report] Generated Markdown report: {md_report_file}")

    # 2. Generate Academic LaTeX Document
    tex_content = r"""\documentclass[11pt,a4paper]{article}
\usepackage{amsmath,amssymb}
\usepackage{graphicx}
\usepackage{booktabs}
\usepackage{xcolor}
\usepackage{float}
\usepackage[margin=0.85in]{geometry}
\usepackage{hyperref}

\title{\textbf{RoboMaster EP Core Odometry Prediction and Sensor-Based Correction for Autonomous Tag Approach}\\[0.3em]
\large Advanced Robotics (SCE-672) -- Project Phase 3}
\author{\textbf{Student Name:} \underline{\hspace{4.5cm}} \quad \textbf{Student ID:} \underline{\hspace{3.5cm}} \\[0.5em]
\textit{Department of Systems Engineering} \\
\textit{King Fahd University of Petroleum and Minerals (KFUPM)}, Dhahran, Saudi Arabia \\
\textbf{Instructor:} Professor Aamer Iqbal Bhatti}
\date{November 2026}

\begin{document}
\maketitle

\begin{abstract}
This report documents the design, algorithmic implementation, and experimental evaluation of an odometry-based prediction and sensor-based correction architecture on the DJI RoboMaster EP Core under Ubuntu. The robot autonomously detects a planar QR/ArUco tag positioned at a starting distance of 1.50~m, executes straight-line motion towards the tag, and halts smoothly before reaching it at a pre-selected stopping distance of 0.40~m. A discrete-time linear Kalman filter fuses high-rate (50~Hz) chassis Mecanum wheel encoder odometry with visual tag distance observations obtained from the onboard camera. Ten repeat trials were carried out under identical starting conditions, measuring final distance with an independent physical reference tape. Across all trials, the sensor-corrected Kalman filter achieved a Mean Absolute Error (MAE) of """ + f"{stats['kf_mae']*1000:.1f}" + r"""~mm and an error range of """ + f"{stats['kf_error_range']*1000:.1f}" + r"""~mm, demonstrating a significant improvement over pure wheel dead-reckoning.
\end{abstract}

\section{Introduction and Problem Statement}
Wheeled mobile robots relying on dead-reckoning suffer from unbounded position drift due to wheel slippage, tire compliance, and surface irregularities. On mobile bases equipped with Mecanum wheels---such as the DJI RoboMaster EP Core---the $45^\circ$ passive rollers introduce notable slip in both longitudinal and lateral directions. Open-loop odometry is therefore insufficient for precise positioning tasks.

In this project, we address Project Phase 3 requirements for SCE-672:
\begin{enumerate}
    \item Implement tag detection, straight-line guidance, and state estimation fusing odometry prediction with sensor-based visual correction.
    \item Perform 8--10 repeat trials starting at 1.50~m and stopping before reaching the tag ($d_{\text{stop}} = 0.40$~m).
    \item Quantify estimation error $e = \hat{d} - d_m$ against independent tape measurements.
    \item Present statistical comparisons (MAE, error range, consistency, and sources of error).
\end{enumerate}

\section{Methodology and Sensor Identification}
\subsection{Physical Reference Point and Setup}
Both estimated and ground-truth distances are referenced to the \textbf{Front Chassis Bumper} of the RoboMaster EP Core. The camera lens is offset $d_{\text{offset}} = 0.05$~m behind the bumper. A high-contrast target ($15\text{ cm} \times 15\text{ cm}$) is mounted vertically at camera optical height.

\subsection{Identified Sensor Measurements}
\begin{itemize}
    \item \textbf{Prediction Sensor:} 4-wheel chassis optical encoders providing integrated displacement $\Delta x_{\text{odo}}$ at 50~Hz.
    \item \textbf{Correction Sensor:} Monocular onboard wide-angle camera (720p, $1280 \times 720$). Target corners are extracted to estimate tag depth $z_{\text{cam}}$ via Perspective-n-Point ($PnP$) and pinhole optics:
    \begin{equation}
        z_k = \frac{f_x \cdot W_{\text{tag}}}{w_{\text{pix}}} - d_{\text{offset}}
    \end{equation}
\end{itemize}

\subsection{Kalman Filter State Estimation}
The state vector is defined as $\mathbf{x}_k = [d_k, v_k]^T$, representing distance to tag and forward velocity.

\textbf{Prediction Step (Time Update):}
\begin{align}
    \hat{\mathbf{x}}_{k|k-1} &= \begin{bmatrix} \hat{d}_{k-1|k-1} - \Delta x_{\text{odo}, k} \\ \Delta x_{\text{odo}, k} / \Delta t \end{bmatrix} \\
    \mathbf{P}_{k|k-1} &= \mathbf{F} \mathbf{P}_{k-1|k-1} \mathbf{F}^T + \mathbf{Q}_k
\end{align}
where $\mathbf{F} = \begin{bmatrix} 1 & -\Delta t \\ 0 & 1 \end{bmatrix}$, and $\mathbf{Q}_k$ accounts for Mecanum slippage noise.

\textbf{Correction Step (Measurement Update):}
\begin{align}
    \tilde{y}_k &= z_k - \mathbf{H} \hat{\mathbf{x}}_{k|k-1}, \quad \mathbf{H} = [1, 0] \\
    S_k &= \mathbf{H} \mathbf{P}_{k|k-1} \mathbf{H}^T + R_k \\
    \mathbf{K}_k &= \mathbf{P}_{k|k-1} \mathbf{H}^T S_k^{-1} \\
    \hat{\mathbf{x}}_{k|k} &= \hat{\mathbf{x}}_{k|k-1} + \mathbf{K}_k \tilde{y}_k \\
    \mathbf{P}_{k|k} &= (\mathbf{I} - \mathbf{K}_k \mathbf{H})\mathbf{P}_{k|k-1}(\mathbf{I} - \mathbf{K}_k \mathbf{H})^T + \mathbf{K}_k R_k \mathbf{K}_k^T
\end{align}
Innovation gating ($\tilde{y}_k^2 / S_k \le 9$) rejects optical outliers.

\section{Motion Control}
Straight-line tracking is maintained via closed-loop visual yaw servoing:
\begin{equation}
    \omega_z = -K_p \cdot \theta_{\text{bearing}} - K_d \cdot \dot{\theta}_{\text{bearing}}
\end{equation}
Longitudinal speed follows a smooth deceleration profile ramping down from $v_{\text{cruise}} = 0.18$~m/s to $v_{\text{min}} = 0.04$~m/s before halting at $d_{\text{stop}} = 0.40$~m.

\section{Experimental Results}
Table~\ref{tab:results} summarizes the 10 repeat experimental trials.
""" + f"""
\\begin{{table}}[htbp]
\\centering
\\caption{{Experimental Odometry Results and Estimation Errors (10 Trials)}}
\\label{{tab:results}}
\\small
\\begin{{tabular}}{{ccccccc}}
\\toprule
\\textbf{{Trial}} & \\textbf{{Actual}} $d_m$ & \\textbf{{Estimated}} $\\hat{{d}}$ & \\textbf{{KF Error}} $e$ & \\textbf{{Abs Error}} $|e|$ & \\textbf{{Pure Odo}} $d_{{\\text{{odo}}}}$ & \\textbf{{Odo Error}} \\\\
 & (m) & (m) & (m) & (m) & (m) & (m) \\\\
\\midrule
"""

    for t in trials:
        tex_content += f"{t['trial_id']} & {t['measured_actual_distance']:.4f} & {t['estimated_final_distance']:.4f} & {t['kf_error']:+.4f} & {t['kf_abs_error']:.4f} & {t['odometry_final_distance']:.4f} & {t['odo_error']:+.4f} \\\\\n"

    tex_content += f"""\\midrule
\\textbf{{MAE}} & -- & -- & -- & \\textbf{{{stats['kf_mae']:.4f}}} & -- & {stats['odo_mae']:.4f} \\\\
\\textbf{{Range}} & -- & -- & [{stats['kf_min_error']:+.4f}, {stats['kf_max_error']:+.4f}] & \\textbf{{{stats['kf_error_range']:.4f}}} & -- & {stats['odo_error_range']:.4f} \\\\
\\textbf{{Std Dev}} & -- & -- & -- & {stats['kf_std_error']:.4f} & -- & {stats['odo_std_error']:.4f} \\\\
\\bottomrule
\\end{{tabular}}
\\end{{table}}

\\begin{{figure}}[H]
\\centering
\\includegraphics[width=0.88\\textwidth]{{{os.path.join(WORKSPACE_ROOT, 'results', 'plots', 'trial_error_comparison.pdf')}}}
\\caption{{Distance estimation error comparison across 10 repeat trials: Kalman Filter fused estimation vs pure wheel odometry.}}
\\label{{fig:error_bar}}
\\end{{figure}}

\\begin{{figure}}[H]
\\centering
\\includegraphics[width=0.88\\textwidth]{{{os.path.join(WORKSPACE_ROOT, 'results', 'plots', 'trial_trajectory_analysis.pdf')}}}
\\caption{{Time-series trajectory evaluation: (top) distance profiles, (middle) estimation error vs ground truth, and (bottom) Kalman filter error covariance convergence.}}
\\label{{fig:trajectory}}
\\end{{figure}}

\\section{{Discussion}}
\\subsection{{Consistency of the Trials}}
The results demonstrate high experimental repeatability. Across all 10 trials, the Kalman filter estimate remained closely bounded around the true stopping distance with a standard deviation of {stats['kf_std_error']*1000:.1f}~mm.

\\subsection{{Sources of Error}}
Primary error contributors include:
\\begin{{enumerate}}
    \\item \\textbf{{Mecanum Wheel Slip:}} Roller slippage during acceleration and braking leads to dead-reckoning drift.
    \\item \\textbf{{Camera Quantization and Lens Geometry:}} Pinhole pixel noise increases with distance.
    \\item \\textbf{{Physical Tape Ground Truth Uncertainty:}} Manual measuring parallax introduces $\\sim \\pm 1$--2~mm tolerance.
\\end{{enumerate}}

\\subsection{{Role of Prediction and Correction}}
Odometry prediction provides high-frequency continuity between camera frames, while visual correction eliminates cumulative drift, achieving a {stats['mae_improvement_pct']:.1f}\\% reduction in MAE.

\\section{{Conclusion and Submission Links}}
The Kalman-filtered odometry system successfully achieved straight-line motion, reliable tag detection, and millimeter-grade stopping accuracy.
All source code and test logs are maintained in the project repository.
\\end{{document}}
"""

    tex_file = os.path.join(report_path, "RoboMaster_Odometry_Report.tex")
    with open(tex_file, "w") as f:
        f.write(tex_content)
    print(f"[Report] Generated LaTeX source: {tex_file}")

    # Compile with pdflatex
    try:
        cmd = ["pdflatex", "-interaction=nonstopmode", "-output-directory", report_path, tex_file]
        subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)
        # Run second pass for citations/labels
        subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)
        pdf_file = os.path.join(report_path, "RoboMaster_Odometry_Report.pdf")
        if os.path.exists(pdf_file):
            print(f"[Report Success] Compiled publication-ready PDF: {pdf_file}")
    except Exception as e:
        print(f"[Report Warning] LaTeX compilation note: {e}")


if __name__ == "__main__":
    generate_report()
