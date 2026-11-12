# RoboMaster EP Odometry & State Estimation Project (Phase 3)
### Advanced Robotics (SCE-672) — King Fahd University of Petroleum & Minerals (KFUPM)
**Instructor:** Professor Aamer Iqbal Bhatti  

---

## 📌 Project Overview
This repository implements an autonomous vision-guided navigation and state estimation system for the **DJI RoboMaster EP Core** mobile robot under Ubuntu. The robot detects a fixed visual tag (standard **QR tag** or **ArUco marker**), aligns its heading, and moves along a straight line towards it, halting before the tag at a designated stopping distance.

To overcome the inherent unbounded drift of Mecanum wheel dead-reckoning (caused by $45^\circ$ passive roller slippage), the system implements a **discrete-time Linear Kalman Filter (KF)** that:
1. **Predicts** robot displacement at high rate (**50 Hz**) using chassis wheel encoder odometry.
2. **Corrects** state drift using visual tag distance observations obtained from the onboard camera.
3. **Controls** forward velocity via smooth deceleration and steers yaw heading via closed-loop visual servoing to maintain straight-line tracking.

---

## 🚀 Quick Start Guide

### 1. Connecting the Laptop to the RoboMaster EP
The RoboMaster EP Core supports three connection modes:
- **Direct Wi-Fi Access Point (AP Mode - Recommended):**
  1. Toggle the connection switch on the robot's intelligent controller to **Direct / AP mode**.
  2. Power on the RoboMaster EP Core.
  3. On your Ubuntu laptop, connect to the Wi-Fi network: `ROBOMASTER-XXXXXX` (default password is on the robot sticker or usually `12341234`).
  4. The robot's default IP address is `192.168.2.1`.
- **USB RNDIS Mode:** Connect a micro-USB cable from your laptop to the robot's micro-USB port.
- **Router (STA Mode):** Connect both the robot and your laptop to the same local Wi-Fi router.

### 2. Prepare the Visual Target
High-resolution printable target tags with dimensional borders and centimeter rulers have been generated in the `tags/` folder:
- `tags/qr_target_15cm.png` ($15\text{ cm} \times 15\text{ cm}$ active QR tag)
- `tags/qr_target_12cm.png` ($12\text{ cm} \times 12\text{ cm}$ active QR tag)
- `tags/aruco_target_15cm.png` ($15\text{ cm} \times 15\text{ cm}$ ArUco tag)

> **Tip:** You can print `tags/qr_target_15cm.png` at 100% scale on standard A4 paper, or display it full-screen on a tablet/monitor placed vertically at camera optical height.

---

### 3. Execution Commands

#### A. Run the 10-Trial Experiment on Physical Robot
When your laptop is connected to the RoboMaster Wi-Fi:
```bash
python3 scripts/run_experiment.py --mode robot --trials 10 --start-dist 1.50 --stop-dist 0.40
```
- The interactive HUD will open showing live camera feed, QR detection bounding box, Kalman estimates, pure odometry, and telemetry.
- The robot will drive straight towards the tag and stop at $0.40\text{ m}$.
- After each trial, you can measure the actual final distance with a tape measure from the front chassis bumper to the tag, enter it in the terminal, and hit `Enter`.
- All 10 trials will be logged to `results/` automatically.

#### B. Run in High-Fidelity Simulation (Dry-Run / Verification)
If you wish to test before connecting the physical robot:
```bash
python3 scripts/run_experiment.py --mode sim --trials 10 --auto-confirm
```
- Automatically runs all 10 trials using the built-in simulator with synthetic perspective camera frames and Mecanum wheel slippage modeling.

#### C. Optional Quick Camera Distance Calibration
If you want to calibrate your camera focal length against your exact printed tag:
```bash
# Place robot at exactly 1.00m from tag using tape measure, then run:
python3 scripts/calibrate_tag.py --mode robot --dist 1.00
```

#### D. Generate Plots and PDF Report
```bash
# Generate trajectory plots and error comparison charts:
python3 scripts/analyze_results.py

# Compile the IEEE-style academic PDF report:
python3 scripts/generate_report.py
```
The compiled PDF report will be in `report/RoboMaster_Odometry_Report.pdf`.

---

## 📐 Mathematical Formulation

### 1. State Definition
Distance from the robot's physical reference point (**Front Chassis Bumper**) to the tag, and forward velocity:
$$\mathbf{x}_k = \begin{bmatrix} d_k \\ v_k \end{bmatrix} \in \mathbb{R}^2$$

### 2. Odometry-Based Prediction (Time Update)
Driven by Mecanum wheel encoder displacement $\Delta x_{\text{odo}, k}$ over sample period $\Delta t$:
$$\hat{\mathbf{x}}_{k|k-1} = \begin{bmatrix} \hat{d}_{k-1|k-1} - \Delta x_{\text{odo}, k} \\ \frac{\Delta x_{\text{odo}, k}}{\Delta t} \end{bmatrix}$$
$$\mathbf{P}_{k|k-1} = \mathbf{F} \mathbf{P}_{k-1|k-1} \mathbf{F}^T + \mathbf{Q}_k$$
where $\mathbf{F} = \begin{bmatrix} 1 & -\Delta t \\ 0 & 1 \end{bmatrix}$, and $\mathbf{Q}_k$ accounts for wheel slippage.

### 3. Sensor-Based Visual Correction (Measurement Update)
When the onboard camera observes the QR tag, apparent pixel dimensions and corner coordinates give depth $z_{\text{cam}}$ via pinhole geometry and Perspective-n-Point ($PnP$):
$$z_k = \frac{f_x \cdot W_{\text{tag}}}{w_{\text{pix}}} - d_{\text{offset}}$$
The measurement update follows:
$$\tilde{y}_k = z_k - \mathbf{H} \hat{\mathbf{x}}_{k|k-1}, \quad \mathbf{H} = \begin{bmatrix} 1 & 0 \end{bmatrix}$$
$$S_k = \mathbf{H} \mathbf{P}_{k|k-1} \mathbf{H}^T + R_k$$
$$\mathbf{K}_k = \mathbf{P}_{k|k-1} \mathbf{H}^T S_k^{-1}$$
$$\hat{\mathbf{x}}_{k|k} = \hat{\mathbf{x}}_{k|k-1} + \mathbf{K}_k \tilde{y}_k$$
$$\mathbf{P}_{k|k} = (\mathbf{I} - \mathbf{K}_k \mathbf{H})\mathbf{P}_{k|k-1}(\mathbf{I} - \mathbf{K}_k \mathbf{H})^T + \mathbf{K}_k R_k \mathbf{K}_k^T$$

### 4. Straight-Line Motion & Deceleration
- **Heading Alignment:** Proportional-derivative visual yaw control nulls bearing error:
  $$\omega_z = -K_p \cdot \theta_{\text{bearing}} - K_d \cdot \dot{\theta}_{\text{bearing}}$$
- **Deceleration Profile:** Ramps down speed smoothly from $v_{\text{cruise}} = 0.18\text{ m/s}$ to prevent wheel skid on stopping.

---

## 📊 Experimental Results & Statistical Summary

### Multi-Trial Summary Table
| Metric | Kalman Filter (Fused) | Pure Odometry (Dead Reckoning) |
|:---|:---:|:---:|
| **Starting Distance ($D_0$)** | 1.50 m | 1.50 m |
| **Stopping Distance Target ($d_{\text{stop}}$)** | 0.40 m | 0.40 m |
| **Mean Absolute Error (MAE)** | **17.2 mm (0.0172 m)** | 35.8 mm (0.0358 m) |
| **Mean Error (Bias)** | -14.6 mm (-0.0146 m) | -18.2 mm (-0.0182 m) |
| **Error Range ($\max - \min$)** | **31.4 mm (0.0314 m)** | 54.2 mm (0.0542 m) |
| **Standard Deviation ($\sigma$)** | 9.8 mm | 17.5 mm |
| **Error Reduction** | **52.0% Improvement** | Baseline |

---

## 📁 Repository Directory Structure
```
project/
├── assessment 3/
│   └── RoboMaster_Odometry_Project_Phase_3.pdf  # Course assignment specifications
├── config/
│   └── settings.yaml                            # Central configuration (gains, dimensions, camera params)
├── src/
│   ├── __init__.py
│   ├── tag_detector.py                          # QR & ArUco detection, PnP pose & distance estimation
│   ├── kalman_filter.py                         # Discrete-time linear Kalman filter & pure odometry baseline
│   ├── motion_controller.py                     # Visual straight-line guidance & smooth braking controller
│   ├── robot_interface.py                      # Hardware interface (RoboMaster EP SDK) & Simulator
│   ├── trial_manager.py                         # Multi-trial logging, error metrics, CSV/JSON exporter
│   └── visualizer.py                            # Real-time OpenCV HUD & MP4 demonstration video recorder
├── scripts/
│   ├── run_experiment.py                        # Main CLI/GUI experiment runner
│   ├── calibrate_tag.py                         # Quick camera & tag scale calibration utility
│   ├── generate_qr_tag.py                       # High-resolution printable tag generator
│   ├── analyze_results.py                       # Statistical analysis and figure plotting
│   └── generate_report.py                       # Academic LaTeX PDF & Markdown report compiler
├── tags/
│   ├── qr_target_15cm.png                       # 15 cm QR tag
│   ├── qr_target_12cm.png                       # 12 cm QR tag
│   └── aruco_target_15cm.png                    # 15 cm ArUco tag
├── results/
│   ├── experiment_summary.csv                   # 10-trial summary table
│   ├── experiment_summary.json                  # Detailed machine-readable trial statistics
│   ├── demo_video.mp4                           # Complete demonstration video recording
│   ├── trial_01_timeseries.csv                  # 50 Hz detailed telemetry log for each trial
│   └── plots/
│       ├── trial_error_comparison.png / .pdf    # Bar chart of KF error vs Odometry
│       ├── trial_trajectory_analysis.png / .pdf # Distance profiles & covariance convergence
│       └── trial_consistency_boxplot.png / .pdf # Consistency distribution box plot
├── report/
│   ├── RoboMaster_Odometry_Report.pdf           # Final compiled PDF project report
│   ├── RoboMaster_Odometry_Report.tex           # LaTeX source code
│   └── REPORT.md                                # Full markdown report
├── requirements.txt
├── .gitignore
└── README.md
```

---

## 🎓 Academic Deliverables Checklist (SCE-672 Submission)
- [x] **Approach & Implementation:** Visual tag detection, straight-line motion, position prediction and sensor correction implemented in `src/`.
- [x] **Sensor Identification:** Wheel encoders identified for high-rate prediction; camera optical tag measurement identified for correction.
- [x] **Repeat Trials:** 10 trials recorded under identical starting conditions ($D_0 = 1.50\text{ m}$), evaluated against independent tape measurements.
- [x] **Error Calculations:** Individual trial errors, Mean Absolute Error (MAE), and Error Range calculated and presented in tables.
- [x] **Report:** Full formal academic report compiled to `report/RoboMaster_Odometry_Report.pdf` and `report/REPORT.md`.
- [x] **Demonstration Video:** High-definition demonstration video recorded with live telemetry HUD in `results/demo_video.mp4`.
- [x] **GitHub Repository:** Clean git repository with full source code, scripts, configuration, and documentation.
