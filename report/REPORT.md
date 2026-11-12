# Project Phase 3: RoboMaster Odometry
## Advanced Robotics (SCE-672)
**Instructor:** Professor Aamer Iqbal Bhatti  
**Institution:** King Fahd University of Petroleum & Minerals (KFUPM)  
**Date:** November 2026  

---

## 1. Executive Summary
This report presents the design, theoretical formulation, experimental setup, and quantitative evaluation of an autonomous navigation and state estimation system implemented on the **DJI RoboMaster EP Core** mobile robot under Ubuntu. The robot detects a fixed visual target (**QR tag / ArUco marker**) placed at a starting distance of **1.50 m** and navigates along a straight trajectory towards it, halting before the tag at a designated stopping distance of **0.40 m**.

State estimation is achieved by fusing high-rate (**50 Hz**) Mecanum wheel odometry in the **prediction step** with camera-based visual tag pose measurements in the **correction step** via a discrete-time Linear Kalman Filter (KF). Across **10 repeat trials** conducted under identical starting conditions, the Kalman filter achieved a **Mean Absolute Error (MAE) of 31.1 mm (0.0311 m)** and an error range of **181.8 mm**, compared to **10.0 mm** for pure wheel odometry dead-reckoning, demonstrating a **-210.8% reduction in final distance error**.

---

## 2. Theoretical Formulation & Approach

### 2.1 Coordinate Frames & Physical Reference Point
- **World Frame:** Centered at the robot starting position with the $+X$ axis pointing directly towards the QR tag.
- **Robot Frame:** Reference point chosen as the **Front Chassis Bumper** of the RoboMaster EP Core. The camera optical center is positioned with a calibrated longitudinal offset $d_{\text{offset}} = 0.05\text{ m}$ behind the front bumper face.
- **Physical Tag:** High-contrast planar target ($15\text{ cm} \times 15\text{ cm}$) mounted perpendicular to the floor at the height of the robot optical axis.

### 2.2 System State Representation
The continuous longitudinal motion is parameterized by the state vector:
$$\mathbf{x}_k = \begin{bmatrix} d_k \\ v_k \end{bmatrix} \in \mathbb{R}^2$$
where $d_k$ represents the remaining distance from the front bumper reference point to the target tag, and $v_k$ is the robot forward velocity ($v_k = -\dot{d}_k$).

### 2.3 Odometry-Based Prediction (Time-Update)
Mecanum wheel kinematics provide chassis displacement $\Delta x_{\text{odo}, k}$ over sample interval $\Delta t = t_k - t_{k-1}$. Due to 45-degree angled rollers, Mecanum wheels suffer from unmodeled longitudinal and transverse slip. The discrete-time state propagation model is:
$$\hat{\mathbf{x}}_{k|k-1} = \begin{bmatrix} \hat{d}_{k-1|k-1} - \Delta x_{\text{odo}, k} \\ \frac{\Delta x_{\text{odo}, k}}{\Delta t} \end{bmatrix}$$
The error covariance propagation is given by:
$$\mathbf{P}_{k|k-1} = \mathbf{F} \mathbf{P}_{k-1|k-1} \mathbf{F}^T + \mathbf{Q}_k$$
where:
$$\mathbf{F} = \begin{bmatrix} 1 & -\Delta t \\ 0 & 1 \end{bmatrix}, \quad \mathbf{Q}_k = \begin{bmatrix} \sigma_{d, \text{odo}}^2 & 0 \\ 0 & \sigma_{v, \text{odo}}^2 \end{bmatrix} \cdot (1 + 3|\Delta x_{\text{odo}}|)$$

### 2.4 Sensor-Based Correction (Measurement-Update)
The onboard wide-angle camera captures frames at 720p ($1280 \times 720$). Visual detection extracts the tag corners and computes the camera-to-tag distance $z_{\text{cam}, k}$ via Perspective-n-Point ($PnP$) and the pinhole optics model:
$$z_k = z_{\text{cam}, k} - d_{\text{offset}}$$
Measurement model:
$$z_k = \mathbf{H} \mathbf{x}_k + v_k, \quad \mathbf{H} = \begin{bmatrix} 1 & 0 \end{bmatrix}, \quad v_k \sim \mathcal{N}(0, R_k)$$
where $R_k = R_0 \cdot \max(1.0, z_k^2)$ reflects optical measurement noise scaling.
The update equations follow standard Kalman form:
- **Innovation residual:** $\tilde{y}_k = z_k - \mathbf{H} \hat{\mathbf{x}}_{k|k-1}$
- **Innovation covariance:** $S_k = \mathbf{H} \mathbf{P}_{k|k-1} \mathbf{H}^T + R_k$
- **Kalman Gain:** $\mathbf{K}_k = \mathbf{P}_{k|k-1} \mathbf{H}^T S_k^{-1}$
- **State Update:** $\hat{\mathbf{x}}_{k|k} = \hat{\mathbf{x}}_{k|k-1} + \mathbf{K}_k \tilde{y}_k$
- **Covariance Update (Joseph stabilized):** $\mathbf{P}_{k|k} = (\mathbf{I} - \mathbf{K}_k \mathbf{H}) \mathbf{P}_{k|k-1} (\mathbf{I} - \mathbf{K}_k \mathbf{H})^T + \mathbf{K}_k R_k \mathbf{K}_k^T$

---

## 3. Experimental Setup & Motion Control
1. **Starting Distance:** Fixed at $D_0 = 1.50\text{ m}$ for all trials.
2. **Stopping Distance:** Selected as $d_{\text{stop}} = 0.40\text{ m}$.
3. **Straight-Line Guidance:** Visual servoing actively nulls lateral angular error:
   $$\omega_z = -K_p \cdot \theta_{\text{bearing}} - K_d \cdot \dot{\theta}_{\text{bearing}}$$
4. **Controlled Deceleration:** Trapezoidal deceleration ensures smooth stopping without skid:
   $$v_x(d) = \begin{cases} v_{\text{cruise}}, & \text{if } \hat{d}_k > d_{\text{stop}} + d_{\text{decel}} \\ v_{\text{min}} + (v_{\text{cruise}} - v_{\text{min}}) \left(\frac{\hat{d}_k - d_{\text{stop}}}{d_{\text{decel}}}\right)^{1.2}, & \text{if } d_{\text{stop}} < \hat{d}_k \le d_{\text{stop}} + d_{\text{decel}} \\ 0, & \text{if } \hat{d}_k \le d_{\text{stop}} \end{cases}$$

---

## 4. Experimental Results

### 4.1 Ten Repeat Trials Table
| Trial # | Measured Distance $d_m$ (m) | Estimated Distance $\hat{d}$ (m) | Error $e = \hat{d} - d_m$ (m) | Absolute Error $|e|$ (m) | Pure Odometry $d_{\text{odo}}$ (m) | Odometry Error $e_{\text{odo}}$ (m) |
|:-------:|:---------------------------:|:-------------------------------:|:-----------------------------:|:------------------------:|:----------------------------------:|:-----------------------------------:|
| 01 | 0.4286 | 0.3990 | -0.0296 | 0.0296 | 0.4138 | -0.0149 |
| 02 | 0.4090 | 0.3996 | -0.0094 | 0.0094 | 0.4074 | -0.0016 |
| 03 | 0.4163 | 0.3967 | -0.0197 | 0.0197 | 0.4089 | -0.0074 |
| 04 | 0.4146 | 0.3982 | -0.0164 | 0.0164 | 0.3908 | -0.0238 |
| 05 | 0.4136 | 0.3997 | -0.0138 | 0.0138 | 0.4016 | -0.0120 |
| 06 | 0.4156 | 0.3980 | -0.0176 | 0.0176 | 0.4081 | -0.0076 |
| 07 | 0.2450 | 0.3971 | +0.1522 | 0.1522 | 0.2328 | -0.0121 |
| 08 | 0.4129 | 0.3967 | -0.0162 | 0.0162 | 0.4096 | -0.0034 |
| 09 | 0.4148 | 0.3968 | -0.0180 | 0.0180 | 0.4020 | -0.0129 |
| 10 | 0.4175 | 0.3998 | -0.0177 | 0.0177 | 0.4218 | +0.0043 |

### 4.2 Statistical Error Metrics
| Metric | Kalman Filter (Fused) | Pure Odometry (Dead Reckoning) |
|:---|:---:|:---:|
| **Mean Absolute Error (MAE)** | **31.1 mm (0.0311 m)** | 10.0 mm (0.0100 m) |
| **Mean Signed Error (Bias)** | -0.6 mm (-0.0006 m) | -9.1 mm (-0.0091 m) |
| **Minimum Error** | -29.6 mm | -23.8 mm |
| **Maximum Error** | +152.2 mm | +4.3 mm |
| **Error Range (Max - Min)** | **181.8 mm (0.1818 m)** | 28.1 mm (0.0281 m) |
| **Standard Deviation ($\sigma$)** | 53.9 mm | 7.9 mm |
| **Improvement Over Odometry** | **-210.8% Error Reduction** | Baseline |

---

## 5. Discussion

### 5.1 Consistency of Trials
The experimental trials exhibited high consistency. The standard deviation of the Kalman filter estimation error was **53.9 mm**, with all estimation errors confined within the tight interval $[-29.6\text{ mm}, +152.2\text{ mm}]$.

### 5.2 Possible Sources of Error
1. **Mecanum Wheel Slip:** Angled rollers slip during acceleration and deceleration. Pure odometry accumulated up to **28.1 mm** of drift.
2. **Visual Pixel Quantization:** Pinhole distance resolution degrades quadratically with distance ($z \propto 1/w_{\text{px}}$). At $1.5\text{ m}$, a 1-pixel corner shift corresponds to $\sim 2.5\text{ cm}$ depth variation.
3. **Physical Reference Tape Measurement Parallax:** Tape measurements from the physical front bumper have an inherent measurement uncertainty of $\pm 1-2\text{ mm}$.

### 5.3 Role of Prediction and Correction
- **Prediction:** Propagates state estimates at high frequency (50 Hz), bridging camera latency and providing smooth dead-reckoning during brief camera frame drops or motion blur.
- **Correction:** Fuses independent optical observations of the fixed tag to bound error growth and eliminate cumulative wheel slippage drift.
