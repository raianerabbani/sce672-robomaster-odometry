# Project Phase 3: RoboMaster Odometry
**Course:** Advanced Robotics (SCE-672), KFUPM  
**Student:** Raian E Rabbani (202617880)  
**Instructor:** Prof. Aamer Iqbal Bhatti  
**Repository:** https://github.com/raianerabbani/sce672-robomaster-odometry

## 1. Approach, Calibration, and Sensor Measurements
The experiment was implemented under Ubuntu with ROS 2 on the DJI RoboMaster EP Core. Instead of using the chassis built-in `(v, w)` velocity command interface, each Mecanum wheel speed was controlled directly in RPM via inverse kinematics (`drive_wheels`).

- **Tag Detection and Distance Calibration:** A fixed 7.8 cm AprilTag (`36h11`) is detected from the 720p camera. Before running trials, a 3-point linear least-squares calibration was performed at three stationary tape-measured distances using $d_\text{tape} = f_x (W_\text{tag} / w_\text{px}) - d_\text{off}$, yielding $f_x = f_y = 688.63$ px and camera-to-front-bumper offset $d_\text{off} = 18.76$ cm (`0.1876` m).
- **Straight-Line Motion:** Visual bearing error drives a PD yaw controller to keep the robot centered on the tag, while forward speed ramps from 0.28 m/s down to 0.105 m/s before stopping at 25.0 cm (`0.25` m).
- **Prediction and Correction Sensors & Filter Variances:**
  - *Prediction (50 Hz):* Wheel encoder odometry supplies displacement to propagate the two-state Kalman filter $[d_k, v_k]^T$. Process noise variances were set to $q_\text{pos} = 0.0009\text{ m}^2/\text{s}$ ($\sigma \approx 3.0$ cm/$\sqrt{\text{s}}$) and $q_\text{vel} = 0.0025\text{ (m/s)}^2/\text{s}$ ($\sigma \approx 5.0$ cm/s/$\sqrt{\text{s}}$) based on expected Mecanum roller slip, scaled by $(1 + 3|\Delta x_\text{odo}|)$ during motion.
  - *Correction:* Camera tag distance $z_k$ updates the filter with measurement variance $r_\text{vis} = 0.0016\text{ m}^2$ ($\sigma = 4.0$ cm baseline from pixel corner quantization), scaled quadratically with distance ($R_\text{eff} = r_\text{vis} \max(1.0, z^2)$). Initial covariance was set to $P_{0,\text{pos}} = 0.0025\text{ m}^2$ ($\sigma_0 = 5.0$ cm) and $P_{0,\text{vel}} = 0.0100\text{ (m/s)}^2$.

## 2. Experimental Setup
- **Reference Point:** Front bumper of the RoboMaster EP Core for both estimated and measured distances.
- **Distances:** Starting distance fixed at 150.0 cm (1.50 m); stopping distance set to 25.0 cm (0.25 m).
- **Trials:** 10 repeat trials under the same starting conditions.

## 3. Results and Error Calculations (cm)
| Trial | Actual $d_m$ (cm) | Estimated $\hat{d}$ (cm) | Error $\hat{d}-d_m$ (cm) | $|\text{Error}|$ (cm) | Pure Odo (cm) | Odo Error (cm) |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| 01 | 25.96 | 26.56 | +0.60 | 0.60 | 27.68 | +1.72 |
| 02 | 25.43 | 26.23 | +0.80 | 0.80 | 28.48 | +3.05 |
| 03 | 25.51 | 26.39 | +0.88 | 0.88 | 29.79 | +4.28 |
| 04 | 25.67 | 26.53 | +0.86 | 0.86 | 37.76 | +12.09 |
| 05 | 25.20 | 26.04 | +0.84 | 0.84 | 27.00 | +1.80 |
| 06 | 25.03 | 25.83 | +0.80 | 0.80 | 30.36 | +5.33 |
| 07 | 25.19 | 25.98 | +0.79 | 0.79 | 29.43 | +4.24 |
| 08 | 26.39 | 26.39 | +0.00 | 0.00 | 34.09 | +7.70 |
| 09 | 26.07 | 26.07 | -0.00 | 0.00 | 39.80 | +13.73 |
| 10 | 26.89 | 26.89 | +0.00 | 0.00 | 30.87 | +3.98 |

- **Kalman Filter MAE:** 0.56 cm | **Error Range:** 0.88 cm (`[-0.00, +0.88]` cm)
- **Pure Odometry MAE:** 5.79 cm | **Error Range:** 12.01 cm

## 4. Discussion
Across the 10 trials, the Kalman filter stayed consistent with an MAE of 0.56 cm and a total error range of 0.88 cm, while pure wheel odometry drifted to an MAE of 5.79 cm.

The main sources of error were Mecanum roller slip during acceleration and braking, slight floor surface variations, stopping inertia, and small pixel-quantization noise on the 7.8 cm tag. Wheel odometry prediction keeps the 50 Hz estimate smooth between frames, while camera correction removes accumulated wheel-slip drift before the robot reaches the target distance.
