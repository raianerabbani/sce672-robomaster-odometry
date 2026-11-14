# Project Phase 3: RoboMaster Odometry
**Course:** Advanced Robotics (SCE-672), KFUPM | **Instructor:** Prof. Aamer Iqbal Bhatti

## 1. Approach & Sensor Measurements
- **Tag Detection:** Fixed planar visual tag (7.8 cm black square, AprilTag `36h11`) detected from the onboard 720p camera using calibrated pinhole geometry ($f_x = 688.6$ px, camera-to-bumper offset $d_\text{off} = 0.1876$ m).
- **Straight-Line Motion:** Visual bearing error drives closed-loop PD yaw alignment ($\omega_z$) with cosine deceleration ($v_\text{cruise} = 0.28$ m/s to $v_\text{min} = 0.105$ m/s), converted to individual wheel RPMs via Mecanum inverse kinematics (`drive_wheels`).
- **Prediction & Correction Sensors:**
  - *Prediction (50 Hz):* 4-wheel chassis encoder odometry (`sub_position`) projected onto the trial heading propagates distance $d_k = d_{k-1} - \Delta x_\text{odo}$.
  - *Correction:* Monocular camera tag distance $z_k = z_\text{cam} - d_\text{off}$ updates the 2-state Kalman Filter $[d_k, v_k]^T$ whenever a new frame arrives.

## 2. Experimental Setup
- **Platform:** DJI RoboMaster EP Core under Ubuntu.
- **Reference Point:** Front chassis bumper for both estimated and actual distances.
- **Distances:** Starting distance $D_0 \approx 1.50$ m; target stopping distance $d_\text{stop} = 0.25$ m; **10 repeat trials**.

## 3. Results & Error Calculations
| Trial | Actual $d_m$ (m) | Estimated $\hat{d}$ (m) | Error $\hat{d}-d_m$ (m) | $|\text{Error}|$ (m) | Pure Odo (m) | Odo Error (m) |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| 01 | 0.2587 | 0.2632 | +0.0045 | 0.0045 | 0.2728 | +0.0141 |
| 02 | 0.2593 | 0.2638 | +0.0046 | 0.0046 | 0.2953 | +0.0361 |
| 03 | 0.2610 | 0.2630 | +0.0019 | 0.0019 | 0.2975 | +0.0365 |
| 04 | 0.2605 | 0.2619 | +0.0014 | 0.0014 | 0.2878 | +0.0272 |
| 05 | 0.2584 | 0.2604 | +0.0020 | 0.0020 | 0.3219 | +0.0635 |
| 06 | 0.2609 | 0.2635 | +0.0026 | 0.0026 | 0.3256 | +0.0647 |
| 07 | 0.2590 | 0.2674 | +0.0085 | 0.0085 | 0.3111 | +0.0522 |
| 08 | 0.2584 | 0.2589 | +0.0005 | 0.0005 | 0.3028 | +0.0444 |
| 09 | 0.2514 | 0.2549 | +0.0036 | 0.0036 | 0.3694 | +0.1181 |
| 10 | 0.2522 | 0.2560 | +0.0038 | 0.0038 | 0.3471 | +0.0949 |

- **Kalman Filter MAE:** **0.33 cm** (`0.0033` m) | **Error Range:** **0.80 cm** (`[+0.0005, +0.0085]` m)
- **Pure Odometry MAE:** **5.52 cm** (`0.0552` m) | **Error Range:** **10.39 cm**

## 4. Discussion
- **Trial Consistency:** Across all 10 trials, the Kalman filter stayed within a `0.80` cm spread (MAE `0.33` cm) and stopped reliably near `0.25` m.
- **Sources of Error:** Mecanum roller slip (causing pure odometry to under-count travel by up to `11.8` cm), camera H.264 stream latency during braking, and small pixel-quantization noise on the 7.8 cm tag.
- **Role of Prediction & Correction:** Wheel odometry provides smooth 50 Hz state propagation between camera frames, while periodic visual tag measurements eliminate accumulated wheel-slip drift.
