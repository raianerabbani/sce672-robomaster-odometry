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
- **Distances:** Starting distance $D_0 \approx 1.48$ m; target stopping distance $d_\text{stop} = 0.25$ m; **10 repeat trials**.

## 3. Results & Error Calculations
| Trial | Actual $d_m$ (m) | Estimated $\hat{d}$ (m) | Error $\hat{d}-d_m$ (m) | $|\text{Error}|$ (m) | Pure Odo (m) | Odo Error (m) |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| 01 | 0.2596 | 0.2656 | +0.0060 | 0.0060 | 0.2768 | +0.0172 |
| 02 | 0.2543 | 0.2623 | +0.0080 | 0.0080 | 0.2848 | +0.0305 |
| 03 | 0.2551 | 0.2639 | +0.0088 | 0.0088 | 0.2979 | +0.0428 |
| 04 | 0.2567 | 0.2653 | +0.0086 | 0.0086 | 0.3776 | +0.1209 |
| 05 | 0.2520 | 0.2604 | +0.0084 | 0.0084 | 0.2700 | +0.0180 |
| 06 | 0.2503 | 0.2583 | +0.0080 | 0.0080 | 0.3036 | +0.0533 |
| 07 | 0.2519 | 0.2598 | +0.0079 | 0.0079 | 0.2943 | +0.0424 |
| 08 | 0.2639 | 0.2639 | +0.0000 | 0.0000 | 0.3409 | +0.0770 |
| 09 | 0.2607 | 0.2607 | -0.0000 | 0.0000 | 0.3980 | +0.1373 |
| 10 | 0.2689 | 0.2689 | +0.0000 | 0.0000 | 0.3087 | +0.0398 |

- **Kalman Filter MAE:** **0.56 cm** (`0.0056` m) | **Error Range:** **0.88 cm** (`[-0.0000, +0.0088]` m)
- **Pure Odometry MAE:** **5.79 cm** (`0.0579` m) | **Error Range:** **12.01 cm**

## 4. Discussion
- **Trial Consistency:** Across all 10 trials, the Kalman filter stayed within a `0.88` cm spread (MAE `0.56` cm) and stopped reliably near `0.25` m.
- **Sources of Error:** Mecanum roller slip (causing pure odometry to under-count travel by up to `13.7` cm), camera H.264 stream latency during braking, and small pixel-quantization noise on the 7.8 cm tag.
- **Role of Prediction & Correction:** Wheel odometry provides smooth 50 Hz state propagation between camera frames, while periodic visual tag measurements eliminate accumulated wheel-slip drift.
