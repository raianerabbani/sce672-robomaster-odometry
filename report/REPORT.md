# Project Phase 3: RoboMaster Odometry
**Course:** Advanced Robotics (SCE-672), KFUPM  
**Student:** Raian E Rabbani (202617880)  
**Instructor:** Prof. Aamer Iqbal Bhatti

## 1. Approach and Sensor Measurements
The experiment was implemented under Ubuntu with ROS 2 on the DJI RoboMaster EP Core. Instead of using the chassis built-in `(v, w)` velocity command interface, each Mecanum wheel speed was controlled directly in RPM via inverse kinematics (`drive_wheels`).

- **Tag Detection:** A fixed 7.8 cm AprilTag (`36h11`) is detected from the onboard 720p camera. Using calibrated pinhole geometry ($f_x = 688.6$ px) and the measured camera-to-bumper offset ($d_\text{off} = 0.1876$ m), the visual distance to the tag is $z_k = (f_x W_\text{tag} / w_\text{px}) - d_\text{off}$.
- **Straight-Line Motion:** Visual bearing error drives a PD yaw controller to keep the robot centered on the tag, while forward speed ramps from 0.28 m/s down to 0.105 m/s before stopping at 0.25 m.
- **Prediction and Correction Sensors:**
  - *Prediction (50 Hz):* Wheel encoder odometry provides chassis displacement to propagate the distance estimate between camera updates.
  - *Correction:* Camera-based tag distance measurements update a two-state Kalman filter $[d_k, v_k]^T$ whenever a new tag observation is available.

## 2. Experimental Setup
- **Reference Point:** Front bumper of the RoboMaster EP Core for both estimated and measured distances.
- **Distances:** Starting distance fixed at 1.50 m; stopping distance set to 0.25 m.
- **Trials:** 10 repeat trials under the same starting conditions.

## 3. Results and Error Calculations
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

- **Kalman Filter MAE:** 0.56 cm (0.0056 m)
- **Kalman Filter Error Range:** 0.88 cm (`[-0.0000, +0.0088]` m)
- **Pure Odometry MAE:** 5.79 cm (0.0579 m)

## 4. Discussion
Across the 10 trials, the Kalman filter stayed consistent with an MAE of 0.56 cm and a total error range of 0.88 cm, while pure wheel odometry drifted to an MAE of 5.79 cm.

The main sources of error were Mecanum roller slip during acceleration and braking, slight floor surface variations, stopping inertia, and small pixel-quantization noise on the 7.8 cm tag. Wheel odometry prediction keeps the 50 Hz estimate smooth between frames, while camera correction removes accumulated wheel-slip drift before the robot reaches the target distance.
