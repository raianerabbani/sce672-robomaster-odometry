# RoboMaster EP Odometry (SCE-672 Phase 3)

Single-file 50 Hz odometry prediction + AprilTag visual correction and direct Mecanum wheel RPM control (`drive_wheels`) for the DJI RoboMaster EP Core.

## Usage
```bash
pip install -r requirements.txt
python3 robomaster_odometry.py --trials 10
python3 robomaster_odometry.py --sim --trials 1
```

## Structure
- `robomaster_odometry.py`: complete pipeline (odometry & 720p AprilTag sensors, 2-state Kalman filter, PD + cosine-ramp motion controller, Mecanum inverse kinematics, `drive_wheels` actuation, and CSV export).
- `results/`: 10-trial hardware dataset, error scatterplot, and demo video.
