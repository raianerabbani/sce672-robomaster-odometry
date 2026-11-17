import argparse, csv, os, threading, time
import cv2, numpy as np
from robomaster import robot, camera

START_D, STOP_D, TAG_SIZE, CAM_OFF = 1.50, 0.25, 0.078, 0.1876
FX_1280, TARGET_IDS = 688.63, {1, 2, 3}
Q_POS, Q_VEL, R_VIS = 0.0009, 0.0025, 0.0016
V_CRUISE, V_MIN, DECEL_D, MAX_ACCEL = 0.28, 0.105, 0.55, 0.60
KP_YAW, KD_YAW, MAX_WZ, DEADBAND = 0.65, 0.04, 20.0, 1.2

def mecanum_ik(vx, vy, wz_deg):
    wz = np.radians(wz_deg) * 0.21
    rpms = [v * (1000.0 / 3.50) for v in (vx - vy + wz, vx + vy - wz, vx - vy - wz, vx + vy + wz)]
    peak = max(abs(r) for r in rpms)
    if peak > 1000.0:
        rpms = [r * 1000.0 / peak for r in rpms]
    return tuple(int(round(r)) for r in rpms)

class TagSensor:
    def __init__(self):
        d = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_APRILTAG_36h11)
        p = cv2.aruco.DetectorParameters()
        p.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_APRILTAG
        self.det = cv2.aruco.ArucoDetector(d, p)
    def detect(self, frame):
        if frame is None or frame.size == 0:
            return False, 0.0, 0.0
        h, w = frame.shape[:2]
        fx, cx = FX_1280 * (w / 1280.0), w / 2.0
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if frame.ndim == 3 else frame
        corners, ids, _ = self.det.detectMarkers(gray)
        if ids is None:
            return False, 0.0, 0.0
        for c, tid in zip(corners, ids.flatten()):
            if int(tid) not in TARGET_IDS:
                continue
            pts = c.reshape(4, 2)
            w_px = 0.5 * (np.linalg.norm(pts[0] - pts[1]) + np.linalg.norm(pts[3] - pts[2]))
            h_px = 0.5 * (np.linalg.norm(pts[0] - pts[3]) + np.linalg.norm(pts[1] - pts[2]))
            if w_px < 12.0 or h_px < 12.0 or max(w_px / h_px, h_px / w_px) > 2.2:
                continue
            return True, float((fx * TAG_SIZE) / w_px), float(np.degrees(np.arctan2(float(np.mean(pts[:, 0])) - cx, fx)))
        return False, 0.0, 0.0

class KalmanFilter1D:
    def __init__(self, d0):
        self.x, self.P, self.H = np.array([[float(d0)], [0.0]]), np.diag([0.0025, 0.0100]), np.array([[1.0, 0.0]])
        self.pure_odo, self.last_odo = float(d0), None
    def predict(self, odo_pos, dt):
        dt = max(dt, 0.001)
        dx = 0.0 if self.last_odo is None else float(odo_pos - self.last_odo)
        self.last_odo, self.pure_odo = float(odo_pos), max(0.0, self.pure_odo - dx)
        F = np.array([[1.0, -dt], [0.0, 1.0]])
        Q = np.diag([Q_POS, Q_VEL]) * dt * (1.0 + 3.0 * abs(dx))
        self.x = np.array([[self.x[0, 0] - dx], [dx / dt]])
        P_new = F @ self.P @ F.T + Q
        self.P = 0.5 * (P_new + P_new.T)
    def correct(self, z_cam):
        z = float(z_cam) - CAM_OFF
        R = R_VIS * max(1.0, z ** 2)
        y = z - (self.H @ self.x)[0, 0]
        S = (self.H @ self.P @ self.H.T)[0, 0] + R
        if (y ** 2) / S > 9.0 and abs(y) > 0.25:
            return
        K = (self.P @ self.H.T) / S
        self.x = self.x + K * y
        IKH = np.eye(2) - K @ self.H
        P_new = IKH @ self.P @ IKH.T + K * R @ K.T
        self.P = 0.5 * (P_new + P_new.T)

class MotionController:
    def __init__(self):
        self.vx, self.wz, self.b_filt, self.last_b, self.outliers, self.reached = 0.0, 0.0, None, None, 0, False
    def step(self, dist, bearing, dt):
        if self.reached or dist <= STOP_D + 0.025 + 0.25 * max(self.vx, 0.0):
            self.reached, self.vx, self.wz = True, 0.0, 0.0
            return (0, 0, 0, 0), True
        d_rem = dist - STOP_D
        vx_t = V_CRUISE if d_rem >= DECEL_D else V_MIN + (V_CRUISE - V_MIN) * 0.5 * (1.0 - np.cos(np.pi * np.clip(d_rem / DECEL_D, 0.0, 1.0)))
        self.vx += float(np.clip(vx_t - self.vx, -MAX_ACCEL * dt, MAX_ACCEL * dt))
        if bearing is not None:
            if self.b_filt is None:
                self.b_filt, self.outliers = bearing, 0
            elif abs(bearing - self.b_filt) > 20.0 and self.outliers < 2:
                self.outliers += 1
            else:
                self.b_filt, self.outliers = 0.25 * bearing + 0.75 * self.b_filt, 0
            rate = 0.0 if self.last_b is None else float(np.clip((self.b_filt - self.last_b) / dt, -60.0, 60.0))
            self.last_b = self.b_filt
            wz_t = 0.0 if abs(self.b_filt) < DEADBAND else float(np.clip(KP_YAW * (self.b_filt - np.sign(self.b_filt) * DEADBAND) + KD_YAW * rate, -MAX_WZ, MAX_WZ))
        else:
            self.last_b, self.outliers, wz_t = None, 0, 0.0
        self.wz += float(np.clip(wz_t - self.wz, -90.0 * dt, 90.0 * dt))
        return mecanum_ik(self.vx, 0.0, self.wz), False


class RobotHW:
    def __init__(self, sim=False):
        self.sim, self.ep, self.x, self.y, self.yaw, self.x0, self.y0, self.yaw0 = sim, None, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
        self.sim_d, self.sim_vx, self.t_last = START_D, 0.0, time.time()
        if not sim:
            self.ep = robot.Robot()
            self.ep.initialize(conn_type="ap")
            self.ep.chassis.sub_position(cs=0, freq=50, callback=self._on_pos)
            self.ep.camera.start_video_stream(display=False, resolution=camera.STREAM_720P)
            time.sleep(0.5)
        self.reset_odo()
    def _on_pos(self, p):
        self.x, self.y, self.yaw = float(p[0]), float(p[1]), float(p[2]) if len(p) > 2 else 0.0
    def reset_odo(self):
        self.x0, self.y0, self.yaw0 = self.x, self.y, self.yaw
    def odo_forward(self):
        if self.sim:
            now = time.time()
            self.sim_d -= self.sim_vx * (now - self.t_last)
            self.t_last = now
            return START_D - self.sim_d
        y0 = np.radians(self.yaw0)
        return float((self.x - self.x0) * np.cos(y0) + (self.y - self.y0) * np.sin(y0))
    def get_frame(self):
        if self.sim:
            img = np.full((720, 1280, 3), 255, dtype=np.uint8)
            w_px = int(np.clip(FX_1280 * TAG_SIZE / max(self.sim_d + CAM_OFF, 0.2), 20, 400))
            tag = cv2.resize(cv2.aruco.generateImageMarker(cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_APRILTAG_36h11), 1, w_px), (w_px, w_px))
            x0, y0 = 640 - w_px // 2, 360 - w_px // 2
            img[y0:y0 + w_px, x0:x0 + w_px] = cv2.cvtColor(tag, cv2.COLOR_GRAY2BGR)
            return img
        return self.ep.camera.read_cv2_image(strategy="newest", timeout=0.5)
    def drive_wheels(self, w1, w2, w3, w4):
        if self.sim:
            self.sim_vx = ((w1 + w2 + w3 + w4) / 4.0) * (3.50 / 1000.0)
        else:
            self.ep.chassis.drive_wheels(w1=int(w1), w2=int(w2), w3=int(w3), w4=int(w4))
    def close(self):
        self.drive_wheels(0, 0, 0, 0)
        if not self.sim and self.ep:
            self.ep.camera.stop_video_stream()
            self.ep.close()

def run_trial(hw, sensor, trial_id):
    vis_state, lock, running = [False, 0.0, 0.0, 0], threading.Lock(), [True]
    def cam_loop():
        while running[0]:
            ok, d, b = sensor.detect(hw.get_frame())
            with lock:
                vis_state[0], vis_state[1], vis_state[2], vis_state[3] = ok, d, b, vis_state[3] + 1
            time.sleep(0.01)
    t = threading.Thread(target=cam_loop, daemon=True)
    t.start()
    time.sleep(0.4)
    with lock:
        d0 = max(0.2, vis_state[1] - CAM_OFF) if vis_state[0] else START_D
    hw.reset_odo()
    kf, ctrl, last_t, last_seq, settle_at = KalmanFilter1D(d0), MotionController(), time.time(), -1, None
    while True:
        t0 = time.time()
        dt = float(np.clip(t0 - last_t, 0.005, 0.08))
        last_t = t0
        with lock:
            ok, vd, vb, seq = vis_state
        kf.predict(hw.odo_forward(), dt)
        if ok and seq != last_seq:
            kf.correct(vd)
            last_seq = seq
        if settle_at is None:
            wheels, reached = ctrl.step(kf.x[0, 0], vb if ok else None, dt)
            hw.drive_wheels(*wheels)
            if reached:
                settle_at = t0 + 0.40
        elif t0 >= settle_at:
            break
        time.sleep(max(0.0, 0.020 - (time.time() - t0)))
    running[0] = False
    t.join(timeout=0.3)
    ok, vd, _ = sensor.detect(hw.get_frame())
    actual = max(0.0, vd - CAM_OFF) if ok else float(kf.x[0, 0])
    kf_est, odo_est = float(kf.x[0, 0]), kf.pure_odo
    print(f"Trial {trial_id}: Start={d0:.3f}m | KF={kf_est:.4f}m | Actual={actual:.4f}m | Err={(kf_est-actual)*100:+.2f}cm | Odo={odo_est:.4f}m")
    return [trial_id, round(d0, 3), STOP_D, round(kf_est, 4), round(actual, 4), round(kf_est - actual, 4), round(odo_est, 4), round(odo_est - actual, 4)]

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--trials", type=int, default=1)
    ap.add_argument("--sim", action="store_true")
    args = ap.parse_args()
    hw, sensor, rows = RobotHW(sim=args.sim), TagSensor(), []
    try:
        for i in range(1, args.trials + 1):
            rows.append(run_trial(hw, sensor, i))
            if i < args.trials:
                input("Reposition robot to 1.50 m and press Enter...")
    finally:
        hw.close()
    os.makedirs("results", exist_ok=True)
    with open("results/single_file_summary.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["Trial", "Start_m", "Stop_m", "KF_Est_m", "Actual_m", "KF_Err_m", "Pure_Odo_m", "Odo_Err_m"])
        w.writerows(rows)
