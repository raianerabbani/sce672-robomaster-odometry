"""Tag Detector module for RoboMaster EP Odometry Project (Phase 3).

Detects visual tags (standard QR codes, ArUco markers, or RoboMaster markers)
from camera frames, extracts corner points, and calculates 3D pose and distance
using Perspective-n-Point (PnP) and pinhole projection models.
"""

from dataclasses import dataclass
from typing import Optional, Tuple, List
import numpy as np
import cv2


@dataclass
class TagDetection:
    """Structured representation of a visual tag observation."""
    detected: bool = False
    tag_type: str = "none"
    tag_id: str = ""
    corners: Optional[np.ndarray] = None          # (4, 2) in image coordinates (px)
    center: Tuple[float, float] = (0.0, 0.0)      # (cx, cy) in px
    width_px: float = 0.0                         # Apparent width (px)
    height_px: float = 0.0                        # Apparent height (px)
    distance: float = 0.0                         # Forward distance Z in camera frame (meters)
    distance_euclidean: float = 0.0               # Euclidean distance sqrt(X^2 + Y^2 + Z^2) (meters)
    bearing_deg: float = 0.0                      # Horizontal bearing angle (deg), positive = right
    elevation_deg: float = 0.0                    # Vertical angle (deg), positive = down
    rvec: Optional[np.ndarray] = None             # Rotation vector from solvePnP
    tvec: Optional[np.ndarray] = None             # Translation vector [X, Y, Z]^T (meters)


class TagDetector:
    """Robust multi-modal tag detector supporting standard QR codes and ArUco markers."""

    def __init__(
        self,
        tag_type: str = "apriltag",
        tag_size: float = 0.15,
        camera_matrix: Optional[np.ndarray] = None,
        dist_coeffs: Optional[np.ndarray] = None,
        aruco_dict_name: str = "DICT_4X4_50",
        aruco_id: int = 0,
        apriltag_family: str = "DICT_APRILTAG_36h11",
        target_ids: Optional[List[int]] = None,
        expected_qr_payload: Optional[str] = None,
    ):
        """Initialize tag detector.

        Args:
            tag_type: "apriltag", "qr", "aruco", or "auto".
            tag_size: Physical width and height of tag outer black square in meters.
            camera_matrix: 3x3 intrinsic matrix.
            dist_coeffs: Lens distortion coefficients.
            aruco_dict_name: ArUco dictionary identifier.
            aruco_id: ArUco marker ID to match (None to match any).
            apriltag_family: AprilTag dictionary name (e.g. "DICT_APRILTAG_36h11") or "auto".
            target_ids: List of allowed tag IDs (e.g. [1, 2, 3]) or None to match any.
            expected_qr_payload: Specific string to match in QR payload (None to accept any).
        """
        self.tag_type = tag_type.lower()
        self.tag_size = float(tag_size)
        self.aruco_id = aruco_id
        self.target_ids = list(target_ids) if target_ids is not None else None
        self.expected_qr_payload = expected_qr_payload
        # Ratio of detected finder-pattern quad to outer physical QR dimensions (OpenCV QRCodeDetector standard)
        self.qr_scale_factor = 1.0 / 1.20833

        # Default camera matrix for RoboMaster EP 720p stream if not specified
        if camera_matrix is None:
            self.camera_matrix = np.array([
                [580.0, 0.0, 640.0],
                [0.0, 580.0, 360.0],
                [0.0, 0.0, 1.0]
            ], dtype=np.float64)
        else:
            self.camera_matrix = np.array(camera_matrix, dtype=np.float64)

        self._base_camera_matrix = self.camera_matrix.copy()
        self._base_width = max(2.0 * float(self._base_camera_matrix[0, 2]), 1.0)
        self._base_height = max(2.0 * float(self._base_camera_matrix[1, 2]), 1.0)

        if dist_coeffs is None:
            self.dist_coeffs = np.zeros((5, 1), dtype=np.float64)
        else:
            self.dist_coeffs = np.array(dist_coeffs, dtype=np.float64)

        # 3D object points in tag coordinate frame (centered at origin, z=0)
        # Order: Top-Left, Top-Right, Bottom-Right, Bottom-Left
        half = self.tag_size / 2.0
        self.object_points = np.array([
            [-half, -half, 0.0],
            [ half, -half, 0.0],
            [ half,  half, 0.0],
            [-half,  half, 0.0]
        ], dtype=np.float64)

        # Initialize QR detector
        self.qr_detector = cv2.QRCodeDetector()

        # Initialize ArUco detector
        try:
            dict_attr = getattr(cv2.aruco, aruco_dict_name, cv2.aruco.DICT_4X4_50)
            self.aruco_dict = cv2.aruco.getPredefinedDictionary(dict_attr)
            self.aruco_params = cv2.aruco.DetectorParameters()
            self.aruco_detector = cv2.aruco.ArucoDetector(self.aruco_dict, self.aruco_params)
        except Exception:
            self.aruco_detector = None

        # Initialize AprilTag detector(s) with sub-pixel corner refinement
        # Note: 16h5 / 25h9 have tiny bit grids that cause false positives on room geometry;
        # restrict "auto" to 36h11.
        self.apriltag_detectors = []
        families = (
            ["DICT_APRILTAG_36h11"]
            if apriltag_family.lower() == "auto"
            else [apriltag_family]
        )
        for fam in families:
            try:
                dict_attr = getattr(cv2.aruco, fam)
                at_dict = cv2.aruco.getPredefinedDictionary(dict_attr)
                at_params = cv2.aruco.DetectorParameters()
                at_params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_APRILTAG
                self.apriltag_detectors.append((fam, cv2.aruco.ArucoDetector(at_dict, at_params)))
            except Exception:
                pass

    def detect(self, frame: np.ndarray) -> TagDetection:
        """Detect tag in frame and estimate 3D pose and distance.

        Args:
            frame: BGR or Grayscale image from camera.

        Returns:
            TagDetection populated with metrics or detected=False.
        """
        if frame is None or frame.size == 0:
            return TagDetection()

        h, w = frame.shape[:2]
        if w > 0 and h > 0:
            scale_x = float(w) / self._base_width
            scale_y = float(h) / self._base_height
            self.camera_matrix[0, 0] = self._base_camera_matrix[0, 0] * scale_x
            self.camera_matrix[1, 1] = self._base_camera_matrix[1, 1] * scale_y
            self.camera_matrix[0, 2] = float(w) / 2.0
            self.camera_matrix[1, 2] = float(h) / 2.0

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if len(frame.shape) == 3 else frame

        # Try AprilTag detection
        if self.tag_type in ("apriltag", "auto") and self.apriltag_detectors:
            det = self._detect_apriltag(gray)
            if det.detected:
                return det

        # Try QR detection
        if self.tag_type in ("qr", "auto"):
            det = self._detect_qr(gray)
            if det.detected:
                return det

        # Try ArUco detection
        if self.tag_type in ("aruco", "auto") and self.aruco_detector is not None:
            det = self._detect_aruco(gray)
            if det.detected:
                return det

        return TagDetection()

    def _detect_qr(self, gray: np.ndarray) -> TagDetection:
        """Detect standard QR code."""
        retval, decoded_info, points, _ = self.qr_detector.detectAndDecodeMulti(gray)
        if not retval or points is None or len(points) == 0:
            # Fall back to single detection
            val, pts, _ = self.qr_detector.detectAndDecode(gray)
            if pts is None or len(pts) == 0:
                # Try simple detect without decode
                ret_d, pts_d = self.qr_detector.detect(gray)
                if not ret_d or pts_d is None:
                    return TagDetection()
                pts = pts_d
                val = ""
            decoded_info = [val]
            points = pts if pts.ndim == 3 else np.expand_dims(pts, axis=0)

        for i, pts in enumerate(points):
            corners = pts.reshape(-1, 2).astype(np.float64)
            if len(corners) != 4:
                continue

            payload = decoded_info[i] if i < len(decoded_info) else ""
            if self.expected_qr_payload and payload and self.expected_qr_payload not in payload:
                continue

            return self._compute_pose_from_corners(corners, tag_type="QR", tag_id=payload)

        return TagDetection()

    def _detect_apriltag(self, gray: np.ndarray) -> TagDetection:
        """Detect AprilTag marker (e.g. DICT_APRILTAG_36h11 IDs 1, 2, 3)."""
        for fam_name, detector in self.apriltag_detectors:
            corners_list, ids, _ = detector.detectMarkers(gray)
            if ids is None or len(corners_list) == 0:
                continue

            for i, marker_id in enumerate(ids.flatten()):
                if self.target_ids is not None and int(marker_id) not in self.target_ids:
                    continue
                corners = corners_list[i].reshape(4, 2).astype(np.float64)
                return self._compute_pose_from_corners(corners, tag_type="AprilTag", tag_id=f"ID-{int(marker_id)}")

        return TagDetection()

    def _detect_aruco(self, gray: np.ndarray) -> TagDetection:
        """Detect ArUco marker."""
        corners_list, ids, _ = self.aruco_detector.detectMarkers(gray)
        if ids is None or len(corners_list) == 0:
            return TagDetection()

        for i, marker_id in enumerate(ids.flatten()):
            if self.aruco_id is not None and marker_id != self.aruco_id and self.tag_type != "auto":
                continue
            corners = corners_list[i].reshape(4, 2).astype(np.float64)
            return self._compute_pose_from_corners(corners, tag_type="ArUco", tag_id=f"ID-{marker_id}")

        # If specific ID was not matched in auto mode, take first detected marker
        corners = corners_list[0].reshape(4, 2).astype(np.float64)
        return self._compute_pose_from_corners(corners, tag_type="ArUco", tag_id=f"ID-{ids[0][0]}")

    def _compute_pose_from_corners(
        self,
        corners: np.ndarray,
        tag_type: str,
        tag_id: str
    ) -> TagDetection:
        """Calculate 3D position, distance, and bearing from 4 tag corners."""
        # Order corners consistently: Top-Left, Top-Right, Bottom-Right, Bottom-Left
        # Compute centroid
        center_x = float(np.mean(corners[:, 0]))
        center_y = float(np.mean(corners[:, 1]))

        # Sort corners by polar angle relative to centroid to ensure CCW/CW ordering
        angles = np.arctan2(corners[:, 1] - center_y, corners[:, 0] - center_x)
        # Top-left is around -135 deg (-3pi/4), Top-right -45 deg, Bottom-right +45 deg, Bottom-left +135 deg
        sort_idx = np.argsort(angles)
        sorted_corners = corners[sort_idx]
        # Re-align so top-left is index 0
        tl_idx = np.argmin(np.sum(sorted_corners, axis=1))  # min(x + y) is top-left
        ordered_corners = np.roll(sorted_corners, -tl_idx, axis=0)

        # Width and height in pixels
        w1 = np.linalg.norm(ordered_corners[0] - ordered_corners[1])
        w2 = np.linalg.norm(ordered_corners[3] - ordered_corners[2])
        h1 = np.linalg.norm(ordered_corners[0] - ordered_corners[3])
        h2 = np.linalg.norm(ordered_corners[1] - ordered_corners[2])
        w_px = float((w1 + w2) / 2.0)
        h_px = float((h1 + h2) / 2.0)

        # Reject tiny or heavily skewed false-positive quads
        if w_px < 12.0 or h_px < 12.0 or max(w_px / h_px, h_px / w_px) > 2.2:
            return TagDetection()

        fx = self.camera_matrix[0, 0]
        fy = self.camera_matrix[1, 1]
        cx = self.camera_matrix[0, 2]
        cy = self.camera_matrix[1, 2]

        # Effective square size for detected quad (QR vs ArUco)
        effective_size = self.tag_size * self.qr_scale_factor if tag_type.upper() == "QR" else self.tag_size
        half = effective_size / 2.0
        obj_pts_active = np.array([
            [-half, -half, 0.0],
            [ half, -half, 0.0],
            [ half,  half, 0.0],
            [-half,  half, 0.0]
        ], dtype=np.float64)

        # Pinhole geometry baseline distance
        tz_pinhole = (fx * effective_size) / max(w_px, 1.0)

        # Perspective-n-Point pose estimation using robust Levenberg-Marquardt iterative solver
        success, rvec, tvec = cv2.solvePnP(
            obj_pts_active,
            ordered_corners,
            self.camera_matrix,
            self.dist_coeffs,
            flags=cv2.SOLVEPNP_ITERATIVE
        )

        if success and tvec is not None and tvec[2, 0] > 0.15 and abs(tvec[2, 0] - tz_pinhole) < 0.35 * tz_pinhole:
            tx = float(tvec[0, 0])
            ty = float(tvec[1, 0])
            tz = float(tvec[2, 0])
            euclidean = float(np.sqrt(tx**2 + ty**2 + tz**2))
        else:
            # Fallback to direct pinhole model
            tz = tz_pinhole
            tx = (center_x - cx) * tz / fx
            ty = (center_y - cy) * tz / fy
            euclidean = float(np.sqrt(tx**2 + ty**2 + tz**2))
            rvec = np.zeros((3, 1), dtype=np.float64)
            tvec = np.array([[tx], [ty], [tz]], dtype=np.float64)

        # Bearing angle: angle between camera optical axis and ray to tag center
        bearing_rad = np.arctan2(center_x - cx, fx)
        bearing_deg = float(np.degrees(bearing_rad))

        elevation_rad = np.arctan2(center_y - cy, fy)
        elevation_deg = float(np.degrees(elevation_rad))

        return TagDetection(
            detected=True,
            tag_type=tag_type,
            tag_id=tag_id,
            corners=ordered_corners,
            center=(center_x, center_y),
            width_px=w_px,
            height_px=h_px,
            distance=tz,
            distance_euclidean=euclidean,
            bearing_deg=bearing_deg,
            elevation_deg=elevation_deg,
            rvec=rvec,
            tvec=tvec
        )

    def draw_detection(self, frame: np.ndarray, det: TagDetection) -> np.ndarray:
        """Annotate camera frame with bounding box, corners, centroid, and metrics."""
        vis = frame.copy()
        if not det.detected or det.corners is None:
            return vis

        pts = det.corners.astype(np.int32).reshape((-1, 1, 2))
        # Draw bounding polygon
        cv2.polylines(vis, [pts], isClosed=True, color=(0, 255, 0), thickness=2)

        # Draw corners with circle dots
        for i, pt in enumerate(det.corners):
            color = (0, 0, 255) if i == 0 else (255, 200, 0)
            cv2.circle(vis, (int(pt[0]), int(pt[1])), 4, color, -1)

        # Draw center crosshair
        cx, cy = int(det.center[0]), int(det.center[1])
        cv2.drawMarker(vis, (cx, cy), (0, 255, 255), cv2.MARKER_CROSS, 14, 2)

        # Draw coordinate axes if rvec/tvec available
        if det.rvec is not None and det.tvec is not None:
            try:
                cv2.drawFrameAxes(vis, self.camera_matrix, self.dist_coeffs, det.rvec, det.tvec, self.tag_size * 0.5, 2)
            except Exception:
                pass

        # Text banner
        label = f"{det.tag_type} {det.tag_id} | Dist: {det.distance:.3f}m | Yaw: {det.bearing_deg:+.1f}deg"
        cv2.putText(vis, label, (max(10, cx - 120), max(25, cy - 15)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(vis, label, (max(10, cx - 120), max(25, cy - 15)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 120), 1, cv2.LINE_AA)

        return vis
