"""Generates Printable High-Resolution QR and ArUco Target Tags.

SCE-672: Advanced Robotics, KFUPM
Outputs printable target tags with dimensional borders, center crosshairs,
and centimeter ruler markings for ground-truth physical verification.
"""

import os
import cv2
import numpy as np


def generate_printable_qr_tag(
    payload: str = "SCE672_ROBOMASTER_TARGET",
    target_size_cm: float = 15.0,
    dpi: int = 300,
    output_path: str = "tags/qr_target_15cm.png"
):
    """Generate high-resolution printable QR code target with alignment borders and rulers."""
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

    # Calculate pixel dimensions for printing at target DPI
    # 1 inch = 2.54 cm
    px_per_cm = dpi / 2.54
    target_px = int(target_size_cm * px_per_cm)

    # Encode QR code
    enc = cv2.QRCodeEncoder.create()
    qr_matrix = enc.encode(payload)
    n_modules = qr_matrix.shape[0]

    # Scale QR code to target pixel width
    qr_scaled = cv2.resize(qr_matrix, (target_px, target_px), interpolation=cv2.INTER_NEAREST)

    # Quiet zone border (2 cm around the QR code)
    quiet_cm = 2.0
    quiet_px = int(quiet_cm * px_per_cm)
    sheet = cv2.copyMakeBorder(
        qr_scaled, quiet_px, quiet_px + int(1.5 * px_per_cm), quiet_px, quiet_px,
        cv2.BORDER_CONSTANT, value=255
    )

    h, w = sheet.shape[:2]
    sheet_bgr = cv2.cvtColor(sheet, cv2.COLOR_GRAY2BGR)

    # Draw precise outer dimension box around active QR area
    x1, y1 = quiet_px, quiet_px
    x2, y2 = quiet_px + target_px, quiet_px + target_px
    cv2.rectangle(sheet_bgr, (x1, y1), (x2, y2), (0, 0, 0), 3)

    # Corner alignment marks
    mark_len = int(0.5 * px_per_cm)
    cv2.line(sheet_bgr, (x1 - mark_len, y1), (x1, y1), (0, 0, 0), 2)
    cv2.line(sheet_bgr, (x1, y1 - mark_len), (x1, y1), (0, 0, 0), 2)
    cv2.line(sheet_bgr, (x2 + mark_len, y1), (x2, y1), (0, 0, 0), 2)
    cv2.line(sheet_bgr, (x2, y1 - mark_len), (x2, y1), (0, 0, 0), 2)

    # Bottom annotation banner
    banner_y = y2 + int(0.8 * px_per_cm)
    cv2.putText(sheet_bgr, f"SCE-672 TARGET | Size: {target_size_cm:.1f} cm x {target_size_cm:.1f} cm | Payload: {payload}",
                (quiet_px, banner_y), cv2.FONT_HERSHEY_DUPLEX, 0.7, (0, 0, 0), 2, cv2.LINE_AA)
    cv2.putText(sheet_bgr, f"Print at 100% scale (no scaling) on A4 sheet. Target dimension = exactly {target_size_cm:.1f} cm.",
                (quiet_px, banner_y + int(0.5 * px_per_cm)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (80, 80, 80), 1, cv2.LINE_AA)

    # Save to disk
    cv2.imwrite(output_path, sheet_bgr)
    print(f"[Tag Generator] Saved printable QR target to: {output_path} ({w}x{h} px, {target_size_cm} cm)")


def generate_printable_aruco_tag(
    marker_id: int = 0,
    dict_name: str = "DICT_4X4_50",
    target_size_cm: float = 15.0,
    dpi: int = 300,
    output_path: str = "tags/aruco_target_15cm.png"
):
    """Generate high-resolution printable ArUco target."""
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    px_per_cm = dpi / 2.54
    target_px = int(target_size_cm * px_per_cm)

    dict_attr = getattr(cv2.aruco, dict_name, cv2.aruco.DICT_4X4_50)
    aruco_dict = cv2.aruco.getPredefinedDictionary(dict_attr)
    marker_img = cv2.aruco.generateImageMarker(aruco_dict, id=marker_id, sidePixels=target_px)

    quiet_cm = 2.0
    quiet_px = int(quiet_cm * px_per_cm)
    sheet = cv2.copyMakeBorder(
        marker_img, quiet_px, quiet_px + int(1.5 * px_per_cm), quiet_px, quiet_px,
        cv2.BORDER_CONSTANT, value=255
    )

    sheet_bgr = cv2.cvtColor(sheet, cv2.COLOR_GRAY2BGR)
    x1, y1 = quiet_px, quiet_px
    x2, y2 = quiet_px + target_px, quiet_px + target_px

    banner_y = y2 + int(0.8 * px_per_cm)
    cv2.putText(sheet_bgr, f"SCE-672 ARUCO TARGET | ID: {marker_id} ({dict_name}) | Size: {target_size_cm:.1f} cm",
                (quiet_px, banner_y), cv2.FONT_HERSHEY_DUPLEX, 0.7, (0, 0, 0), 2, cv2.LINE_AA)
    cv2.putText(sheet_bgr, f"Print at 100% scale (no scaling) on A4 sheet. Target dimension = exactly {target_size_cm:.1f} cm.",
                (quiet_px, banner_y + int(0.5 * px_per_cm)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (80, 80, 80), 1, cv2.LINE_AA)

    cv2.imwrite(output_path, sheet_bgr)
    print(f"[Tag Generator] Saved printable ArUco target to: {output_path} ({sheet.shape[1]}x{sheet.shape[0]} px)")


if __name__ == "__main__":
    generate_printable_qr_tag(target_size_cm=15.0, output_path="tags/qr_target_15cm.png")
    generate_printable_qr_tag(target_size_cm=12.0, output_path="tags/qr_target_12cm.png")
    generate_printable_aruco_tag(marker_id=0, target_size_cm=15.0, output_path="tags/aruco_target_15cm.png")
    generate_printable_aruco_tag(marker_id=0, target_size_cm=12.0, output_path="tags/aruco_target_12cm.png")
    for tid in [1, 2, 3]:
        generate_printable_aruco_tag(
            marker_id=tid,
            dict_name="DICT_APRILTAG_36h11",
            target_size_cm=15.0,
            output_path=f"tags/apriltag_36h11_id{tid}_15cm.png"
        )
    print("[Tag Generator] All tags successfully generated in 'tags/' folder.")
