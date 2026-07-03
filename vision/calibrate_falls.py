"""Script temporal de calibración: loggea descriptores frame a frame de
FallImmobilityDetector sobre videos/NN.mp4 (clips de caída) para ajustar
los umbrales con datos reales. No es parte del pipeline productivo.

Uso:
    python calibrate_falls.py
Genera fall_calibration.csv en el directorio actual.
"""
from __future__ import annotations

import csv
import glob

import cv2

from src.pose_pipeline import build_pose_sample, detect_person
from ultralytics import YOLO


VELOCITY_WINDOW_S = 0.4  # ventana para medir velocidad vertical (no frame-a-frame, no todo el historial)


def main() -> None:
    model = YOLO("yolo11n-pose.pt")
    rows = []

    paths = sorted(glob.glob("../videos/fall*/*.mp4"))

    for path in paths:
        cap = cv2.VideoCapture(path)
        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        history: list = []  # [(t, hip_y)]
        frame_i = 0
        while True:
            ok, image = cap.read()
            if not ok:
                break
            t = frame_i / fps
            h, w = image.shape[:2]
            detection = detect_person(model, image)
            if detection is not None:
                kpts, box = detection
                sample = build_pose_sample(kpts, box, w, h, t)

                # velocidad vertical vs. la muestra mas cercana a (t - VELOCITY_WINDOW_S)
                vertical_velocity = ""
                if history:
                    target_t = t - VELOCITY_WINDOW_S
                    ref = min(history, key=lambda h_: abs(h_[0] - target_t))
                    dt = max(t - ref[0], 1e-3)
                    if dt >= VELOCITY_WINDOW_S / 2:  # evita ventanas demasiado cortas al arrancar
                        vertical_velocity = round((sample.hip_y - ref[1]) / dt, 3)

                rows.append({
                    "video": path,
                    "frame": frame_i,
                    "t": round(t, 2),
                    "hip_y": round(sample.hip_y, 4),
                    "aspect_ratio": round(sample.aspect_ratio, 3),
                    "torso_angle_deg": round(sample.torso_angle_deg, 1),
                    "vertical_velocity": vertical_velocity,
                })
                history.append((t, sample.hip_y))
                history[:] = [x for x in history if t - x[0] <= 5.0]
            frame_i += 1
        cap.release()
        print(f"{path}: {frame_i} frames procesados")

    with open("fall_calibration.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Guardado fall_calibration.csv ({len(rows)} filas)")


if __name__ == "__main__":
    main()
