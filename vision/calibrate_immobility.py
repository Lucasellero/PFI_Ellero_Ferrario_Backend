"""Script temporal de calibración: loggea desplazamiento de keypoints frame a
frame sobre segmentos reales de Toyota Smarthome (dataset "untrimmed", con
anotación por evento) para calibrar IMMOBILITY_MOVEMENT_THRESHOLD e
IMMOBILITY_WINDOW_S con datos reales. No es parte del pipeline productivo.

Los segmentos "quieto" son huecos Sit_down -> Get_up sin ningún evento
etiquetado en el medio (ver ../videos/README.md sobre el dataset untrimmed):
la persona se sienta, no hace nada catalogado por un rato, se levanta — eso
es sedentarismo real, no un artefacto de anotación. Los segmentos "moviendo"
son tramos `Walk` de >=2.5s de los mismos clips, como contraste.

Requiere tener descargado el dataset untrimmed + anotaciones de Toyota
Smarthome (ver conversación del proyecto) en las rutas configuradas abajo.

Uso:
    python calibrate_immobility.py
Genera immobility_calibration.csv en el directorio actual.
"""
from __future__ import annotations

import csv

import cv2
from ultralytics import YOLO

from src.pose_pipeline import build_pose_sample, detect_person
from src.posture import classify_posture

VIDEOS_DIR = r"D:\DESCARGAS\Videos_mp4"
SAMPLE_EVERY_N_FRAMES = 12  # ~2fps a 25fps nativo: alcanza para desplazamiento en ventanas de minutos, corre ~12x más rápido

# (video_id, start_frame, end_frame, label) — frames en la numeración original del clip (25fps)
SEGMENTS = [
    # "quieto": hueco Sit_down -> Get_up sin eventos etiquetados en el medio
    ("P10T14C07", 12744, 19768, "quieto"),
    ("P15T06C05", 8192, 14743, "quieto"),
    ("P12T13C07", 12939, 19232, "quieto"),
    ("P20T07C04", 18770, 23989, "quieto"),
    ("P07T01C05", 6749, 10556, "quieto"),
    ("P11T12C03", 2213, 5655, "quieto"),
    ("P10T07C05", 6530, 9767, "quieto"),
    ("P07T02C05", 10184, 13063, "quieto"),
    ("P15T10C01", 8260, 10716, "quieto"),
    ("P10T01C05", 20476, 22072, "quieto"),
    # "moviendo": tramos Walk >=2.5s de los mismos clips, como contraste
    ("P10T14C07", 706, 841, "moviendo"),
    ("P15T06C05", 2997, 3089, "moviendo"),
    ("P12T13C07", 4174, 4269, "moviendo"),
    ("P20T07C04", 711, 779, "moviendo"),
    ("P07T01C05", 6010, 6080, "moviendo"),
    ("P11T12C03", 22460, 22580, "moviendo"),
    ("P07T02C05", 10062, 10148, "moviendo"),
    ("P15T10C01", 1924, 2152, "moviendo"),
    ("P10T01C05", 11388, 12757, "moviendo"),
]


def main() -> None:
    model = YOLO("yolo11n-pose.pt")
    rows = []

    for video_id, start_frame, end_frame, label in SEGMENTS:
        path = f"{VIDEOS_DIR}/{video_id}.mp4"
        cap = cv2.VideoCapture(path)
        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

        prev_sample = None
        frame_i = start_frame
        n_samples = 0
        while frame_i < end_frame:
            ok, image = cap.read()
            if not ok:
                break
            if (frame_i - start_frame) % SAMPLE_EVERY_N_FRAMES == 0:
                h, w = image.shape[:2]
                detection = detect_person(model, image)
                if detection is not None:
                    kpts, box, _ = detection
                    sample = build_pose_sample(kpts, box, w, h, frame_i / fps)
                    displacement = ""
                    if prev_sample is not None:
                        import numpy as np
                        dt = sample.timestamp_s - prev_sample.timestamp_s
                        displacement = round(
                            float(np.nanmean(np.abs(sample.keypoints - prev_sample.keypoints))) / dt, 5
                        )
                    rows.append({
                        "video": video_id,
                        "label": label,
                        "frame": frame_i,
                        "t_in_segment": round((frame_i - start_frame) / fps, 2),
                        "posture": classify_posture(sample),
                        "aspect_ratio": round(sample.aspect_ratio, 3),
                        "displacement_per_s": displacement,
                    })
                    prev_sample = sample
                    n_samples += 1
                else:
                    prev_sample = None
            frame_i += 1
        cap.release()
        print(f"{video_id} [{label}] frames {start_frame}-{end_frame}: {n_samples} muestras")

    with open("immobility_calibration.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Guardado immobility_calibration.csv ({len(rows)} filas)")


if __name__ == "__main__":
    main()
