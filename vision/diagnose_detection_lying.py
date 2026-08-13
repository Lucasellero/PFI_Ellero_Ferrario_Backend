"""Por qué la detección de persona cae a ~4-9% en los clips de `Lay_down`
largo usados para validar la rama `acostado` (ver vision/README.md,
"Inmovilidad prolongada", 2026-08-13) — mucho más bajo que en clips de
caída (99.6-100% detección, ver validación de esa rama). Distingue dos
causas posibles, MUY distintas en cómo se arreglan:

  (a) YOLO no encuentra ninguna persona en el frame (oclusión real, pose
      fuera de lo que el modelo reconoce, o simplemente cámara muy lejos).
  (b) YOLO SÍ encuentra a la persona pero con confianza de keypoints por
      debajo de MIN_KEYPOINT_CONFIDENCE (0.2) — en ese caso el problema es
      el umbral, no el modelo, y subirlo/bajarlo es una decisión de
      trade-off (menos rechazo = más ruido en los keypoints que sí se usan).

Corre el modelo SIN el filtro de confianza sobre una muestra de frames del
clip con peor detección (P25T05C04, ~4%) y reporta la distribución de
confianza promedio de keypoints frame a frame, separando "sin detección
alguna" de "detectado con confianza X".

Uso:
    vision/.venv/Scripts/python.exe diagnose_detection_lying.py
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import cv2
import numpy as np
from ultralytics import YOLO

VIDEOS_DIR = Path(r"D:\DESCARGAS\Videos_mp4")
VIDEO_ID = "P25T05C04"
START_FRAME = 14555 - 75
END_FRAME = 27649
SAMPLE_EVERY = 5  # cada 5 frames (~5fps de 25fps nativo) para no tardar demasiado


def main() -> None:
    model = YOLO("yolo11n-pose.pt")
    model.to("cuda")

    path = VIDEOS_DIR / f"{VIDEO_ID}.mp4"
    cap = cv2.VideoCapture(str(path))
    cap.set(cv2.CAP_PROP_POS_FRAMES, START_FRAME)

    no_box = 0
    low_conf = []  # confianzas promedio que caen debajo de 0.2
    ok_conf = []   # confianzas promedio que pasan 0.2
    n_checked = 0
    example_low_conf_frame = None
    example_low_conf_val = 1.0

    frame_i = START_FRAME
    while frame_i < END_FRAME:
        ok, image = cap.read()
        if not ok:
            break
        if (frame_i - START_FRAME) % SAMPLE_EVERY == 0:
            n_checked += 1
            result = model(image, verbose=False)[0]
            if result.keypoints is None or len(result.keypoints.xy) == 0:
                no_box += 1
            else:
                boxes = result.boxes.xyxy.cpu().numpy()
                areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
                idx = int(areas.argmax())
                conf = float(result.keypoints.conf[idx].cpu().numpy().mean()) if result.keypoints.conf is not None else 1.0
                if conf < 0.2:
                    low_conf.append(conf)
                    if conf < example_low_conf_val:
                        example_low_conf_val = conf
                        example_low_conf_frame = (frame_i, image.copy())
                else:
                    ok_conf.append(conf)
        frame_i += 1
    cap.release()

    print(f"{VIDEO_ID} [{START_FRAME}-{END_FRAME}], muestreado cada {SAMPLE_EVERY} frames: {n_checked} chequeados\n")
    print(f"Sin ninguna detección (YOLO no encuentra persona): {no_box} ({no_box/n_checked:.1%})")
    print(f"Detectado pero confianza < 0.2 (filtrado por umbral): {len(low_conf)} ({len(low_conf)/n_checked:.1%})")
    if low_conf:
        print(f"  confianza en ese grupo: media={np.mean(low_conf):.3f}, mediana={np.median(low_conf):.3f}, "
              f"p90={np.percentile(low_conf, 90):.3f}")
    print(f"Detectado con confianza >= 0.2 (pasa el filtro hoy): {len(ok_conf)} ({len(ok_conf)/n_checked:.1%})")
    if ok_conf:
        print(f"  confianza en ese grupo: media={np.mean(ok_conf):.3f}, mediana={np.median(ok_conf):.3f}")

    if example_low_conf_frame is not None:
        frame_i, img = example_low_conf_frame
        out_path = ROOT / "vision" / f"frame_lowconf_{VIDEO_ID}_{frame_i}.jpg"
        cv2.imwrite(str(out_path), img)
        print(f"\nFrame de ejemplo (confianza más baja, {example_low_conf_val:.3f}) guardado en {out_path}")


if __name__ == "__main__":
    main()
