"""Mismo diagnóstico que diagnose_immobility_displacement.py (full vs. core
keypoints, ver vision/README.md "Inmovilidad prolongada") pero sobre los 3
clips reales de `Lay_down` largo encontrados para la rama `acostado`
(validate_immobility_acostado.py) — la inmovilidad tampoco se disparó ahí,
esto busca saber si es la misma causa (movimiento de manos/cabeza sin
cambiar de postura) que en el caso sedentario.

Uso:
    vision/.venv/Scripts/python.exe diagnose_immobility_acostado.py
"""
from __future__ import annotations

import pickle
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import cv2
from ultralytics import YOLO

from diagnose_immobility_displacement import CORE_IDX, displacement_series, summarize
from src.pose_pipeline import IMMOBILITY_MOVEMENT_THRESHOLD, IMMOBILITY_WINDOW_S, build_pose_sample, detect_person
from validate_immobility_acostado import CASES, VIDEOS_DIR

CORE_THRESHOLD = 0.014  # umbral recalibrado para core (ver README) — no reusar 0.02, que era para "full"


def main() -> None:
    model = YOLO("yolo11n-pose.pt")
    model.to("cuda")

    for video_id, start_frame, end_frame in CASES:
        path = VIDEOS_DIR / f"{video_id}.mp4"
        cap = cv2.VideoCapture(str(path))
        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        cap.set(cv2.CAP_PROP_POS_FRAMES, max(start_frame, 0))

        samples = []
        frame_i = max(start_frame, 0)
        t0 = time.time()
        while frame_i < end_frame:
            ok, image = cap.read()
            if not ok:
                break
            h, w = image.shape[:2]
            detection = detect_person(model, image)
            if detection is not None:
                kpts, box, _ = detection
                sample = build_pose_sample(kpts, box, w, h, (frame_i - max(start_frame, 0)) / fps)
                samples.append((sample.timestamp_s, sample.keypoints))
            frame_i += 1
        cap.release()
        elapsed = time.time() - t0

        full_series = displacement_series(samples, idx=None)
        core_series = displacement_series(samples, idx=CORE_IDX)

        duration_s = (frame_i - max(start_frame, 0)) / fps
        print(f"=== {video_id} ({duration_s:.0f}s, {len(samples)} muestras, {elapsed:.0f}s de cómputo) ===")
        full_moving, full_longest = summarize(full_series, IMMOBILITY_MOVEMENT_THRESHOLD)
        core_moving, core_longest = summarize(core_series, CORE_THRESHOLD)
        print(f"  full (17 kpts, umbral {IMMOBILITY_MOVEMENT_THRESHOLD}): {full_moving:.1%} 'moviéndose', racha más larga = {full_longest:.0f}s")
        print(f"  core (hombros+cadera, umbral {CORE_THRESHOLD}):        {core_moving:.1%} 'moviéndose', racha más larga = {core_longest:.0f}s")
        print(f"  (umbral de disparo de inmovilidad: {IMMOBILITY_WINDOW_S:.0f}s)")
        print()

        with open(f"diagnose_acostado_{video_id}.pkl", "wb") as f:
            pickle.dump({"full": full_series, "core": core_series}, f)


if __name__ == "__main__":
    main()
