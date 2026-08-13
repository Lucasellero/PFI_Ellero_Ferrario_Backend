"""Diagnóstico (no calibración) para el gap documentado en vision/README.md
(2026-08-13, sección "Inmovilidad prolongada"): la inmovilidad no se disparó
en 3 de los 4 clips largos de `videos/ToyotaSmarthome_FP/` aun después de
integrar el filtro CLIP que casi elimina los falsos positivos de caída.
Pregunta a responder: ¿la persona realmente se mueve todo el rato (el
heurístico está bien), o el criterio de "quieto" es demasiado estricto
porque promedia desplazamiento sobre TODOS los keypoints —incluyendo manos y
codos— y una persona sentada durante una hora igual mueve las manos
(comer, usar el celular, gesticular) sin cambiar de postura?

Calcula, para cada clip, dos series de "tasa de desplazamiento" con el
mismo método que `_smoothed_displacement_rate` (comparar contra la muestra
más cercana a 2s atrás, normalizado por dt real):
  - "full": los 17 keypoints COCO (igual que producción hoy).
  - "core": solo hombros + cadera (índices 5,6,11,12) — la postura del
    torso, sin manos/codos/rodillas/tobillos.

Para cada serie reporta: % de muestras "moviéndose" (rate > umbral), y la
racha continua más larga de "quieto" (mismo criterio que el contador de
`_track_immobility`: se resetea apenas una muestra supera el umbral).

Muestrea a ~2fps (cada 12 frames de 25fps nativo), igual criterio que
calibrate_immobility.py — la ventana de suavizado (2s) no necesita más
resolución que esa.

Uso:
    vision/.venv/Scripts/python.exe diagnose_immobility_displacement.py
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import cv2
import numpy as np

from src.pose_pipeline import (
    IMMOBILITY_MOVEMENT_THRESHOLD,
    IMMOBILITY_SAMPLE_WINDOW_S,
    IMMOBILITY_WINDOW_S,
    build_pose_sample,
    detect_person,
)
from ultralytics import YOLO

CLIPS_DIR = ROOT / "videos" / "ToyotaSmarthome_FP"
CLIPS = ["P09T05C05.mp4", "P07T05C05.mp4", "P09T06C05.mp4"]  # los 3 que nunca dispararon inmovilidad
# OJO (2026-08-13): con SAMPLE_EVERY_N_FRAMES=12 (~2fps, igual criterio que
# calibrate_immobility.py) las rachas "quietas" salían de 212-238s — pero el
# detector real evalúa el desplazamiento en CADA frame (25fps) y nunca
# disparó en estos mismos clips. Un movimiento breve entre dos muestras de
# 2fps no se detecta a esa resolución, así que esas rachas estaban infladas
# (mismo tipo de bug de subsampleo que MAX_FRAMES_PER_CLIP, ver arriba en el
# módulo de recalibración). Corrido a 1 = framerate nativo, para que sea
# comparable 1:1 con el comportamiento real de producción.
SAMPLE_EVERY_N_FRAMES = 1

CORE_IDX = [5, 6, 11, 12]  # hombros + cadera (índices COCO)


def displacement_series(samples: list, idx: list[int] | None) -> list[tuple[float, float | None]]:
    """Reproduce _smoothed_displacement_rate frame a frame sobre `samples`
    (lista de (t, keypoints)), opcionalmente restringido a un subconjunto de
    keypoints (`idx`)."""
    out = []
    history: list[tuple[float, np.ndarray]] = []
    for t, kpts in samples:
        target_t = t - IMMOBILITY_SAMPLE_WINDOW_S
        rate = None
        if history:
            ref_t, ref_kpts = min(history, key=lambda s: abs(s[0] - target_t))
            dt = t - ref_t
            if dt >= IMMOBILITY_SAMPLE_WINDOW_S / 2:
                a = kpts if idx is None else kpts[idx]
                b = ref_kpts if idx is None else ref_kpts[idx]
                rate = float(np.nanmean(np.abs(a - b))) / dt
        out.append((t, rate))
        history.append((t, kpts))
    return out


def summarize(series: list[tuple[float, float | None]], threshold: float) -> tuple[float, float]:
    """Devuelve (% de muestras "moviéndose", racha continua más larga de
    "quieto" en segundos), mismo criterio de reset que _track_immobility."""
    valid = [(t, r) for t, r in series if r is not None]
    if not valid:
        return 0.0, 0.0
    moving_frac = sum(1 for _, r in valid if r > threshold) / len(valid)

    longest = 0.0
    immobile_since = None
    for t, r in valid:
        if r > threshold:
            immobile_since = None
        else:
            if immobile_since is None:
                immobile_since = t
            longest = max(longest, t - immobile_since)
    return moving_frac, longest


def main() -> None:
    model = YOLO("yolo11n-pose.pt")
    model.to("cuda")

    for clip_name in CLIPS:
        path = CLIPS_DIR / clip_name
        cap = cv2.VideoCapture(str(path))
        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0

        samples = []
        frame_i = 0
        t0 = time.time()
        while True:
            ok, image = cap.read()
            if not ok:
                break
            if frame_i % SAMPLE_EVERY_N_FRAMES == 0:
                h, w = image.shape[:2]
                detection = detect_person(model, image)
                if detection is not None:
                    kpts, box, _ = detection
                    sample = build_pose_sample(kpts, box, w, h, frame_i / fps)
                    samples.append((sample.timestamp_s, sample.keypoints))
            frame_i += 1
        cap.release()
        elapsed = time.time() - t0

        full_series = displacement_series(samples, idx=None)
        core_series = displacement_series(samples, idx=CORE_IDX)

        duration_s = frame_i / fps
        print(f"=== {clip_name} ({duration_s:.0f}s, {len(samples)} muestras, {elapsed:.0f}s de cómputo) ===")
        for thr in (IMMOBILITY_MOVEMENT_THRESHOLD, 0.014):
            full_moving, full_longest = summarize(full_series, thr)
            core_moving, core_longest = summarize(core_series, thr)
            print(f"  umbral={thr}:")
            print(f"    full (17 keypoints):   {full_moving:.1%} 'moviéndose', racha quieta más larga = {full_longest:.0f}s")
            print(f"    core (hombros+cadera): {core_moving:.1%} 'moviéndose', racha quieta más larga = {core_longest:.0f}s")
        print(f"  (umbral de disparo de inmovilidad: {IMMOBILITY_WINDOW_S:.0f}s)")
        print()

        import pickle
        with open(f"diagnose_series_{clip_name}.pkl", "wb") as f:
            pickle.dump({"full": full_series, "core": core_series}, f)


if __name__ == "__main__":
    main()
