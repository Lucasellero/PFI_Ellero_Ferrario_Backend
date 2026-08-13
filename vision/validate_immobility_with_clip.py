"""Revalida si el filtro de confirmación CLIP (linear probe, ver
vlm/src/fall_confirmation.py, integrado 2026-08-13) resuelve el hallazgo
documentado en vision/README.md ("Inmovilidad prolongada", sección final):
de 12 tramos "sentado quieto" de Toyota Smarthome untrimmed, 4 nunca
llegaron a probar la lógica de inmovilidad porque el detector de caídas
geométrico disparaba primero, siempre falso positivo (P07T10C02, P09T05C05,
P07T05C05, P09T06C05 — los mismos 4 clips largos usados como negativos
duros en `videos/ToyotaSmarthome_FP/`).

Esta corrida NO reproduce los sub-tramos exactos de la validación original
(esos límites no quedaron guardados en ningún script versionado) — corre el
pipeline completo (heurística geométrica + confirmación CLIP + tracking de
inmovilidad) sobre los 4 clips COMPLETOS y mide, de forma honesta y
reproducible:
  1. Eventos de caída geométricos crudos (antes del filtro CLIP).
  2. Cuántos de esos sobreviven la confirmación CLIP (falsos positivos
     reales que llegarían al store de eventos en producción).
  3. Si la inmovilidad llega a dispararse en algún momento del clip, y con
     qué postura/severidad.

Uso:
    vision/.venv/Scripts/python.exe validate_immobility_with_clip.py
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import cv2

from src.pose_pipeline import FallImmobilityDetector
from vlm.src.clip_classifier import ClipActivityClassifier, crop_to_person
from vlm.src.fall_confirmation import FallConfirmationClassifier

CLIPS_DIR = ROOT / "videos" / "ToyotaSmarthome_FP"
CLIPS = ["P07T10C02.mp4", "P09T05C05.mp4", "P07T05C05.mp4", "P09T06C05.mp4"]


def main() -> None:
    clip_model = ClipActivityClassifier()
    fall_confirm = FallConfirmationClassifier(clip_model)
    print(f"Filtro de confirmación: {'probe' if fall_confirm._probe_W is not None else 'zero-shot'}\n")

    for clip_name in CLIPS:
        path = CLIPS_DIR / clip_name
        cap = cv2.VideoCapture(str(path))
        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0

        detector = FallImmobilityDetector(model_path="yolo11n-pose.pt")
        raw_falls = []
        confirmed_falls = []
        immobility_events = []

        t0 = time.time()
        frame_i = 0
        while True:
            ok, image = cap.read()
            if not ok:
                break
            t = frame_i / fps
            for raw in detector.process_frame(image, t):
                if raw["event_type"] == "fall":
                    raw_falls.append(raw["video_timestamp_s"])
                    if detector.last_detection is not None:
                        kpts, box = detector.last_detection
                        crop = crop_to_person(image, box)
                        confirmed, margin = fall_confirm.confirm(crop)
                        if confirmed:
                            confirmed_falls.append((raw["video_timestamp_s"], round(margin, 3)))
                elif raw["event_type"] == "immobility":
                    immobility_events.append(raw)
            frame_i += 1
        cap.release()
        elapsed = time.time() - t0

        print(f"=== {clip_name} ({frame_i / fps:.0f}s, {frame_i} frames, {elapsed:.0f}s de cómputo) ===")
        print(f"  Caídas geométricas crudas: {len(raw_falls)}")
        print(f"  Sobreviven el filtro CLIP: {len(confirmed_falls)}"
              + (f"  {confirmed_falls[:5]}{'...' if len(confirmed_falls) > 5 else ''}" if confirmed_falls else ""))
        if immobility_events:
            for ev in immobility_events:
                print(f"  Inmovilidad disparada @ {ev['video_timestamp_s']:.1f}s: {ev['metadata']}")
        else:
            print(f"  Inmovilidad: nunca se disparó en todo el clip")
        print()


if __name__ == "__main__":
    main()
