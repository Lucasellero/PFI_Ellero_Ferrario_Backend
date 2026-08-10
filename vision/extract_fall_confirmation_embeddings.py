"""Prototipo dev-only: extrae embeddings ONNX de CLIP (no scores zero-shot)
en cada evento geométrico de caída, para entrenar un linear probe en vez de
usar el margen contra prompts fijos (ver `train_fall_confirmation_probe.py`).

Motivación: el filtro de confirmación zero-shot (`calibrate_fall_confirmation.py`,
`vlm/src/fall_confirmation.py`) ya funciona (82.0% recall, -67% FP), pero
compara contra prompts de texto fijos. Un linear probe entrenado sobre los
mismos embeddings —la misma técnica que ya mejoró la clasificación de
actividades en `vlm/` (`train_category_probe.py`)— podría generalizar mejor,
sin agregar ningún modelo nuevo al pipeline: reusa el mismo encoder ONNX que
`vlm/` ya necesita cargar para actividades.

Corre el detector geométrico completo sobre TODOS los clips (`videos/**/Fall.csv`,
mismo `load_ground_truth()` que `evaluate_falls.py`), y en cada evento
geométrico recorta la persona y guarda su embedding CLIP (512-dim, ONNX
INT8 — el mismo modelo cuantizado que corre en producción, no el prototipo
fp32/torch) junto con video/t/etiqueta. No es parte del pipeline productivo.

Uso:
    vlm/.venv/Scripts/python.exe extract_fall_confirmation_embeddings.py
Genera fall_confirmation_embeddings.npz (features, labels, is_in_nonfall, video, t).
"""
from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from evaluate_falls import load_ground_truth
from src.pose_pipeline import FallImmobilityDetector
from vlm.src.clip_classifier import ClipActivityClassifier, crop_to_person


def main() -> None:
    encoder = ClipActivityClassifier(quantized=True)

    cases = load_ground_truth()
    if "--skip-toyota" in sys.argv:
        # Los 4 clips de ToyotaSmarthome_FP duran ~1100-1600s cada uno (vs.
        # ~100-300 frames del resto) — carísimos de correr frame a frame en
        # CPU. Se pueden excluir para una corrida rápida sobre GMDCSA-24 +
        # CAUCAFall (128 de los 128 casos de caída), dejando el chequeo de FP
        # en el setup problemático para una corrida aparte sin apuro.
        cases = [c for c in cases if "ToyotaSmarthome_FP" not in c["video"]]
    else:
        max_toyota = 4
        for arg in sys.argv:
            if arg.startswith("--max-toyota-clips="):
                max_toyota = int(arg.split("=", 1)[1])
        if max_toyota < 4:
            kept_toyota = 0
            filtered = []
            for c in cases:
                if "ToyotaSmarthome_FP" in c["video"]:
                    if kept_toyota >= max_toyota:
                        continue
                    kept_toyota += 1
                filtered.append(c)
            cases = filtered
    features: list[np.ndarray] = []
    videos: list[str] = []
    ts: list[float] = []
    is_real_fall: list[bool] = []
    is_in_nonfall: list[bool] = []

    out_path = Path(__file__).parent / "fall_confirmation_embeddings.npz"

    def save() -> None:
        np.savez(
            out_path,
            features=np.array(features, dtype=np.float32),
            video=np.array(videos),
            t=np.array(ts, dtype=np.float32),
            is_real_fall=np.array(is_real_fall),
            is_in_nonfall=np.array(is_in_nonfall),
        )

    for i, case in enumerate(cases):
        det = FallImmobilityDetector()
        cap = cv2.VideoCapture(case["video"])
        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        frame_i = 0
        n_events = 0
        while True:
            ok, image = cap.read()
            if not ok:
                break
            t = frame_i / fps
            events = det.process_frame(image, t)
            for e in events:
                if e["event_type"] != "fall":
                    continue
                n_events += 1
                box = det.last_detection[1]
                crop = crop_to_person(image, box)
                feat = encoder.encode_image(crop)

                real_fall = any(s <= t <= en for s, en in case["fall_segments"])
                in_nonfall = any(s <= t <= en for s, en in case["nonfall_segments"])

                features.append(feat)
                videos.append(case["video"])
                ts.append(round(t, 2))
                is_real_fall.append(real_fall)
                is_in_nonfall.append(in_nonfall)
            frame_i += 1
        cap.release()
        print(f"  [{i + 1}/{len(cases)}] {case['video']}: {n_events} eventos geométricos", flush=True)
        save()  # checkpoint incremental: permite cortar el proceso sin perder lo ya hecho

    print(f"\nGuardado {out_path} ({len(features)} eventos geométricos evaluados)")


if __name__ == "__main__":
    main()
