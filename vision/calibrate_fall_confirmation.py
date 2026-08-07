"""Prototipo dev-only: filtro de confirmación con CLIP zero-shot para los
eventos de caída que dispara la heurística geométrica (`FallImmobilityDetector`).

Motivación: el grid search de umbrales (`recalibrate_fall_thresholds.py`)
demostró que NINGUNA combinación de FALL_VERTICAL_VELOCITY/FALL_ASPECT_RATIO
resuelve los falsos positivos del setup de cámara Toyota Smarthome (persona
sentada en un sillón bajo/piso) — es un límite estructural de usar solo
geometría (aspect ratio + velocidad), no un problema de calibración fina. Ver
vision/README.md, sección "Detección de caída como evento".

Este script corre el detector geométrico completo sobre TODOS los clips
(`videos/**/Fall.csv`), y en cada frame donde dispara un evento de caída,
recorta la persona y le pregunta a CLIP zero-shot si la escena parece una
caída accidental o una postura voluntaria (sentado/acostado a propósito).
Guarda los scores en un CSV para poder elegir el umbral de confirmación por
separado del recall/FP geométrico ya calibrado.

No es parte del pipeline productivo — usa el paquete `clip` completo
(PyTorch), no el ONNX de `vlm/`. Si el filtro demuestra ser efectivo, la
integración productiva va en `runner/` (que ya importa vision/ y vlm/ juntos,
ver README raíz "Cómo trabajar en paralelo"), reusando el encoder ONNX que
`vlm/` ya exporta en vez de cargar un segundo modelo CLIP completo.

Uso:
    pip install -r requirements-dev.txt
    python calibrate_fall_confirmation.py
Genera fall_confirmation_scores.csv.
"""
from __future__ import annotations

import csv

import clip
import cv2
import torch
from PIL import Image

from evaluate_falls import load_ground_truth
from src.pose_pipeline import FallImmobilityDetector

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Prompt ensembling, mismo criterio que vlm/src/clip_classifier.py: varios
# templates por clase promediados, más robusto que un solo prompt.
FALL_PROMPTS = [
    "a photo of a person having an accidental fall",
    "a person collapsing and falling down to the floor",
    "someone tripping and falling on the ground",
    "a person in the middle of falling, losing balance",
]
NOTFALL_PROMPTS = [
    "a photo of a person sitting down calmly on a low chair or the floor",
    "a person intentionally lying down to rest",
    "someone sitting still on a sofa or ottoman watching TV",
    "a person calmly seated on the ground",
]


def crop_to_person(image_bgr, box, margin: float = 0.25):
    """Copia de vlm/src/clip_classifier.py:crop_to_person — duplicado a
    propósito, no se importa vlm/ desde vision/ (ver README raíz, límites de
    módulo). Es un recorte de 15 líneas, no vale la pena romper la
    independencia de los módulos por esto."""
    h, w = image_bgr.shape[:2]
    x1, y1, x2, y2 = (float(v) for v in box)
    mx = (x2 - x1) * margin
    my = (y2 - y1) * margin
    x1 = max(0, int(x1 - mx))
    y1 = max(0, int(y1 - my))
    x2 = min(w, int(x2 + mx))
    y2 = min(h, int(y2 + my))
    if x2 - x1 < 32 or y2 - y1 < 32:
        return image_bgr
    return image_bgr[y1:y2, x1:x2]


def main() -> None:
    print(f"Device: {DEVICE}")
    clip_model, clip_preprocess = clip.load("ViT-B/32", device=DEVICE)
    clip_model.eval()

    with torch.no_grad():
        fall_tokens = clip.tokenize(FALL_PROMPTS).to(DEVICE)
        notfall_tokens = clip.tokenize(NOTFALL_PROMPTS).to(DEVICE)
        fall_feat = clip_model.encode_text(fall_tokens).float()
        fall_feat /= fall_feat.norm(dim=-1, keepdim=True)
        fall_feat = fall_feat.mean(dim=0)
        fall_feat /= fall_feat.norm()
        notfall_feat = clip_model.encode_text(notfall_tokens).float()
        notfall_feat /= notfall_feat.norm(dim=-1, keepdim=True)
        notfall_feat = notfall_feat.mean(dim=0)
        notfall_feat /= notfall_feat.norm()

    cases = load_ground_truth()
    rows = []

    for i, case in enumerate(cases):
        det = FallImmobilityDetector()
        det._model.to(DEVICE)
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
                pil = clip_preprocess(Image.fromarray(crop[..., ::-1])).unsqueeze(0).to(DEVICE)
                with torch.no_grad():
                    img_feat = clip_model.encode_image(pil).float()
                    img_feat /= img_feat.norm(dim=-1, keepdim=True)
                    fall_score = float((img_feat @ fall_feat).item())
                    notfall_score = float((img_feat @ notfall_feat).item())

                is_real_fall = any(s <= t <= en for s, en in case["fall_segments"])
                is_in_nonfall_segment = any(s <= t <= en for s, en in case["nonfall_segments"])
                rows.append({
                    "video": case["video"],
                    "t": round(t, 2),
                    "is_real_fall": is_real_fall,
                    "is_in_nonfall_segment": is_in_nonfall_segment,
                    "clip_fall_score": round(fall_score, 4),
                    "clip_notfall_score": round(notfall_score, 4),
                    "clip_margin": round(fall_score - notfall_score, 4),
                })
            frame_i += 1
        cap.release()
        print(f"  [{i + 1}/{len(cases)}] {case['video']}: {n_events} eventos geométricos", flush=True)

    with open("fall_confirmation_scores.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nGuardado fall_confirmation_scores.csv ({len(rows)} eventos geométricos evaluados)")


if __name__ == "__main__":
    main()
