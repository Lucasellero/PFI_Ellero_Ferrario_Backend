"""Evalúa la clasificación de actividades CLIP contra el ground truth del
nombre de archivo (los videos llevan la clase como prefijo: Walk_...,
Laydown_...).

Compara, en ventanas de 3 muestras:
  1. viejo: frame completo, 1 muestra, resize distorsionado (enfoque original)
  2. crop persona + promedio de 3, resize distorsionado
  3. crop persona + promedio de 3 + pad-a-cuadrado (runner actual)
y para el enfoque actual reporta además la precisión a nivel de categoría
gruesa (COARSE_CATEGORIES), que es lo que el runner emite y el familiar ve.

Corre desde la raíz del proyecto:

    python -m vlm.evaluate_activity                                # usa ../videos/, todos los videos
    python -m vlm.evaluate_activity <otra_carpeta>                  # ej. dataset externo más grande
    python -m vlm.evaluate_activity <carpeta> --per-class 10 --seed 1

`--per-class N` muestrea como máximo N videos por actividad (en vez de usar
la carpeta entera) — pensado para iterar tipo cross-validation sobre un
dataset grande: correr con `--seed 1`, ajustar prompts/umbrales según lo que
falla, correr de nuevo con `--seed 2` sobre otra muestra, etc.

Solo se evalúan videos cuyo prefijo de actividad tiene prompts definidos en
ACTIVITY_PROMPTS (ver src/clip_classifier.py) — el resto se ignora, no se
cuenta como error.

Resultado con los 39 videos etiquetados (2026-07-16): ver tabla en vlm/README.md.
"""
import argparse
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'vision'))

import cv2
import numpy as np
from vlm.src.clip_classifier import (
    ClipActivityClassifier,
    LABEL_TO_CATEGORY,
    crop_to_hands,
    crop_to_person,
)
from src.pose_pipeline import detect_person
from ultralytics import YOLO

parser = argparse.ArgumentParser()
parser.add_argument('dir', nargs='?', default=None, help='Carpeta con videos (default: ../videos/)')
parser.add_argument('--per-class', type=int, default=None,
                     help='Muestrear como máximo N videos por actividad en vez de usar todos')
parser.add_argument('--seed', type=int, default=0, help='Semilla del muestreo (--per-class)')
parser.add_argument('--models-dir', default=None,
                     help='Carpeta de modelos ONNX a usar (default: vlm/models/ = ViT-B/32 producción). '
                          'Ej. vlm/models_vit-b-16 para comparar otra variante.')
args = parser.parse_args()

models_dir = Path(args.models_dir) if args.models_dir else None
clf_nopad = ClipActivityClassifier(quantized=True, pad_square=False, models_dir=models_dir)
clf_pad = ClipActivityClassifier(quantized=True, pad_square=True, models_dir=models_dir)
yolo = YOLO(str(ROOT / 'yolo11n-pose.pt'))

videos_dir = Path(args.dir) if args.dir else ROOT / 'videos'

by_label: dict[str, list[Path]] = {}
skipped = 0
for f in sorted(videos_dir.glob('*.mp4')):
    label = f.stem.split('_')[0]
    if label not in LABEL_TO_CATEGORY:
        skipped += 1
        continue
    by_label.setdefault(label, []).append(f)
if skipped:
    print(f'({skipped} videos con actividad sin prompts definidos, ignorados)')

rng = random.Random(args.seed)
videos: list[tuple[Path, str]] = []
for label, files in sorted(by_label.items()):
    chosen = rng.sample(files, args.per_class) if args.per_class and len(files) > args.per_class else files
    videos.extend((f, label) for f in chosen)
    print(f'  {label:22s} {len(chosen)}/{len(files)} videos')

old_ok = crop_ok = pad_ok = cat_ok = total_windows = 0
for path, truth in videos:
    truth_cat = LABEL_TO_CATEGORY[truth]
    cap = cv2.VideoCapture(str(path))
    nframes = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    # 9 frames repartidos → 3 ventanas de 3 muestras
    idxs = np.linspace(0, max(nframes - 1, 0), 9).astype(int)
    samples = []   # (scores_full, scores_crop, scores_crop_pad, category_scores_pad) por frame
    for i in idxs:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(i))
        ok, frame = cap.read()
        if not ok:
            continue
        det = detect_person(yolo, frame)
        full = clf_nopad.classify(frame)['scores']
        if det:
            kpts, box, kpts_conf = det
            crop = crop_to_person(frame, box)
            hand_crop = crop_to_hands(frame, kpts, box, keypoints_conf=kpts_conf)
            crop_scores = clf_nopad.classify(crop)['scores']
            pad_result = clf_pad.classify(crop, hand_crop)
            pad_scores = pad_result['scores']
            pad_cat_scores = pad_result['category_scores']
        else:
            crop_scores = pad_scores = pad_cat_scores = None
        samples.append((full, crop_scores, pad_scores, pad_cat_scores))
    cap.release()

    v_old = v_crop = v_pad = v_cat = v_tot = 0
    for w in range(0, len(samples) - 2, 3):
        window = samples[w:w + 3]
        old_pred = max(window[0][0], key=window[0][0].get)
        crops = [(c, p, cs) for _, c, p, cs in window if c is not None]
        if not crops:
            continue
        avg_crop = {k: sum(c[k] for c, _, _ in crops) / len(crops) for k in crops[0][0]}
        avg_pad = {k: sum(p[k] for _, p, _ in crops) / len(crops) for k in crops[0][0]}
        all_cats = {c for _, _, cs in crops for c in cs}
        cat_scores = {c: sum(cs.get(c, 0.0) for _, _, cs in crops) / len(crops) for c in all_cats}
        v_tot += 1
        v_old += old_pred == truth
        v_crop += max(avg_crop, key=avg_crop.get) == truth
        v_pad += max(avg_pad, key=avg_pad.get) == truth
        v_cat += max(cat_scores, key=cat_scores.get) == truth_cat
    old_ok += v_old; crop_ok += v_crop; pad_ok += v_pad; cat_ok += v_cat
    total_windows += v_tot
    print(f'{path.stem:45s} {truth:22s} viejo {v_old}/{v_tot}  crop {v_crop}/{v_tot}  '
          f'+pad {v_pad}/{v_tot}  categoría {v_cat}/{v_tot}', flush=True)

print(f'\n=== TOTAL ({len(videos)} videos, {total_windows} ventanas) ===')
print(f'viejo (frame completo, 1 muestra):    {old_ok}/{total_windows} = {old_ok/total_windows:.1%}')
print(f'crop persona + promedio 3:            {crop_ok}/{total_windows} = {crop_ok/total_windows:.1%}')
print(f'crop + promedio 3 + pad (fino):       {pad_ok}/{total_windows} = {pad_ok/total_windows:.1%}')
print(f'crop + pad, categoría gruesa (runner): {cat_ok}/{total_windows} = {cat_ok/total_windows:.1%}')
