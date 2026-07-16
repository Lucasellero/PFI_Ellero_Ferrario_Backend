"""Evalúa la clasificación de actividades CLIP contra el ground truth del
nombre de archivo (los videos de videos/{acostado,parado,sentado} llevan la
clase como prefijo: Walk_..., Laydown_...).

Compara, en ventanas de 3 muestras:
  1. viejo: frame completo, 1 muestra, resize distorsionado (enfoque original)
  2. crop persona + promedio de 3, resize distorsionado
  3. crop persona + promedio de 3 + pad-a-cuadrado (runner actual)
y para el enfoque actual reporta además la precisión a nivel de categoría
gruesa (COARSE_CATEGORIES), que es lo que el runner emite y el familiar ve.

Corre desde la raíz del proyecto:

    python -m vlm.evaluate_activity

Resultado con los 39 videos etiquetados (2026-07-16): ver tabla en vlm/README.md.
"""
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
    aggregate_to_categories,
    crop_to_person,
)
from src.pose_pipeline import detect_person
from ultralytics import YOLO

clf_nopad = ClipActivityClassifier(quantized=True, pad_square=False)
clf_pad = ClipActivityClassifier(quantized=True, pad_square=True)
yolo = YOLO(str(ROOT / 'yolo11n-pose.pt'))

videos = []
for folder in ('acostado', 'parado', 'sentado'):
    for f in sorted((ROOT / 'videos' / folder).glob('*.mp4')):
        label = f.stem.split('_')[0]
        videos.append((f, label))

old_ok = crop_ok = pad_ok = cat_ok = total_windows = 0
for path, truth in videos:
    truth_cat = LABEL_TO_CATEGORY[truth]
    cap = cv2.VideoCapture(str(path))
    nframes = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    # 9 frames repartidos → 3 ventanas de 3 muestras
    idxs = np.linspace(0, max(nframes - 1, 0), 9).astype(int)
    samples = []   # (scores_full, scores_crop, scores_crop_pad) por frame
    for i in idxs:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(i))
        ok, frame = cap.read()
        if not ok:
            continue
        det = detect_person(yolo, frame)
        full = clf_nopad.classify(frame)['scores']
        if det:
            crop = crop_to_person(frame, det[1])
            crop_scores = clf_nopad.classify(crop)['scores']
            pad_scores = clf_pad.classify(crop)['scores']
        else:
            crop_scores = pad_scores = None
        samples.append((full, crop_scores, pad_scores))
    cap.release()

    v_old = v_crop = v_pad = v_cat = v_tot = 0
    for w in range(0, len(samples) - 2, 3):
        window = samples[w:w + 3]
        old_pred = max(window[0][0], key=window[0][0].get)
        crops = [(c, p) for _, c, p in window if c is not None]
        if not crops:
            continue
        avg_crop = {k: sum(c[k] for c, _ in crops) / len(crops) for k in crops[0][0]}
        avg_pad = {k: sum(p[k] for _, p in crops) / len(crops) for k in crops[0][0]}
        cat_scores = aggregate_to_categories(avg_pad)
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
