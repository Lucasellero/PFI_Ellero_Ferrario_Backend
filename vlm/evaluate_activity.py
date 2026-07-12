"""Evalúa la clasificación de actividades CLIP contra el ground truth del
nombre de archivo (los videos de videos/{acostado,parado,sentado} llevan la
clase como prefijo: Walk_..., Laydown_...).

Compara el enfoque viejo (frame completo, 1 muestra por ventana) contra el
del runner actual (recorte a la persona + promedio de 3 muestras por
ventana). Corre desde la raíz del proyecto:

    python -m vlm.evaluate_activity

Resultado con los 39 videos etiquetados (2026-07-12):
    viejo (frame completo, 1 muestra): 25.0% de ventanas correctas
    nuevo (crop persona + promedio 3): 40.2%
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'vision'))

import cv2
import numpy as np
from vlm.src.clip_classifier import ClipActivityClassifier, crop_to_person
from src.pose_pipeline import detect_person
from ultralytics import YOLO
clf = ClipActivityClassifier(quantized=True)
yolo = YOLO(str(ROOT / 'yolo11n-pose.pt'))

videos = []
for folder in ('acostado', 'parado', 'sentado'):
    for f in sorted((ROOT / 'videos' / folder).glob('*.mp4')):
        label = f.stem.split('_')[0]
        videos.append((f, label))

old_ok = new_ok = total_windows = 0
per_video = []
for path, truth in videos:
    cap = cv2.VideoCapture(str(path))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25
    nframes = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    # 9 frames repartidos → 3 ventanas de 3 muestras
    idxs = np.linspace(0, max(nframes - 1, 0), 9).astype(int)
    samples = []   # (scores_full, scores_crop) por frame con persona
    for i in idxs:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(i))
        ok, frame = cap.read()
        if not ok:
            continue
        det = detect_person(yolo, frame)
        full = clf.classify(frame)['scores']
        crop = clf.classify(crop_to_person(frame, det[1]))['scores'] if det else None
        samples.append((full, crop))
    cap.release()
    v_old = v_new = v_tot = 0
    for w in range(0, len(samples) - 2, 3):
        window = samples[w:w+3]
        # viejo: primera muestra de la ventana, frame completo
        old_pred = max(window[0][0], key=window[0][0].get)
        # nuevo: promedio de los crops disponibles (fallback a full si no hubo persona)
        crops = [c for _, c in window if c is not None]
        if not crops:
            continue
        avg = {k: sum(c[k] for c in crops) / len(crops) for k in crops[0]}
        new_pred = max(avg, key=avg.get)
        v_tot += 1
        v_old += old_pred == truth
        v_new += new_pred == truth
    old_ok += v_old; new_ok += v_new; total_windows += v_tot
    per_video.append((path.stem, truth, v_old, v_new, v_tot))
    print(f'{path.stem:45s} {truth:22s} viejo {v_old}/{v_tot}  nuevo {v_new}/{v_tot}', flush=True)

print(f'\n=== TOTAL ({len(videos)} videos, {total_windows} ventanas) ===')
print(f'viejo (frame completo, 1 muestra): {old_ok}/{total_windows} = {old_ok/total_windows:.1%}')
print(f'nuevo (crop persona + promedio 3): {new_ok}/{total_windows} = {new_ok/total_windows:.1%}')
