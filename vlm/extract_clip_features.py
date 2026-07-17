"""Extrae embeddings CLIP (frozen, mismo encoder ONNX de producción) de un
dataset externo grande de actividades (ej. Toyota Smarthome completo), con
el mismo preprocesado que usa el runner (crop_to_person + pad_square).
Cachea a un .npz para poder entrenar/evaluar un linear probe (ver
train_category_probe.py) sin recomputar YOLO+CLIP en cada experimento.

El dataset de referencia (../videos/, 39 videos) es demasiado chico para
entrenar un clasificador — este script apunta a una carpeta externa con
muchos más videos por clase (no versionada en git, ver vlm/README.md).

Uso:
    python extract_clip_features.py <carpeta_dataset> [--max-per-class N] [--frames-per-video N]
"""
import argparse
import random
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'vision'))

import cv2
import numpy as np
from vlm.src.clip_classifier import ClipActivityClassifier, crop_to_person, LABEL_TO_CATEGORY
from src.pose_pipeline import detect_person
from ultralytics import YOLO

parser = argparse.ArgumentParser()
parser.add_argument('dataset_dir', help='Carpeta con videos <Actividad>_pXX_...mp4')
parser.add_argument('--max-per-class', type=int, default=40, help='Tope de videos por actividad')
parser.add_argument('--frames-per-video', type=int, default=5)
parser.add_argument('--seed', type=int, default=7)
parser.add_argument('-o', '--output', default=None, help='Default: vlm/clip_features.npz')
args = parser.parse_args()

DATASET_DIR = Path(args.dataset_dir)
OUT_PATH = Path(args.output) if args.output else Path(__file__).parent / "clip_features.npz"

clf = ClipActivityClassifier(quantized=True, pad_square=True)
yolo = YOLO(str(ROOT / 'yolo11n-pose.pt'))


def participant_of(stem: str) -> str:
    m = re.search(r'_p(\d+)_', stem)
    return m.group(1) if m else "unknown"


def image_features(image_bgr: np.ndarray) -> np.ndarray:
    tensor = clf._preprocess(image_bgr)
    feats = clf._session.run(None, {clf._input_name: tensor})[0][0]
    return feats / np.linalg.norm(feats)


by_label: dict[str, list[Path]] = {}
for f in sorted(DATASET_DIR.glob('*.mp4')):
    label = f.stem.split('_')[0]
    if label in LABEL_TO_CATEGORY:
        by_label.setdefault(label, []).append(f)

rng = random.Random(args.seed)
rows_feat, rows_label, rows_participant, rows_video = [], [], [], []

for label, files in sorted(by_label.items()):
    chosen = rng.sample(files, args.max_per_class) if len(files) > args.max_per_class else files
    print(f'{label:22s} {len(chosen)}/{len(files)} videos', flush=True)
    for f in chosen:
        cap = cv2.VideoCapture(str(f))
        nframes = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        idxs = np.linspace(0, max(nframes - 1, 0), args.frames_per_video).astype(int)
        for i in idxs:
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(i))
            ok, frame = cap.read()
            if not ok:
                continue
            det = detect_person(yolo, frame)
            if not det:
                continue
            crop = crop_to_person(frame, det[1])
            feat = image_features(crop)
            rows_feat.append(feat)
            rows_label.append(label)
            rows_participant.append(participant_of(f.stem))
            rows_video.append(f.stem)
        cap.release()

features = np.stack(rows_feat).astype(np.float32)
labels = np.array(rows_label)
participants = np.array(rows_participant)
videos = np.array(rows_video)

np.savez(OUT_PATH, features=features, labels=labels, participants=participants, videos=videos)
print(f'\nGuardado {OUT_PATH}: {features.shape[0]} muestras, {features.shape[1]}-dim')
print('Por clase:', {l: int((labels == l).sum()) for l in sorted(set(labels))})
