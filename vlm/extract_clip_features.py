"""Extrae embeddings CLIP (frozen, mismo encoder ONNX de producción) de un
dataset externo grande de actividades (ej. Toyota Smarthome completo), con
el mismo preprocesado que usa el runner (crop_to_person + pad_square).
Cachea a un .npz para poder entrenar/evaluar un linear probe (ver
train_category_probe.py) sin recomputar YOLO+CLIP en cada experimento.

El dataset de referencia (../videos/, 39 videos) es demasiado chico para
entrenar un clasificador — este script apunta a una carpeta externa con
muchos más videos por clase (no versionada en git, ver vlm/README.md).

Con `--object-context`, suma además un vector de presencia de objetos
COCO cerca de la persona (`fusion/src/object_context.py`, Fase 2 de
`fusion/PLAN_sustitucion_clip.md`) — pensado para medir si esa señal ayuda
a separar categorías que CLIP solo confunde por falta de contexto espacial
(ver "El probe de producción había perdido 2 categorías..." en
vlm/README.md, 2026-09-20: `Cook.Cleandishes`/`Cutbread` vs
`EatingDrinkingMedication`). Cuesta una segunda pasada de YOLO (detección,
no pose) por frame muestreado — no gratis, por eso es opcional.

Con `--zone-context <archivo.yaml>`, suma un vector one-hot de zona del
hogar (`fusion/src/zone_context.py`, Fase 1 del mismo plan) — geometría
pura sobre los keypoints que ya calculó `detect_person`, sin modelo
adicional. El `.yaml` mapea ID de cámara (parseado del nombre de archivo,
sufijo "_cNN") a polígonos normalizados por zona — ver
`fusion/toyota_smarthome_zones.yaml` para el caso de Toyota Smarthome (7
cámaras fijas de un mismo departamento).

Uso:
    python extract_clip_features.py <carpeta_dataset> [--max-per-class N] [--frames-per-video N]
                                     [--object-context] [--zone-context archivo.yaml]
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
import yaml
from vlm.src.clip_classifier import (
    ClipActivityClassifier,
    crop_to_hands,
    crop_to_person,
    LABEL_TO_CATEGORY,
)
from src.pose_pipeline import detect_person
from ultralytics import YOLO

from fusion.src.object_context import object_presence_vector
from fusion.src.zone_context import zone_one_hot

parser = argparse.ArgumentParser()
parser.add_argument('dataset_dir', help='Carpeta con videos <Actividad>_pXX_...mp4')
parser.add_argument('--max-per-class', type=int, default=40, help='Tope de videos por actividad')
parser.add_argument('--frames-per-video', type=int, default=5)
parser.add_argument('--seed', type=int, default=7)
parser.add_argument(
    '--object-context', action='store_true',
    help='Suma vector de presencia de objetos COCO cerca de la persona (fusion/src/object_context.py)',
)
parser.add_argument(
    '--zone-context', default=None, metavar='ARCHIVO.YAML',
    help='Suma vector one-hot de zona (fusion/src/zone_context.py). Archivo: '
         'ID de cámara -> {zona: [[x,y], ...]} (ver fusion/toyota_smarthome_zones.yaml)',
)
parser.add_argument('-o', '--output', default=None, help='Default: vlm/clip_features.npz')
args = parser.parse_args()

DATASET_DIR = Path(args.dataset_dir)
OUT_PATH = Path(args.output) if args.output else Path(__file__).parent / "clip_features.npz"

clf = ClipActivityClassifier(quantized=True, pad_square=True)
yolo = YOLO(str(ROOT / 'yolo11n-pose.pt'))
yolo_obj = YOLO(str(ROOT / 'yolo11n.pt')) if args.object_context else None
zones_by_camera = yaml.safe_load(Path(args.zone_context).read_text()) if args.zone_context else None


def camera_of(stem: str) -> str | None:
    m = re.search(r'_c(\d+)$', stem)
    return m.group(1) if m else None


def participant_of(stem: str) -> str:
    m = re.search(r'_p(\d+)_', stem)
    return m.group(1) if m else "unknown"


def image_features(image_bgr: np.ndarray) -> np.ndarray:
    tensor = clf._preprocess(image_bgr)
    feats = clf._session.run(None, {clf._input_name: tensor})[0][0]
    return feats / np.linalg.norm(feats)


by_label: dict[str, list[Path]] = {}
for f in sorted(DATASET_DIR.glob('**/*.mp4')):
    label = f.stem.split('_')[0]
    if label in LABEL_TO_CATEGORY:
        by_label.setdefault(label, []).append(f)

rng = random.Random(args.seed)
rows_feat, rows_hand_feat, rows_obj_feat, rows_zone_feat = [], [], [], []
rows_label, rows_participant, rows_video = [], [], []

for label, files in sorted(by_label.items()):
    chosen = rng.sample(files, args.max_per_class) if len(files) > args.max_per_class else files
    print(f'{label:22s} {len(chosen)}/{len(files)} videos', flush=True)
    for f in chosen:
        camera_zones = zones_by_camera.get(camera_of(f.stem)) if zones_by_camera else None
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
            kpts, box, kpts_conf = det
            crop = crop_to_person(frame, box)
            feat = image_features(crop)
            # Mismo fallback que ClipActivityClassifier.classify(): sin crop
            # de manos válido, se duplica el body embedding.
            hand_crop = crop_to_hands(frame, kpts, box, keypoints_conf=kpts_conf)
            hand_feat = image_features(hand_crop) if hand_crop is not None else feat
            rows_feat.append(feat)
            rows_hand_feat.append(hand_feat)
            if yolo_obj is not None:
                rows_obj_feat.append(object_presence_vector(yolo_obj, frame, box))
            if zones_by_camera is not None:
                h, w = frame.shape[:2]
                hip = np.nanmean(kpts[[11, 12]], axis=0)  # mismos índices que _L_HIP/_R_HIP en keypoint_sequence.py
                point_norm = (float(hip[0] / w), float(hip[1] / h))
                rows_zone_feat.append(zone_one_hot(camera_zones, point_norm))
            rows_label.append(label)
            rows_participant.append(participant_of(f.stem))
            rows_video.append(f.stem)
        cap.release()

features = np.stack(rows_feat).astype(np.float32)
hand_features = np.stack(rows_hand_feat).astype(np.float32)
labels = np.array(rows_label)
participants = np.array(rows_participant)
videos = np.array(rows_video)

extra = {}
if yolo_obj is not None:
    extra['object_features'] = np.stack(rows_obj_feat).astype(np.float32)
if zones_by_camera is not None:
    extra['zone_features'] = np.stack(rows_zone_feat).astype(np.float32)

np.savez(
    OUT_PATH, features=features, hand_features=hand_features,
    labels=labels, participants=participants, videos=videos, **extra,
)
suffix = ''.join(f' + {k}' for k in extra)
print(f'\nGuardado {OUT_PATH}: {features.shape[0]} muestras, {features.shape[1]}-dim (+ hand_features{suffix})')
print('Por clase:', {l: int((labels == l).sum()) for l in sorted(set(labels))})
