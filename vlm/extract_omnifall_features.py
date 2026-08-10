"""Extrae embeddings CLIP de segmentos anotados por OmniFall
(huggingface.co/datasets/simplexsigil2/omnifall) sobre los videos de
CAUCAFall que ya están en ../videos/CAUCAFall/ — sin descargar nada nuevo.

OmniFall no redistribuye los videos originales de los datasets que unifica
(son de terceros, con sus propias licencias), solo anotaciones por segmento
armonizadas a una taxonomía de 16 clases (ver LABELS.md del dataset). Este
script resuelve cada fila de labels/caucafall.csv a un archivo local dentro
de CAUCAFall (mismo trazado que un video de Toyota Smarthome: nombre de
carpeta de actividad + subject → archivo .avi) y extrae frames dentro de la
ventana [start, end] de cada segmento.

Solo se extraen las clases de interés: `Standing` (parado inactivo, no
cubierto por ninguna de las 31 clases de Toyota Smarthome) y `StandUp`
(fusionada con `Getup`, mismo gesto de incorporarse). El resto de las 16
clases de OmniFall (Walk, Fall, Fallen, Sitting, etc.) ya están cubiertas
por Toyota Smarthome o por el detector geométrico de vision/, así que no
aportan nada nuevo acá — ver vlm/README.md para el detalle completo.

Uso:
    curl -sL https://huggingface.co/datasets/simplexsigil2/omnifall/resolve/main/labels/caucafall.csv \
        -o caucafall_omnifall.csv
    python extract_omnifall_features.py [--csv caucafall_omnifall.csv]
"""
import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'vision'))

import cv2
import numpy as np
import pandas as pd
from vlm.src.clip_classifier import ClipActivityClassifier, crop_to_hands, crop_to_person
from src.pose_pipeline import detect_person
from ultralytics import YOLO

CAUCAFALL_DIR = ROOT / 'videos' / 'CAUCAFall' / 'CAUCAFall'

# Carpeta real (con espacios) para cada clave de actividad tal como aparece
# en el 'path' de caucafall.csv (ej. "adl/PickupobjectS1" → clave "Pickupobject").
FOLDER_OF_ACTIVITY = {
    'Hop': 'Hop',
    'Kneel': 'Kneel',
    'Pickupobject': 'Pick up object',
    'SitDown': 'Sit down',
    'Walk': 'Walk',
    'FallBackwards': 'Fall backwards',
    'FallForward': 'Fall forward',
    'FallLeft': 'Fall left',
    'FallRight': 'Fall right',
    'FallSitting': 'Fall sitting',
}

# label2id de OmniFall (ver LABELS.md): solo nos interesan estas dos.
LABEL_ID_TO_NAME = {7: 'StandUp', 8: 'Standing'}
# StandUp es el mismo gesto que la clase 'Getup' ya definida en
# ACTIVITY_PROMPTS/COARSE_CATEGORIES (src/clip_classifier.py) — se etiqueta
# así para reforzar esa clase en vez de crear una redundante.
OUTPUT_LABEL = {'StandUp': 'Getup', 'Standing': 'Standing'}

parser = argparse.ArgumentParser()
parser.add_argument('--csv', default=None, help='Default: vlm/caucafall_omnifall.csv')
parser.add_argument('-o', '--output', default=None, help='Default: vlm/omnifall_features.npz')
args = parser.parse_args()

CSV_PATH = Path(args.csv) if args.csv else Path(__file__).parent / 'caucafall_omnifall.csv'
OUT_PATH = Path(args.output) if args.output else Path(__file__).parent / 'omnifall_features.npz'

clf = ClipActivityClassifier(quantized=True, pad_square=True)
yolo = YOLO(str(ROOT / 'yolo11n-pose.pt'))


def image_features(image_bgr: np.ndarray) -> np.ndarray:
    tensor = clf._preprocess(image_bgr)
    feats = clf._session.run(None, {clf._input_name: tensor})[0][0]
    return feats / np.linalg.norm(feats)


def resolve_video_path(path: str, subject: int) -> Path | None:
    stem = path.split('/')[-1]                       # ej. "PickupobjectS1"
    suffix = f"S{subject}"
    if not stem.endswith(suffix):
        return None
    activity_key = stem[: -len(suffix)]               # ej. "Pickupobject"
    folder = FOLDER_OF_ACTIVITY.get(activity_key)
    if folder is None:
        return None
    return CAUCAFALL_DIR / f"Subject.{subject}" / folder / f"{stem}.avi"


df = pd.read_csv(CSV_PATH)
df = df[df['label'].isin(LABEL_ID_TO_NAME)]

rows_feat, rows_hand_feat, rows_label, rows_participant, rows_video = [], [], [], [], []
skipped_missing_file, skipped_no_person = 0, 0

for i, row in df.iterrows():
    video_path = resolve_video_path(row['path'], int(row['subject']))
    if video_path is None or not video_path.exists():
        skipped_missing_file += 1
        continue

    label = OUTPUT_LABEL[LABEL_ID_TO_NAME[row['label']]]
    start_s, end_s = float(row['start']), float(row['end'])
    duration = end_s - start_s
    # Segmentos cortos (frecuente, algunos <1s): 1 muestra. Más largos: 2,
    # en los extremos de la ventana para variar la pose dentro del gesto.
    sample_times = [start_s + duration / 2] if duration < 2.0 else [
        start_s + duration * 0.3, start_s + duration * 0.7,
    ]

    cap = cv2.VideoCapture(str(video_path))
    video_id = f"{row['path']}_{i}"
    got_sample = False
    for t in sample_times:
        cap.set(cv2.CAP_PROP_POS_MSEC, t * 1000)
        ok, frame = cap.read()
        if not ok:
            continue
        det = detect_person(yolo, frame)
        if not det:
            continue
        kpts, box = det
        crop = crop_to_person(frame, box)
        feat = image_features(crop)
        hand_crop = crop_to_hands(frame, kpts, box)
        hand_feat = image_features(hand_crop) if hand_crop is not None else feat
        rows_feat.append(feat)
        rows_hand_feat.append(hand_feat)
        rows_label.append(label)
        rows_participant.append(f"cauca_s{row['subject']}")
        rows_video.append(video_id)
        got_sample = True
    cap.release()
    if not got_sample:
        skipped_no_person += 1

features = np.stack(rows_feat).astype(np.float32)
hand_features = np.stack(rows_hand_feat).astype(np.float32)
labels = np.array(rows_label)
participants = np.array(rows_participant)
videos = np.array(rows_video)

np.savez(
    OUT_PATH, features=features, hand_features=hand_features,
    labels=labels, participants=participants, videos=videos,
)
print(f'Guardado {OUT_PATH}: {features.shape[0]} muestras, {features.shape[1]}-dim (+ hand_features)')
print('Por clase:', {l: int((labels == l).sum()) for l in sorted(set(labels))})
print(f'Segmentos sin archivo local: {skipped_missing_file}, sin persona detectada: {skipped_no_person}')
