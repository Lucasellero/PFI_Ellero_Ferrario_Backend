"""Extrae, en un solo pase de YOLO-pose por video, tanto ventanas de
keypoints (para el clasificador temporal, ver `src/keypoint_sequence.py`)
como embeddings CLIP (cuerpo + manos, mismo esquema que
`vlm/extract_clip_features.py`) — cachea todo a un `.npz` para poder
entrenar y evaluar los tres enfoques (CLIP solo / esqueleto solo / híbrido)
sin recomputar YOLO+CLIP en cada experimento.

Un solo pase porque las ventanas de esqueleto necesitan la secuencia
continua de frames anteriores al punto de muestreo (a diferencia de
`extract_clip_features.py`, que puede saltar directo con `cap.set()` a
frames arbitrarios porque CLIP no usa historial) — acá se lee el video
frame a frame, subsampleado a `--fps-cap` para no correr YOLO más seguido
de lo que correría en vivo en el Pi.

Uso:
    python extract_dataset.py <carpeta_dataset> [--max-per-class N] [--samples-per-video N]
                               [--window-s 2.0] [--n-frames 16] [--fps-cap 10]
                               [-o fusion/dataset_features.npz]

`<carpeta_dataset>` con videos `<Actividad>_pXX_...mp4` (mismo patrón que
`vlm/extract_clip_features.py` — para sumar `Standing`, agregar clips
`Standing_pXX_...mp4` a la carpeta, ver `fusion/README.md` sección "Datos").
"""
from __future__ import annotations

import argparse
import random
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "vision"))

import cv2
import numpy as np
from ultralytics import YOLO

from src.pose_pipeline import build_pose_sample, detect_person
from vlm.src.clip_classifier import ClipActivityClassifier, crop_to_hands, crop_to_person

from fusion.src.keypoint_sequence import SkeletonSequenceBuffer, frame_features
from fusion.src.labels import FINE_LABELS

parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument("dataset_dir", help="Carpeta con videos <Actividad>_pXX_...mp4")
parser.add_argument("--max-per-class", type=int, default=40, help="Tope de videos por actividad")
parser.add_argument(
    "--samples-per-video", type=int, default=3,
    help="Ventanas muestreadas por video (espaciadas uniformemente entre los puntos con cobertura temporal suficiente)",
)
parser.add_argument("--window-s", type=float, default=2.0, help="Ventana temporal del clasificador de esqueleto (segundos)")
parser.add_argument("--n-frames", type=int, default=16, help="Frames resampleados dentro de la ventana")
parser.add_argument(
    "--fps-cap", type=float, default=10.0,
    help="FPS máximo al que se corre YOLO-pose (subsamplea el video nativo — más rápido y más parecido al framerate real esperable en el Pi)",
)
parser.add_argument("--min-coverage", type=float, default=0.6, help="Ver SkeletonSequenceBuffer.build_window")
parser.add_argument("--seed", type=int, default=7)
parser.add_argument("-o", "--output", default=None, help="Default: fusion/dataset_features.npz")
args = parser.parse_args()

DATASET_DIR = Path(args.dataset_dir)
OUT_PATH = Path(args.output) if args.output else Path(__file__).parent / "dataset_features.npz"

clip_clf = ClipActivityClassifier(quantized=True, pad_square=True)
yolo = YOLO(str(ROOT / "yolo11n-pose.pt"))


def participant_of(stem: str) -> str:
    m = re.search(r"_p(\d+)_", stem)
    return m.group(1) if m else "unknown"


def extract_video(path: Path) -> list[tuple[float, np.ndarray, np.ndarray, np.ndarray, np.ndarray | None]]:
    """Primer pase: corre YOLO-pose frame a frame (subsampleado a
    --fps-cap) y devuelve, para cada punto con ventana de esqueleto válida,
    (timestamp_s, window, kpts, box, kpts_conf) — todavía sin el embedding
    CLIP, que se calcula en un segundo pase solo sobre los puntos elegidos
    (ver main loop) para no cargar en memoria un frame completo por cada
    frame subsampleado de un video largo.
    """
    cap = cv2.VideoCapture(str(path))
    native_fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    stride = max(1, round(native_fps / args.fps_cap))

    buffer = SkeletonSequenceBuffer(window_s=args.window_s, n_frames=args.n_frames)
    candidates = []
    frame_idx = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if frame_idx % stride == 0:
            timestamp_s = frame_idx / native_fps
            det = detect_person(yolo, frame)
            if det is not None:
                kpts, box, kpts_conf = det
                h, w = frame.shape[:2]
                ps = build_pose_sample(kpts, box, w, h, timestamp_s)
                feat = frame_features(
                    kpts, kpts_conf, box, ps.aspect_ratio, ps.torso_angle_deg, ps.hip_y, ps.bbox_height_frac,
                )
                buffer.push(timestamp_s, feat)
                window = buffer.build_window(timestamp_s, min_coverage=args.min_coverage)
                if window is not None:
                    candidates.append((frame_idx, timestamp_s, window, kpts, box, kpts_conf))
        frame_idx += 1
    cap.release()
    return candidates


by_label: dict[str, list[Path]] = {}
for f in sorted(DATASET_DIR.glob("*.mp4")):
    label = f.stem.split("_")[0]
    if label in FINE_LABELS:
        by_label.setdefault(label, []).append(f)

rng = random.Random(args.seed)
rows_window, rows_body, rows_hand = [], [], []
rows_label, rows_participant, rows_video, rows_ts = [], [], [], []
n_skipped_no_window = 0

for label, files in sorted(by_label.items()):
    chosen_files = rng.sample(files, args.max_per_class) if len(files) > args.max_per_class else files
    print(f"{label:22s} {len(chosen_files)}/{len(files)} videos", flush=True)
    for f in chosen_files:
        candidates = extract_video(f)
        if not candidates:
            n_skipped_no_window += 1
            continue

        n = min(args.samples_per_video, len(candidates))
        chosen_idx = sorted(set(np.linspace(0, len(candidates) - 1, n).astype(int)))
        chosen = [candidates[i] for i in chosen_idx]

        cap2 = cv2.VideoCapture(str(f))
        for frame_idx, timestamp_s, window, kpts, box, kpts_conf in chosen:
            cap2.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ok, frame = cap2.read()
            if not ok:
                continue
            crop = crop_to_person(frame, box)
            hand_crop = crop_to_hands(frame, kpts, box, keypoints_conf=kpts_conf)
            body_feat = clip_clf.encode_image(crop)
            # Mismo fallback que ClipActivityClassifier.classify() y
            # extract_clip_features.py: sin crop de manos válido, se
            # duplica el body embedding (misma distribución de input que
            # ve el probe en entrenamiento y en producción).
            hand_feat = clip_clf.encode_image(hand_crop) if hand_crop is not None else body_feat

            rows_window.append(window)
            rows_body.append(body_feat)
            rows_hand.append(hand_feat)
            rows_label.append(label)
            rows_participant.append(participant_of(f.stem))
            rows_video.append(f.stem)
            rows_ts.append(timestamp_s)
        cap2.release()

if not rows_window:
    raise SystemExit(
        "Ningún video produjo una ventana de esqueleto válida — revisar "
        "--window-s/--fps-cap o el contenido de la carpeta."
    )

windows = np.stack(rows_window).astype(np.float32)
body_features = np.stack(rows_body).astype(np.float32)
hand_features = np.stack(rows_hand).astype(np.float32)
labels = np.array(rows_label)
participants = np.array(rows_participant)
videos = np.array(rows_video)
timestamps = np.array(rows_ts, dtype=np.float32)

np.savez(
    OUT_PATH,
    windows=windows,
    body_features=body_features,
    hand_features=hand_features,
    labels=labels,
    participants=participants,
    videos=videos,
    timestamps=timestamps,
    window_s=np.array(args.window_s),
    n_frames=np.array(args.n_frames),
)
print(f"\nGuardado {OUT_PATH}: {windows.shape[0]} ventanas, {windows.shape[1:]} shape (+ embeddings CLIP)")
if n_skipped_no_window:
    print(f"({n_skipped_no_window} videos sin ninguna ventana con cobertura temporal suficiente, ignorados)")
print("Por clase:", {lbl: int((labels == lbl).sum()) for lbl in sorted(set(labels))})
