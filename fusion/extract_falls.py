"""Extrae ventanas de keypoints etiquetadas para el clasificador temporal de
caída (`src/fall_temporal.py`), a partir de los `videos/**/Fall.csv` que ya
usa `vision/evaluate_falls.py`.

Un solo pase de YOLO-pose por video (subsampleado a `--fps-cap`, igual
criterio que `extract_dataset.py`), armando ventanas con
`SkeletonSequenceBuffer` en cada frame con detección. Cada ventana se
etiqueta según el segmento del `Fall.csv` en que cae su timestamp final:

  - dentro de un `Fall[start, end]`                    -> `caida_en_curso`
  - hasta `--post-fall-s` segundos después de un Fall  -> `caido`
  - dentro de un tramo explícito no-caída, o clip sin ninguna caída -> `normal`
  - resto (sin anotación that lo cubra)                -> se descarta

El parseo del CSV se reusa de `vision/evaluate_falls.py::parse_classes` (mismo
formato, mismas inconsistencias toleradas). Grupo para el split = el video
(nunca el mismo video en train y test — ver `train_fall_temporal.py`).

Cachea los candidatos crudos por video en `fall_windows_cache.pkl` (keyed por
ruta + mtime + parámetros de ventaneo) para no recomputar YOLO en cada
reejecución — mismo criterio que `vision/fall_descriptors_cache.pkl`.

Uso:
    python extract_falls.py <carpeta_videos> [--fps-cap 8] [--window-s 2.0]
        [--n-frames 16] [--post-fall-s 3.0] [--max-normal-per-video 40]
        [--only CAUCAFall,URFD] [--skip ToyotaSmarthome_FP] [-o fall_windows.npz]
"""
from __future__ import annotations

import argparse
import csv as csv_mod
import pickle
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "vision"))

import cv2
import numpy as np

from src.pose_pipeline import FallImmobilityDetector, build_pose_sample
from evaluate_falls import parse_classes  # de vision/

from fusion.src.keypoint_sequence import SkeletonSequenceBuffer, frame_features

parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument("videos_dir", help="Carpeta raíz con subcarpetas que tienen Fall.csv")
parser.add_argument("--fps-cap", type=float, default=8.0, help="FPS máximo al que se corre YOLO-pose")
parser.add_argument("--window-s", type=float, default=2.0)
parser.add_argument("--n-frames", type=int, default=16)
parser.add_argument("--min-coverage", type=float, default=0.6, help="Ver SkeletonSequenceBuffer.build_window")
parser.add_argument("--post-fall-s", type=float, default=3.0, help="Ventana tras el fin de un Fall que se etiqueta 'caido'")
parser.add_argument("--max-normal-per-video", type=int, default=40, help="Tope de ventanas 'normal' aleatorias por video (evita que los clips largos de ToyotaSmarthome_FP dominen)")
parser.add_argument("--mine-hard-negatives", action="store_true", help="Además de las 'normal' aleatorias, agregar como 'normal' las ventanas en los instantes donde la heurística geométrica de vision/ dispara un FP (tramo no-caída) — sin tope. Ataca el caso que el modelo no aprende a rechazar (sillón bajo = firma de caída).")
parser.add_argument("--only", default=None, help="Lista separada por comas de subcarpetas a incluir (default: todas)")
parser.add_argument("--skip", default=None, help="Lista separada por comas de subcarpetas a excluir")
parser.add_argument("--no-cache", action="store_true", help="Ignora fall_windows_cache.pkl y recomputa")
parser.add_argument("-o", "--output", default=None, help="Default: fusion/fall_windows.npz")
args = parser.parse_args()

VIDEOS_DIR = Path(args.videos_dir)
OUT_PATH = Path(args.output) if args.output else Path(__file__).parent / "fall_windows.npz"
CACHE_PATH = Path(__file__).parent / "fall_windows_cache.pkl"
ONLY = set(args.only.split(",")) if args.only else None
SKIP = set(args.skip.split(",")) if args.skip else set()

_cache: dict = {}
if CACHE_PATH.exists() and not args.no_cache:
    _cache = pickle.loads(CACHE_PATH.read_bytes())


def cache_key(path: Path) -> tuple:
    # "v2": el cache ahora guarda también los timestamps donde la heurística
    # geométrica dispara (para --mine-hard-negatives).
    return ("v2", str(path), round(path.stat().st_mtime, 1), args.fps_cap, args.window_s, args.n_frames, args.min_coverage)


def label_for(t: float, fall_segments, nonfall_segments) -> str | None:
    for start, end in fall_segments:
        if start <= t <= end:
            return "caida_en_curso"
    for start, end in fall_segments:
        if end < t <= end + args.post_fall_s:
            return "caido"
    for start, end in nonfall_segments:
        if start <= t <= end:
            return "normal"
    if not fall_segments and not nonfall_segments:
        return "normal"  # clip anotado NoFall de punta a punta (ToyotaSmarthome_FP)
    return None


def extract_video(path: Path) -> dict:
    """Devuelve {"candidates": [(t, window), ...], "fire_times": [t, ...]}.

    `fire_times` = instantes donde la heurística geométrica de
    `FallImmobilityDetector` dispara un evento de caída (para
    --mine-hard-negatives). Se reusa el YOLO del propio detector — no hay
    doble inferencia.
    """
    key = cache_key(path)
    if key in _cache:
        return _cache[key]

    cap = cv2.VideoCapture(str(path))
    native_fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    stride = max(1, round(native_fps / args.fps_cap))
    buffer = SkeletonSequenceBuffer(window_s=args.window_s, n_frames=args.n_frames)
    det = FallImmobilityDetector(model_path=str(ROOT / "yolo11n-pose.pt"))

    candidates: list[tuple[float, np.ndarray]] = []
    fire_times: list[float] = []
    frame_idx = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if frame_idx % stride == 0:
            t = frame_idx / native_fps
            evs = det.process_frame(frame, t)
            if any(e["event_type"] == "fall" for e in evs):
                fire_times.append(t)
            if det.last_detection is not None:
                kpts, box = det.last_detection
                kpts_conf = det.last_detection_conf
                h, w = frame.shape[:2]
                ps = build_pose_sample(kpts, box, w, h, t, kpts_conf=kpts_conf)
                feat = frame_features(
                    kpts, kpts_conf, box, ps.aspect_ratio, ps.torso_angle_deg, ps.hip_y, ps.bbox_height_frac,
                )
                buffer.push(t, feat)
                window = buffer.build_window(t, min_coverage=args.min_coverage)
                if window is not None:
                    candidates.append((t, window))
            else:
                buffer.reset()
        frame_idx += 1
    cap.release()

    _cache[key] = {"candidates": candidates, "fire_times": fire_times}
    return _cache[key]


def main() -> None:
    csv_paths = sorted(VIDEOS_DIR.glob("**/Fall.csv"))
    if not csv_paths:
        raise SystemExit(f"No se encontró ningún Fall.csv bajo {VIDEOS_DIR}")

    rows_window, rows_label, rows_video, rows_ts, rows_dataset = [], [], [], [], []
    rows_hard: list[int] = []   # índices de ventanas que son hard negatives minados
    rng = np.random.default_rng(7)

    for csv_path in csv_paths:
        # dataset = nombre de la subcarpeta de primer nivel bajo videos/
        # (ej. "GMDCSA-24" para videos/GMDCSA-24/fall1/Fall.csv).
        try:
            dataset = csv_path.relative_to(VIDEOS_DIR).parts[0]
        except ValueError:
            dataset = csv_path.parent.name
        if ONLY is not None and dataset not in ONLY:
            continue
        if dataset in SKIP:
            continue

        folder = csv_path.parent
        for row in csv_mod.DictReader(open(csv_path, encoding="utf-8-sig")):
            fname = row["File Name"]
            video = folder / fname
            if not video.exists():
                print(f"  [aviso] falta {video}, se ignora")
                continue
            classes_field = row.get(" Classes") or row.get("Classes") or ""
            segments = parse_classes(classes_field, video_id=f"{dataset}/{fname}")
            fall_segments = [(s[1], s[2]) for s in segments if s[3]]
            nonfall_segments = [(s[1], s[2]) for s in segments if not s[3]]

            res = extract_video(video)
            candidates, fire_times = res["candidates"], res["fire_times"]
            per_video_normal = []
            n_pos = 0
            for t, window in candidates:
                lbl = label_for(t, fall_segments, nonfall_segments)
                if lbl is None:
                    continue
                if lbl == "normal":
                    per_video_normal.append((t, window))
                    continue
                rows_window.append(window); rows_label.append(lbl)
                rows_video.append(video.stem); rows_ts.append(t); rows_dataset.append(dataset)
                n_pos += 1

            # Subsamplear los 'normal' aleatorios de este video al tope configurado.
            if len(per_video_normal) > args.max_normal_per_video:
                idx = rng.choice(len(per_video_normal), args.max_normal_per_video, replace=False)
                per_video_normal = [per_video_normal[i] for i in sorted(idx)]
            for t, window in per_video_normal:
                rows_window.append(window); rows_label.append("normal")
                rows_video.append(video.stem); rows_ts.append(t); rows_dataset.append(dataset)

            # Hard negatives: ventana en cada instante donde la heurística
            # geométrica dispara DENTRO de un tramo no-caída (o clip NoFall
            # entero). Es el caso que el modelo no aprende con muestreo
            # aleatorio: la geometría dispara justo en el momento más
            # parecido a una caída (bbox colapsa + pico de velocidad al
            # dejarse caer en un sillón bajo).
            n_hard = 0
            if args.mine_hard_negatives and candidates:
                cand_ts = np.array([t for t, _ in candidates])
                for ft in fire_times:
                    if label_for(ft, fall_segments, nonfall_segments) != "normal":
                        continue  # el disparo cae en un tramo de caída real: no es hard negative
                    j = int(np.abs(cand_ts - ft).argmin())
                    if abs(cand_ts[j] - ft) > args.window_s:
                        continue  # sin ventana cercana al disparo
                    t, window = candidates[j]
                    rows_window.append(window); rows_label.append("normal")
                    rows_video.append(video.stem); rows_ts.append(t); rows_dataset.append(dataset)
                    rows_hard.append(len(rows_window) - 1)
                    n_hard += 1

            hn = f" (+{n_hard} hard-neg)" if n_hard else ""
            print(f"{dataset}/{fname}: {n_pos} caída, {len(per_video_normal)} normal{hn}", flush=True)

    if not args.no_cache:
        CACHE_PATH.write_bytes(pickle.dumps(_cache))

    if not rows_window:
        raise SystemExit("Ninguna ventana etiquetada — revisar --window-s/--fps-cap/--only.")

    windows = np.stack(rows_window).astype(np.float32)
    labels = np.array(rows_label)
    videos = np.array(rows_video)
    datasets = np.array(rows_dataset)
    timestamps = np.array(rows_ts, dtype=np.float32)
    is_hard_negative = np.zeros(len(labels), dtype=bool)
    if rows_hard:
        is_hard_negative[np.array(rows_hard)] = True

    np.savez(
        OUT_PATH,
        windows=windows,
        labels=labels,
        videos=videos,
        datasets=datasets,
        timestamps=timestamps,
        is_hard_negative=is_hard_negative,
        window_s=np.array(args.window_s),
        n_frames=np.array(args.n_frames),
    )
    print(f"\nGuardado {OUT_PATH}: {windows.shape[0]} ventanas {windows.shape[1:]}")
    print("Por clase (fina):", {lbl: int((labels == lbl).sum()) for lbl in sorted(set(labels))})
    print(f"Hard negatives minados: {int(is_hard_negative.sum())}")
    print("Por dataset:", {d: int((datasets == d).sum()) for d in sorted(set(datasets))})


if __name__ == "__main__":
    main()
