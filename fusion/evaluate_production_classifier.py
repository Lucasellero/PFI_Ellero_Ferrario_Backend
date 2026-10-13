"""Matriz de confusión del clasificador de actividades tal como lo arma `runner/`, sobre videos
que no se usaron para entrenarlo.

`evaluate_hybrid.py` compara los tres enfoques sobre un cache de features y con las categorías
del dataset local (4). Este script mide otra cosa: la decisión que toma el runner en una ventana
(`runner/camera_worker.py::_finish_clip_window`), con los modelos que hay en esta máquina y con
el contexto de zona, sobre el dataset completo de Toyota Smarthome.

- **Modelos**: los que carga `runner/main.py`: `ClipActivityClassifier(quantized=True)` (el
  probe de 6 categorías con zona, `vlm/models/category_probe.npz`) y, si está exportado,
  `SkeletonActivityClassifier(quantized=True)` fusionado con `ScoreFuser(skeleton_weight=0.5)`.
  Si el modelo de esqueleto no está (`fusion/models/` no se versiona), el runner cae a CLIP
  solo y este script mide eso: el modo queda escrito en el archivo de salida (`modo`).
- **Zonas**: `fusion/toyota_smarthome_zones.yaml`, por la cámara del nombre de archivo (`_cNN`).
- **Holdout**: los videos del dataset cuyo nombre no está en el cache con el que se entrenó el
  probe (`--train-features`, por defecto `vlm/clip_features_full_zone.npz`). Son videos no
  vistos, pero de los mismos participantes y del mismo departamento: el probe final se entrenó
  con todos los participantes. Las clases finas con 120 videos o menos entraron enteras al
  entrenamiento y no tienen ninguno acá. De cada clase fina se toman hasta `--max-per-class`.
- **Decisión por video**: un video es una ventana. Se toman 3 muestras (como
  `_CLIP_SAMPLES_PER_WINDOW`) al 20 %, 50 % y 80 % del clip — los clips son recortes de 1 a
  pocos segundos, no entran 3 muestras separadas 1 s —, se promedian las categorías y se aplica
  lo mismo que el runner: `Background` o confianza menor que `clip_min_confidence` (0,4) no
  emite nada; `EatingDrinkingMedication` con menos de 0,55 pasa a la segunda categoría si esa
  llega a 0,4. Con esqueleto, las ventanas se arman leyendo el clip entero a `--fps-cap`.

Guarda un JSON con las cuentas (real \\ predicho): `matriz` (solo las ventanas que emiten
evento, 6 x 6; es la que lee `baseline/evaluate_production_vocab.py --matriz`), `sin_evento`
(por categoría real, cuántas ventanas no emiten nada) y `sin_persona` (videos donde YOLO no
detectó a nadie en ninguna muestra).

Uso:
    python evaluate_production_classifier.py <carpeta con los .mp4 de Toyota Smarthome>
                                             [--max-per-class 40] [-o confusion_holdout_produccion.json]
"""
from __future__ import annotations

import argparse
import json
import random
import re
import sys
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "vision"))

import cv2
import numpy as np
import yaml
from ultralytics import YOLO

from src.pose_pipeline import build_pose_sample, detect_person
from vlm.src.clip_classifier import LABEL_TO_CATEGORY, ClipActivityClassifier, crop_to_hands, crop_to_person

from fusion.src.fusion import HybridActivityClassifier, ScoreFuser
from fusion.src.keypoint_sequence import SkeletonSequenceBuffer, frame_features
from fusion.src.zone_context import zone_one_hot

CATEGORIES = ("Cooking", "EatingDrinkingMedication", "Moving", "Phone", "ReadingScreen", "Resting")
# Los mismos valores que `runner/`: `_CLIP_SAMPLES_PER_WINDOW`, `clip_min_confidence`,
# `_EATING_CONFIRM_THRESHOLD` y el `skeleton_weight` por defecto.
SAMPLES_PER_WINDOW = 3
MIN_CONFIDENCE = 0.4
EATING_CONFIRM_THRESHOLD = 0.55
SKELETON_WEIGHT = 0.5
SAMPLE_POSITIONS = (0.2, 0.5, 0.8)


def camera_of(stem: str) -> str | None:
    m = re.search(r"_c(\d+)$", stem)
    return m.group(1) if m else None


def build_classifier() -> HybridActivityClassifier:
    """El mismo armado que `CameraWorker.__init__`: híbrido si hay modelo de esqueleto, CLIP solo si no."""
    clip = ClipActivityClassifier(quantized=True)
    try:
        from fusion.src.temporal_classifier import SkeletonActivityClassifier
        skeleton = SkeletonActivityClassifier(quantized=True)
    except FileNotFoundError as e:
        print(f"Modelo de esqueleto no encontrado ({e}): el runner corre CLIP solo, se mide eso.")
        return HybridActivityClassifier(mode="clip_only", clip_classifier=clip)
    return HybridActivityClassifier(
        mode="hybrid", skeleton_classifier=skeleton, clip_classifier=clip, fuser=ScoreFuser(skeleton_weight=SKELETON_WEIGHT),
    )


def decide(cat_scores_por_muestra: list[dict]) -> str | None:
    """`_finish_clip_window`: la categoría que emitiría el runner, o None si no emite nada."""
    if not cat_scores_por_muestra:
        return None
    n = len(cat_scores_por_muestra)
    cats = {c for s in cat_scores_por_muestra for c in s}
    scores = {c: sum(s.get(c, 0.0) for s in cat_scores_por_muestra) / n for c in cats}
    top = max(scores, key=scores.get)
    if top == "Background" or scores[top] < MIN_CONFIDENCE:
        return None
    if top == "EatingDrinkingMedication" and scores[top] < EATING_CONFIRM_THRESHOLD:
        segunda = max((c for c in scores if c not in ("Background", top)), key=scores.get, default=None)
        if segunda is not None and scores[segunda] >= MIN_CONFIDENCE:
            top = segunda
    return top


def classify_video(path: Path, clf: HybridActivityClassifier, yolo: YOLO, zones: dict | None, fps_cap: float) -> list[dict]:
    """Las `category_scores` de cada muestra del video en la que se detectó a una persona."""
    cap = cv2.VideoCapture(str(path))
    nframes = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    objetivos = sorted({int(round(p * max(nframes - 1, 0))) for p in SAMPLE_POSITIONS})
    con_esqueleto = clf.mode == "hybrid"
    stride = max(1, round(fps / fps_cap))
    buffer = SkeletonSequenceBuffer() if con_esqueleto else None

    muestras: list[dict] = []
    frame_idx = 0
    while True:
        if not con_esqueleto:
            # Sin esqueleto no hace falta el historial: se salta directo a los frames de muestra.
            if not objetivos:
                break
            frame_idx = objetivos.pop(0)
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ok, frame = cap.read()
        if not ok:
            break
        es_muestra = (not con_esqueleto) or frame_idx in objetivos
        if es_muestra or frame_idx % stride == 0:
            ts = frame_idx / fps
            det = detect_person(yolo, frame)
            h, w = frame.shape[:2]
            if det is None:
                if buffer is not None:
                    buffer.reset()
            else:
                kpts, box, kpts_conf = det
                if buffer is not None:
                    ps = build_pose_sample(kpts, box, w, h, ts, kpts_conf=kpts_conf)
                    buffer.push(ts, frame_features(
                        kpts, kpts_conf, box, ps.aspect_ratio, ps.torso_angle_deg, ps.hip_y, ps.bbox_height_frac,
                    ))
                if es_muestra:
                    zone_vector = None
                    if zones:
                        hip = (kpts[11] + kpts[12]) / 2.0
                        zone_vector = zone_one_hot(zones, (float(hip[0] / w), float(hip[1] / h)))
                    window = buffer.build_window(ts) if buffer is not None else None
                    fused = clf.classify(
                        window, crop_to_person(frame, box), crop_to_hands(frame, kpts, box, keypoints_conf=kpts_conf), zone_vector,
                    )
                    if fused["category_scores"]:
                        muestras.append(fused["category_scores"])
        frame_idx += 1
    cap.release()
    return muestras


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("dataset_dir", type=Path, help="Carpeta con videos <Actividad>_pXX_..._cNN.mp4")
    ap.add_argument("--train-features", type=Path, default=ROOT / "vlm" / "clip_features_full_zone.npz",
                    help="Cache con el que se entrenó el probe: sus videos quedan fuera del holdout")
    ap.add_argument("--zones", type=Path, default=Path(__file__).parent / "toyota_smarthome_zones.yaml")
    ap.add_argument("--max-per-class", type=int, default=40, help="Tope de videos por clase fina")
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--fps-cap", type=float, default=10.0, help="Solo con esqueleto: FPS al que se arma la ventana")
    ap.add_argument("-o", "--output", type=Path, default=Path(__file__).parent / "confusion_holdout_produccion.json")
    args = ap.parse_args()

    vistos = set(np.unique(np.load(args.train_features, allow_pickle=True)["videos"]).tolist())
    zones_by_camera = yaml.safe_load(args.zones.read_text(encoding="utf-8"))
    por_clase: dict[str, list[Path]] = {}
    total_por_clase: dict[str, int] = {}
    for f in sorted(args.dataset_dir.glob("**/*.mp4")):
        label = f.stem.split("_")[0]
        if LABEL_TO_CATEGORY.get(label) in CATEGORIES:
            total_por_clase[label] = total_por_clase.get(label, 0) + 1
            if f.stem not in vistos:
                por_clase.setdefault(label, []).append(f)

    clf = build_classifier()
    yolo = YOLO(str(ROOT / "yolo11n-pose.pt"))
    rng = random.Random(args.seed)
    idx = {c: i for i, c in enumerate(CATEGORIES)}
    matriz = np.zeros((len(CATEGORIES), len(CATEGORIES)), dtype=int)
    sin_evento = np.zeros(len(CATEGORIES), dtype=int)
    sin_persona = np.zeros(len(CATEGORIES), dtype=int)
    videos_por_clase: dict[str, int] = {}
    filas = []

    for label in sorted(total_por_clase):
        files = por_clase.get(label, [])
        elegidos = rng.sample(files, args.max_per_class) if len(files) > args.max_per_class else files
        videos_por_clase[label] = len(elegidos)
        print(f"{label:22s} {len(elegidos):3d} de {len(files)} no vistos ({total_por_clase[label]} en el dataset)", flush=True)
        real = LABEL_TO_CATEGORY[label]
        for f in elegidos:
            muestras = classify_video(f, clf, yolo, zones_by_camera.get(camera_of(f.stem)), args.fps_cap)
            predicha = decide(muestras)
            if not muestras:
                sin_persona[idx[real]] += 1
            if predicha is None:
                sin_evento[idx[real]] += 1
            else:
                matriz[idx[real], idx[predicha]] += 1
            filas.append({"video": f.stem, "clase": label, "real": real, "predicha": predicha, "muestras": len(muestras)})

    emitidos = matriz.sum(axis=1)
    print(f"\nModo: {clf.mode}. Videos: {int(emitidos.sum() + sin_evento.sum())}  (emiten evento: {int(emitidos.sum())})")
    print("Real \\ Predicho".ljust(26) + "".join(c[:10].ljust(12) for c in CATEGORIES) + "sin evento")
    for i, c in enumerate(CATEGORIES):
        print(c.ljust(26) + "".join(f"{matriz[i, j]}".ljust(12) for j in range(len(CATEGORIES))) + f"{sin_evento[i]}")
    print("\nPorcentaje sobre las ventanas que emiten evento:")
    for i, c in enumerate(CATEGORIES):
        fila = matriz[i] / max(emitidos[i], 1) * 100
        print(c.ljust(26) + "".join(f"{v:.0f} %".ljust(12) for v in fila))
    print(f"\nAciertos sobre las que emiten: {np.trace(matriz) / max(emitidos.sum(), 1):.1%}"
          f"   sobre todos los videos: {np.trace(matriz) / max(emitidos.sum() + sin_evento.sum(), 1):.1%}")

    args.output.write_text(json.dumps({
        "fecha": date.today().isoformat(),
        "modo": clf.mode,
        "descripcion": "Cuentas real \\ predicho por video (una ventana de 3 muestras por video), decisión de runner/camera_worker.py::_finish_clip_window",
        "dataset": str(args.dataset_dir),
        "holdout": f"videos que no están en {args.train_features.name}; hasta {args.max_per_class} por clase fina, semilla {args.seed}",
        "zonas": args.zones.name,
        "categorias": list(CATEGORIES),
        "matriz": matriz.tolist(),
        "sin_evento": sin_evento.tolist(),
        "sin_persona": sin_persona.tolist(),
        "videos_por_clase_fina": videos_por_clase,
        "videos": filas,
    }, indent=1, ensure_ascii=False), encoding="utf-8")
    print(f"Guardado: {args.output}")


if __name__ == "__main__":
    main()
