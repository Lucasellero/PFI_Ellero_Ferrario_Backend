"""Por qué histéresis y no "bajar el umbral" para el bloqueante de persona
acostada (ver vision/README.md, "Histéresis de detección", 2026-10-04).

Corre YOLO una sola vez por frame con umbral de caja muy bajo (0.02), guarda
todas las cajas crudas y después compara offline, sobre las mismas cajas:

  1. Umbral GLOBAL de caja (0.25 = actual, hasta 0.03): % de frames con
     alguna caja en clips de alguien acostándose/acostado, contra cuántas
     cajas espurias deja pasar en clips de control. Caja espuria = caja sobre
     el umbral que no solapa (IoU < 0.1) con la caja principal de alta
     confianza (>= 0.5) del mismo frame; los clips son de un solo residente,
     así que cualquier otra caja es ruido.
  2. HISTÉRESIS (lo integrado en `FallImmobilityDetector._detect`): una
     detección normal de confianza alta fija un "ancla"; el umbral bajo solo
     sirve para mantener el seguimiento si la caja débil solapa con el ancla.
     No puede crear cajas espurias nuevas por construcción; el riesgo es otro,
     un seguimiento "fantasma" que sobrevive colgado de un objeto cuando la
     persona ya no está. Se mide de forma adversarial: se toma el ancla al
     final de un clip A y se le dan SOLO las cajas débiles de otro clip B (la
     persona "desapareció").

Con un segundo modelo más grande como referencia (--ref yolo11m-pose.pt)
reporta además cuántas cajas débiles aceptadas coinciden (IoU >= 0.3) con
donde ese modelo ve a la persona. Sin esa columna, el % de frames "con
persona" no dice si la caja está sobre la persona o sobre un mueble: la
primera versión de la histéresis (crear a 0.25, encadenar débil con débil)
daba 82% y en el clip más largo era un seguimiento colgado de una planta.

No es parte del pipeline productivo.

Uso:
    python sweep_detection_hysteresis.py [--ref yolo11m-pose.pt]
"""
from __future__ import annotations

import argparse
import glob
import itertools
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO

from src.pose_pipeline import (
    MIN_KEYPOINT_CONFIDENCE,
    PERSON_BOX_CONFIDENCE,
    TRACK_CREATE_BOX_CONFIDENCE,
    TRACK_KEEP_BOX_CONFIDENCE,
    TRACK_KEEP_MAX_GAP_S,
    TRACK_KEEP_MIN_IOU,
    _iou,
)

_ROOT = Path(__file__).resolve().parents[1]
SAMPLE_EVERY = 4  # clips a 20fps => 5fps
DUMP_CONFIDENCE = 0.02
GLOBAL_THRESHOLDS = [0.25, 0.15, 0.10, 0.05, 0.03]
GHOST_TRIALS = 400


def clip_sets() -> dict[str, list[str]]:
    videos = _ROOT / "videos"
    return {
        "acostado": sorted(glob.glob(str(videos / "acostado" / "*.mp4"))) + sorted(glob.glob(str(videos / "acostarse" / "*.mp4"))),
        "tele": sorted(glob.glob(str(videos / "mirar tele" / "*.mp4")))[:40],
        "caminar": sorted(glob.glob(str(videos / "caminar" / "*.mp4")))[:60],
    }


def dump_boxes(model: YOLO, path: str) -> tuple[float, list[tuple[np.ndarray, np.ndarray, np.ndarray]]]:
    """Devuelve (segundos entre muestras, [(cajas xyxy, confianza de caja,
    confianza promedio de keypoints), ...])."""
    cap = cv2.VideoCapture(path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 20.0
    frames = []
    i = 0
    while True:
        ok, image = cap.read()
        if not ok:
            break
        if i % SAMPLE_EVERY == 0:
            result = model(image, verbose=False, conf=DUMP_CONFIDENCE)[0]
            boxes = result.boxes.xyxy.cpu().numpy()
            kpts_conf = result.keypoints.conf.cpu().numpy().mean(axis=1) if len(boxes) else np.zeros(0)
            frames.append((boxes, result.boxes.conf.cpu().numpy(), kpts_conf))
        i += 1
    cap.release()
    return SAMPLE_EVERY / fps, frames


def simulate(frames, dt, low, iou_min, max_gap, anchor=None, only_weak=False):
    """Misma regla que `FallImmobilityDetector._detect` (sin el tope
    TRACK_WEAK_MAX_S, que acá no llega a actuar: los clips son más cortos).
    Devuelve (por frame: (0 nada / 1 normal / 2 débil, caja aceptada), ancla)."""
    out = []
    last_t = 0.0 if anchor is not None else float("-inf")
    for k, (boxes, conf, kpts_conf) in enumerate(frames):
        t = k * dt
        state, accepted = 0, None
        strong = [] if only_weak else np.flatnonzero(conf >= PERSON_BOX_CONFIDENCE)
        normal = None
        if len(strong):
            areas = (boxes[strong, 2] - boxes[strong, 0]) * (boxes[strong, 3] - boxes[strong, 1])
            j = int(strong[areas.argmax()])
            if kpts_conf[j] >= MIN_KEYPOINT_CONFIDENCE:
                normal = j
        if normal is not None:
            state, accepted = 1, boxes[normal]
            if conf[normal] >= TRACK_CREATE_BOX_CONFIDENCE:
                anchor, last_t = boxes[normal], t
            elif anchor is not None and _iou(boxes[normal], anchor) >= iou_min:
                last_t = t
        elif low is not None and anchor is not None and t - last_t <= max_gap:
            cand = [j for j in range(len(conf)) if conf[j] >= low and not (only_weak and conf[j] >= PERSON_BOX_CONFIDENCE)]
            overlaps = [_iou(boxes[j], anchor) for j in cand]
            if overlaps and max(overlaps) >= iou_min:
                state, accepted, last_t = 2, boxes[cand[int(np.argmax(overlaps))]], t
        out.append((state, accepted))
    return out, anchor


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=str(_ROOT / "yolo11n-pose.pt"))
    parser.add_argument("--ref", default=None, help="modelo de pose más grande, usado como referencia de dónde está la persona")
    args = parser.parse_args()
    model = YOLO(args.model)
    ref_model = YOLO(args.ref) if args.ref else None

    data: dict[str, list[tuple[float, list]]] = {}
    ref_boxes: list[list[np.ndarray]] = []  # por clip acostado, por frame: cajas >= 0.25 del modelo de referencia
    for name, files in clip_sets().items():
        data[name] = [dump_boxes(model, f) for f in files]
        print(f"{name}: {len(files)} clips, {sum(len(fr) for _, fr in data[name])} frames muestreados")
        if name == "acostado" and ref_model is not None:
            for f in files:
                _, frames = dump_boxes(ref_model, f)
                ref_boxes.append([boxes[conf >= PERSON_BOX_CONFIDENCE] for boxes, conf, _ in frames])

    print(f"\n=== 1. Umbral global de caja ({Path(args.model).name}) ===")
    print("umbral | acostado: frames c/alguna caja | espurias cada 1000 frames (tele / caminar)")
    for th in GLOBAL_THRESHOLDS:
        n = hit = 0
        for _, frames in data["acostado"]:
            for _, conf, _ in frames:
                n += 1
                hit += bool(len(conf) and conf.max() >= th)
        spurious = []
        for name in ("tele", "caminar"):
            ref = extra = 0
            for _, frames in data[name]:
                for boxes, conf, _ in frames:
                    if not len(conf) or conf.max() < 0.5:
                        continue
                    ref += 1
                    main_box = boxes[conf.argmax()]
                    extra += sum(1 for b, c in zip(boxes, conf) if th <= c < conf.max() and _iou(b, main_box) < 0.1)
            spurious.append(1000 * extra / max(ref, 1))
        print(f" {th:.2f}  | {hit / n:6.1%}                        | {spurious[0]:6.1f} / {spurious[1]:6.1f}")

    print(f"\n=== 2. Histéresis (ancla: detección normal con caja >= {TRACK_CREATE_BOX_CONFIDENCE}; mantener: caja >= low con IoU >= iou contra el ancla) ===")
    print(" low   iou  gap | acostado: normal + débil = total | débiles que coinciden con la referencia | fantasma: p90 / máx (s), % > 5s")
    rng = np.random.default_rng(0)
    all_clips = [c for clips in data.values() for c in clips]
    others = data["tele"] + data["caminar"]
    grid = [(None, 0.0, 0.0)] + list(itertools.product([0.10, 0.05, 0.03], [0.3, 0.5], [3.0, 15.0]))
    for low, iou_min, max_gap in grid:
        total = normal = weak = agree = verifiable = 0
        for c, (dt, frames) in enumerate(data["acostado"]):
            out, _ = simulate(frames, dt, low, iou_min, max_gap)
            total += len(out)
            normal += sum(1 for s, _ in out if s == 1)
            weak += sum(1 for s, _ in out if s == 2)
            if ref_boxes:
                for (state, box), ref in zip(out, ref_boxes[c]):
                    if state == 2 and len(ref):
                        verifiable += 1
                        agree += max(_iou(box, r) for r in ref) >= 0.3
        ghost = [0.0]
        if low is not None:
            ghost = []
            for _ in range(GHOST_TRIALS):
                dt_a, a = all_clips[rng.integers(len(all_clips))]
                dt_b, b = others[rng.integers(len(others))]
                _, anchor = simulate(a, dt_a, low, iou_min, max_gap)
                if anchor is None:
                    continue
                out, _ = simulate(b, dt_b, low, iou_min, max_gap, anchor=anchor, only_weak=True)
                ghost.append((max((k for k, (s, _) in enumerate(out) if s == 2), default=-1) + 1) * dt_b)
        ghost = np.array(ghost)
        chosen = (low, iou_min, max_gap) == (TRACK_KEEP_BOX_CONFIDENCE, TRACK_KEEP_MIN_IOU, TRACK_KEEP_MAX_GAP_S)
        label = "actual (sin)   " if low is None else f"{low:.2f}  {iou_min:.1f}  {max_gap:3.0f}"
        ref_col = f"{agree:4d}/{verifiable:<4d} = {agree / verifiable:4.0%}" if verifiable else "      -        "
        print(
            f"{label} | {normal / total:6.1%} + {weak / total:6.1%} = {(normal + weak) / total:6.1%}"
            f" | {ref_col}"
            f" | {np.percentile(ghost, 90):5.1f} / {ghost.max():5.1f}, {(ghost > 5).mean():5.1%}"
            + ("   <- integrado" if chosen else "")
        )


if __name__ == "__main__":
    main()
