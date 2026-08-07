"""Recalibra FALL_VERTICAL_VELOCITY (y confirma FALL_ASPECT_RATIO) tras el
cambio a velocidad normalizada por escala de persona (ver pose_pipeline.py).

Corre YOLO-pose una sola vez por clip (GMDCSA-24 + CAUCAFall, incluyendo los
clips 100% sin caída) y cachea los descriptores por frame en un .pkl, para
poder barrer decenas de combinaciones de umbrales en segundos en vez de
volver a correr inferencia por cada candidato.

Uso:
    python recalibrate_fall_thresholds.py            # usa cache si existe
    python recalibrate_fall_thresholds.py --rebuild   # fuerza recomputar descriptores
"""
from __future__ import annotations

import os

os.environ.setdefault("OPENCV_FFMPEG_LOGLEVEL", "-8")  # silencia el spam de FFmpeg en .avi corruptos

import pickle
import sys
from pathlib import Path

import cv2

from evaluate_falls import load_ground_truth
from src.pose_pipeline import build_pose_sample, detect_person
from ultralytics import YOLO

CACHE_PATH = Path("fall_descriptors_cache.pkl")

FALL_VELOCITY_WINDOW_S = 0.4
FALL_CONFIRM_WINDOW_S = 1.5
FALL_COOLDOWN_S = 3.0
MAX_FRAMES_PER_CLIP = 10_000  # tope de seguridad: un .avi corrupto no debe poder colgar la extracción


def extract_descriptors(video_path: str, model: YOLO) -> list[tuple[float, float, float]]:
    """Devuelve [(timestamp_s, hip_y, bbox_height_frac), ...] para un clip."""
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    out = []
    frame_i = 0
    while frame_i < MAX_FRAMES_PER_CLIP:
        ok, image = cap.read()
        if not ok:
            break
        t = frame_i / fps
        h, w = image.shape[:2]
        detection = detect_person(model, image)
        if detection is not None:
            kpts, box = detection
            sample = build_pose_sample(kpts, box, w, h, t)
            out.append((sample.timestamp_s, sample.hip_y, sample.bbox_height_frac, sample.aspect_ratio))
        frame_i += 1
    cap.release()
    return out


def build_cache(cases: list[dict]) -> dict[str, list]:
    model = YOLO("yolo11n-pose.pt")
    model.to("cuda")
    cache = {}
    for i, case in enumerate(cases):
        cache[case["video"]] = extract_descriptors(case["video"], model)
        print(f"  [{i + 1}/{len(cases)}] {case['video']}: {len(cache[case['video']])} frames con detección", flush=True)
    return cache


def simulate(descriptors: list[tuple[float, float, float, float]], velocity_threshold: float, aspect_threshold: float) -> list[float]:
    """Reproduce la lógica de dos fases de FallImmobilityDetector._detect_fall
    sobre descriptores ya calculados. Devuelve los timestamps de los eventos 'fall'."""
    history: list[tuple[float, float, float]] = []  # (t, hip_y, bbox_height_frac)
    last_velocity_spike_t: float | None = None
    cooldown_until = float("-inf")
    events = []

    for t, hip_y, bbox_height_frac, aspect_ratio in descriptors:
        history.append((t, hip_y, bbox_height_frac))
        history[:] = [s for s in history if t - s[0] <= 1.5]

        if t >= cooldown_until:
            target_t = t - FALL_VELOCITY_WINDOW_S
            ref = min(history, key=lambda s: abs(s[0] - target_t))
            dt = t - ref[0]
            if dt >= FALL_VELOCITY_WINDOW_S / 2:
                scale = (bbox_height_frac + ref[2]) / 2
                if scale > 1e-6:
                    velocity = (hip_y - ref[1]) / scale / dt
                    if velocity > velocity_threshold:
                        last_velocity_spike_t = t

            if (
                last_velocity_spike_t is not None
                and aspect_ratio > aspect_threshold
                and t - last_velocity_spike_t <= FALL_CONFIRM_WINDOW_S
            ):
                last_velocity_spike_t = None
                cooldown_until = t + FALL_COOLDOWN_S
                events.append(t)

    return events


def score(cases: list[dict], cache: dict[str, list], velocity_threshold: float, aspect_threshold: float) -> tuple[int, int, int]:
    """Devuelve (true_positives, false_negatives, false_positives) sobre todos los casos."""
    tp = fn = fp = 0
    for case in cases:
        event_times = simulate(cache[case["video"]], velocity_threshold, aspect_threshold)

        if case["fall_segments"]:
            hit = any(start <= t <= end for t in event_times for start, end in case["fall_segments"])
            if hit:
                tp += 1
            else:
                fn += 1

        for t in event_times:
            if any(start <= t <= end for start, end in case["nonfall_segments"]):
                fp += 1
                break  # 1 FP por clip alcanza para el ranking del barrido

    return tp, fn, fp


def main() -> None:
    cases = load_ground_truth()
    n_fall_cases = sum(1 for c in cases if c["fall_segments"])

    if CACHE_PATH.exists() and "--rebuild" not in sys.argv:
        print(f"Usando cache existente: {CACHE_PATH}")
        cache = pickle.loads(CACHE_PATH.read_bytes())
    else:
        print(f"Extrayendo descriptores de {len(cases)} clips (esto corre YOLO una vez por clip)...")
        cache = build_cache(cases)
        CACHE_PATH.write_bytes(pickle.dumps(cache))
        print(f"Cache guardado en {CACHE_PATH}")

    print(f"\n{n_fall_cases} casos con caída, {len(cases) - n_fall_cases} clips sin caída\n")
    print(f"{'vel_thr':>8} {'asp_thr':>8} {'recall':>8} {'FP':>6}")

    velocity_candidates = [round(0.02 + 0.01 * i, 3) for i in range(14)]  # 0.02 .. 0.15
    aspect_candidates = [round(1.0 + 0.05 * i, 2) for i in range(13)]  # 1.0 .. 1.6 (cubre el umbral desplegado, 1.4)

    results = []
    for vt in velocity_candidates:
        for at in aspect_candidates:
            tp, fn, fp = score(cases, cache, vt, at)
            recall = tp / n_fall_cases if n_fall_cases else 0.0
            results.append((vt, at, recall, fp, tp, fn))

    results.sort(key=lambda r: (-r[2], r[3]))  # más recall primero, después menos FP
    for vt, at, recall, fp, tp, fn in results[:30]:
        print(f"{vt:>8} {at:>8} {recall * 100:>7.1f}% {fp:>6}")

    best = results[0]
    print(f"\nMejor combinación: FALL_VERTICAL_VELOCITY={best[0]}, FALL_ASPECT_RATIO={best[1]} "
          f"-> recall {best[2]*100:.1f}% ({best[4]}/{n_fall_cases}), FP {best[3]}")


if __name__ == "__main__":
    main()
