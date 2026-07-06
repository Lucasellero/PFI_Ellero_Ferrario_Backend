"""Extiende `fall_descriptors_cache.pkl` con los clips nuevos que aparezcan en
`load_ground_truth()` (ej. tras sumar `videos/Kaggle/Fall.csv`) sin recalcular
los que ya están cacheados.

Uso:
    python extend_descriptor_cache.py
"""
from __future__ import annotations

import pickle
from pathlib import Path

from evaluate_falls import load_ground_truth
from recalibrate_fall_thresholds import extract_descriptors, CACHE_PATH
from ultralytics import YOLO


def main() -> None:
    cases = load_ground_truth()
    cache = pickle.loads(CACHE_PATH.read_bytes()) if CACHE_PATH.exists() else {}

    missing = [c for c in cases if c["video"] not in cache]
    print(f"{len(cases)} casos totales, {len(missing)} nuevos sin cachear")
    if not missing:
        return

    model = YOLO("yolo11n-pose.pt")
    for i, case in enumerate(missing):
        cache[case["video"]] = extract_descriptors(case["video"], model)
        print(f"  [{i + 1}/{len(missing)}] {case['video']}: {len(cache[case['video']])} frames con detección", flush=True)
        # Se guarda después de cada clip (no solo al final): si se corta el
        # proceso a mitad de camino, lo ya calculado no se pierde.
        CACHE_PATH.write_bytes(pickle.dumps(cache))

    print(f"Cache actualizado: {len(cache)} clips en total")


if __name__ == "__main__":
    main()
