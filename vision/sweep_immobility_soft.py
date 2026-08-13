"""Barrido acotado para el modo `immobility_soft` de `simulate()`
(vision/recalibrate_fall_thresholds.py) — la idea pendiente del README
(2026-08-11/12): en vez de cancelar el candidato de inmovilidad apenas UN
frame se mueve más que el umbral ("strict", ya probado y descartado: 27.8%
recall con 2s), promediar el desvío en toda la ventana antes de decidir,
para tolerar jitter puntual de keypoints sin dejar de exigir quietud
sostenida.

Usa el mismo cache de descriptores que recalibrate_fall_thresholds.py
(fall_descriptors_cache.pkl) — no vuelve a correr YOLO.

Uso:
    vision/.venv/Scripts/python.exe sweep_immobility_soft.py
"""
from __future__ import annotations

import pickle

from evaluate_falls import load_ground_truth
from recalibrate_fall_thresholds import CACHE_PATH, score

cases = load_ground_truth()
n_fall_cases = sum(1 for c in cases if c["fall_segments"])
cache = pickle.loads(CACHE_PATH.read_bytes())
print(f"{n_fall_cases} casos con caída, {len(cases) - n_fall_cases} clips sin caída\n")

VEL, ASP = 0.12, 1.4

baseline_tp, baseline_fn, baseline_fp = score(cases, cache, VEL, ASP)
baseline_recall = baseline_tp / n_fall_cases
print(f"Baseline (sin inmovilidad): recall {baseline_recall:.1%} ({baseline_tp}/{n_fall_cases}), FP {baseline_fp}")
print(f"Referencia — strict a 2.0s (ya medido, recalibrate_run7.log): recall 27.8%, FP 58\n")

immobility_candidates = [0.5, 1.0, 1.5, 2.0, 3.0]
hip_thresh_candidates = [0.05, 0.08, 0.12]
height_ratio_candidates = [0.15, 0.25, 0.35]

print("=== Modo soft (promedio de desvío en toda la ventana, sin height_drop) ===")
print(f"{'immob_s':>8} {'hip_th':>7} {'hgt_ratio':>9} {'recall':>8} {'FP':>6}")
results = []
for im in immobility_candidates:
    for ht in hip_thresh_candidates:
        for hr in height_ratio_candidates:
            tp, fn, fp = score(
                cases, cache, VEL, ASP,
                immobility_confirm_s=im, immobility_soft=True,
                immobility_hip_thresh=ht, immobility_height_ratio=hr,
            )
            recall = tp / n_fall_cases
            results.append((im, ht, hr, recall, fp, tp))
            print(f"{im:>8} {ht:>7} {hr:>9} {recall * 100:>7.1f}% {fp:>6}")

RESULTS_PATH = "sweep_immobility_soft_results.pkl"
with open(RESULTS_PATH, "wb") as f:
    pickle.dump(results, f)

pareto = []
for r in results:
    dominated = any(
        (o[3] >= r[3] and o[4] <= r[4]) and (o[3] > r[3] or o[4] < r[4])
        for o in results
    )
    if not dominated:
        pareto.append(r)
pareto.sort(key=lambda r: r[4])

print(f"\n=== Frente de Pareto ({len(pareto)} combos no dominados, ordenados por FP creciente) ===")
print(f"{'immob_s':>8} {'hip_th':>7} {'hgt_ratio':>9} {'recall':>8} {'FP':>6}")
for im, ht, hr, recall, fp, tp in pareto:
    print(f"{im:>8} {ht:>7} {hr:>9} {recall * 100:>7.1f}% {fp:>6}")
print(f"\n(comparar contra baseline: recall {baseline_recall:.1%}, FP {baseline_fp})")
