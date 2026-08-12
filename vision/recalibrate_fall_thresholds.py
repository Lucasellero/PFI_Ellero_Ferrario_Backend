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
MAX_FRAMES_PER_CLIP = 100_000  # tope de seguridad: un .avi corrupto no debe poder colgar la extracción.
# OJO (2026-08-11): con el valor viejo (10_000 = 400s a 25fps) los 4 clips largos de
# ToyotaSmarthome_FP (1100-1600s reales) quedaban truncados a ~25-35% de su duración,
# subestimando sus FP por ~3x sin ningún aviso - la causa real de que el barrido diera
# números mucho más optimistas que evaluate_falls.py (que no tiene este tope). 100_000
# frames = ~66min a 25fps, cubre los 4 clips largos completos.


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
            kpts, box, _ = detection
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


BASELINE_WINDOW_S = 2.0  # ventana antes del pico de velocidad para estimar el alto "de pie" de referencia


def simulate(
    descriptors: list[tuple[float, float, float, float]],
    velocity_threshold: float,
    aspect_threshold: float,
    height_drop_ratio: float | None = None,
    floor_hip_y: float | None = None,
    immobility_confirm_s: float | None = None,
    immobility_hip_thresh: float = 0.05,
    immobility_height_ratio: float = 0.15,
) -> list[float]:
    """Reproduce la lógica de dos fases de FallImmobilityDetector._detect_fall
    sobre descriptores ya calculados. Devuelve los timestamps de los eventos 'fall'.

    Confirmación por postura: dos condiciones alternativas (OR), no solo
    aspect_ratio. En una caída hacia adelante/hacia la cámara la persona
    termina encogida (bbox angosto), no "ancha" como en una caída lateral —
    aspect_ratio nunca cruza el umbral aunque la caída sea real (ver hallazgo
    en vision/README.md, forward falls de CAUCAFall). height_drop_ratio agrega
    una condición alternativa: el bbox se derrumba en altura respecto a la
    "altura de pie" observada justo antes del pico de velocidad, sin importar
    si se ensancha o no. height_drop_ratio=None reproduce el comportamiento
    viejo (solo aspect_ratio) para poder comparar ambos en el mismo barrido.

    height_drop_ratio solo (probado antes) resultó poco selectivo: sentarse o
    agacharse en una actividad normal también derrumba el bbox en altura, así
    que dispara todo el tiempo fuera de caídas reales. floor_hip_y agrega un
    tercer requisito cuando se usa junto con height_drop_ratio: no alcanza con
    que el bbox se achique, la cadera además tiene que terminar cerca del piso
    del frame (hip_y alto en términos absolutos, no relativo al bbox) — al
    sentarse en una silla/sillón la cadera se queda a media altura, al
    agacharse a agarrar algo se recupera enseguida; en una caída real la
    persona termina en el piso. floor_hip_y=None desactiva este requisito
    extra (deja el comportamiento de height_drop_ratio solo).

    immobility_confirm_s agrega una CUARTA fase, para evitar el disparo
    repetido que tuvo height_drop_ratio+floor_hip_y en video largo continuo
    (124 FP en un solo clip, ver vision/README.md 2026-08-11): en vez de
    confirmar el evento apenas posture_ok da True, se abre un "candidato" y
    se espera immobility_confirm_s segundos verificando que la persona se
    quede quieta (hip_y y bbox_height_frac no cambian más que
    immobility_hip_thresh/immobility_height_ratio respecto al momento del
    candidato) antes de confirmar. Si se mueve antes de que pase la ventana,
    el candidato se cancela y NO se reporta evento — separa "se agachó y
    siguió" de "se cayó y no se levantó" sin exigir nada en el instante.
    immobility_confirm_s=None desactiva esta fase (comportamiento viejo,
    confirma en el instante).
    """
    history: list[tuple[float, float, float]] = []  # (t, hip_y, bbox_height_frac)
    heights_history: list[tuple[float, float]] = []  # (t, bbox_height_frac), sin trimear a 1.5s
    last_velocity_spike_t: float | None = None
    baseline_height: float | None = None
    cooldown_until = float("-inf")
    candidate: tuple[float, float, float] | None = None  # (t, hip_y, bbox_height_frac) del momento del candidato
    events = []

    for t, hip_y, bbox_height_frac, aspect_ratio in descriptors:
        history.append((t, hip_y, bbox_height_frac))
        history[:] = [s for s in history if t - s[0] <= 1.5]
        heights_history.append((t, bbox_height_frac))
        heights_history[:] = [s for s in heights_history if t - s[0] <= BASELINE_WINDOW_S + FALL_CONFIRM_WINDOW_S]

        # Candidato pendiente de confirmación por inmovilidad: chequear ANTES
        # de evaluar un nuevo pico/postura, así un movimiento real lo cancela.
        if candidate is not None:
            cand_t, cand_hip_y, cand_height = candidate
            moved = (
                abs(hip_y - cand_hip_y) > immobility_hip_thresh
                or abs(bbox_height_frac - cand_height) > cand_height * immobility_height_ratio
            )
            if moved:
                candidate = None
            elif t - cand_t >= immobility_confirm_s:
                events.append(cand_t)
                cooldown_until = t + FALL_COOLDOWN_S
                candidate = None
                last_velocity_spike_t = None
                baseline_height = None
                continue  # confirmado: no re-evaluar pico/postura en el mismo frame

        if t >= cooldown_until and candidate is None:
            target_t = t - FALL_VELOCITY_WINDOW_S
            ref = min(history, key=lambda s: abs(s[0] - target_t))
            dt = t - ref[0]
            if dt >= FALL_VELOCITY_WINDOW_S / 2:
                scale = (bbox_height_frac + ref[2]) / 2
                if scale > 1e-6:
                    velocity = (hip_y - ref[1]) / scale / dt
                    if velocity > velocity_threshold:
                        last_velocity_spike_t = t
                        pre_spike = [h for tt, h in heights_history if t - tt <= BASELINE_WINDOW_S]
                        baseline_height = max(pre_spike) if pre_spike else None

            posture_ok = aspect_ratio > aspect_threshold
            if not posture_ok and height_drop_ratio is not None and baseline_height:
                posture_ok = bbox_height_frac < baseline_height * height_drop_ratio
                if posture_ok and floor_hip_y is not None:
                    posture_ok = hip_y > floor_hip_y

            if (
                last_velocity_spike_t is not None
                and posture_ok
                and t - last_velocity_spike_t <= FALL_CONFIRM_WINDOW_S
            ):
                last_velocity_spike_t = None
                if immobility_confirm_s is not None:
                    candidate = (t, hip_y, bbox_height_frac)
                    continue
                baseline_height = None
                cooldown_until = t + FALL_COOLDOWN_S
                events.append(t)

    return events


def score(
    cases: list[dict],
    cache: dict[str, list],
    velocity_threshold: float,
    aspect_threshold: float,
    height_drop_ratio: float | None = None,
    floor_hip_y: float | None = None,
    immobility_confirm_s: float | None = None,
) -> tuple[int, int, int]:
    """Devuelve (true_positives, false_negatives, false_positives) sobre todos los casos."""
    tp = fn = fp = 0
    for case in cases:
        if case["video"] not in cache:
            continue  # cache parcial (ej. interrumpido a mitad del rebuild)
        event_times = simulate(
            cache[case["video"]], velocity_threshold, aspect_threshold,
            height_drop_ratio, floor_hip_y, immobility_confirm_s,
        )

        if case["fall_segments"]:
            hit = any(start <= t <= end for t in event_times for start, end in case["fall_segments"])
            if hit:
                tp += 1
            else:
                fn += 1

        for t in event_times:
            if any(start <= t <= end for start, end in case["nonfall_segments"]):
                fp += 1  # cuenta CADA evento, no capea a 1 por clip: un heurístico
                # que dispara 100 veces en el mismo video largo es mucho peor que
                # uno que dispara 1 vez, y el ranking del barrido tiene que verlo
                # (ver vision/README.md, hallazgo de 124 FP en un solo clip).

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

    # Grilla enfocada en la idea de inmovilidad sostenida (ver vision/README.md
    # 2026-08-11): la combinación height_drop+floor_hip_y ya probó que puede
    # recuperar forward-falls, pero dispara repetido en video largo continuo.
    # Acá se prueba si exigir que la persona se quede quieta 1-2s antes de
    # confirmar filtra ese disparo repetido sin perder la mejora de recall.
    velocity_candidates = [0.12]
    aspect_candidates = [1.4]
    height_drop_candidates = [None, 0.5, 0.55, 0.6]
    floor_hip_y_candidates = [0.75, 0.8, 0.85]
    immobility_candidates = [None, 1.0, 1.5, 2.0]

    print(f"=== Barrido combinado (height_drop x floor_hip_y x immobility_confirm_s) ===")
    results = []
    for vt in velocity_candidates:
        for at in aspect_candidates:
            for hd in height_drop_candidates:
                floors = floor_hip_y_candidates if hd is not None else [None]
                for fl in floors:
                    for im in immobility_candidates:
                        tp, fn, fp = score(cases, cache, vt, at, height_drop_ratio=hd, floor_hip_y=fl, immobility_confirm_s=im)
                        recall = tp / n_fall_cases if n_fall_cases else 0.0
                        results.append((vt, at, hd, fl, im, recall, fp, tp, fn))

    RESULTS_PATH = Path("recalibrate_results.pkl")
    RESULTS_PATH.write_bytes(pickle.dumps(results))
    print(f"Resultados completos guardados en {RESULTS_PATH} ({len(results)} combos)\n")

    baseline = next(r for r in results if r[0] == 0.12 and r[1] == 1.4 and r[2] is None and r[4] is None)
    print(f"Baseline actual (vel=0.12, asp=1.4, sin height_drop/floor/immobility): recall {baseline[5]*100:.1f}% ({baseline[7]}/{n_fall_cases}), FP {baseline[6]}\n")

    # Frente de Pareto: para cada combo, ¿hay otro con recall>= Y fp<= estrictamente mejor en alguno? si no, es Pareto-óptimo.
    pareto = []
    for r in results:
        dominated = any(
            (o[5] >= r[5] and o[6] <= r[6]) and (o[5] > r[5] or o[6] < r[6])
            for o in results
        )
        if not dominated:
            pareto.append(r)
    pareto.sort(key=lambda r: r[6])  # de menos a más FP

    print(f"=== Frente de Pareto ({len(pareto)} combos no dominados, ordenados por FP creciente) ===")
    print(f"{'vel_thr':>8} {'asp_thr':>8} {'hgt_drop':>8} {'floor_y':>8} {'immob_s':>8} {'recall':>8} {'FP':>6}")
    for vt, at, hd, fl, im, recall, fp, tp, fn in pareto:
        marca = "  <- baseline actual" if (vt, at, hd, fl, im) == (0.12, 1.4, None, None, None) else ""
        print(f"{vt:>8} {at:>8} {str(hd):>8} {str(fl):>8} {str(im):>8} {recall * 100:>7.1f}% {fp:>6}{marca}")


if __name__ == "__main__":
    main()
