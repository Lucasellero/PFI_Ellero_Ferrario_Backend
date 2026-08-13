"""Recalibra IMMOBILITY_MOVEMENT_THRESHOLD para una variante "core-only" del
desplazamiento (solo hombros+cadera, índices COCO 5,6,11,12), en vez de los
17 keypoints completos que usa hoy `_smoothed_displacement_rate`.

Motivación (ver vision/README.md, "Inmovilidad prolongada", 2026-08-13):
`diagnose_immobility_displacement.py` mostró que en 1 de 3 clips largos que
nunca dispararon inmovilidad, restringir el cálculo a hombros+cadera
alcanzaba para cruzar los 180s de quietud sostenida (145s→210s) — la
persona se queda sentada, pero el promedio sobre los 17 keypoints se infla
por el movimiento de manos/codos (comer, gesticular, usar el celular) y
nunca deja acumular la ventana completa.

Antes de integrar ese cambio a producción, esto reusa el mismo dataset ya
calibrado en `calibrate_immobility.py` (10 tramos "quieto" reales de Toyota
Smarthome untrimmed + 9 tramos "moviendo" de contraste) para chequear que
la variante core-only separa igual o mejor esas dos distribuciones — si las
mezclara, el cambio sería peor, no mejor, aunque ayude en el caso puntual
que lo motivó.

Uso:
    vision/.venv/Scripts/python.exe calibrate_immobility_core.py
Requiere el dataset untrimmed de Toyota Smarthome en VIDEOS_DIR (ver
calibrate_immobility.py).
"""
from __future__ import annotations

import numpy as np
from ultralytics import YOLO

from calibrate_immobility import SAMPLE_EVERY_N_FRAMES, SEGMENTS, VIDEOS_DIR
from src.pose_pipeline import build_pose_sample, detect_person

import cv2

CORE_IDX = [5, 6, 11, 12]  # hombros + cadera


def percentile_summary(values: list[float]) -> tuple[float, float, float]:
    arr = np.array(values)
    return float(np.median(arr)), float(np.percentile(arr, 90)), float(np.percentile(arr, 95))


def main() -> None:
    model = YOLO("yolo11n-pose.pt")
    model.to("cuda")

    full_by_label: dict[str, list[float]] = {"quieto": [], "moviendo": []}
    core_by_label: dict[str, list[float]] = {"quieto": [], "moviendo": []}

    for video_id, start_frame, end_frame, label in SEGMENTS:
        path = f"{VIDEOS_DIR}/{video_id}.mp4"
        cap = cv2.VideoCapture(path)
        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

        prev_sample = None
        frame_i = start_frame
        n = 0
        while frame_i < end_frame:
            ok, image = cap.read()
            if not ok:
                break
            if (frame_i - start_frame) % SAMPLE_EVERY_N_FRAMES == 0:
                h, w = image.shape[:2]
                detection = detect_person(model, image)
                if detection is not None:
                    kpts, box, _ = detection
                    sample = build_pose_sample(kpts, box, w, h, frame_i / fps)
                    if prev_sample is not None:
                        dt = sample.timestamp_s - prev_sample.timestamp_s
                        full_rate = float(np.nanmean(np.abs(sample.keypoints - prev_sample.keypoints))) / dt
                        core_rate = float(np.nanmean(np.abs(
                            sample.keypoints[CORE_IDX] - prev_sample.keypoints[CORE_IDX]
                        ))) / dt
                        full_by_label[label].append(full_rate)
                        core_by_label[label].append(core_rate)
                    prev_sample = sample
                    n += 1
                else:
                    prev_sample = None
            frame_i += 1
        cap.release()
        print(f"{video_id} [{label}] frames {start_frame}-{end_frame}: {n} muestras", flush=True)

    print("\n=== full (17 keypoints, referencia — igual a la tabla ya calibrada) ===")
    print(f"{'':>10} {'mediana':>10} {'p90':>10} {'p95':>10}")
    for label in ("quieto", "moviendo"):
        med, p90, p95 = percentile_summary(full_by_label[label])
        print(f"{label:>10} {med:>10.4f} {p90:>10.4f} {p95:>10.4f}")

    print("\n=== core (hombros+cadera, candidata) ===")
    print(f"{'':>10} {'mediana':>10} {'p90':>10} {'p95':>10}")
    for label in ("quieto", "moviendo"):
        med, p90, p95 = percentile_summary(core_by_label[label])
        print(f"{label:>10} {med:>10.4f} {p90:>10.4f} {p95:>10.4f}")

    # Ratio de separación (walking mediana / quieto mediana) — más alto = mejor separación.
    full_ratio = np.median(full_by_label["moviendo"]) / np.median(full_by_label["quieto"])
    core_ratio = np.median(core_by_label["moviendo"]) / np.median(core_by_label["quieto"])
    print(f"\nRatio de separación (mediana moviendo / mediana quieto): full={full_ratio:.1f}x, core={core_ratio:.1f}x")

    # Con el umbral actual (0.02, calibrado para "full"), % de error en cada
    # distribución, para ambas variantes.
    from src.pose_pipeline import IMMOBILITY_MOVEMENT_THRESHOLD
    for name, by_label in (("full", full_by_label), ("core", core_by_label)):
        quieto_err = np.mean(np.array(by_label["quieto"]) > IMMOBILITY_MOVEMENT_THRESHOLD)
        moviendo_err = np.mean(np.array(by_label["moviendo"]) <= IMMOBILITY_MOVEMENT_THRESHOLD)
        print(f"{name}: con umbral {IMMOBILITY_MOVEMENT_THRESHOLD} -> "
              f"{quieto_err:.1%} de 'quieto' lo supera (falso movimiento), "
              f"{moviendo_err:.1%} de 'moviendo' queda por debajo (falso quieto)")

    # Barrido de umbral propio para "core" — 0.02 se calibró para "full", no hay
    # razón para asumir que sigue siendo el mejor punto para la variante nueva.
    print("\n=== Barrido de umbral para 'core' (no reusar 0.02 sin chequear) ===")
    print(f"{'umbral':>8} {'quieto_err':>11} {'moviendo_err':>13}")
    for thr in [0.010, 0.012, 0.014, 0.016, 0.018, 0.020, 0.022, 0.025, 0.030]:
        q_err = np.mean(np.array(core_by_label["quieto"]) > thr)
        m_err = np.mean(np.array(core_by_label["moviendo"]) <= thr)
        print(f"{thr:>8.3f} {q_err:>10.1%} {m_err:>12.1%}")

    import pickle
    with open("calibrate_immobility_core_raw.pkl", "wb") as f:
        pickle.dump({"full": full_by_label, "core": core_by_label}, f)


if __name__ == "__main__":
    main()
