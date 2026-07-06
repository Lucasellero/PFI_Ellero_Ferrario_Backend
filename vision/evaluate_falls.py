"""Evaluación con ground truth real: usa los CSV videos/**/Fall.csv
(ej. GMDCSA-24/fall1/Fall.csv, CAUCAFall/Fall.csv; inicio/fin exacto de cada
caída y de las actividades previas) para medir
recall real (¿el evento cayó dentro de la ventana de caída anotada?) y
falsos positivos (¿disparó durante una actividad marcada como no-caída,
ej. Sitting/Walking/Standing?). No es parte del pipeline productivo.

Uso:
    python evaluate_falls.py
"""
from __future__ import annotations

import csv
import glob
import re
from pathlib import Path

import cv2

from src.pose_pipeline import FallImmobilityDetector

BRACKET_RE = re.compile(r"\[(.*?)\]")
NUMBER_RE = re.compile(r"[\d.]+")


def parse_classes(raw: str, video_id: str = "") -> list[tuple[str, float, float, bool]]:
    """Devuelve [(label, start_s, end_s, is_fall), ...] a partir del campo Classes.

    El CSV tiene algunas inconsistencias menores (":" sobrante, "to" faltante
    entre los números) que se toleran acá; segmentos sin rango de tiempo
    parseable se descartan con un aviso — no se inventan timestamps.
    """
    segments = []
    for part in raw.split(";"):
        part = part.strip()
        if not part:
            continue
        bracket = BRACKET_RE.search(part)
        if not bracket:
            print(f"  [aviso] {video_id}: segmento sin rango de tiempo, se ignora: {part!r}")
            continue
        label = part[: bracket.start()].strip()
        numbers = NUMBER_RE.findall(bracket.group(1))
        if len(numbers) < 2:
            print(f"  [aviso] {video_id}: no se pudieron extraer 2 números: {part!r}")
            continue
        start, end = float(numbers[0]), float(numbers[1])
        is_fall = label.lower().startswith("fall")
        segments.append((label, start, end, is_fall))
    return segments


def load_ground_truth() -> list[dict]:
    """Recorre videos/**/Fall.csv (cualquier profundidad, ej. GMDCSA-24/fall1/
    o CAUCAFall/) y arma la lista de casos a evaluar.

    Incluye también los clips sin ninguna caída (ej. Walk/Hop/Kneel de
    CAUCAFall): no cuentan para el recall, pero sí para falsos positivos —
    antes se descartaban del todo, lo que dejaba 50 clips de actividades
    normales sin evaluar nunca como posible fuente de FP.
    """
    cases = []
    for csv_path in sorted(glob.glob("../videos/**/Fall.csv", recursive=True)):
        folder = Path(csv_path).parent
        for row in csv.DictReader(open(csv_path, encoding="utf-8-sig")):
            classes_field = row.get(" Classes") or row.get("Classes") or ""
            segments = parse_classes(classes_field, video_id=f"{folder.name}/{row['File Name']}")
            fall_segments = [s for s in segments if s[3]]
            nonfall_segments = [s for s in segments if not s[3]]
            if not fall_segments and not nonfall_segments:
                continue
            cases.append({
                "video": str(folder / row["File Name"]),
                "fall_segments": [(s[1], s[2]) for s in fall_segments],
                "nonfall_segments": [(s[1], s[2]) for s in nonfall_segments],
            })
    return cases


def run_detector(video_path: str) -> list[dict]:
    det = FallImmobilityDetector()
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    events = []
    frame_i = 0
    while True:
        ok, img = cap.read()
        if not ok:
            break
        events.extend(det.process_frame(img, frame_i / fps))
        frame_i += 1
    cap.release()
    return [e for e in events if e["event_type"] == "fall"]


def main() -> None:
    cases = load_ground_truth()
    fall_cases = [c for c in cases if c["fall_segments"]]
    n_csv = len(glob.glob('../videos/**/Fall.csv', recursive=True))
    print(f"{len(fall_cases)} casos con caída anotada, {len(cases) - len(fall_cases)} clips sin caída (de {n_csv} carpetas)\n")

    true_positives = 0
    false_negatives = 0
    false_positives_in_nonfall_segments = 0  # eventos fuera de la ventana de caída anotada (incluye clips 100% sin caída)
    latencies = []

    for case in cases:
        events = run_detector(case["video"])
        event_times = [e["video_timestamp_s"] for e in events]

        # Recall: algún evento cae dentro de [start, end] de alguna caída anotada
        hit = False
        for t in event_times:
            for start, end in case["fall_segments"]:
                if start <= t <= end:
                    hit = True
                    latencies.append(t - start)
                    break
        if case["fall_segments"]:
            if hit:
                true_positives += 1
            else:
                false_negatives += 1

        # Falsos positivos: eventos que caen en un tramo marcado explícitamente
        # como no-caída (Sitting/Walking/Standing/etc., o el clip entero si no
        # tiene ninguna caída)
        fp_here = 0
        for t in event_times:
            for start, end in case["nonfall_segments"]:
                if start <= t <= end:
                    fp_here += 1
                    break
        false_positives_in_nonfall_segments += fp_here

        status = "OK" if hit else ("MISS" if case["fall_segments"] else "-")
        fp_note = f" ({fp_here} FP en tramo no-caída)" if fp_here else ""
        print(f"{case['video']}: {status}  eventos={[round(t,2) for t in event_times]}{fp_note}")

    total = len(fall_cases)
    recall = true_positives / total if total else 0.0
    avg_latency = sum(latencies) / len(latencies) if latencies else float("nan")
    print("\n=== RESUMEN ===")
    print(f"Recall: {true_positives}/{total} ({recall*100:.1f}%)")
    print(f"Falsos negativos: {false_negatives}")
    print(f"Falsos positivos en tramos no-caída (de {len(cases)} clips evaluados): {false_positives_in_nonfall_segments}")
    print(f"Latencia promedio (evento - inicio real de la caída): {avg_latency:.2f}s")


if __name__ == "__main__":
    main()
