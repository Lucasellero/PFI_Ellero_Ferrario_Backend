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

import argparse
import csv
import glob
import re
import sys
from pathlib import Path

import cv2

from src.pose_pipeline import FallImmobilityDetector, build_pose_sample

_ROOT = Path(__file__).resolve().parents[1]

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


def load_ground_truth(only: set[str] | None = None) -> list[dict]:
    """Recorre videos/**/Fall.csv (cualquier profundidad, ej. GMDCSA-24/fall1/
    o CAUCAFall/) y arma la lista de casos a evaluar.

    `only`: si se pasa, sólo incluye clips cuya subcarpeta de primer nivel bajo
    videos/ esté en el set (ej. {"GMDCSA-24", "URFD"}).

    Incluye también los clips sin ninguna caída (ej. Walk/Hop/Kneel de
    CAUCAFall): no cuentan para el recall, pero sí para falsos positivos —
    antes se descartaban del todo, lo que dejaba 50 clips de actividades
    normales sin evaluar nunca como posible fuente de FP.
    """
    cases = []
    for csv_path in sorted(glob.glob("../videos/**/Fall.csv", recursive=True)):
        if only is not None:
            try:
                ds = Path(csv_path).resolve().relative_to((_ROOT / "videos").resolve()).parts[0]
            except ValueError:
                ds = Path(csv_path).parent.name
            if ds not in only:
                continue
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


def _load_temporal():
    """Carga el clasificador temporal de caída de fusion/ (2ª etapa,
    reemplaza la confirmación CLIP). Devuelve (clf, Buffer, frame_features) o
    lanza si fusion/ no está entrenado/exportado."""
    sys.path.insert(0, str(_ROOT))
    from fusion.src.fall_temporal import FallTemporalClassifier, FALL_POSITIVE_LABELS  # noqa
    from fusion.src.keypoint_sequence import SkeletonSequenceBuffer, frame_features
    return FallTemporalClassifier(quantized=True), SkeletonSequenceBuffer, frame_features


_TEMPORAL_REJECT = 0.15  # fall_score <= esto => el temporal descarta el evento geométrico (mismo criterio que runner/camera_worker.py::_FALL_TEMPORAL_REJECT)


def run_detector(video_path: str, temporal=None, fps_cap: float | None = None) -> list[dict]:
    det = FallImmobilityDetector()
    buffer = frame_features = None
    if temporal is not None:
        clf, BufferCls, frame_features = temporal
        buffer = BufferCls()
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    # Subsampleo opcional: en clips muy largos (ToyotaSmarthome_FP, ~1600s)
    # procesar cada frame es carísimo. El timestamp real se mantiene, así que
    # las ventanas temporales y la velocidad de caída siguen bien normalizadas.
    stride = max(1, round(fps / fps_cap)) if fps_cap else 1
    events = []
    frame_i = 0
    while True:
        ok = cap.grab()  # avanza sin decodificar
        if not ok:
            break
        if frame_i % stride != 0:
            frame_i += 1
            continue
        ok, img = cap.retrieve()
        if not ok:
            break
        t = frame_i / fps
        raw = det.process_frame(img, t)
        if temporal is not None:
            if det.last_detection is not None:
                kpts, box = det.last_detection
                h, w = img.shape[:2]
                ps = build_pose_sample(kpts, box, w, h, t, kpts_conf=det.last_detection_conf)
                buffer.push(t, frame_features(
                    kpts, det.last_detection_conf, box,
                    ps.aspect_ratio, ps.torso_angle_deg, ps.hip_y, ps.bbox_height_frac,
                ))
            else:
                buffer.reset()
            for e in raw:
                if e["event_type"] != "fall" or e["metadata"].get("evidence") == "partial":
                    continue
                window = buffer.build_window(t)
                if window is None:
                    continue  # sin cobertura: se deja pasar (caería a CLIP en producción)
                res = clf.classify(window)
                e["metadata"]["fall_temporal_score"] = round(res["fall_score"], 4)
        events.extend(raw)
        frame_i += 1
    cap.release()
    # Devuelve TODOS los eventos de caída con su score temporal adjunto; el
    # filtrado por umbral lo hace main() (así --tau y --dump-scores funcionan
    # sin re-correr YOLO).
    return [e for e in events if e["event_type"] == "fall"]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--temporal", action="store_true",
                    help="Corre además el clasificador temporal de fusion/ como 2ª etapa (descarta caídas geométricas que el temporal marca 'normal')")
    ap.add_argument("--only", default=None, help="Datasets a evaluar (coma-separado, ej. GMDCSA-24,CAUCAFall,URFD)")
    ap.add_argument("--fps-cap", type=float, default=None, help="Subsamplear a este FPS (para clips largos; el timestamp real se mantiene)")
    ap.add_argument("--tau", type=float, default=_TEMPORAL_REJECT, help=f"fall_score <= tau => el temporal descarta el evento (default {_TEMPORAL_REJECT})")
    ap.add_argument("--dump-scores", default=None, help="CSV con (video, t, evidence, temporal_score, tipo) por evento de caída — para barrer tau offline")
    a = ap.parse_args()
    only = set(a.only.split(",")) if a.only else None
    temporal = _load_temporal() if a.temporal else None
    dump_rows: list[tuple] = []
    if temporal is not None:
        print("Clasificador temporal de caída: ACTIVO (2ª etapa)\n")

    cases = load_ground_truth(only=only)
    fall_cases = [c for c in cases if c["fall_segments"]]
    n_csv = len(glob.glob('../videos/**/Fall.csv', recursive=True))
    print(f"{len(fall_cases)} casos con caída anotada, {len(cases) - len(fall_cases)} clips sin caída (de {n_csv} carpetas)\n")

    true_positives = 0
    false_negatives = 0
    false_positives_in_nonfall_segments = 0  # eventos fuera de la ventana de caída anotada (incluye clips 100% sin caída)
    latencies = []
    # Desglose por calidad de encuadre (ver "evidence" en pose_pipeline.py):
    # cuánto recall aporta la rama de evidencia parcial y cuánto FP mete.
    tp_by_ev = {"full": 0, "partial": 0}
    fp_by_ev = {"full": 0, "partial": 0}

    for case in cases:
        all_events = run_detector(case["video"], temporal=temporal, fps_cap=a.fps_cap)

        def _kind(t: float) -> str:
            if any(s <= t <= e for s, e in case["fall_segments"]):
                return "TP"
            if any(s <= t <= e for s, e in case["nonfall_segments"]):
                return "FP"
            return "otro"

        for e in all_events:
            score = e["metadata"].get("fall_temporal_score")
            ev = e["metadata"].get("evidence", "full")
            if a.dump_scores is not None:
                dump_rows.append((case["video"], round(e["video_timestamp_s"], 2), ev,
                                  score if score is not None else "", _kind(e["video_timestamp_s"])))

        # Filtro temporal: descarta eventos 'full' con score <= tau (los
        # 'partial' nunca pasan por el temporal — van a sospecha, ver runner).
        events = [
            e for e in all_events
            if not (temporal is not None
                    and e["metadata"].get("evidence", "full") != "partial"
                    and e["metadata"].get("fall_temporal_score") is not None
                    and e["metadata"]["fall_temporal_score"] <= a.tau)
        ]
        event_times = [e["video_timestamp_s"] for e in events]
        event_evidence = {e["video_timestamp_s"]: e.get("metadata", {}).get("evidence", "full") for e in events}

        # Recall: algún evento cae dentro de [start, end] de alguna caída anotada
        hit = False
        hit_evidence = set()
        for t in event_times:
            for start, end in case["fall_segments"]:
                if start <= t <= end:
                    hit = True
                    hit_evidence.add(event_evidence.get(t, "full"))
                    latencies.append(t - start)
                    break
        if case["fall_segments"]:
            if hit:
                true_positives += 1
                # Al clip lo salva "full" si algún evento full cayó en ventana;
                # si sólo lo salva un evento de evidencia parcial, va a esa cuenta.
                tp_by_ev["full" if "full" in hit_evidence else "partial"] += 1
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
                    fp_by_ev[event_evidence.get(t, "full")] += 1
                    break
        false_positives_in_nonfall_segments += fp_here

        status = "OK" if hit else ("MISS" if case["fall_segments"] else "-")
        fp_note = f" ({fp_here} FP en tramo no-caída)" if fp_here else ""
        print(f"{case['video']}: {status}  eventos={[round(t,2) for t in event_times]}{fp_note}", flush=True)

    total = len(fall_cases)
    recall = true_positives / total if total else 0.0
    avg_latency = sum(latencies) / len(latencies) if latencies else float("nan")
    print("\n=== RESUMEN ===")
    print(f"Recall: {true_positives}/{total} ({recall*100:.1f}%)")
    print(f"Falsos negativos: {false_negatives}")
    print(f"Falsos positivos en tramos no-caída (de {len(cases)} clips evaluados): {false_positives_in_nonfall_segments}")
    print(f"Latencia promedio (evento - inicio real de la caída): {avg_latency:.2f}s")
    print("\n--- Desglose por evidencia de encuadre ---")
    print(f"Recall salvado por evento 'full': {tp_by_ev['full']}/{total}  |  sólo por evento 'partial': {tp_by_ev['partial']}/{total}")
    print(f"Falsos positivos 'full': {fp_by_ev['full']}  |  'partial': {fp_by_ev['partial']}")
    if temporal is not None:
        print(f"(umbral temporal tau={a.tau})")

    if a.dump_scores is not None and dump_rows:
        with open(a.dump_scores, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["video", "t", "evidence", "temporal_score", "kind"])
            w.writerows(dump_rows)
        print(f"\nScores por evento → {a.dump_scores} ({len(dump_rows)} eventos). "
              f"Barré tau con: python -c \"import csv,collections;...\"")


if __name__ == "__main__":
    main()
