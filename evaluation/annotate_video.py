"""Genera un video anotado (esqueleto de pose + eventos superpuestos) para
inspección visual manual. Ver README.md de esta carpeta.

Por default, superpone los eventos ya guardados en `../events/events.db`
para ese video (los que haya: `fall`/`immobility` de `vision/`, `activity`
de `vlm/`/`fusion/`, `routine_deviation` de `baseline/`, lo que haya
corrido antes sobre ese video_id). Con --rerun, corre el pipeline de
`vision/` (caída/inmovilidad) desde cero sobre el video y guarda esos
eventos nuevos en el store antes de generar el video — útil para un video
que todavía no tiene nada en events.db, pero no reproduce eventos de
`vlm/`/`fusion/`/`baseline/` (pipelines separados, fuera de alcance acá).

Uso:
    python annotate_video.py --video <nombre>.mp4
    python annotate_video.py --video <nombre>.mp4 --rerun
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import cv2

sys.path.append(str(Path(__file__).resolve().parents[1] / "events"))
sys.path.append(str(Path(__file__).resolve().parents[1] / "vision"))
from schema import EventType, MonitoringEvent  # noqa: E402
from store import EventStore  # noqa: E402
from src.pose_pipeline import FallImmobilityDetector, draw_pose  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

VIDEOS_DIR = Path(__file__).resolve().parents[1] / "videos"
OUTPUT_DIR = Path(__file__).resolve().parent / "output"

# Los eventos son instantáneos (un timestamp, no un rango), así que el texto
# se mantiene en pantalla este tiempo después de dispararse para poder leerlo
# en reproducción normal, no solo cuadro a cuadro.
EVENT_BANNER_S = 3.0
EVENT_COLORS = {
    "fall": (0, 0, 255),
    "immobility": (0, 165, 255),
    "activity": (0, 200, 0),
    "routine_deviation": (255, 0, 255),
}


def find_video(name: str) -> Path:
    matches = list(VIDEOS_DIR.rglob(name))
    if not matches:
        raise FileNotFoundError(f"No se encontró '{name}' dentro de {VIDEOS_DIR}")
    return matches[0]


def format_event_text(event: MonitoringEvent) -> str:
    label = event.event_type.value.upper()
    if event.event_type is EventType.FALL:
        extra = f" (evidence={event.metadata.get('evidence', '?')})"
    elif event.event_type is EventType.IMMOBILITY:
        extra = f" (severity={event.metadata.get('severity', '?')}, postura={event.metadata.get('posture', '?')})"
    elif event.event_type is EventType.ACTIVITY:
        extra = f" ({event.metadata.get('activity', event.metadata.get('label', '?'))})"
    elif event.event_type is EventType.ROUTINE_DEVIATION:
        extra = f" ({event.metadata.get('reason', '?')})"
    else:
        extra = ""
    return f"{label}{extra}"


def annotate(
    video_path: Path,
    model_path: str,
    output_path: Path,
    rerun: bool,
    store: EventStore,
) -> int:
    video_id = video_path.stem
    stored_events = sorted(store.all(video_id=video_id), key=lambda e: e.video_timestamp_s)
    if not rerun and not stored_events:
        logger.warning(
            "No hay eventos guardados para '%s' en events.db — el video va a "
            "salir solo con el esqueleto superpuesto. Usar --rerun para generar "
            "eventos de fall/immobility desde cero.", video_id,
        )

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"No se pudo abrir el video: {video_path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(output_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))

    detector = FallImmobilityDetector(model_path=model_path)
    due_events = stored_events if not rerun else []
    next_due_idx = 0
    active_events: list[MonitoringEvent] = []
    total_events = len(stored_events)

    frame_idx = 0
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            timestamp_s = frame_idx / fps
            frame_idx += 1

            raw_events = detector.process_frame(frame, timestamp_s)
            if detector.last_detection is not None:
                kpts, box = detector.last_detection
                frame = draw_pose(frame, kpts, box)

            if rerun:
                for raw_event in raw_events:
                    event = MonitoringEvent(
                        video_id=video_id,
                        event_type=EventType(raw_event["event_type"]),
                        video_timestamp_s=raw_event["video_timestamp_s"],
                        metadata=raw_event["metadata"],
                    )
                    store.add(event)
                    due_events.append(event)
                    total_events += 1

            while next_due_idx < len(due_events) and due_events[next_due_idx].video_timestamp_s <= timestamp_s:
                active_events.append(due_events[next_due_idx])
                next_due_idx += 1
            active_events = [e for e in active_events if timestamp_s - e.video_timestamp_s <= EVENT_BANNER_S]

            cv2.putText(frame, f"t={timestamp_s:6.2f}s", (10, 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            for i, event in enumerate(active_events):
                color = EVENT_COLORS.get(event.event_type.value, (255, 255, 255))
                cv2.putText(frame, format_event_text(event), (10, 55 + i * 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

            writer.write(frame)
    finally:
        cap.release()
        writer.release()

    return total_events


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Genera un video anotado (esqueleto + eventos) para inspección visual manual",
    )
    parser.add_argument("--video", required=True, help="Nombre del archivo en ../videos/")
    parser.add_argument("--model", default="yolo11n-pose.pt", help="Pesos de YOLO a usar para el esqueleto")
    parser.add_argument(
        "--rerun", action="store_true",
        help="Corre el detector de fall/immobility de vision/ desde cero y guarda los eventos nuevos en events.db "
             "(no reproduce eventos de vlm/fusion/baseline, que son pipelines separados)",
    )
    args = parser.parse_args()

    video_path = find_video(args.video)
    output_path = OUTPUT_DIR / f"{video_path.stem}_annotated.mp4"

    store = EventStore()
    try:
        total_events = annotate(video_path, args.model, output_path, args.rerun, store)
    finally:
        store.close()

    logger.info("Video anotado guardado en %s (%d evento(s) superpuestos)", output_path, total_events)


if __name__ == "__main__":
    main()
