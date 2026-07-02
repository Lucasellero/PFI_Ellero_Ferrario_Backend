"""Punto de entrada: corre el pipeline de pose sobre el dataset local y guarda eventos.

Uso:
    python main.py                         # procesa todos los videos de ../videos/
    python main.py --video Walk_p07_r00_v04_c07.mp4
    python main.py --folder <subcarpeta>   # si ../videos/ está organizada en subcarpetas
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1] / "events"))
from schema import EventType, MonitoringEvent  # noqa: E402
from store import EventStore  # noqa: E402

from src.capture import VideoFile
from src.pose_pipeline import FallImmobilityDetector

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

VIDEOS_DIR = Path(__file__).resolve().parents[1] / "videos"
VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov"}


def find_videos(video: str | None, folder: str | None) -> list[Path]:
    if video:
        matches = list(VIDEOS_DIR.rglob(video))
        if not matches:
            raise FileNotFoundError(f"No se encontró '{video}' dentro de {VIDEOS_DIR}")
        return matches
    base = VIDEOS_DIR / folder if folder else VIDEOS_DIR
    return sorted(p for p in base.rglob("*") if p.suffix.lower() in VIDEO_EXTENSIONS)


def process_video(path: Path, detector: FallImmobilityDetector, store: EventStore) -> int:
    video = VideoFile(path)
    count = 0
    try:
        for frame in video.frames():
            for raw_event in detector.process_frame(frame.image, frame.timestamp_s):
                event = MonitoringEvent(
                    video_id=video.video_id,
                    event_type=EventType(raw_event["event_type"]),
                    video_timestamp_s=raw_event["video_timestamp_s"],
                    metadata=raw_event["metadata"],
                )
                store.add(event)
                logger.info("Evento en %s @ %.2fs: %s", video.video_id,
                            event.video_timestamp_s, event.event_type.value)
                count += 1
    finally:
        video.release()
    return count


def main() -> None:
    parser = argparse.ArgumentParser(description="Pipeline de pose estimation sobre el dataset local")
    parser.add_argument("--video", help="Nombre de un único archivo a procesar")
    parser.add_argument("--folder", help="Subcarpeta de videos/ a procesar, si existe")
    parser.add_argument("--model", default="yolo11n-pose.pt", help="Pesos de YOLO a usar")
    args = parser.parse_args()

    videos = find_videos(args.video, args.folder)
    if not videos:
        logger.warning("No se encontraron videos para procesar en %s", VIDEOS_DIR)
        return

    detector = FallImmobilityDetector(model_path=args.model)
    store = EventStore()

    total_events = 0
    try:
        for path in videos:
            logger.info("Procesando %s", path.relative_to(VIDEOS_DIR))
            total_events += process_video(path, detector, store)
    finally:
        store.close()

    logger.info("Listo: %d video(s) procesados, %d evento(s) detectados", len(videos), total_events)


if __name__ == "__main__":
    main()
