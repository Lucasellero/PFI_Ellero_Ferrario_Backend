"""Punto de entrada: clasifica actividades con CLIP sobre el dataset local y guarda eventos.

Uso:
    python main.py                          # procesa todos los videos de ../videos/
    python main.py --video Walk_p07_r00_v04_c07.mp4
    python main.py --interval 5            # muestrea un frame cada 5 segundos (default: 2)
    python main.py --model ViT-L/14        # variante más precisa de CLIP
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Iterator

import cv2
import numpy as np

sys.path.append(str(Path(__file__).resolve().parents[1] / "events"))
from schema import EventType, MonitoringEvent  # noqa: E402
from store import EventStore  # noqa: E402

from src.clip_classifier import ClipActivityClassifier

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


def sample_frames(path: Path, interval_s: float) -> Iterator[tuple[np.ndarray, float]]:
    """Yield (frame_bgr, timestamp_s) saltando interval_s segundos entre muestras."""
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise RuntimeError(f"No se pudo abrir el video: {path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    step = max(1, int(fps * interval_s))
    try:
        frame_idx = 0
        while frame_idx < total_frames:
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ok, image = cap.read()
            if not ok:
                break
            yield image, frame_idx / fps
            frame_idx += step
    finally:
        cap.release()


def process_video(
    path: Path,
    classifier: ClipActivityClassifier,
    store: EventStore,
    interval_s: float,
) -> int:
    video_id = path.stem
    count = 0
    for image, timestamp_s in sample_frames(path, interval_s):
        result = classifier.classify(image)
        event = MonitoringEvent(
            video_id=video_id,
            event_type=EventType.ACTIVITY,
            video_timestamp_s=timestamp_s,
            metadata=result,
        )
        store.add(event)
        logger.info(
            "Evento en %s @ %.2fs: %s (%.0f%%)",
            video_id,
            timestamp_s,
            result["activity_label"],
            result["confidence"] * 100,
        )
        count += 1
    return count


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Clasificación de actividades con CLIP sobre el dataset local"
    )
    parser.add_argument("--video", help="Nombre de un único archivo a procesar")
    parser.add_argument("--folder", help="Subcarpeta de videos/ a procesar, si existe")
    parser.add_argument(
        "--interval",
        type=float,
        default=10.0,
        help="Segundos entre frames muestreados (default: 10)",
    )
    parser.add_argument(
        "--model",
        default="ViT-B/32",
        help="Variante de CLIP: ViT-B/32, ViT-B/16, ViT-L/14 (default: ViT-B/32)",
    )
    args = parser.parse_args()

    videos = find_videos(args.video, args.folder)
    if not videos:
        logger.warning("No se encontraron videos para procesar en %s", VIDEOS_DIR)
        return

    logger.info("Cargando CLIP (%s)...", args.model)
    classifier = ClipActivityClassifier(args.model)
    store = EventStore()

    total_events = 0
    try:
        for path in videos:
            logger.info("Procesando %s", path.relative_to(VIDEOS_DIR))
            total_events += process_video(path, classifier, store, args.interval)
    finally:
        store.close()

    logger.info(
        "Listo: %d video(s) procesados, %d evento(s) registrados",
        len(videos),
        total_events,
    )


if __name__ == "__main__":
    main()
