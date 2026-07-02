"""Abstracción de fuente de video: archivo local (con loop) o stream en vivo (IP/webcam).

VideoFileLoop  — lee un .mp4/.avi y lo reinicia indefinidamente al terminar.
LiveStream     — lee desde URL RTSP/HTTP o índice de webcam; reconecta ante errores.

Ambas clases exponen el mismo iterador frames() → Iterator[Frame].
Los timestamps son wall-clock (time.monotonic desde que arrancó la fuente),
no el tiempo relativo al video, para que el detector de inmovilidad funcione
de forma coherente incluso cuando el video hace loop.
"""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import cv2
import numpy as np

logger = logging.getLogger(__name__)

RECONNECT_DELAY_S = 3.0   # espera entre intentos de reconexión en streams en vivo


@dataclass
class Frame:
    image: np.ndarray
    timestamp_s: float   # segundos desde que arrancó esta fuente (wall-clock)


def _is_live_source(source: str) -> bool:
    """Devuelve True si la fuente es un stream en vivo (URL o índice de webcam)."""
    if source.startswith(("rtsp://", "rtsps://", "http://", "https://")):
        return True
    try:
        int(source)   # índice de webcam, e.g. "0"
        return True
    except ValueError:
        return False


class VideoFileLoop:
    """Lee un archivo de video y lo reinicia al terminar, indefinidamente."""

    def __init__(self, path: str, realtime: bool = False):
        self._path = path
        self._realtime = realtime
        self._start = time.monotonic()

    def frames(self) -> Iterator[Frame]:
        loop_count = 0
        while True:
            cap = cv2.VideoCapture(self._path)
            if not cap.isOpened():
                raise RuntimeError(f"No se pudo abrir el video: {self._path}")

            fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
            frame_duration = 1.0 / fps
            loop_count += 1
            if loop_count > 1:
                logger.debug("Loop #%d en %s", loop_count, self._path)

            try:
                while True:
                    t0 = time.monotonic()
                    ok, image = cap.read()
                    if not ok:
                        break   # fin del video → reiniciar loop

                    yield Frame(image=image, timestamp_s=t0 - self._start)

                    if self._realtime:
                        elapsed = time.monotonic() - t0
                        wait = frame_duration - elapsed
                        if wait > 0:
                            time.sleep(wait)
            finally:
                cap.release()


class LiveStream:
    """Lee desde un stream en vivo (RTSP, HTTP, webcam). Reconecta ante cortes."""

    def __init__(self, source: str, realtime: bool = False):
        # Para webcam se pasa como int, para URLs como str
        try:
            self._source: int | str = int(source)
        except ValueError:
            self._source = source
        self._realtime = realtime
        self._start = time.monotonic()

    def frames(self) -> Iterator[Frame]:
        fps = 25.0
        frame_duration = 1.0 / fps

        while True:
            cap = cv2.VideoCapture(self._source)
            if not cap.isOpened():
                logger.warning("No se pudo conectar a %s, reintentando en %.0fs...",
                               self._source, RECONNECT_DELAY_S)
                time.sleep(RECONNECT_DELAY_S)
                continue

            fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
            frame_duration = 1.0 / fps
            logger.info("Conectado a %s (%.1f FPS)", self._source, fps)

            try:
                while True:
                    t0 = time.monotonic()
                    ok, image = cap.read()
                    if not ok:
                        logger.warning("Stream cortado en %s, reconectando...", self._source)
                        break

                    yield Frame(image=image, timestamp_s=t0 - self._start)

                    if self._realtime:
                        elapsed = time.monotonic() - t0
                        wait = frame_duration - elapsed
                        if wait > 0:
                            time.sleep(wait)
            finally:
                cap.release()

            time.sleep(RECONNECT_DELAY_S)


_PROJECT_ROOT = Path(__file__).resolve().parents[1]


def make_source(source: str, realtime: bool = False, loop: bool = True) -> VideoFileLoop | LiveStream:
    if _is_live_source(source):
        return LiveStream(source, realtime=realtime)
    # Resuelve rutas relativas al root del proyecto
    path = Path(source)
    if not path.is_absolute():
        path = _PROJECT_ROOT / path
    return VideoFileLoop(str(path), realtime=realtime)
