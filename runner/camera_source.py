"""Abstracción de fuente de video: archivo local (con loop) o stream en vivo (IP/webcam).

VideoFileLoop  — lee un .mp4/.avi y opcionalmente lo reinicia al terminar.
LiveStream     — lee desde URL RTSP/HTTP o índice de webcam; reconecta ante errores.

Ambas clases exponen el mismo iterador frames() → Iterator[Frame].
`LiveStream` y `VideoFileLoop` con `realtime=True` entregan timestamps
wall-clock (time.monotonic desde que arrancó la fuente) — necesario porque
ahí wall-clock es, por construcción, equivalente al tiempo real transcurrido
(no hay otra referencia de tiempo para un stream en vivo, y un archivo en
modo realtime ya se acompasa a eso). `VideoFileLoop` con `realtime=False`
usa en cambio `frame_idx / fps` acumulado across loops (nunca wall-clock):
sin acompasamiento, el hardware procesa a la velocidad que pueda (con GPU,
más rápido que la reproducción real), y wall-clock ahí desalinearía los
umbrales temporales de `vision/` contra el contenido real del video — ver
"Trampa real" en README.md.

Sobre `realtime` y por qué importa: el pipeline de pose (YOLO) tarda mucho
más por frame en CPU (~150-300ms) que la duración real de un frame de video
(~33-40ms a 25-30fps). Si se consume cada frame en orden sin descartar
ninguno, el contenido del video "se estira" en el tiempo — se termina viendo
en cámara lenta en vez de a velocidad real, y en una cámara en vivo real
esto se traduce en un atraso (lag) creciente en vez de cámara lenta, porque
los frames siguen llegando aunque no los proceses a tiempo.
`realtime=True` corrige esto en ambas clases: en vez de entregar todos los
frames en orden, se descartan los que ya quedaron "viejos" respecto al
tiempo real transcurrido, y se entrega siempre el más reciente disponible.
Se pierde densidad de muestreo bajo carga, pero la heurística de caída ya
trabaja con ventanas de ~0.4-1.5s, no depende de ver cada frame individual.
"""
from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator

import cv2
import numpy as np

logger = logging.getLogger(__name__)

RECONNECT_DELAY_S = 3.0   # espera entre intentos de reconexión en streams en vivo
LATEST_FRAME_TIMEOUT_S = 1.0  # cuánto esperar por un frame nuevo antes de revisar si hay que cortar


@dataclass
class Frame:
    image: np.ndarray
    timestamp_s: float   # segundos desde que arrancó esta fuente (monotonic)
    # reloj real (epoch Unix, hora local) al momento de leer el frame —
    # independiente de timestamp_s, que es relativo al arranque de la
    # fuente. Lo necesita `baseline/` para ubicar cada evento en el
    # día/hora real del mundo (ver events/schema.py::wall_clock_ts).
    wall_clock_ts: float = field(default_factory=time.time)


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
    """Lee un archivo de video. Si `loop` es True lo reinicia indefinidamente
    al terminar; si es False, la iteración termina cuando el video se acaba."""

    def __init__(self, path: str, realtime: bool = False, loop: bool = True):
        self._path = path
        self._realtime = realtime
        self._loop = loop
        self._start = time.monotonic()

    def frames(self) -> Iterator[Frame]:
        loop_count = 0
        # Cuenta total de frames entregados across loops (nunca se resetea,
        # a diferencia de frame_idx abajo) — con realtime=False es la base
        # del timestamp (ver más abajo por qué no puede ser wall-clock).
        played_frames = 0
        while True:
            cap = cv2.VideoCapture(self._path)
            if not cap.isOpened():
                raise RuntimeError(f"No se pudo abrir el video: {self._path}")

            fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
            frame_duration = 1.0 / fps
            loop_count += 1
            if loop_count > 1:
                logger.debug("Loop #%d en %s", loop_count, self._path)

            playback_start = time.monotonic()
            frame_idx = 0
            try:
                while True:
                    ok, image = cap.read()
                    if not ok:
                        break   # fin del video

                    scheduled_t = frame_idx * frame_duration
                    frame_idx += 1

                    if self._realtime:
                        elapsed = time.monotonic() - playback_start
                        if elapsed > scheduled_t + frame_duration:
                            continue  # vamos atrasados: descartar este frame para alcanzar el tiempo real
                        wait = scheduled_t - elapsed
                        if wait > 0:
                            time.sleep(wait)
                        timestamp_s = time.monotonic() - self._start
                    else:
                        # Con realtime=True el frame ya está acompasado al
                        # tiempo real transcurrido (sleep arriba), así que
                        # wall-clock es equivalente a tiempo de video. Con
                        # realtime=False NO hay acompasamiento — el hardware
                        # procesa tan rápido/lento como pueda (en la práctica,
                        # con GPU, bastante más rápido que la reproducción
                        # real) — usar wall-clock ahí desalinea los umbrales
                        # temporales de vision/ (180s de inmovilidad, ventana
                        # de velocidad de caída, etc.) contra el contenido
                        # real del video. Ver "Trampa real" en README.md:
                        # un clip de 1168.5s se procesó en 449s de reloj real
                        # (ratio 0.38x), lo que requería ~470s de video real
                        # para acumular 180s de inmovilidad "de reloj".
                        timestamp_s = played_frames * frame_duration

                    played_frames += 1
                    yield Frame(image=image, timestamp_s=timestamp_s)
            finally:
                cap.release()

            if not self._loop:
                return


class LiveStream:
    """Lee desde un stream en vivo (RTSP, HTTP, webcam). Reconecta ante cortes.

    Usa un thread lector en background que drena la cámara/stream lo más
    rápido posible y siempre deja el frame más reciente disponible — así
    `frames()` nunca entrega un frame viejo aunque el consumidor (YOLO)
    tarde más que el intervalo real entre frames: el atraso se traduce en
    frames salteados, no en lag acumulado.
    """

    def __init__(self, source: str, realtime: bool = False, stop_event: threading.Event | None = None):
        # Para webcam se pasa como int, para URLs como str
        try:
            self._source: int | str = int(source)
        except ValueError:
            self._source = source
        self._realtime = realtime
        self._start = time.monotonic()
        self._lock = threading.Lock()
        self._new_frame = threading.Event()
        self._latest: tuple[np.ndarray, float] | None = None
        # Se comparte el stop_event del caller (si se pasa uno) en vez de usar
        # siempre un Event propio: así una señal de corte externa (SIGTERM del
        # runner) interrumpe de inmediato tanto el loop de frames() como el
        # de reconexión en _reader_loop, en vez de depender de que primero
        # llegue un frame nuevo para que frames() note que hay que cortar.
        self._stop = stop_event if stop_event is not None else threading.Event()

    def frames(self) -> Iterator[Frame]:
        reader = threading.Thread(target=self._reader_loop, name="live-stream-reader", daemon=True)
        reader.start()
        try:
            while not self._stop.is_set():
                if not self._new_frame.wait(timeout=LATEST_FRAME_TIMEOUT_S):
                    continue
                with self._lock:
                    if self._latest is None:
                        continue
                    image, ts = self._latest
                    self._latest = None
                self._new_frame.clear()
                yield Frame(image=image, timestamp_s=ts)
        finally:
            self._stop.set()
            reader.join(timeout=RECONNECT_DELAY_S)

    def _reader_loop(self) -> None:
        """Corre en su propio thread: lee frames tan rápido como puede la
        fuente y sobreescribe el último frame disponible. Si `realtime` es
        False no hay drop explícito acá — igual el consumidor lento hace que
        `_latest` se sobreescriba varias veces antes de ser leído, así que
        el efecto de "quedarse con el más nuevo" aplica siempre."""
        while not self._stop.is_set():
            cap = cv2.VideoCapture(self._source)
            if not cap.isOpened():
                logger.warning("No se pudo conectar a %s, reintentando en %.0fs...",
                               self._source, RECONNECT_DELAY_S)
                self._stop.wait(RECONNECT_DELAY_S)
                continue

            fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
            logger.info("Conectado a %s (%.1f FPS)", self._source, fps)

            try:
                while not self._stop.is_set():
                    ok, image = cap.read()
                    if not ok:
                        logger.warning("Stream cortado en %s, reconectando...", self._source)
                        break
                    with self._lock:
                        self._latest = (image, time.monotonic() - self._start)
                    self._new_frame.set()
            finally:
                cap.release()

            if self._stop.is_set():
                return
            self._stop.wait(RECONNECT_DELAY_S)


_PROJECT_ROOT = Path(__file__).resolve().parents[1]


def make_source(
    source: str,
    realtime: bool = False,
    loop: bool = True,
    stop_event: threading.Event | None = None,
) -> VideoFileLoop | LiveStream:
    if _is_live_source(source):
        return LiveStream(source, realtime=realtime, stop_event=stop_event)
    # Resuelve rutas relativas al root del proyecto
    path = Path(source)
    if not path.is_absolute():
        path = _PROJECT_ROOT / path
    return VideoFileLoop(str(path), realtime=realtime, loop=loop)
