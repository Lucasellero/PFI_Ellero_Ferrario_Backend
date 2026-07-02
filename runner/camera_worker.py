"""Thread que corre el pipeline completo para una cámara."""
from __future__ import annotations

import logging
import re
import sys
import threading
from pathlib import Path
from typing import Optional

import cv2

# Agrega los módulos hermanos al path
_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT / "events"))
sys.path.insert(0, str(_ROOT / "vision"))

from schema import EventType, MonitoringEvent  # noqa: E402  (de events/)
from store import EventStore                   # noqa: E402  (de events/)
from src.pose_pipeline import FallImmobilityDetector  # noqa: E402  (de vision/)

from .camera_source import make_source
from .config import CameraConfig

_FRAMES_DIR = _ROOT / "frames"
_FRAME_PREVIEW_EVERY_N = 5   # guarda 1 de cada 5 frames (~5fps preview para video a 25fps)

# ClipActivityClassifier se importa como paquete desde la raíz del proyecto
# (disponible porque python -m runner agrega el root al sys.path).
try:
    from vlm.src.clip_classifier import ClipActivityClassifier
    _CLIP_AVAILABLE = True
except ImportError:
    _CLIP_AVAILABLE = False


class CameraWorker(threading.Thread):
    """Un thread por cámara: lee frames y emite eventos al store compartido."""

    def __init__(
        self,
        config: CameraConfig,
        db_path: Path,
        pose_model: str,
        stop_event: threading.Event,
        clip_classifier: Optional["ClipActivityClassifier"] = None,
        clip_interval_s: float = 10.0,
    ):
        super().__init__(name=f"cam-{config.id}", daemon=True)
        self._config = config
        self._db_path = db_path
        self._pose_model = pose_model
        self._stop = stop_event
        self._clip = clip_classifier
        self._clip_interval = clip_interval_s
        self._log = logging.getLogger(f"camera.{config.id}")
        # nombre de archivo seguro para el frame preview
        self._frame_name = re.sub(r'[^a-zA-Z0-9_-]', '_', config.id)
        self._frame_count = 0

    def run(self) -> None:
        store = EventStore(self._db_path)
        detector = FallImmobilityDetector(model_path=self._pose_model)
        source = make_source(self._config.source, self._config.realtime, self._config.loop)

        # Fuerza la primera clasificación CLIP al comenzar
        last_clip_ts: float = -self._clip_interval

        self._log.info("Iniciando [%s] ← %s", self._config.id, self._config.source)
        if self._clip:
            self._log.info("[%s] CLIP activo, muestreando cada %.0fs", self._config.id, self._clip_interval)
        else:
            self._log.info("[%s] CLIP no disponible, solo pose estimation", self._config.id)

        try:
            for frame in source.frames():
                if self._stop.is_set():
                    break

                # ── Frame preview para el dashboard ───────────────────────
                self._frame_count += 1
                if self._frame_count % _FRAME_PREVIEW_EVERY_N == 0:
                    self._save_preview(frame.image)

                # ── Pose pipeline (todos los frames) ──────────────────────
                raw_events = detector.process_frame(frame.image, frame.timestamp_s)
                for raw in raw_events:
                    event = MonitoringEvent(
                        video_id=self._config.id,
                        event_type=EventType(raw["event_type"]),
                        video_timestamp_s=raw["video_timestamp_s"],
                        metadata=raw["metadata"],
                    )
                    store.add(event)
                    self._log.info(
                        "[%s] %s @ %.2fs  meta=%s",
                        self._config.id,
                        event.event_type.value,
                        event.video_timestamp_s,
                        event.metadata,
                    )

                # ── CLIP (cada clip_interval_s) ───────────────────────────
                if self._clip and (frame.timestamp_s - last_clip_ts >= self._clip_interval):
                    last_clip_ts = frame.timestamp_s
                    try:
                        result = self._clip.classify(frame.image)
                        event = MonitoringEvent(
                            video_id=self._config.id,
                            event_type=EventType.ACTIVITY,
                            video_timestamp_s=frame.timestamp_s,
                            metadata=result,
                        )
                        store.add(event)
                        self._log.info(
                            "[%s] actividad @ %.2fs: %s (%.0f%%)",
                            self._config.id,
                            frame.timestamp_s,
                            result["activity_label"],
                            result["confidence"] * 100,
                        )
                    except Exception:
                        self._log.exception("[%s] Error en CLIP", self._config.id)

        except Exception:
            self._log.exception("Error fatal en cámara [%s]", self._config.id)
        finally:
            store.close()
            self._log.info("[%s] detenida.", self._config.id)

    def _save_preview(self, image) -> None:
        """Escribe un JPEG de preview en frames/{cam_id}.jpg de forma atómica."""
        try:
            _FRAMES_DIR.mkdir(exist_ok=True)
            tmp  = _FRAMES_DIR / f"{self._frame_name}.tmp.jpg"
            dest = _FRAMES_DIR / f"{self._frame_name}.jpg"
            cv2.imwrite(str(tmp), image, [cv2.IMWRITE_JPEG_QUALITY, 70])
            tmp.replace(dest)
        except Exception:
            pass
