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
from src.pose_pipeline import FallImmobilityDetector, draw_pose  # noqa: E402  (de vision/)

from .camera_source import make_source
from .config import CameraConfig
from .overlay_state import read_overlay_enabled

_FRAMES_DIR = _ROOT / "frames"
_FRAME_PREVIEW_EVERY_N = 5   # guarda 1 de cada 5 frames (~5fps preview para video a 25fps)

# Muestreo CLIP: por cada ventana de clip_interval_s se clasifican varios
# frames espaciados y se promedian los scores. Un frame único es ruidoso
# (persona de espaldas, a mitad de un movimiento); el promedio estabiliza.
_CLIP_SAMPLES_PER_WINDOW = 3
_CLIP_SAMPLE_SPACING_S = 1.0

# ClipActivityClassifier se importa como paquete desde la raíz del proyecto
# (disponible porque python -m runner agrega el root al sys.path).
try:
    from vlm.src.clip_classifier import ClipActivityClassifier, ACTIVITY_PROMPTS, crop_to_person
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
        clip_min_confidence: float = 0.4,
        clip_heartbeat_s: float = 60.0,
    ):
        super().__init__(name=f"cam-{config.id}", daemon=True)
        self._config = config
        self._db_path = db_path
        self._pose_model = pose_model
        self._stop = stop_event
        self._clip = clip_classifier
        self._clip_interval = clip_interval_s
        self._clip_min_confidence = clip_min_confidence
        self._clip_heartbeat = clip_heartbeat_s
        self._log = logging.getLogger(f"camera.{config.id}")
        # nombre de archivo seguro para el frame preview
        self._frame_name = re.sub(r'[^a-zA-Z0-9_-]', '_', config.id)
        self._frame_count = 0

        # Estado del muestreo CLIP (ver _clip_tick)
        self._clip_sampling = False
        self._clip_scores: list[dict] = []
        self._clip_skipped = 0
        self._clip_next_window_ts = 0.0
        self._clip_last_sample_ts = float("-inf")
        self._clip_prev_ts = float("-inf")
        self._last_activity_label: Optional[str] = None
        self._last_activity_emit_ts = float("-inf")

    def run(self) -> None:
        store = EventStore(self._db_path)
        detector = FallImmobilityDetector(model_path=self._pose_model)
        source = make_source(self._config.source, self._config.realtime, self._config.loop)

        self._log.info("Iniciando [%s] ← %s", self._config.id, self._config.source)
        if self._clip:
            self._log.info(
                "[%s] CLIP activo: %d muestras/ventana cada %.0fs, confianza mínima %.0f%%",
                self._config.id, _CLIP_SAMPLES_PER_WINDOW, self._clip_interval,
                self._clip_min_confidence * 100,
            )
        else:
            self._log.info("[%s] CLIP no disponible, solo pose estimation", self._config.id)

        try:
            for frame in source.frames():
                if self._stop.is_set():
                    break

                # ── Pose pipeline (todos los frames) ──────────────────────
                raw_events = detector.process_frame(frame.image, frame.timestamp_s)

                # ── Frame preview para el dashboard (con esqueleto de YOLO) ─
                self._frame_count += 1
                if self._frame_count % _FRAME_PREVIEW_EVERY_N == 0:
                    preview = frame.image
                    if detector.last_detection is not None and read_overlay_enabled():
                        kpts, box = detector.last_detection
                        preview = draw_pose(frame.image, kpts, box)
                    self._save_preview(preview)
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

                # ── CLIP (ventanas de muestreo cada clip_interval_s) ──────
                if self._clip:
                    try:
                        self._clip_tick(frame, detector, store)
                    except Exception:
                        self._log.exception("[%s] Error en CLIP", self._config.id)

        except Exception:
            self._log.exception("Error fatal en cámara [%s]", self._config.id)
        finally:
            store.close()
            self._log.info("[%s] detenida.", self._config.id)

    def _clip_tick(self, frame, detector, store: EventStore) -> None:
        """Muestreo CLIP por ventanas.

        Cada clip_interval_s se abre una ventana que clasifica
        _CLIP_SAMPLES_PER_WINDOW frames (espaciados _CLIP_SAMPLE_SPACING_S)
        recortados a la persona detectada, y promedia los scores. Frames sin
        persona no se clasifican: una habitación vacía no tiene actividad.
        """
        ts = frame.timestamp_s

        # Video en loop: el timestamp se reinicia — resetear estado temporal
        if ts < self._clip_prev_ts - 1.0:
            self._clip_sampling = False
            self._clip_next_window_ts = ts
            self._last_activity_emit_ts = float("-inf")
        self._clip_prev_ts = ts

        if not self._clip_sampling:
            if ts < self._clip_next_window_ts:
                return
            self._clip_sampling = True
            self._clip_scores = []
            self._clip_skipped = 0
            self._clip_last_sample_ts = float("-inf")
            self._clip_next_window_ts = ts + self._clip_interval

        if ts - self._clip_last_sample_ts < _CLIP_SAMPLE_SPACING_S:
            return
        self._clip_last_sample_ts = ts

        if detector.last_detection is None:
            self._clip_skipped += 1
        else:
            _, box = detector.last_detection
            result = self._clip.classify(crop_to_person(frame.image, box))
            self._clip_scores.append(result["scores"])

        if len(self._clip_scores) + self._clip_skipped >= _CLIP_SAMPLES_PER_WINDOW:
            self._clip_sampling = False
            self._finish_clip_window(ts, store)

    def _finish_clip_window(self, ts: float, store: EventStore) -> None:
        if not self._clip_scores:
            # Sin persona en toda la ventana: al reaparecer, volver a emitir
            self._last_activity_label = None
            return

        labels = self._clip_scores[0].keys()
        n = len(self._clip_scores)
        avg = {label: sum(s[label] for s in self._clip_scores) / n for label in labels}
        top_label = max(avg, key=avg.get)
        confidence = avg[top_label]

        if confidence < self._clip_min_confidence:
            self._log.info(
                "[%s] actividad incierta @ %.2fs (mejor: %s %.0f%%) — descartada",
                self._config.id, ts, top_label, confidence * 100,
            )
            return

        # Deduplicación: emitir solo en cambio de actividad, con heartbeat
        # periódico para que el análisis registre que la actividad continúa.
        changed = top_label != self._last_activity_label
        heartbeat_due = ts - self._last_activity_emit_ts >= self._clip_heartbeat
        if not changed and not heartbeat_due:
            return

        event = MonitoringEvent(
            video_id=self._config.id,
            event_type=EventType.ACTIVITY,
            video_timestamp_s=ts,
            metadata={
                "activity_label": top_label,
                "description": ACTIVITY_PROMPTS[top_label][0],
                "confidence": confidence,
                "samples": n,
                "ongoing": not changed,
                "scores": {label: round(v, 4) for label, v in avg.items()},
            },
        )
        store.add(event)
        self._log.info(
            "[%s] actividad @ %.2fs: %s (%.0f%%, %d muestras%s)",
            self._config.id, ts, top_label, confidence * 100, n,
            ", continúa" if not changed else "",
        )
        self._last_activity_label = top_label
        self._last_activity_emit_ts = ts

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
