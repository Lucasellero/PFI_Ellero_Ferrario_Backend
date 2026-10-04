"""Thread que corre el pipeline completo para una cámara."""
from __future__ import annotations

import logging
import re
import sys
import threading
import time
from pathlib import Path
from typing import Optional

import cv2

# Agrega los módulos hermanos al path
_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT / "events"))
sys.path.insert(0, str(_ROOT / "vision"))

from schema import EventType, MonitoringEvent  # noqa: E402  (de events/)
from store import EventStore                   # noqa: E402  (de events/)
from src.pose_pipeline import FallImmobilityDetector, build_pose_sample, draw_pose  # noqa: E402  (de vision/)

from .atomic_file import replace_atomic
from .camera_source import make_source
from .config import CameraConfig
from .focus_state import read_focused_camera
from .motion_gate import MotionDetector
from .motion_state import write_motion_state
from .overlay_state import read_overlay_enabled

_FRAMES_DIR = _ROOT / "frames"
_FRAME_PREVIEW_EVERY_N = 5   # guarda 1 de cada 5 frames (~5fps preview para video a 25fps)
_FOCUS_POLL_S = 0.25         # cada cuánto se relee runner_state/focus.json (no en cada frame)

# Muestreo CLIP: por cada ventana de clip_interval_s se clasifican varios
# frames espaciados y se promedian los scores. Un frame único es ruidoso
# (persona de espaldas, a mitad de un movimiento); el promedio estabiliza.
_CLIP_SAMPLES_PER_WINDOW = 3
_CLIP_SAMPLE_SPACING_S = 1.0

# EatingDrinkingMedication es la categoría más floja del probe (46%, ver
# vlm/README.md) y la más sensible de reportar mal (medicación) — exige más
# confianza que clip_min_confidence antes de afirmarla. Por debajo de este
# umbral se degrada a la segunda categoría más probable, si esa sí supera
# clip_min_confidence.
_EATING_CONFIRM_THRESHOLD = 0.55

# Confirmación de caída con el clasificador temporal (fusion/src/fall_temporal.py).
# fall_score = P(caida). Por encima de _CONFIRM se confirma; por debajo de
# _REJECT se descarta; en el medio, si hay CLIP se usa como desempate, si no se
# emite con confirmed=False (no se descarta una caída geométrica por una lectura
# ambigua). Calibrado sobre el volcado de scores de vision/evaluate_falls.py
# --temporal --dump-scores (432 eventos, 5 datasets, modelo con hard-negative
# mining): los FP que llegan a tener ventana de esqueleto puntúan <=0.35
# (mediana ~0), las caídas reales >0.35 (mediana ~1.0). τ=0.15 mantiene el
# recall end-to-end (133/158) y rechaza 94/94 de los FP evaluables (1/113
# caídas reales rechazada de más). Los FP que igual pasan (~74, casi todos de
# ToyotaSmarthome_FP) son eventos SIN ventana: YOLO11n-pose no detecta a la
# persona en esa escena (sillón, cámara ancha/elevada — límite conocido, ver
# vision/README.md); esos caen a la confirmación CLIP de más abajo.
_FALL_TEMPORAL_CONFIRM = 0.45
_FALL_TEMPORAL_REJECT = 0.15

# ClipActivityClassifier se importa como paquete desde la raíz del proyecto
# (disponible porque python -m runner agrega el root al sys.path).
try:
    from vlm.src.clip_classifier import (
        ClipActivityClassifier,
        CATEGORY_DESCRIPTIONS,
        crop_to_hands,
        crop_to_person,
    )
    _CLIP_AVAILABLE = True
except ImportError:
    _CLIP_AVAILABLE = False

# Filtro de confirmación de caídas (ver vlm/src/fall_confirmation.py y
# vision/README.md, sección "Detección de caída como evento"): opcional,
# solo se activa si además de ClipActivityClassifier existe
# fall_confirmation_features.npy (generado por export_onnx.py).
try:
    from vlm.src.fall_confirmation import FallConfirmationClassifier
    _FALL_CONFIRMATION_AVAILABLE = True
except ImportError:
    _FALL_CONFIRMATION_AVAILABLE = False

# Clasificador temporal de esqueleto (ver fusion/README.md, sección
# "Integración"): opcional, degrada a CLIP solo si fusion/ no está
# entrenado/exportado todavía (fusion/train_temporal_classifier.py +
# fusion/export_temporal_onnx.py) — mismo patrón try/except que CLIP arriba.
try:
    from fusion.src.fusion import HybridActivityClassifier, ScoreFuser
    from fusion.src.keypoint_sequence import SkeletonSequenceBuffer, frame_features
    from fusion.src.zone_context import zone_one_hot
    _SKELETON_AVAILABLE = True
except ImportError:
    _SKELETON_AVAILABLE = False


class CameraWorker(threading.Thread):
    """Un thread por cámara: lee frames y emite eventos al store compartido."""

    def __init__(
        self,
        config: CameraConfig,
        db_path: Path,
        pose_model: str,
        stop_event: threading.Event,
        clip_classifier: Optional["ClipActivityClassifier"] = None,
        skeleton_classifier=None,
        fall_temporal_classifier=None,
        skeleton_weight: float = 0.5,
        clip_interval_s: float = 10.0,
        clip_min_confidence: float = 0.4,
        clip_heartbeat_s: float = 60.0,
        motion_gate_enabled: bool = True,
        motion_min_area_ratio: float = 0.005,
        motion_absence_timeout_s: float = 45.0,
    ):
        super().__init__(name=f"cam-{config.id}", daemon=True)
        self._config = config
        self._db_path = db_path
        self._pose_model = pose_model
        # OJO: no llamar self._stop — threading.Thread ya usa ese nombre
        # internamente (método privado), pisarlo rompe Thread.join().
        self._stop_event = stop_event
        self._clip = clip_classifier
        self._fall_confirm = None
        if self._clip is not None and _FALL_CONFIRMATION_AVAILABLE:
            try:
                self._fall_confirm = FallConfirmationClassifier(self._clip)
            except FileNotFoundError:
                pass  # fall_confirmation_features.npy no generado — solo geometría, ver README

        # Fusión esqueleto+CLIP (ver fusion/README.md, "Integración"): el
        # buffer acumula features de pose por frame (ver run()) y arma
        # ventanas en cada tick de _clip_tick. Si fusion/ no está disponible
        # (paquete no importable o modelo no exportado), self._skeleton_buffer
        # queda None y _hybrid degrada a mode="clip_only" — mismo
        # comportamiento que antes de este enganche.
        # Clasificador temporal de caída (ver fusion/src/fall_temporal.py):
        # segunda etapa del evento de caída, en reemplazo de la confirmación
        # CLIP cuando está disponible. Necesita el buffer de esqueleto igual
        # que la fusión de actividad — se crea abajo si hace falta para
        # cualquiera de los dos.
        self._fall_temporal = fall_temporal_classifier if _SKELETON_AVAILABLE else None

        self._skeleton_buffer = None
        self._hybrid = None
        need_buffer = _SKELETON_AVAILABLE and (
            (self._clip is not None and skeleton_classifier is not None) or self._fall_temporal is not None
        )
        if need_buffer:
            self._skeleton_buffer = SkeletonSequenceBuffer()
        if self._clip is not None:
            if _SKELETON_AVAILABLE and skeleton_classifier is not None:
                self._hybrid = HybridActivityClassifier(
                    mode="hybrid",
                    skeleton_classifier=skeleton_classifier,
                    clip_classifier=self._clip,
                    fuser=ScoreFuser(skeleton_weight=skeleton_weight),
                )
            elif _SKELETON_AVAILABLE:
                self._hybrid = HybridActivityClassifier(mode="clip_only", clip_classifier=self._clip)
            # si no _SKELETON_AVAILABLE, self._hybrid queda None y _clip_tick
            # usa self._clip.classify() directo (fallback más abajo).

        self._clip_interval = clip_interval_s
        self._clip_min_confidence = clip_min_confidence
        self._clip_heartbeat = clip_heartbeat_s
        self._motion_gate_enabled = motion_gate_enabled
        self._motion_min_area_ratio = motion_min_area_ratio
        self._motion_absence_timeout = motion_absence_timeout_s
        self._log = logging.getLogger(f"camera.{config.id}")
        # nombre de archivo seguro para el frame preview
        self._frame_name = re.sub(r'[^a-zA-Z0-9_-]', '_', config.id)
        self._frame_count = 0
        # Cache de _is_focused() (ver _FOCUS_POLL_S)
        self._focused = False
        self._focus_checked_at = 0.0

        # Estado del muestreo CLIP (ver _clip_tick)
        self._clip_sampling = False
        self._clip_scores: list[dict] = []
        self._clip_cat_scores: list[dict] = []
        self._clip_zs_cat_scores: list[dict] = []
        self._clip_skipped = 0
        self._clip_next_window_ts = 0.0
        self._clip_last_sample_ts = float("-inf")
        self._clip_prev_ts = float("-inf")
        self._last_activity_label: Optional[str] = None
        self._last_activity_emit_ts = float("-inf")

        # Estado del gate de movimiento (ver _update_motion_gate)
        self._motion: Optional[MotionDetector] = None
        self._awake = True
        self._last_person_seen_ts = float("-inf")

    def run(self) -> None:
        store = EventStore(self._db_path)
        detector = FallImmobilityDetector(model_path=self._pose_model)
        source = make_source(
            self._config.source, self._config.realtime, self._config.loop,
            stop_event=self._stop_event,
        )

        if self._motion_gate_enabled:
            self._motion = MotionDetector(min_area_ratio=self._motion_min_area_ratio)
            self._awake = False
            # Publica el estado inicial ya mismo — si no, el dashboard no
            # muestra nada para esta cámara hasta el primer preview (ver
            # motion_state.py).
            write_motion_state(self._config.id, awake=False)

        self._log.info("Iniciando [%s] ← %s", self._config.id, self._config.source)
        if self._clip:
            self._log.info(
                "[%s] CLIP activo: %d muestras/ventana cada %.0fs, confianza mínima %.0f%%",
                self._config.id, _CLIP_SAMPLES_PER_WINDOW, self._clip_interval,
                self._clip_min_confidence * 100,
            )
        else:
            self._log.info("[%s] CLIP no disponible, solo pose estimation", self._config.id)
        self._log.info(
            "[%s] fusión de esqueleto (fusion/): %s",
            self._config.id, "activa (modo hybrid)" if self._skeleton_buffer is not None else "inactiva (solo CLIP)",
        )
        self._log.info(
            "[%s] confirmación de caídas con CLIP: %s",
            self._config.id, "activa" if self._fall_confirm else "inactiva (solo heurística geométrica)",
        )
        if self._motion:
            self._log.info(
                "[%s] gate de movimiento activo: YOLO/CLIP arrancan dormidos hasta detectar movimiento, "
                "y vuelven a dormir tras %.0fs sin persona detectada",
                self._config.id, self._motion_absence_timeout,
            )

        try:
            for frame in source.frames():
                if self._stop_event.is_set():
                    break

                # ── Gate de movimiento: decide si este frame corre YOLO/CLIP ─
                if self._motion is not None:
                    self._update_motion_gate(frame)

                if not self._awake:
                    self._frame_count += 1
                    on_cadence = self._frame_count % _FRAME_PREVIEW_EVERY_N == 0
                    if on_cadence or self._is_focused():
                        self._save_preview(frame.image)
                    if on_cadence:
                        write_motion_state(self._config.id, awake=False)
                    continue

                # ── Pose pipeline (todos los frames, mientras está despierto) ─
                raw_events = detector.process_frame(frame.image, frame.timestamp_s)

                # ── Buffer de esqueleto para fusion/ (mismo criterio que el
                # historial de inmovilidad de vision/: push por frame con
                # detección, reset en cada hueco — no interpolar sobre el
                # vacío, ver keypoint_sequence.py) ─
                if self._skeleton_buffer is not None:
                    if detector.last_detection is not None:
                        kpts, box = detector.last_detection
                        h, w = frame.image.shape[:2]
                        ps = build_pose_sample(
                            kpts, box, w, h, frame.timestamp_s,
                            kpts_conf=detector.last_detection_conf,
                        )
                        feat = frame_features(
                            kpts, detector.last_detection_conf, box,
                            ps.aspect_ratio, ps.torso_angle_deg, ps.hip_y, ps.bbox_height_frac,
                        )
                        self._skeleton_buffer.push(frame.timestamp_s, feat)
                    else:
                        self._skeleton_buffer.reset()

                if self._motion is not None:
                    if detector.last_detection is not None:
                        self._last_person_seen_ts = frame.timestamp_s
                    elif frame.timestamp_s - self._last_person_seen_ts > self._motion_absence_timeout:
                        self._sleep()

                # ── Frame preview para el dashboard (con esqueleto de YOLO) ─
                # Mientras la cámara está en foco (vista única, ver focus_state.py),
                # se guarda cada frame en vez de 1 de cada _FRAME_PREVIEW_EVERY_N —
                # la pose ya corre sobre todos los frames estando despierta (línea de
                # arriba), así que esto no agrega costo de inferencia, solo de
                # encode+escritura del JPEG.
                self._frame_count += 1
                on_cadence = self._frame_count % _FRAME_PREVIEW_EVERY_N == 0
                if on_cadence or self._is_focused():
                    preview = frame.image
                    if detector.last_detection is not None and read_overlay_enabled():
                        kpts, box = detector.last_detection
                        preview = draw_pose(frame.image, kpts, box)
                    self._save_preview(preview)
                if on_cadence and self._motion is not None:
                    write_motion_state(self._config.id, awake=True)
                for raw in raw_events:
                    if raw["event_type"] == "fall":
                        # Evidencia parcial (persona fuera de cuadro / torso no
                        # visible): la firma geométrica es poco confiable y un
                        # recorte truncado hace que CLIP tampoco sirva. No se
                        # descarta ni se confirma con CLIP — se emite como
                        # sospecha (severity baja, confirmed=False) y el
                        # detector la escala a severidad alta si la persona
                        # queda inmóvil (evento immobility trigger=post_fall_sospecha).
                        if raw["metadata"].get("evidence") == "partial":
                            raw["metadata"]["severity"] = "sospecha"
                            raw["metadata"]["confirmed"] = False
                            self._log.info(
                                "[%s] caída de evidencia parcial @ %.2fs — se emite como sospecha "
                                "(escala por inmovilidad si la persona queda quieta)",
                                self._config.id, raw["video_timestamp_s"],
                            )
                        elif not self._confirm_fall_event(raw, frame, detector):
                            self._log.info(
                                "[%s] caída descartada @ %.2fs  meta=%s",
                                self._config.id, raw["video_timestamp_s"], raw["metadata"],
                            )
                            continue

                    event = MonitoringEvent(
                        video_id=self._config.id,
                        event_type=EventType(raw["event_type"]),
                        video_timestamp_s=raw["video_timestamp_s"],
                        wall_clock_ts=frame.wall_clock_ts,
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

    def _update_motion_gate(self, frame) -> None:
        """Corre el detector de movimiento barato y, si hay movimiento y el
        worker estaba dormido, lo despierta. No hace nada si ya está
        despierto — volver a dormir lo decide `run()` en base a si YOLO
        sigue viendo a alguien, no en base a este chequeo (ver docstring de
        motion_gate.py)."""
        motion_detected = self._motion.detect(frame.image)
        if motion_detected and not self._awake:
            self._wake(frame.timestamp_s)

    def _wake(self, ts: float) -> None:
        self._awake = True
        self._last_person_seen_ts = ts
        # Arranca una ventana de CLIP nueva en vez de retomar una a medio
        # llenar con muestras de antes de dormir (ver motion_gate.py).
        self._clip_sampling = False
        self._clip_next_window_ts = ts
        if self._skeleton_buffer is not None:
            self._skeleton_buffer.reset()
        write_motion_state(self._config.id, awake=True)
        self._log.info("[%s] movimiento detectado — YOLO/CLIP activados", self._config.id)

    def _sleep(self) -> None:
        self._awake = False
        write_motion_state(self._config.id, awake=False)
        self._log.info(
            "[%s] sin persona detectada por %.0fs — YOLO/CLIP en pausa hasta el próximo movimiento",
            self._config.id, self._motion_absence_timeout,
        )

    def _confirm_fall(self, frame, detector) -> tuple[bool, float]:
        """Segunda etapa del evento 'fall': confirma con CLIP zero-shot antes
        de reportarlo (ver vlm/src/fall_confirmation.py). Si no hay detección
        de persona en este frame exacto (no debería pasar, el evento se
        generó a partir de una detección), se confirma por defecto — mejor un
        falso positivo ocasional que perder una caída real por una falla de
        este chequeo adicional."""
        if detector.last_detection is None:
            return True, 0.0
        _, box = detector.last_detection
        crop = crop_to_person(frame.image, box)
        return self._fall_confirm.confirm(crop)

    def _confirm_fall_event(self, raw: dict, frame, detector) -> bool:
        """Segunda etapa de un evento de caída de evidencia completa.

        Prioridad: clasificador temporal de esqueleto (mira la *forma del
        movimiento*, la señal que distingue una caída de sentarse/acostarse a
        propósito) → si no está disponible o la ventana no tiene cobertura
        suficiente, confirmación CLIP (apariencia de un frame) → si tampoco,
        se confirma por defecto. CLIP también actúa de desempate cuando el
        temporal queda en zona incierta. Escribe en `raw["metadata"]` lo que
        cada etapa haya evaluado. Devuelve True si el evento se mantiene."""
        meta = raw["metadata"]
        window = (
            self._skeleton_buffer.build_window(frame.timestamp_s)
            if self._skeleton_buffer is not None else None
        )

        temporal_verdict: Optional[bool] = None  # None = indeciso / no evaluado
        if self._fall_temporal is not None and window is not None:
            res = self._fall_temporal.classify(window)
            meta["fall_temporal_label"] = res["label"]
            meta["fall_temporal_score"] = round(res["fall_score"], 4)
            if res["fall_score"] >= _FALL_TEMPORAL_CONFIRM:
                temporal_verdict = True
            elif res["fall_score"] <= _FALL_TEMPORAL_REJECT:
                temporal_verdict = False

        if temporal_verdict is True:
            meta["confirmed"] = True
            return True
        if temporal_verdict is False:
            meta["confirmed"] = False
            return False

        # Temporal indeciso o ausente → CLIP (desempate / camino previo).
        if self._fall_confirm is not None:
            confirmed, margin = self._confirm_fall(frame, detector)
            meta["clip_confirmation_margin"] = round(margin, 4)
            meta["confirmed"] = confirmed
            return confirmed

        # Sin CLIP: si el temporal corrió pero quedó en zona incierta, se
        # mantiene el evento pero marcado confirmed=False; si no corrió nada,
        # se confirma por defecto (comportamiento previo de _confirm_fall).
        meta["confirmed"] = "fall_temporal_label" not in meta
        return True

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
            if self._skeleton_buffer is not None:
                self._skeleton_buffer.reset()
        self._clip_prev_ts = ts

        if not self._clip_sampling:
            if ts < self._clip_next_window_ts:
                return
            self._clip_sampling = True
            self._clip_scores = []
            self._clip_cat_scores = []
            self._clip_zs_cat_scores = []
            self._clip_skipped = 0
            self._clip_last_sample_ts = float("-inf")
            self._clip_next_window_ts = ts + self._clip_interval

        if ts - self._clip_last_sample_ts < _CLIP_SAMPLE_SPACING_S:
            return
        self._clip_last_sample_ts = ts

        if detector.last_detection is None:
            self._clip_skipped += 1
        else:
            kpts, box = detector.last_detection
            kpts_conf = detector.last_detection_conf
            hand_crop = crop_to_hands(frame.image, kpts, box, keypoints_conf=kpts_conf)
            crop = crop_to_person(frame.image, box)

            # Contexto de zona del hogar (ver "Fase 1 de PLAN_sustitucion_clip.md..."
            # en vlm/README.md, 2026-09-21): geometría sobre los mismos
            # keypoints de cadera que ya calcula build_pose_sample más
            # arriba, sin modelo adicional. None si esta cámara no tiene
            # zones: configurado en cameras.yaml — degrada solo (ver
            # ClipActivityClassifier.classify), no rompe nada.
            zone_vector = None
            if _SKELETON_AVAILABLE and self._config.zones:
                h, w = frame.image.shape[:2]
                hip = (kpts[11] + kpts[12]) / 2.0
                zone_vector = zone_one_hot(self._config.zones, (float(hip[0] / w), float(hip[1] / h)))

            # Con esqueleto disponible, category_scores sale fusionado
            # (HybridActivityClassifier, ver fusion/src/fusion.py); si no,
            # cae a CLIP solo — mismo resultado que antes de este enganche.
            if self._hybrid is not None:
                window = (
                    self._skeleton_buffer.build_window(ts)
                    if self._skeleton_buffer is not None else None
                )
                fused = self._hybrid.classify(window, crop, hand_crop, zone_vector)
                clip_result = fused["clip"]
                cat_scores = fused["category_scores"]
            else:
                clip_result = self._clip.classify(crop, hand_crop, zone_vector)
                cat_scores = clip_result["category_scores"]

            self._clip_scores.append(clip_result["scores"])
            self._clip_cat_scores.append(cat_scores)
            self._clip_zs_cat_scores.append(clip_result["zeroshot_category_scores"])

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
        fine_label = max(avg, key=avg.get)

        # Categoría gruesa: promedio de category_scores por frame. Cada
        # frame usa el linear probe entrenado (ver train_category_probe.py)
        # cuando no es Background, o la agregación zero-shot cuando sí lo es
        # (el probe no tiene clase Background, ver ClipActivityClassifier.classify).
        all_cats = {c for s in self._clip_cat_scores for c in s}
        cat_scores = {c: sum(s.get(c, 0.0) for s in self._clip_cat_scores) / n for c in all_cats}
        top_cat = max(cat_scores, key=cat_scores.get)
        confidence = cat_scores[top_cat]

        # Categoría por la otra señal (agregación zero-shot de prompts de
        # texto, sin pasar por el probe) — mismo promedio de ventana. Se usa
        # solo como doble check de confianza, no reemplaza al probe (ver
        # "Doble check probe/zero-shot" en runner/README.md).
        zs_all_cats = {c for s in self._clip_zs_cat_scores for c in s}
        zs_cat_scores = {c: sum(s.get(c, 0.0) for s in self._clip_zs_cat_scores) / n for c in zs_all_cats}
        zs_top_cat = max(zs_cat_scores, key=zs_cat_scores.get) if zs_cat_scores else None

        if top_cat == "Background":
            # El recorte no muestra una persona reconocible: no hay actividad
            self._log.info(
                "[%s] ventana sin persona clara @ %.2fs (%.0f%%) — descartada",
                self._config.id, ts, confidence * 100,
            )
            return

        if confidence < self._clip_min_confidence:
            self._log.info(
                "[%s] actividad incierta @ %.2fs (mejor: %s %.0f%%) — descartada",
                self._config.id, ts, top_cat, confidence * 100,
            )
            return

        confirmed = True
        if top_cat == "EatingDrinkingMedication" and confidence < _EATING_CONFIRM_THRESHOLD:
            runner_up_cat = max(
                (c for c in cat_scores if c not in ("Background", top_cat)),
                key=cat_scores.get, default=None,
            )
            if runner_up_cat is not None and cat_scores[runner_up_cat] >= self._clip_min_confidence:
                self._log.info(
                    "[%s] EatingDrinkingMedication de baja confianza @ %.2fs (%.0f%% < %.0f%%) — "
                    "se reporta %s (%.0f%%) en su lugar",
                    self._config.id, ts, confidence * 100, _EATING_CONFIRM_THRESHOLD * 100,
                    runner_up_cat, cat_scores[runner_up_cat] * 100,
                )
                top_cat, confidence = runner_up_cat, cat_scores[runner_up_cat]

        # Deduplicación: emitir solo en cambio de actividad, con heartbeat
        # periódico para que el análisis registre que la actividad continúa.
        changed = top_cat != self._last_activity_label
        heartbeat_due = ts - self._last_activity_emit_ts >= self._clip_heartbeat
        if not changed and not heartbeat_due:
            return

        event = MonitoringEvent(
            video_id=self._config.id,
            event_type=EventType.ACTIVITY,
            video_timestamp_s=ts,
            wall_clock_ts=time.time(),
            metadata={
                "activity_label": top_cat,
                "fine_label": fine_label,
                "description": CATEGORY_DESCRIPTIONS[top_cat],
                "confidence": confidence,
                "confirmed": confirmed,
                "fusion_mode": self._hybrid.mode if self._hybrid is not None else "clip_only",
                "samples": n,
                "ongoing": not changed,
                "scores": {label: round(v, 4) for label, v in avg.items()},
                "category_scores": {c: round(v, 4) for c, v in cat_scores.items()},
                # Categoría de la señal zero-shot, guardada aparte del probe
                # para poder auditar desacuerdos al analizar video real sin
                # ground truth (ver "Doble check probe/zero-shot" en
                # runner/README.md) sin tener que reprocesar nada.
                "zeroshot_category": zs_top_cat,
            },
        )
        store.add(event)
        self._log.info(
            "[%s] actividad @ %.2fs: %s [%s] (%.0f%%, %d muestras%s%s)",
            self._config.id, ts, top_cat, fine_label, confidence * 100, n,
            ", continúa" if not changed else "",
            ", sin confirmar" if not confirmed else "",
        )
        self._last_activity_label = top_cat
        self._last_activity_emit_ts = ts

    def _is_focused(self) -> bool:
        """True si esta cámara es la que está en vista única en el dashboard."""
        now = time.monotonic()
        if now - self._focus_checked_at >= _FOCUS_POLL_S:
            self._focused = read_focused_camera() == self._config.id
            self._focus_checked_at = now
        return self._focused

    def _save_preview(self, image) -> None:
        """Escribe un JPEG de preview en frames/{cam_id}.jpg de forma atómica."""
        try:
            _FRAMES_DIR.mkdir(exist_ok=True)
            tmp  = _FRAMES_DIR / f"{self._frame_name}.tmp.jpg"
            dest = _FRAMES_DIR / f"{self._frame_name}.jpg"
            cv2.imwrite(str(tmp), image, [cv2.IMWRITE_JPEG_QUALITY, 70])
            replace_atomic(tmp, dest)
        except Exception:
            pass
