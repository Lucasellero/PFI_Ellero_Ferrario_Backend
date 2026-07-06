"""Pose estimation + heurística de caída/inmovilidad con YOLOv11n-pose (Ultralytics).

Descriptores según Giri et al. (2025), citado en el informe (Sec. 2.2.7):
velocidad vertical de la cadera y aspect ratio del bounding box.

Heurística de caída calibrada empíricamente: la velocidad vertical y el
aspect ratio "acostado" NO ocurren en el mismo frame — el pico de velocidad
pasa durante la caída (~0.3-1s) y el aspect ratio recién se estabiliza
después, cuando el cuerpo termina de asentarse en el piso. Por eso la
detección es en dos fases (pico de velocidad, después confirmación por
postura), no una condición simultánea. El ángulo de torso, calibrado
inicialmente según Giri et al., se descartó de la condición porque en
caídas laterales no llega a valores altos y hacía perder ~2/16 casos sin
mejorar la precisión de los que sí detectaba.

La velocidad vertical se normaliza por el alto del bbox de la persona (no
por el alto del frame): sin eso, el umbral quedaba atado a la distancia
cámara-sujeto de un único setup. Los umbrales actuales (`FALL_VERTICAL_VELOCITY`,
`FALL_ASPECT_RATIO`) se recalibraron con un grid search sobre 128 clips con
caída + 51 sin caída de dos datasets con cámaras distintas (`videos/GMDCSA-24/`,
mismo encuadre fijo; `videos/CAUCAFall/`, encuadre/distancia variable por
sujeto — ver `recalibrate_fall_thresholds.py`), para generalizar entre
setups de cámara en vez de sobreajustar a uno solo. Recall combinado 85.2%
(vs. 66.4% con los umbrales viejos y la velocidad sin normalizar), FP 18/179
clips evaluados.
"""
from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass

import numpy as np
from ultralytics import YOLO

# Índices de keypoints COCO (formato usado por los modelos de pose de YOLO).
L_SHOULDER, R_SHOULDER, L_HIP, R_HIP = 5, 6, 11, 12

# Pares de keypoints COCO que forman el esqueleto, para dibujar overlays de debug.
SKELETON_EDGES = [
    (5, 6), (5, 11), (6, 12), (11, 12),          # torso
    (5, 7), (7, 9), (6, 8), (8, 10),              # brazos
    (11, 13), (13, 15), (12, 14), (14, 16),       # piernas
    (0, 5), (0, 6),                                # cabeza-hombros
]

FALL_VELOCITY_WINDOW_S = 0.4          # ventana para medir la velocidad vertical de la cadera
FALL_VERTICAL_VELOCITY = 0.12         # "alturas de persona" por segundo (hip_y / bbox_height_frac) => posible caída en curso
FALL_ASPECT_RATIO = 1.4               # width/height por encima de esto => postura final acostada
FALL_CONFIRM_WINDOW_S = 1.5           # tiempo máx. entre el pico de velocidad y la postura final
FALL_COOLDOWN_S = 3.0                 # tras confirmar una caída, no reportar otra hasta pasado esto
MIN_KEYPOINT_CONFIDENCE = 0.2         # confianza promedio mínima de keypoints para no descartar el frame
IMMOBILITY_WINDOW_S = 30.0            # segundos de movimiento casi nulo => evento de inmovilidad
IMMOBILITY_MOVEMENT_THRESHOLD = 0.02  # desplazamiento normalizado de keypoints entre frames


@dataclass
class PoseSample:
    timestamp_s: float
    hip_y: float               # normalizado (0 = arriba, 1 = abajo del frame)
    bbox_height_frac: float    # alto del bbox / alto del frame, para normalizar la velocidad por la escala de la persona
    aspect_ratio: float
    torso_angle_deg: float
    keypoints: np.ndarray      # normalizados a [0, 1] por ancho/alto del frame


def detect_person(model: YOLO, image: np.ndarray) -> tuple[np.ndarray, np.ndarray] | None:
    """Corre el modelo de pose sobre un frame y devuelve (keypoints, bbox) de la
    persona detectada, o None si no hay ninguna detección confiable.

    Si hay más de una persona en el frame, se toma la del bounding box más
    grande (más cercana/prominente a cámara) como sujeto principal — el
    informe excluye explícitamente el soporte multi-residente (Sec. 1.2.2),
    así que esto es una heurística de desambiguación, no tracking real.
    Se descartan detecciones con confianza promedio de keypoints por debajo
    de MIN_KEYPOINT_CONFIDENCE (oclusión, movimiento, sujeto lejos de
    cámara) siguiendo el criterio de calidad usado en Benabdennour et al.
    (2026, IEEE Access) para este mismo problema.
    """
    result = model(image, verbose=False)[0]
    if result.keypoints is None or len(result.keypoints.xy) == 0:
        return None

    boxes = result.boxes.xyxy.cpu().numpy()
    areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    idx = int(areas.argmax())

    if result.keypoints.conf is not None:
        mean_confidence = float(result.keypoints.conf[idx].mean())
        if mean_confidence < MIN_KEYPOINT_CONFIDENCE:
            return None

    kpts = result.keypoints.xy[idx].cpu().numpy()
    box = boxes[idx]
    return kpts, box


def build_pose_sample(kpts: np.ndarray, box: np.ndarray, w: int, h: int, timestamp_s: float) -> PoseSample:
    x1, y1, x2, y2 = box
    bw, bh = max(x2 - x1, 1e-6), max(y2 - y1, 1e-6)
    # float(...) porque box/kpts vienen de numpy (float32), que no es
    # serializable a JSON al guardar el evento en events/store.py.
    aspect_ratio = float(bw / bh)

    hip = np.nanmean([kpts[L_HIP], kpts[R_HIP]], axis=0)
    shoulder = np.nanmean([kpts[L_SHOULDER], kpts[R_SHOULDER]], axis=0)
    dx, dy = shoulder[0] - hip[0], shoulder[1] - hip[1]
    torso_angle_deg = float(math.degrees(math.atan2(abs(dx), abs(dy) + 1e-6)))

    return PoseSample(
        timestamp_s=timestamp_s,
        hip_y=float(hip[1] / h),
        bbox_height_frac=float(bh / h),
        aspect_ratio=aspect_ratio,
        torso_angle_deg=torso_angle_deg,
        keypoints=kpts / np.array([w, h]),
    )


def draw_pose(image: np.ndarray, kpts: np.ndarray, box: np.ndarray) -> np.ndarray:
    """Dibuja el bounding box y el esqueleto de una detección sobre una copia
    del frame. Solo para debug/visualización (ej. preview del dashboard) —
    no se usa en la lógica de detección."""
    import cv2

    out = image.copy()
    x1, y1, x2, y2 = box.astype(int)
    cv2.rectangle(out, (x1, y1), (x2, y2), (0, 255, 0), 2)

    for a, b in SKELETON_EDGES:
        if a >= len(kpts) or b >= len(kpts):
            continue
        xa, ya = kpts[a]
        xb, yb = kpts[b]
        if (xa <= 0 and ya <= 0) or (xb <= 0 and yb <= 0):
            continue
        cv2.line(out, (int(xa), int(ya)), (int(xb), int(yb)), (0, 200, 255), 2)

    for x, y in kpts:
        if x <= 0 and y <= 0:
            continue
        cv2.circle(out, (int(x), int(y)), 3, (0, 0, 255), -1)

    return out


class FallImmobilityDetector:
    """Consume frames de un video y emite eventos 'fall' / 'immobility'.

    Umbrales de caída recalibrados sobre 179 clips de dos datasets con
    cámaras distintas, ver docstring del módulo y
    `recalibrate_fall_thresholds.py`. Los de inmovilidad siguen siendo un
    punto de partida de la literatura, sin calibrar todavía.
    """

    def __init__(self, model_path: str = "yolo11n-pose.pt", history_seconds: float = 1.5):
        self._model = YOLO(model_path)
        self._history: deque[PoseSample] = deque()
        self._history_seconds = history_seconds
        self._immobile_since: float | None = None
        self._immobility_reported = False
        self._last_velocity_spike_t: float | None = None
        self._fall_cooldown_until: float = float("-inf")
        self.last_detection: tuple[np.ndarray, np.ndarray] | None = None

    def process_frame(self, image: np.ndarray, timestamp_s: float) -> list[dict]:
        events: list[dict] = []
        h, w = image.shape[:2]
        detection = detect_person(self._model, image)
        self.last_detection = detection
        if detection is None:
            return events
        kpts, box = detection

        sample = build_pose_sample(kpts, box, w, h, timestamp_s)
        self._history.append(sample)
        self._trim_history(sample.timestamp_s)

        if self._detect_fall(sample):
            events.append({
                "event_type": "fall",
                "video_timestamp_s": sample.timestamp_s,
                "metadata": {
                    "aspect_ratio": sample.aspect_ratio,
                    "torso_angle_deg": sample.torso_angle_deg,
                },
            })

        events.extend(self._track_immobility(sample))
        return events

    def _trim_history(self, now: float) -> None:
        while self._history and now - self._history[0].timestamp_s > self._history_seconds:
            self._history.popleft()

    def _detect_fall(self, sample: PoseSample) -> bool:
        """Dos fases: (1) un pico de velocidad vertical marca una posible caída
        en curso, (2) si dentro de FALL_CONFIRM_WINDOW_S el aspect ratio se
        estabiliza en postura acostada, se confirma el evento. No exige que
        ambas condiciones ocurran en el mismo frame (ver docstring del módulo).
        """
        if sample.timestamp_s < self._fall_cooldown_until:
            return False

        velocity = self._vertical_velocity(sample)
        if velocity is not None and velocity > FALL_VERTICAL_VELOCITY:
            self._last_velocity_spike_t = sample.timestamp_s

        if (
            self._last_velocity_spike_t is not None
            and sample.aspect_ratio > FALL_ASPECT_RATIO
            and sample.timestamp_s - self._last_velocity_spike_t <= FALL_CONFIRM_WINDOW_S
        ):
            self._last_velocity_spike_t = None
            self._fall_cooldown_until = sample.timestamp_s + FALL_COOLDOWN_S
            return True
        return False

    def _vertical_velocity(self, sample: PoseSample) -> float | None:
        """Velocidad vertical de la cadera vs. la muestra más cercana a
        FALL_VELOCITY_WINDOW_S segundos atrás (no frame-a-frame, no el
        historial completo: ambos extremos diluyen el pico real de la caída).

        Se normaliza por el alto del bbox de la persona (promedio entre la
        muestra actual y la de referencia), no por el alto del frame: sin
        esto, el mismo movimiento real da velocidades distintas según la
        distancia cámara-sujeto (más lejos => bbox más chico => se subestima
        la velocidad y se pierden caídas; más cerca => se sobreestima y
        dispara con movimientos normales). Confirmado empíricamente al
        evaluar sobre CAUCAFall, que varía la distancia de cámara por sujeto
        (recall 28% vs. 91% en los clips propios, todos a la misma distancia).
        """
        if not self._history:
            return None
        target_t = sample.timestamp_s - FALL_VELOCITY_WINDOW_S
        ref = min(self._history, key=lambda s: abs(s.timestamp_s - target_t))
        dt = sample.timestamp_s - ref.timestamp_s
        if dt < FALL_VELOCITY_WINDOW_S / 2:
            return None
        scale = (sample.bbox_height_frac + ref.bbox_height_frac) / 2
        if scale <= 1e-6:
            return None
        return (sample.hip_y - ref.hip_y) / scale / dt

    def _track_immobility(self, sample: PoseSample) -> list[dict]:
        events: list[dict] = []
        if len(self._history) >= 2:
            prev = self._history[-2]
            displacement = float(np.nanmean(np.abs(sample.keypoints - prev.keypoints)))
            moving = displacement > IMMOBILITY_MOVEMENT_THRESHOLD
        else:
            moving = True

        if moving:
            self._immobile_since = None
            self._immobility_reported = False
        else:
            if self._immobile_since is None:
                self._immobile_since = sample.timestamp_s
            elapsed = sample.timestamp_s - self._immobile_since
            if elapsed > IMMOBILITY_WINDOW_S and not self._immobility_reported:
                self._immobility_reported = True
                events.append({
                    "event_type": "immobility",
                    "video_timestamp_s": sample.timestamp_s,
                    "metadata": {"duration_s": elapsed},
                })
        return events
