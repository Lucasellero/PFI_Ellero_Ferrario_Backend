"""Pose estimation + heurística de caída/inmovilidad con YOLOv11n-pose (Ultralytics).

Descriptores según Giri et al. (2025), citado en el informe (Sec. 2.2.7):
velocidad vertical de la cadera, aspect ratio del bounding box y ángulo de
orientación del torso, calculados sobre una ventana corta de frames.
"""
from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass

import numpy as np
from ultralytics import YOLO

# Índices de keypoints COCO (formato usado por los modelos de pose de YOLO).
L_SHOULDER, R_SHOULDER, L_HIP, R_HIP = 5, 6, 11, 12

FALL_ASPECT_RATIO = 1.4               # width/height por encima de esto => persona acostada
FALL_VERTICAL_VELOCITY = 0.35         # caída normalizada de cadera por segundo
IMMOBILITY_WINDOW_S = 30.0            # segundos de movimiento casi nulo => evento de inmovilidad
IMMOBILITY_MOVEMENT_THRESHOLD = 0.02  # desplazamiento normalizado de keypoints entre frames


@dataclass
class PoseSample:
    timestamp_s: float
    hip_y: float               # normalizado (0 = arriba, 1 = abajo del frame)
    aspect_ratio: float
    torso_angle_deg: float
    keypoints: np.ndarray      # normalizados a [0, 1] por ancho/alto del frame


def detect_person(model: YOLO, image: np.ndarray) -> tuple[np.ndarray, np.ndarray] | None:
    """Corre el modelo de pose sobre un frame y devuelve (keypoints, bbox) de la
    primera persona detectada, o None si no se detectó a nadie.

    Solo se trackea la primera persona: el informe excluye explícitamente el
    soporte multi-residente (Sec. 1.2.2).
    """
    result = model(image, verbose=False)[0]
    if result.keypoints is None or len(result.keypoints.xy) == 0:
        return None
    kpts = result.keypoints.xy[0].cpu().numpy()
    box = result.boxes.xyxy[0].cpu().numpy()
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
        aspect_ratio=aspect_ratio,
        torso_angle_deg=torso_angle_deg,
        keypoints=kpts / np.array([w, h]),
    )


class FallImmobilityDetector:
    """Consume frames de un video y emite eventos 'fall' / 'immobility'.

    Los umbrales de módulo son un punto de partida tomado de la literatura
    citada en el informe; hay que calibrarlos mirando los resultados sobre
    el dataset real (ver evaluation/).
    """

    def __init__(self, model_path: str = "yolo11n-pose.pt", history_seconds: float = 5.0):
        self._model = YOLO(model_path)
        self._history: deque[PoseSample] = deque()
        self._history_seconds = history_seconds
        self._immobile_since: float | None = None
        self._immobility_reported = False

    def process_frame(self, image: np.ndarray, timestamp_s: float) -> list[dict]:
        events: list[dict] = []
        h, w = image.shape[:2]
        detection = detect_person(self._model, image)
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
        if len(self._history) < 2:
            return False
        prev = self._history[0]
        dt = max(sample.timestamp_s - prev.timestamp_s, 1e-3)
        vertical_velocity = (sample.hip_y - prev.hip_y) / dt

        return (
            vertical_velocity > FALL_VERTICAL_VELOCITY
            and sample.aspect_ratio > FALL_ASPECT_RATIO
            and sample.torso_angle_deg > 45
        )

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
