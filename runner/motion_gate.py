"""Detector de movimiento barato para pausar YOLO/CLIP cuando no hay nadie
en cuadro.

`MotionDetector` no reemplaza a la detección de persona de `vision/`
(YOLO11n-pose): es un filtro previo, mucho más liviano (sustracción de fondo
sobre un frame reducido, del orden de 1-2ms en CPU), que decide si vale la
pena correr el pipeline pesado en este frame. Pensado para cámaras que
apuntan a una habitación vacía la mayor parte del día — evita correr YOLO
(~150-300ms/frame) y CLIP sobre frames donde no hay nada que ver.

Solo dispara el "despertar" del pipeline (ver `CameraWorker._wake` en
`camera_worker.py`); no decide por sí solo cuándo volver a dormir. Esa
decisión la toma YOLO mismo, al confirmar ausencia de persona durante
`motion_absence_timeout_s` — si el gate durmiera el pipeline apenas cesa el
movimiento, una persona inmóvil (posible caída, sedentarismo) se
confundiría con una habitación vacía, justo el evento que `vision/` necesita
seguir viendo (ver "Inmovilidad prolongada" en `vision/README.md`).
"""
from __future__ import annotations

import cv2
import numpy as np


class MotionDetector:
    """Sustracción de fondo (MOG2) sobre una copia reducida del frame."""

    def __init__(
        self,
        resize_width: int = 160,
        min_area_ratio: float = 0.005,
        history: int = 300,
        var_threshold: float = 16.0,
        warmup_frames: int = 15,
    ):
        self._resize_width = resize_width
        self._min_area_ratio = min_area_ratio
        self._back_sub = cv2.createBackgroundSubtractorMOG2(
            history=history, varThreshold=var_threshold, detectShadows=False,
        )
        self._warmup_frames = warmup_frames
        self._frame_count = 0

    def detect(self, image: np.ndarray) -> bool:
        """Devuelve True si el frame tiene movimiento por encima del umbral.

        Los primeros `warmup_frames` siempre devuelven False: el modelo de
        fondo de MOG2 arranca vacío y marca el frame entero como
        "movimiento" hasta aprender el fondo real — sin este warmup, toda
        cámara arrancaría en modo despierto sin importar si hay alguien.
        """
        h, w = image.shape[:2]
        scale = self._resize_width / w
        small = cv2.resize(
            image, (self._resize_width, max(1, int(h * scale))),
            interpolation=cv2.INTER_AREA,
        )
        gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
        fg_mask = self._back_sub.apply(gray)

        self._frame_count += 1
        if self._frame_count <= self._warmup_frames:
            return False

        moving_ratio = float(np.count_nonzero(fg_mask)) / fg_mask.size
        return moving_ratio > self._min_area_ratio
