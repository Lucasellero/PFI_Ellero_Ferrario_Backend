"""Ventanas temporales de keypoints para el clasificador de actividad por
movimiento (`temporal_classifier.py` / `temporal_model.py`).

Se apoya en la misma detección de pose que ya corre para caídas/inmovilidad
(`vision.src.pose_pipeline.detect_person`, expuesta en vivo vía
`FallImmobilityDetector.last_detection` / `.last_detection_conf`) — este
módulo NO carga un segundo modelo YOLO ni toca nada de `vision/`. Ver
`fusion/README.md`, sección "Integración", para el punto exacto donde
`runner/camera_worker.py` engancharía esto (mismo lugar donde ya engancha
CLIP hoy).
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass

import numpy as np

# Índices COCO de cadera, reusados de vision/src/pose_pipeline.py (mismo
# formato de keypoints que devuelve detect_person: (17, 2) en píxeles).
_L_HIP, _R_HIP = 11, 12
NUM_KEYPOINTS = 17

# Vector de features por frame, dimensión fija:
#   - 17 keypoints x (x, y) centrados en la cadera media y escalados por el
#     alto del bbox de la persona → invariante a traslación (posición en el
#     frame) y a distancia/zoom de cámara. Mismo criterio de normalización
#     que ya usa vision/src/pose_pipeline.py para la velocidad de caída
#     (_vertical_velocity) y el desplazamiento de inmovilidad
#     (_smoothed_displacement_rate): sin esto, el mismo movimiento real da
#     magnitudes distintas según qué tan lejos esté la cámara del sujeto.
#   - 17 confidencias de keypoint individuales (keypoints_conf de
#     detect_person): le indican al modelo qué puntos son ruido/oclusión en
#     vez de dejar que una posición no predicha, (0, 0), se lea como una
#     posición real más tras centrar/escalar.
#   - 4 descriptores escalares que vision/ ya calcula gratis por frame
#     (build_pose_sample): aspect_ratio, torso_angle_deg (normalizado a
#     [0, 1] dividiendo por 180), hip_y, bbox_height_frac. No cuestan nada
#     extra de cómputo — mismos números que ya alimentan la heurística de
#     caídas/inmovilidad.
FRAME_FEATURE_DIM = NUM_KEYPOINTS * 2 + NUM_KEYPOINTS + 4


def frame_features(
    keypoints_px: np.ndarray,
    keypoints_conf: np.ndarray | None,
    box_xyxy: np.ndarray,
    aspect_ratio: float,
    torso_angle_deg: float,
    hip_y: float,
    bbox_height_frac: float,
) -> np.ndarray:
    """Arma el vector de features de UN frame a partir de la misma
    detección que ya produce `vision.src.pose_pipeline.detect_person` +
    `build_pose_sample`.

    `keypoints_px` viene en coordenadas de **píxel** del frame completo
    (`.xy` de ultralytics, no `.xyn`) — igual que exige `crop_to_hands` en
    `vlm/src/clip_classifier.py` (ver el bug de coordenadas documentado en
    `vlm/README.md`: mezclar píxeles con normalizado ahí rompió el hand-crop
    en silencio durante toda una ronda de entrenamiento).
    """
    x1, y1, x2, y2 = (float(v) for v in box_xyxy)
    box_h = max(y2 - y1, 1e-6)

    hip = np.nanmean(keypoints_px[[_L_HIP, _R_HIP]], axis=0)
    centered = (keypoints_px - hip) / box_h  # (17, 2), invariante a traslación/escala

    conf = (
        keypoints_conf.astype(np.float32)
        if keypoints_conf is not None
        else np.ones(NUM_KEYPOINTS, dtype=np.float32)
    )

    scalars = np.array(
        [aspect_ratio, torso_angle_deg / 180.0, hip_y, bbox_height_frac],
        dtype=np.float32,
    )
    return np.concatenate([centered.flatten(), conf, scalars]).astype(np.float32)


@dataclass
class _SkeletonFrame:
    timestamp_s: float
    features: np.ndarray  # (FRAME_FEATURE_DIM,)


class SkeletonSequenceBuffer:
    """Acumula `_SkeletonFrame` por cámara/video y arma ventanas de tamaño
    fijo (`n_frames`) resampleadas dentro de los últimos `window_s`
    segundos — el input del clasificador temporal.

    Resampleo por vecino más cercano en el tiempo (no por índice de frame):
    mismo criterio que `_vertical_velocity`/`_smoothed_displacement_rate` en
    `vision/src/pose_pipeline.py`, tolerante a que el intervalo entre
    detecciones varíe (frame-dropping en tiempo real en `runner/`, o huecos
    de detección — ver "Bug de historial" en `vision/README.md`, mismo tipo
    de problema que motivó separar el historial de inmovilidad del de
    caídas ahí).

    Una sola implementación para los dos usos: en vivo (`runner/`, un
    `push()` por frame con detección, `build_window()` en cada tick de
    clasificación) y offline (`extract_dataset.py`, recorriendo un video
    frame a frame).
    """

    def __init__(
        self,
        window_s: float = 2.0,
        n_frames: int = 16,
        retention_s: float | None = None,
        gap_tolerance_s: float = 0.0,
    ) -> None:
        self.window_s = window_s
        self.n_frames = n_frames
        self.gap_tolerance_s = gap_tolerance_s
        self._retention_s = retention_s if retention_s is not None else window_s * 1.5
        self._history: deque[_SkeletonFrame] = deque()

    def push(self, timestamp_s: float, features: np.ndarray) -> None:
        self._history.append(_SkeletonFrame(timestamp_s, features))
        while self._history and timestamp_s - self._history[0].timestamp_s > self._retention_s:
            self._history.popleft()

    def mark_missing(self, timestamp_s: float) -> None:
        """Frame sin detección: resetea solo si el hueco ya supera
        `gap_tolerance_s` (0 = resetear al primer frame perdido, el
        comportamiento histórico). Con tolerancia, un parpadeo corto del
        detector no tira los ~2s de historia; `build_window` sigue exigiendo
        `min_coverage`, así que un hueco largo igual deja sin ventana."""
        if self._history and timestamp_s - self._history[-1].timestamp_s > self.gap_tolerance_s:
            self._history.clear()

    def reset(self) -> None:
        """Limpiar al perder la detección de la persona (o al pasar a otro
        video/cámara) — evita mezclar una ventana con contexto de antes de
        un hueco de detección largo o de un video distinto."""
        self._history.clear()

    def build_window(self, now: float, min_coverage: float = 0.6) -> np.ndarray | None:
        """Devuelve `(n_frames, FRAME_FEATURE_DIM)` o `None` si todavía no
        hay suficiente cobertura temporal real (arranque en frío, persona
        recién detectada, o un hueco de detección reciente).

        `min_coverage`: fracción mínima de los `n_frames` objetivo que debe
        tener una muestra real a menos de medio paso de muestreo de
        distancia (no solo "la muestra más cercana que haya, sea cual sea
        su antigüedad") — sin este chequeo, una ventana con una sola
        detección real se rellenaría entera repitiendo esa muestra y el
        modelo vería una secuencia "quieta" falsa.
        """
        if not self._history:
            return None
        step = self.window_s / max(self.n_frames - 1, 1)
        targets = now - self.window_s + np.linspace(0, self.window_s, self.n_frames)
        samples = list(self._history)

        rows: list[np.ndarray] = []
        covered = 0
        for t in targets:
            ref = min(samples, key=lambda f: abs(f.timestamp_s - t))
            if abs(ref.timestamp_s - t) <= step:
                covered += 1
            rows.append(ref.features)

        if covered / self.n_frames < min_coverage:
            return None
        return np.stack(rows).astype(np.float32)
