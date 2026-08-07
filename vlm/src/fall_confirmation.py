"""Filtro de confirmación con CLIP zero-shot para eventos de caída geométricos.

Motivación: `vision/src/pose_pipeline.py` (`FallImmobilityDetector`) detecta
caídas con una heurística puramente geométrica (velocidad vertical de la
cadera + aspect ratio del bounding box). Esa heurística no distingue una
caída real de sentarse/acostarse a propósito en algunos setups de cámara
(bbox ancho y bajo al estar sentado en el piso o un sillón bajo genera la
misma firma geométrica que una caída) — ver "Detección de caída como evento"
en `vision/README.md` para el detalle cuantitativo (recalibrar los umbrales
geométricos con datos de contraste de ese setup no lo resuelve: es un límite
estructural del enfoque, no un problema de calibración fina).

Este filtro es la segunda etapa: solo corre cuando la heurística geométrica
ya disparó (no en cada frame — CLIP es caro), y le pregunta a CLIP zero-shot
si el recorte de la persona en ese momento se parece más a una caída
accidental o a una postura voluntaria. Calibrado sobre 445 eventos
geométricos de 183 clips (`vision/calibrate_fall_confirmation.py`), con el
umbral recalibrado directamente contra el encoder ONNX INT8 real (ver nota
en `CONFIRMATION_MARGIN_THRESHOLD` — el prototipo inicial con CLIP fp32/torch
daba un número distinto que resultó miscalibrado para el modelo cuantizado):
`CONFIRMATION_MARGIN_THRESHOLD = -0.028` retiene 82.0% del recall geométrico
de caídas (105/128 casos, vs. 109/128 sin filtro) y elimina 67% de los
falsos positivos totales (288→96), 70% en el setup de cámara más afectado
(270→81). Ver `vision/README.md` para la tabla completa de trade-offs.

Reusa el encoder de imagen ONNX ya cargado por `ClipActivityClassifier`
(mismo modelo, otro set de prompts) — no carga un segundo modelo en memoria.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

MODELS_DIR = Path(__file__).resolve().parents[1] / "models"

# Prompt ensembling, mismo criterio que ACTIVITY_PROMPTS en clip_classifier.py.
FALL_PROMPTS = [
    "a photo of a person having an accidental fall",
    "a person collapsing and falling down to the floor",
    "someone tripping and falling on the ground",
    "a person in the middle of falling, losing balance",
]
NOTFALL_PROMPTS = [
    "a photo of a person sitting down calmly on a low chair or the floor",
    "a person intentionally lying down to rest",
    "someone sitting still on a sofa or ottoman watching TV",
    "a person calmly seated on the ground",
]

# Calibrado en vision/calibrate_fall_confirmation.py — ver docstring del módulo.
# IMPORTANTE: el prototipo inicial se calibró con CLIP fp32 (paquete `clip`,
# PyTorch) y dio -0.02, pero el modelo que corre en producción es el ONNX
# INT8 cuantizado (clip_image_encoder_int8.onnx) — la cuantización desplaza
# los embeddings lo suficiente como para que ese umbral no sea válido acá
# (con -0.02 sobre márgenes ONNX el recall real cae a 71.1%, no el 82.8%
# calibrado en torch). Este valor está recalibrado directamente contra
# márgenes calculados con ESTE encoder ONNX (ver
# vision/fall_confirmation_onnx_check.csv): recall 82.0% (105/128, -3.2pp vs.
# sin filtro) y elimina 67% de los falsos positivos totales (288→96), 70% en
# el setup de cámara más afectado (270→81).
CONFIRMATION_MARGIN_THRESHOLD = -0.028


class FallConfirmationClassifier:
    """Confirma o descarta un evento de caída geométrico ya disparado."""

    def __init__(self, image_classifier, models_dir: Path | None = None) -> None:
        """`image_classifier` es un `ClipActivityClassifier` ya inicializado
        (se reusa su encoder ONNX vía `encode_image`)."""
        models_dir = models_dir or MODELS_DIR
        features_path = models_dir / "fall_confirmation_features.npy"
        if not features_path.exists():
            raise FileNotFoundError(
                f"No se encontró {features_path}. "
                "Corré export_onnx.py en la máquina de desarrollo primero."
            )
        # [0] = embedding de texto promedio de FALL_PROMPTS, [1] = de NOTFALL_PROMPTS
        self._text_features = np.load(features_path)
        self._image_classifier = image_classifier

    def confirm(self, person_crop_bgr: np.ndarray) -> tuple[bool, float]:
        """Devuelve (confirmado, margin). `person_crop_bgr` ya debe estar
        recortado a la persona (ver `crop_to_person` en `clip_classifier.py`)."""
        image_features = self._image_classifier.encode_image(person_crop_bgr)
        fall_score = float(image_features @ self._text_features[0])
        notfall_score = float(image_features @ self._text_features[1])
        margin = fall_score - notfall_score
        return margin > CONFIRMATION_MARGIN_THRESHOLD, margin
