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
ya disparó (no en cada frame — CLIP es caro), y decide si el recorte de la
persona en ese momento se parece más a una caída accidental o a una postura
voluntaria.

Dos implementaciones, con fallback automático (ver `__init__`):

1. **Linear probe** (`vlm/models/fall_confirmation_probe.npz`, preferido si
   existe): regresión logística entrenada sobre embeddings CLIP de 509
   eventos geométricos de 292 clips (`vision/train_fall_confirmation_probe.py`),
   incluyendo como negativos duros el ruido geométrico real de clips
   domésticos largos y continuos (4 clips de `ToyotaSmarthome_FP`, 1100-1600s
   cada uno). Evaluado out-of-fold (GroupKFold por video, nunca ve en
   entrenamiento el video sobre el que se mide).
2. **Zero-shot** (`fall_confirmation_features.npy`, fallback si no hay
   probe): margen contra prompts fijos de "caída" vs. "postura voluntaria",
   sin entrenamiento. Calibrado originalmente sobre 445 eventos de 183 clips
   (`vision/calibrate_fall_confirmation.py`), umbral recalibrado directamente
   contra el encoder ONNX INT8 real (el prototipo con CLIP fp32/torch daba un
   número distinto, miscalibrado para el modelo cuantizado — ver nota en
   `CONFIRMATION_MARGIN_THRESHOLD`).

**Decisión (2026-08-13): se integró el probe en reemplazo del zero-shot.**
Comparados de forma justa sobre el mismo conjunto de evaluación —algo que
había quedado pendiente, ver `vision/README.md`— con
`vision/compare_confirmation_filters.py` (509 eventos, 292 clips, ambos
medidos igual: FP contra los mismos tramos "no-caída", recall contra el
mismo ground truth; el zero-shot no tiene entrenamiento así que se evalúa
directo, el probe out-of-fold para no favorecerlo con leakage):

| filtro    | recall      | FP  |
|-----------|-------------|-----|
| zero-shot | 82.3%       | 116 |
| probe     | **84.8%**   | **30** |

El probe domina en los dos ejes a la vez (más recall, ~4x menos FP) — no es
un trade-off, así que no hubo que priorizar según el criterio de "menos FP
por sobre más recall" del proyecto, ambos mejoraron juntos. El zero-shot
queda como fallback (por si `fall_confirmation_probe.npz` no está presente,
p. ej. en una instalación que no re-entrenó el probe) pero ya no es el
camino que corre en la práctica cuando el probe está disponible.

Reusa el encoder de imagen ONNX ya cargado por `ClipActivityClassifier`
(mismo modelo, otro set de prompts o pesos) — no carga un segundo modelo en
memoria.
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
    """Confirma o descarta un evento de caída geométrico ya disparado.

    Usa el linear probe (`fall_confirmation_probe.npz`) si está presente —
    domina al zero-shot en recall y FP a la vez, ver docstring del módulo—,
    y cae al filtro zero-shot (`fall_confirmation_features.npy`) si no."""

    def __init__(self, image_classifier, models_dir: Path | None = None) -> None:
        """`image_classifier` es un `ClipActivityClassifier` ya inicializado
        (se reusa su encoder ONNX vía `encode_image`)."""
        models_dir = models_dir or MODELS_DIR
        self._image_classifier = image_classifier

        probe_path = models_dir / "fall_confirmation_probe.npz"
        self._probe_W = self._probe_b = None
        self._text_features = None
        if probe_path.exists():
            probe = np.load(probe_path)
            self._probe_W = probe["W"][0]  # (512,)
            self._probe_b = float(probe["b"][0])
            return

        features_path = models_dir / "fall_confirmation_features.npy"
        if not features_path.exists():
            raise FileNotFoundError(
                f"No se encontró {probe_path} ni {features_path}. "
                "Corré export_onnx.py (y opcionalmente train_fall_confirmation_probe.py) "
                "en la máquina de desarrollo primero."
            )
        # [0] = embedding de texto promedio de FALL_PROMPTS, [1] = de NOTFALL_PROMPTS
        self._text_features = np.load(features_path)

    def confirm(self, person_crop_bgr: np.ndarray) -> tuple[bool, float]:
        """Devuelve (confirmado, margin). `person_crop_bgr` ya debe estar
        recortado a la persona (ver `crop_to_person` en `clip_classifier.py`).

        Con probe, `margin` es el logit de la regresión logística (>0 confirma).
        Con zero-shot, es la diferencia de similitud fall vs. notfall."""
        image_features = self._image_classifier.encode_image(person_crop_bgr)
        if self._probe_W is not None:
            logit = float(image_features @ self._probe_W + self._probe_b)
            return logit > 0.0, logit
        fall_score = float(image_features @ self._text_features[0])
        notfall_score = float(image_features @ self._text_features[1])
        margin = fall_score - notfall_score
        return margin > CONFIRMATION_MARGIN_THRESHOLD, margin
