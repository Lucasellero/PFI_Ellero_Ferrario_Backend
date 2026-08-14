"""Espacio de clases del clasificador temporal (esqueleto).

Usa las mismas 31 clases finas de Toyota Smarthome que `vlm/src/clip_classifier.py`
(`ACTIVITY_PROMPTS`) — así el esqueleto y CLIP predicen sobre el mismo
vocabulario y se pueden fusionar/comparar en igualdad de condiciones — más
una clase `Standing` que CLIP no tiene.

`Standing` se sacó de `vlm/` porque el embedding fotométrico de CLIP no
generalizaba entre cámaras para esa clase: el linear probe daba 97.3% en la
cámara de entrenamiento y 0% de recall en una cámara nueva (ver
"`Standing`: se probó y se descartó" en `vlm/README.md`) — resultó estar
aprendiendo a reconocer la instalación, no la postura. El clasificador de
esqueleto de este módulo parte de una representación normalizada por
traslación/escala (ver `keypoint_sequence.frame_features`), así que en
principio no tiene ese modo de falla: "parado quieto" es la misma geometría
relativa sea cual sea la cámara o el fondo. Es una hipótesis razonable, no
un resultado medido — no hay datos de `Standing` en el dataset local (ver
`fusion/README.md`, sección "Datos") para confirmarla todavía. Se reserva
la clase en el espacio de labels para no tener que retocar la arquitectura/
export cuando se sumen esos datos (mismo criterio que ya se usó al ampliar
`vlm/` de 14 a 31 clases: ver "Vocabulario ampliado..." en `vlm/README.md`).

`LABEL_TO_CATEGORY`/`COARSE_CATEGORIES` extienden los de `vlm/` (import, no
copia) sumando `Standing` como categoría propia — se usan en `fusion.src.fusion`
para poder sumar/fusionar las dos señales en el mismo espacio de categorías
gruesas.
"""
from __future__ import annotations

from vlm.src.clip_classifier import (
    ACTIVITY_PROMPTS,
    COARSE_CATEGORIES as _CLIP_COARSE_CATEGORIES,
    LABEL_TO_CATEGORY as _CLIP_LABEL_TO_CATEGORY,
)

FINE_LABELS: list[str] = sorted(ACTIVITY_PROMPTS.keys()) + ["Standing"]

LABEL_TO_CATEGORY: dict[str, str] = {**_CLIP_LABEL_TO_CATEGORY, "Standing": "Standing"}

COARSE_CATEGORIES: dict[str, list[str]] = {**_CLIP_COARSE_CATEGORIES, "Standing": ["Standing"]}
