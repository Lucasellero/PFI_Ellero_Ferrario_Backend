"""Contexto de objeto vía YOLO-COCO — Fase 2 de
`fusion/PLAN_sustitucion_clip.md`.

Idea: CLIP describe la escena entera en un embedding holístico y no sabe
*dónde* está la persona en la casa — por eso confunde "cortar pan sobre la
mesada" con "servirse comida en la mesa" (ver "El probe de producción había
perdido 2 categorías..." en `vlm/README.md`, 2026-09-20: la mitad de los
errores de `Cooking` van a `EatingDrinkingMedication` justo en esos dos
gestos). Un detector de objetos COCO ya sabe distinguir "pileta de cocina"
de "mesa de comer" sin ambigüedad — este módulo corre esa segunda pasada de
YOLO (detección, no pose) y arma un vector de presencia de objetos
domésticos relevantes cerca de la persona.

Deliberadamente NO se toca `vision/` ni `vlm/` (mismo criterio de módulos
independientes que ya sigue `keypoint_sequence.py`) — esto vive en
`fusion/` y se usa desde los scripts de extracción de dataset como una
señal más, igual que el embedding CLIP.
"""
from __future__ import annotations

import numpy as np

# Subconjunto de las 80 clases COCO que un hogar real puede tener a la vista
# de una cámara doméstica y que ayudan a distinguir las categorías gruesas
# que hoy confunde CLIP (ver PLAN_sustitucion_clip.md, Fase 2). No se usan
# las 80 — el resto (auto, semáforo, animales, etc.) no aporta nada en este
# dominio y solo diluiría el vector.
TRACKED_OBJECTS: tuple[str, ...] = (
    "bed", "couch", "dining table", "tv", "oven", "sink", "refrigerator",
    "microwave", "toilet", "chair", "book", "cup", "bottle", "bowl",
)

NUM_TRACKED_OBJECTS = len(TRACKED_OBJECTS)

# Margen de "cercanía": se agranda la caja de la persona en cada dirección
# antes de comparar contra las cajas de objeto. Un objeto de mobiliario
# (mesa, pileta) casi nunca se superpone literalmente con el bbox de la
# persona (la persona está *al lado*, no *encima*) — sin este margen, la
# señal de contexto más útil (mesada/mesa/heladera) nunca dispararía.
_PROXIMITY_MARGIN_FRAC = 0.6


def _expand_box(box_xyxy: np.ndarray, margin_frac: float) -> np.ndarray:
    x1, y1, x2, y2 = (float(v) for v in box_xyxy)
    w, h = x2 - x1, y2 - y1
    mx, my = w * margin_frac, h * margin_frac
    return np.array([x1 - mx, y1 - my, x2 + mx, y2 + my], dtype=np.float32)


def _boxes_overlap(a: np.ndarray, b: np.ndarray) -> bool:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    return ax1 < bx2 and bx1 < ax2 and ay1 < by2 and by1 < ay2


def object_presence_vector(
    yolo_detector, frame_bgr: np.ndarray, person_box_xyxy: np.ndarray,
    conf_threshold: float = 0.25,
) -> np.ndarray:
    """Corre YOLO-COCO (detección) sobre el frame completo y devuelve un
    vector `(NUM_TRACKED_OBJECTS,)`: para cada clase trackeada, la
    confianza de la detección más segura que esté "cerca" de la persona
    (bbox de persona agrandado por `_PROXIMITY_MARGIN_FRAC`), o 0.0 si no
    se encontró ninguna. Valor continuo (no binario) para no perder la
    diferencia entre "hay una mesa clarísima al lado" y "puede haber una
    mesa, no estoy seguro" — el linear probe ya sabe aprovechar magnitudes
    continuas (igual que hace con los embeddings de CLIP).

    `yolo_detector`: instancia de `ultralytics.YOLO` cargada con pesos de
    detección COCO (`yolo11n.pt`, NO el `-pose.pt` que ya usa `vision/`/
    `keypoint_sequence.py` — es un modelo distinto, deliberadamente
    separado del de pose, ver Fase 2 en `PLAN_sustitucion_clip.md`).
    """
    vicinity = _expand_box(person_box_xyxy, _PROXIMITY_MARGIN_FRAC)
    vec = np.zeros(NUM_TRACKED_OBJECTS, dtype=np.float32)

    results = yolo_detector.predict(frame_bgr, conf=conf_threshold, verbose=False)
    if not results:
        return vec
    result = results[0]
    if result.boxes is None or len(result.boxes) == 0:
        return vec

    names = result.names
    xyxy = result.boxes.xyxy.cpu().numpy()
    conf = result.boxes.conf.cpu().numpy()
    cls_idx = result.boxes.cls.cpu().numpy().astype(int)

    class_to_slot = {name: i for i, name in enumerate(TRACKED_OBJECTS)}
    for box, c, ci in zip(xyxy, conf, cls_idx):
        name = names.get(int(ci), None) if isinstance(names, dict) else names[int(ci)]
        slot = class_to_slot.get(name)
        if slot is None:
            continue
        if not _boxes_overlap(vicinity, box):
            continue
        if c > vec[slot]:
            vec[slot] = float(c)
    return vec
