"""Zona del hogar — Fase 1 de `fusion/PLAN_sustitucion_clip.md`.

A diferencia del contexto de objeto (`object_context.py`, que corre un
segundo modelo YOLO), la zona no se detecta: se **configura una vez** por
cámara (un instalador dibuja polígonos sobre un frame congelado — "esto es
la mesa", "esto es la cocina" — igual que las zonas de interés de cualquier
cámara de seguridad doméstica) y de ahí en más es geometría pura: en qué
polígono cae la posición de la persona. Sin costo de modelo adicional — usa
los keypoints que YOLO-pose ya calculó para el crop de manos.

Este módulo NO decide de dónde salen los polígonos (eso es
`cameras.yaml.zones` en producción, o `fusion/toyota_smarthome_zones.yaml`
para la validación sobre ese dataset) — solo resuelve "punto → zona".
"""
from __future__ import annotations

import numpy as np

# Vocabulario fijo y compartido: el vector de zona siempre tiene esta
# dimensión, sin importar cuántas zonas configuró una cámara en particular
# (una cámara que no configuró "cama" simplemente nunca activa esa columna,
# igual que TRACKED_OBJECTS en object_context.py incluye clases que en un
# video dado nunca aparecen). Vocabulario propuesto en el plan original
# (Fase 1): cocina, mesa, sofa, cama, baño, puerta.
ZONE_VOCAB: tuple[str, ...] = ("cocina", "mesa", "sofa", "cama", "bano", "puerta")

NUM_ZONES = len(ZONE_VOCAB)


def _point_in_polygon(point: tuple[float, float], polygon: list[tuple[float, float]]) -> bool:
    """Ray casting estándar (par/impar de cruces): sin dependencias nuevas
    (no hace falta shapely para polígonos simples, no necesariamente
    convexos, que es lo que un instalador va a dibujar a mano)."""
    x, y = point
    inside = False
    n = len(polygon)
    for i in range(n):
        x1, y1 = polygon[i]
        x2, y2 = polygon[(i + 1) % n]
        if (y1 > y) != (y2 > y):
            x_intersect = x1 + (y - y1) * (x2 - x1) / (y2 - y1)
            if x < x_intersect:
                inside = not inside
    return inside


def zone_one_hot(
    zones: dict[str, list[tuple[float, float]]] | None,
    point_norm: tuple[float, float],
    vocab: tuple[str, ...] = ZONE_VOCAB,
) -> np.ndarray:
    """Vector one-hot `(len(vocab),)`: 1.0 en la zona cuyo polígono contiene
    `point_norm` (coordenadas normalizadas 0-1 del frame), todo 0 si no hay
    zonas configuradas para esta cámara o el punto no cae en ninguna (mismo
    tratamiento que "objeto no detectado" en `object_context.py` — no es un
    error, es información real: "no sabemos en qué zona está").

    Si el punto cae en más de un polígono (zonas mal dibujadas que se
    superponen), se queda con la primera que matchea en el orden de
    `vocab` — determinístico, no hace falta resolverlo mejor para esto.
    """
    vec = np.zeros(len(vocab), dtype=np.float32)
    if not zones:
        return vec
    for i, name in enumerate(vocab):
        polygon = zones.get(name)
        if polygon and _point_in_polygon(point_norm, polygon):
            vec[i] = 1.0
            break
    return vec
