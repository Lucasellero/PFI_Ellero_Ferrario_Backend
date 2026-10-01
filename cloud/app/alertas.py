"""Deriva `Evento.es_alerta` a partir de `event_type`/`metadata` — el informe lo describe como
"un booleano derivado de la severidad del evento, en lugar de modelar una entidad Notificación
aparte" (ver capítulo 3, "Diagrama de Base de Datos"). Un solo lugar para esta regla: la usan
tanto la ingestión (`routers/devices.py`, eventos que vienen de `vision/`/`vlm/`) como
`baseline_runner.py` (eventos `routine_deviation` que genera la nube).
"""
from __future__ import annotations


def es_alerta(event_type: str, metadata: dict) -> bool:
    if event_type == "fall":
        # ver runner/camera_worker.py: una caída de evidencia parcial se marca confirmed=False
        # y escala a inmovilidad después — no es la alerta dura en sí misma todavía.
        return bool(metadata.get("confirmed", True))
    if event_type == "immobility":
        return metadata.get("severity") == "alta"
    if event_type == "routine_deviation":
        return True
    return False  # "activity": nunca es una alerta
