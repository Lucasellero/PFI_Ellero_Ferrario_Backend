"""Esquema de evento: contrato compartido entre vision/, vlm/ y baseline/.

Nota: como estamos procesando un dataset de video grabado (no cámara en
vivo), el timestamp del evento es relativo al video (segundos desde el
inicio del archivo), no un timestamp de reloj real. Ver README.md.

`wall_clock_ts` es el complemento de reloj real para el modo en vivo
(epoch Unix, hora local de la máquina que corre `runner/`): `None` en modo
batch sobre `videos/` (no tiene sentido ahí), poblado por `runner/` cuando
procesa un stream real. `baseline/` lo necesita para saber a qué día/hora
del mundo real pertenece cada evento de actividad.
"""
from __future__ import annotations

from enum import Enum

from pydantic import BaseModel, Field


class EventType(str, Enum):
    FALL = "fall"
    IMMOBILITY = "immobility"
    ACTIVITY = "activity"
    ROUTINE_DEVIATION = "routine_deviation"


class MonitoringEvent(BaseModel):
    video_id: str
    event_type: EventType
    video_timestamp_s: float
    wall_clock_ts: float | None = None
    metadata: dict = Field(default_factory=dict)
