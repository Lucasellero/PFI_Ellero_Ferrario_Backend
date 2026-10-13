"""Esquema de evento: contrato compartido entre vision/, vlm/ y baseline/.

Nota: como estamos procesando un dataset de video grabado (no cámara en
vivo), el timestamp del evento es relativo al video (segundos desde el
inicio del archivo), no un timestamp de reloj real. Ver README.md.

`wall_clock_ts` es el complemento de reloj real para el modo en vivo
(epoch Unix, hora local de la máquina que corre `runner/`): `None` en modo
batch sobre `videos/` (no tiene sentido ahí), poblado por `runner/` cuando
procesa un stream real. `baseline/` lo necesita para saber a qué día/hora
del mundo real pertenece cada evento de actividad.

`event_id` identifica al evento de forma única (UUID en hexadecimal). Lo
asigna `events/store.py` al guardarlo en la controladora, así que quien crea
el evento no necesita pasarlo. Viaja a la nube con el evento para que
`POST /devices/events` reconozca uno que ya recibió y no lo guarde dos veces
(ver "Ingestión idempotente" en `cloud/README.md`). `None` en un evento que
todavía no pasó por el store.
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
    event_id: str | None = None
    metadata: dict = Field(default_factory=dict)
