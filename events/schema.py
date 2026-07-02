"""Esquema de evento: contrato compartido entre vision/, vlm/ y baseline/.

Nota: como estamos procesando un dataset de video grabado (no cámara en
vivo), el timestamp del evento es relativo al video (segundos desde el
inicio del archivo), no un timestamp de reloj real. Ver README.md.
"""
from __future__ import annotations

from enum import Enum

from pydantic import BaseModel, Field


class EventType(str, Enum):
    FALL = "fall"
    IMMOBILITY = "immobility"
    ACTIVITY = "activity"


class MonitoringEvent(BaseModel):
    video_id: str
    event_type: EventType
    video_timestamp_s: float
    metadata: dict = Field(default_factory=dict)
