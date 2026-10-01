"""Esquemas Pydantic de la API. El body de ingestión reusa `MonitoringEvent`/`EventType` de
`events/schema.py` tal cual — el informe pide explícitamente compartir ese esquema entre el
backend local y el de la nube, no inventar uno nuevo para el transporte Pi→nube.
"""
from __future__ import annotations

import sys
from datetime import datetime
from pathlib import Path

from pydantic import BaseModel

_EVENTS_DIR = Path(__file__).resolve().parents[2] / "events"
if str(_EVENTS_DIR) not in sys.path:
    sys.path.insert(0, str(_EVENTS_DIR))

from schema import EventType, MonitoringEvent  # noqa: E402  (de events/)

__all__ = ["EventType", "MonitoringEvent", "EventoOut", "CamaraIn", "CamaraOut", "HogarCreada"]


class EventoOut(BaseModel):
    id: int
    camara_id: int | None
    event_type: str
    occurred_at: datetime
    es_alerta: bool
    metadata: dict

    @staticmethod
    def from_orm_evento(evento) -> "EventoOut":
        """`evento.metadata_json` -> `metadata` en la respuesta — el atributo del modelo
        SQLAlchemy no puede llamarse `metadata` (reservado por `DeclarativeBase`), pero la API
        no tiene por qué exponer ese detalle interno."""
        return EventoOut(
            id=evento.id, camara_id=evento.camara_id, event_type=evento.event_type,
            occurred_at=evento.occurred_at, es_alerta=evento.es_alerta,
            metadata=evento.metadata_json,
        )


class CamaraIn(BaseModel):
    local_id: str
    ubicacion: str | None = None


class CamaraOut(BaseModel):
    id: int
    local_id: str
    ubicacion: str | None
    activa: bool

    model_config = {"from_attributes": True}


class HogarCreada(BaseModel):
    """Se devuelve una sola vez, al crear el hogar — después la API key cruda no se puede
    recuperar (solo se guarda el hash, ver models.Hogar)."""
    id: int
    nombre: str
    device_api_key: str
