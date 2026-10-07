"""Esquemas Pydantic de la API. El body de ingestión reusa `MonitoringEvent`/`EventType` de
`events/schema.py` tal cual — el informe pide explícitamente compartir ese esquema entre el
backend local y el de la nube, no inventar uno nuevo para el transporte Pi→nube.
"""
from __future__ import annotations

import sys
from datetime import datetime
from pathlib import Path

import re

from pydantic import BaseModel, Field, field_validator

_EVENTS_DIR = Path(__file__).resolve().parents[2] / "events"
if str(_EVENTS_DIR) not in sys.path:
    sys.path.insert(0, str(_EVENTS_DIR))

from schema import EventType, MonitoringEvent  # noqa: E402  (de events/)

__all__ = ["EventType", "MonitoringEvent", "EventoOut", "CamaraIn", "CamaraOut",
           "PerfilIn", "PerfilOut", "FamiliarOut", "HogarPerfilOut", "ResidenteOut", "KitResumenOut",
           "VincularKitIn", "KitVinculadoOut", "UbicacionIn", "ResidenteIn",
           "IndicadorRutinaOut", "ResumenRutinaOut", "PushTokenIn",
           "FamiliarVinculadoOut", "InvitarFamiliarIn", "CambiarRolIn"]


class EventoOut(BaseModel):
    id: int
    camara_id: int | None
    event_type: str
    occurred_at: datetime
    es_alerta: bool
    metadata: dict
    ubicacion: str | None  # `Camara.ubicacion`; null si el evento no vino de una cámara o no se cargó
    resuelta: bool  # `Evento.resuelta_at is not None`

    @staticmethod
    def from_orm_evento(evento) -> "EventoOut":
        """`evento.metadata_json` -> `metadata` en la respuesta — el atributo del modelo
        SQLAlchemy no puede llamarse `metadata` (reservado por `DeclarativeBase`), pero la API
        no tiene por qué exponer ese detalle interno."""
        return EventoOut(
            id=evento.id, camara_id=evento.camara_id, event_type=evento.event_type,
            occurred_at=evento.occurred_at, es_alerta=evento.es_alerta,
            metadata=evento.metadata_json,
            ubicacion=(evento.camara.ubicacion or None) if evento.camara is not None else None,
            resuelta=evento.resuelta_at is not None,
        )


class CamaraIn(BaseModel):
    identificador_local: str  # el `id` de la cámara en cameras.yaml de la Pi
    ubicacion: str | None = None


class CamaraOut(BaseModel):
    id: int
    identificador_local: str
    ubicacion: str | None  # "" en la base (sin cargar todavía) -> null en la API
    activa: bool

    model_config = {"from_attributes": True}

    @field_validator("ubicacion", mode="before")
    @classmethod
    def _vacia_es_null(cls, v):
        return v or None


# --- `GET /me` / `POST /me` (routers/me.py) ---------------------------------------------------
# Calcan `Perfil` / `HogarApi` de `src/api/types.ts` del repo Frontend: si cambia uno, cambiar
# el otro.


class PerfilIn(BaseModel):
    """Body de `POST /me`. El email NO viene acá: sale del token de Firebase (no se le cree al
    cliente). Mismo rango que `validarNombre` de la app (2 a 60)."""
    nombre: str = Field(min_length=2, max_length=60)

    @field_validator("nombre", mode="before")
    @classmethod
    def _strip(cls, v):
        return v.strip() if isinstance(v, str) else v


class FamiliarOut(BaseModel):
    id: int
    nombre: str
    email: str


class PushTokenIn(BaseModel):
    """Body de `POST /me/push-token` (RF8). El SDK de FCM en el cliente avisa con un callback
    (`onTokenRefresh`) cuando el token rota — la app lo vuelve a mandar en ese momento, no solo
    al loguearse."""
    token: str = Field(min_length=1, max_length=300)


class ResidenteOut(BaseModel):
    id: int
    nombre: str
    telefono: str | None


class KitResumenOut(BaseModel):
    numero_serie: str
    cantidad_camaras: int


class HogarPerfilOut(BaseModel):
    id: int
    nombre: str
    rol: str  # `HogarFamiliar.rol`
    residente: ResidenteOut | None
    kit: KitResumenOut | None  # `Hogar.kit_activo`


class PerfilOut(BaseModel):
    familiar: FamiliarOut
    # vacío = cuenta recién creada, sin kit vinculado -> la app muestra la vinculación
    hogares: list[HogarPerfilOut]


# --- Vinculación del kit (routers/kits.py, routers/mobile.py) ---------------------------------
# Contrato en `docs/vinculacion-kit.md` del repo Frontend, sección 5. Las reglas de validación
# repiten las de `src/lib/validaciones.ts` de la app (la app valida primero; esto es la última
# barrera).


def _colapsar_espacios(v):
    return " ".join(v.split()) if isinstance(v, str) else v


class VincularKitIn(BaseModel):
    codigo: str  # tal como lo tipeó/escaneó el familiar; se normaliza antes de hashear


class KitVinculadoOut(BaseModel):
    hogar_id: int
    numero_serie: str
    cantidad_camaras: int
    camaras: list[CamaraOut]  # las del plan, creadas en fábrica


class UbicacionIn(BaseModel):
    """`validarTextoCorto(texto, ..., 40)` de la app."""
    ubicacion: str = Field(min_length=2, max_length=40)

    @field_validator("ubicacion", mode="before")
    @classmethod
    def _limpiar(cls, v):
        return _colapsar_espacios(v)


_E164 = re.compile(r"^\+[1-9]\d{7,14}$")


class ResidenteIn(BaseModel):
    nombre: str = Field(min_length=2, max_length=60)
    telefono: str | None = None  # E.164 ("+5491123456789"); la app lo arma con validarTelefonoAR

    @field_validator("nombre", mode="before")
    @classmethod
    def _limpiar(cls, v):
        return _colapsar_espacios(v)

    @field_validator("telefono")
    @classmethod
    def _e164(cls, v):
        if v is not None and not _E164.match(v):
            raise ValueError("El teléfono tiene que estar en formato internacional (ej. +5491123456789).")
        return v


# --- Resumen de rutina (routers/mobile.py, rutina.py) ------------------------------------------
# Calca `IndicadorRutina`/`ResumenRutina` de `src/api/types.ts` del Frontend.


class IndicadorRutinaOut(BaseModel):
    clave: str
    titulo: str
    valor: str
    habitual: str
    estado: str  # "ok" | "desvio"


class ResumenRutinaOut(BaseModel):
    periodo: str  # "hoy" | "semana" | "mes"
    indicadores: list[IndicadorRutinaOut]


# --- CRUD de familiares/roles (routers/familiares.py, RF12) ------------------------------------


class FamiliarVinculadoOut(BaseModel):
    id: int
    nombre: str
    email: str
    rol: str  # ver ROLES en models.py


class InvitarFamiliarIn(BaseModel):
    email: str = Field(min_length=3, max_length=200)
    rol: str = Field(default="familiar")


class CambiarRolIn(BaseModel):
    rol: str
