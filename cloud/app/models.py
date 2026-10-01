"""Modelo de datos cloud — fiel al diagrama de base de datos del informe (capítulo 3,
`fig:diagrama_bd_cloud`): `HOGAR` 1:1 `RESIDENTE` (RNF11: un solo residente por instalación),
`HOGAR` 1:N `CAMARA`, `EVENTO` (FK a `CAMARA` + `RESIDENTE` desnormalizado), `FAMILIAR` N:M
`HOGAR` vía `HOGAR_FAMILIAR` con `rol`.

Tipos de columna portables (sin `JSONB`/`UUID` nativos de Postgres) para poder probar contra
SQLite local sin cambiar el modelo — ver `db.py`.
"""
from __future__ import annotations

import hashlib
import secrets
from datetime import datetime

from sqlalchemy import JSON, Boolean, DateTime, ForeignKey, String
from sqlalchemy.orm import Mapped, mapped_column, relationship

from .db import Base


def _new_api_key() -> str:
    return secrets.token_urlsafe(32)


def hash_api_key(raw_key: str) -> str:
    return hashlib.sha256(raw_key.encode("utf-8")).hexdigest()


class Hogar(Base):
    __tablename__ = "hogares"

    id: Mapped[int] = mapped_column(primary_key=True)
    nombre: Mapped[str] = mapped_column(String(200))
    # hash de la API key que usa runner/cloud_sync.py (header X-API-Key) — nunca se guarda en
    # texto plano. La key cruda se devuelve una sola vez al crear el hogar (ver routers/devices.py).
    device_api_key_hash: Mapped[str] = mapped_column(String(64), unique=True, index=True)

    residente: Mapped["Residente"] = relationship(back_populates="hogar", uselist=False)
    camaras: Mapped[list["Camara"]] = relationship(back_populates="hogar")
    familiares: Mapped[list["HogarFamiliar"]] = relationship(back_populates="hogar")

    @staticmethod
    def generar_api_key() -> tuple[str, str]:
        """Devuelve (api_key_cruda, hash) — la cruda se muestra una sola vez al usuario."""
        raw = _new_api_key()
        return raw, hash_api_key(raw)


class Residente(Base):
    __tablename__ = "residentes"

    id: Mapped[int] = mapped_column(primary_key=True)
    hogar_id: Mapped[int] = mapped_column(ForeignKey("hogares.id"), unique=True)  # 1:1 (RNF11)
    nombre: Mapped[str] = mapped_column(String(200))

    hogar: Mapped[Hogar] = relationship(back_populates="residente")


class Camara(Base):
    __tablename__ = "camaras"

    id: Mapped[int] = mapped_column(primary_key=True)
    hogar_id: Mapped[int] = mapped_column(ForeignKey("hogares.id"), index=True)
    # coincide con el `id` de esa cámara en cameras.yaml del lado de la Pi (ver runner/config.py)
    local_id: Mapped[str] = mapped_column(String(100))
    ubicacion: Mapped[str | None] = mapped_column(String(200), nullable=True)
    activa: Mapped[bool] = mapped_column(Boolean, default=True)

    hogar: Mapped[Hogar] = relationship(back_populates="camaras")


class Evento(Base):
    __tablename__ = "eventos"

    id: Mapped[int] = mapped_column(primary_key=True)
    # nullable: los eventos `routine_deviation` los genera `baseline_runner.py` a partir del
    # historial del residente, no una cámara puntual — no tiene sentido atribuirlos a una.
    camara_id: Mapped[int | None] = mapped_column(ForeignKey("camaras.id"), index=True, nullable=True)
    # desnormalizado a propósito (ver informe, Sección "Diagrama de Base de Datos"): evita
    # recorrer camara -> hogar -> residente cada vez que baseline_runner pide todos los eventos
    # de una persona.
    residente_id: Mapped[int] = mapped_column(ForeignKey("residentes.id"), index=True)
    event_type: Mapped[str] = mapped_column(String(30))  # ver events.schema.EventType
    occurred_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), index=True)
    synced_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))
    es_alerta: Mapped[bool] = mapped_column(Boolean, default=False)
    # atributo renombrado a propósito: `metadata` está reservado por `DeclarativeBase` (Base.metadata)
    metadata_json: Mapped[dict] = mapped_column("metadata", JSON, default=dict)

    camara: Mapped[Camara] = relationship()
    residente: Mapped[Residente] = relationship()


class Familiar(Base):
    __tablename__ = "familiares"

    id: Mapped[int] = mapped_column(primary_key=True)
    firebase_uid: Mapped[str] = mapped_column(String(128), unique=True, index=True)
    nombre: Mapped[str] = mapped_column(String(200))
    email: Mapped[str] = mapped_column(String(200))

    hogares: Mapped[list["HogarFamiliar"]] = relationship(back_populates="familiar")


class HogarFamiliar(Base):
    __tablename__ = "hogar_familiar"

    hogar_id: Mapped[int] = mapped_column(ForeignKey("hogares.id"), primary_key=True)
    familiar_id: Mapped[int] = mapped_column(ForeignKey("familiares.id"), primary_key=True)
    rol: Mapped[str] = mapped_column(String(50), default="responsable")

    hogar: Mapped[Hogar] = relationship(back_populates="familiares")
    familiar: Mapped[Familiar] = relationship(back_populates="hogares")
