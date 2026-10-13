"""Modelo de datos cloud — fiel al diagrama de base de datos del informe (Figura 3.11, "Diagrama
de Base de Datos en la Nube"; detalle campo por campo en `docs/vinculacion-kit.md` del repo
Frontend, sección 4):

- `HOGAR` 1:1 `RESIDENTE` (RNF11: un solo residente por instalación).
- `KIT` (la controladora + sus cámaras, existe desde fábrica): N:1 `HOGAR` (null hasta que un
  familiar escanea el QR), 1:N `CAMARA` (tantas como `cantidad_camaras`). La API key con la que
  la Pi manda eventos es del KIT, no del hogar: si se cambia la controladora, se vincula un kit
  nuevo al mismo hogar sin perder historial.
- `EVENTO` (FK a `CAMARA` —nullable— + `RESIDENTE` desnormalizado; `resuelta_por` FK a FAMILIAR).
- `FAMILIAR` N:M `HOGAR` vía `HOGAR_FAMILIAR` con `rol` (`responsable_principal` | `familiar`).
- `RESIDENTE` 1:1 `LINEA_BASE_COMPORTAMIENTO`.

Tipos de columna portables (sin `JSONB`/`UUID` nativos de Postgres) para poder probar contra
SQLite local sin cambiar el modelo — ver `db.py`.
"""
from __future__ import annotations

import hashlib
import secrets
from datetime import datetime, timezone

from sqlalchemy import JSON, Boolean, DateTime, ForeignKey, Integer, String, UniqueConstraint
from sqlalchemy.orm import Mapped, mapped_column, relationship

from .db import Base

ROL_RESPONSABLE_PRINCIPAL = "responsable_principal"
ROL_FAMILIAR = "familiar"
ROLES = (ROL_RESPONSABLE_PRINCIPAL, ROL_FAMILIAR)

# Crockford base32: sin I, L, O ni U (no se confunden con 1 y 0 al tipear). Mismo alfabeto que
# `src/features/vinculacion/codigoKit.ts` de la app.
_ALFABETO_CROCKFORD = "0123456789ABCDEFGHJKMNPQRSTVWXYZ"
LARGO_CODIGO_VINCULACION = 8


def _ahora() -> datetime:
    return datetime.now(timezone.utc)


def _new_api_key() -> str:
    return secrets.token_urlsafe(32)


def hash_api_key(raw_key: str) -> str:
    return hashlib.sha256(raw_key.encode("utf-8")).hexdigest()


def normalizar_codigo(texto: str) -> str:
    """Igual que `normalizarCodigo` de la app: mayúsculas, sin espacios ni guiones, O→0, I/L→1.
    "7k3p 9qxm" y "7K3P-9QXM" tienen que dar el mismo hash."""
    t = "".join(texto.upper().split()).replace("-", "")
    return t.replace("O", "0").replace("I", "1").replace("L", "1")


def hash_codigo(codigo: str) -> str:
    return hashlib.sha256(normalizar_codigo(codigo).encode("utf-8")).hexdigest()


class Hogar(Base):
    __tablename__ = "hogares"

    id: Mapped[int] = mapped_column(primary_key=True)
    nombre: Mapped[str] = mapped_column(String(200))
    direccion: Mapped[str | None] = mapped_column(String(300), nullable=True)
    fecha_alta: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_ahora)

    residente: Mapped["Residente"] = relationship(back_populates="hogar", uselist=False)
    kits: Mapped[list["Kit"]] = relationship(back_populates="hogar")
    camaras: Mapped[list["Camara"]] = relationship(back_populates="hogar")
    familiares: Mapped[list["HogarFamiliar"]] = relationship(back_populates="hogar")

    @property
    def kit_activo(self) -> "Kit | None":
        """El último kit vinculado (los anteriores quedan si se reemplazó la controladora)."""
        vinculados = [k for k in self.kits if k.vinculado_at is not None]
        return max(vinculados, key=lambda k: k.vinculado_at, default=None)


class Kit(Base):
    """Un kit físico: la controladora (Lar) con sus cámaras (Vela). Se da de alta en fábrica
    (ver `cloud/alta_kit.py`), antes de que exista un hogar."""
    __tablename__ = "kits"

    id: Mapped[int] = mapped_column(primary_key=True)
    numero_serie: Mapped[str] = mapped_column(String(20), unique=True)  # "LR-000123"
    cantidad_camaras: Mapped[int] = mapped_column(Integer)  # plan comprado, 1 a 3
    # SHA-256 del código del QR (normalizado) — el código nunca se guarda en claro
    codigo_vinculacion_hash: Mapped[str] = mapped_column(String(64), unique=True, index=True)
    # SHA-256 de la API key que usa runner/cloud_sync.py (header X-API-Key)
    device_api_key_hash: Mapped[str] = mapped_column(String(64), unique=True, index=True)
    hogar_id: Mapped[int | None] = mapped_column(ForeignKey("hogares.id"), index=True, nullable=True)
    vinculado_por: Mapped[int | None] = mapped_column(ForeignKey("familiares.id"), nullable=True)
    vinculado_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    # la controladora la actualiza en cada request autenticado (ver auth.require_kit_by_api_key)
    ultima_conexion_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    fabricado_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_ahora)

    hogar: Mapped[Hogar | None] = relationship(back_populates="kits")
    camaras: Mapped[list["Camara"]] = relationship(back_populates="kit")

    @staticmethod
    def generar_api_key() -> tuple[str, str]:
        """Devuelve (api_key_cruda, hash) — la cruda se muestra una sola vez (en fábrica)."""
        raw = _new_api_key()
        return raw, hash_api_key(raw)

    @staticmethod
    def generar_codigo_vinculacion() -> tuple[str, str]:
        """Devuelve (código_crudo, hash). El crudo va impreso en la etiqueta/QR del kit."""
        raw = "".join(secrets.choice(_ALFABETO_CROCKFORD) for _ in range(LARGO_CODIGO_VINCULACION))
        return raw, hash_codigo(raw)


class Residente(Base):
    __tablename__ = "residentes"

    id: Mapped[int] = mapped_column(primary_key=True)
    hogar_id: Mapped[int] = mapped_column(ForeignKey("hogares.id"), unique=True)  # 1:1 (RNF11)
    nombre: Mapped[str] = mapped_column(String(200))
    telefono: Mapped[str | None] = mapped_column(String(20), nullable=True)  # E.164, lo valida la app

    hogar: Mapped[Hogar] = relationship(back_populates="residente")
    linea_base: Mapped["LineaBaseComportamiento | None"] = relationship(
        back_populates="residente", uselist=False,
    )


class Camara(Base):
    __tablename__ = "camaras"
    __table_args__ = (
        UniqueConstraint("kit_id", "identificador_local", name="uq_camaras_kit_identificador_local"),
    )

    id: Mapped[int] = mapped_column(primary_key=True)
    kit_id: Mapped[int] = mapped_column(ForeignKey("kits.id"), index=True)
    # null hasta que el kit se vincula a un hogar
    hogar_id: Mapped[int | None] = mapped_column(ForeignKey("hogares.id"), index=True, nullable=True)
    # coincide con el `id` de esa cámara en cameras.yaml del lado de la Pi (ver runner/config.py)
    identificador_local: Mapped[str] = mapped_column(String(100))
    # obligatoria (Figura 3.11); "" = todavía sin cargar (la cámara se crea en fábrica y el
    # familiar la ubica en el paso 3 de la vinculación). La API expone "" como null.
    ubicacion: Mapped[str] = mapped_column(String(200), default="", server_default="")
    activa: Mapped[bool] = mapped_column(Boolean, default=True)

    kit: Mapped[Kit] = relationship(back_populates="camaras")
    hogar: Mapped[Hogar | None] = relationship(back_populates="camaras")


class Evento(Base):
    __tablename__ = "eventos"
    # Ingestión idempotente: una cámara no puede tener dos filas con el mismo `event_id`. Como
    # cada cámara es de un solo kit, equivale a "único por kit". Los eventos sin `event_id`
    # (los `routine_deviation`, o los de una controladora vieja) no chocan entre sí: NULL no
    # cuenta como repetido ni en Postgres ni en SQLite.
    __table_args__ = (UniqueConstraint("camara_id", "event_id", name="uq_eventos_camara_event_id"),)

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
    # "Marcar como resuelta" en la app: cuándo y quién atendió la alerta
    resuelta_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    resuelta_por: Mapped[int | None] = mapped_column(ForeignKey("familiares.id"), nullable=True)
    # `MonitoringEvent.event_id`: lo genera la controladora al guardar el evento y lo repite en
    # cada reenvío (ver `routers/devices.py`)
    event_id: Mapped[str | None] = mapped_column(String(64), nullable=True)

    camara: Mapped[Camara] = relationship()
    residente: Mapped[Residente] = relationship()


class Familiar(Base):
    __tablename__ = "familiares"

    id: Mapped[int] = mapped_column(primary_key=True)
    firebase_uid: Mapped[str] = mapped_column(String(128), unique=True, index=True)
    nombre: Mapped[str] = mapped_column(String(200))
    email: Mapped[str] = mapped_column(String(200))
    # token de FCM del celular (RF8). Uno por familiar: se pisa al iniciar sesión en otro equipo.
    push_token: Mapped[str | None] = mapped_column(String(300), nullable=True)

    hogares: Mapped[list["HogarFamiliar"]] = relationship(back_populates="familiar")


class HogarFamiliar(Base):
    __tablename__ = "hogar_familiar"

    hogar_id: Mapped[int] = mapped_column(ForeignKey("hogares.id"), primary_key=True)
    familiar_id: Mapped[int] = mapped_column(ForeignKey("familiares.id"), primary_key=True)
    # ver ROLES. `responsable_principal` lo asigna la vinculación del kit; por default, solo lectura.
    rol: Mapped[str] = mapped_column(String(50), default=ROL_FAMILIAR)

    hogar: Mapped[Hogar] = relationship(back_populates="familiares")
    familiar: Mapped[Familiar] = relationship(back_populates="hogares")


class LineaBaseComportamiento(Base):
    """Parámetros de la línea base del residente (rangos habituales por actividad). En el
    diagrama del informe; TODAVÍA NO SE ESCRIBE: `baseline_runner.py` la recalcula en cada
    corrida. Queda creada para que Reportes ("habitual 7:00–7:45") tenga de dónde leer."""
    __tablename__ = "lineas_base_comportamiento"

    id: Mapped[int] = mapped_column(primary_key=True)
    residente_id: Mapped[int] = mapped_column(ForeignKey("residentes.id"), unique=True)  # 1:1
    parametros_cluster: Mapped[dict] = mapped_column(JSON, default=dict)
    actualizado_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_ahora)

    residente: Mapped[Residente] = relationship(back_populates="linea_base")
