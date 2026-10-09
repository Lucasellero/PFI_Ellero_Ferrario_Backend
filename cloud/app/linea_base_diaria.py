"""Ejecución diaria de la línea base dentro de la propia API.

Hasta acá `run_baseline_daily.py` existía pero nada lo llamaba en AWS (no hay EventBridge, ni
Lambdas, ni cron en la instancia de Elastic Beanstalk, confirmado en la consola el 2026-10-09).
En vez de sumar infraestructura, la API lanza al arrancar un thread que todos los días a las
03:00 de Argentina evalúa el día anterior de cada hogar con `baseline_runner.process_hogar`, la
misma función que usa el CLI.

- **Horario**: 03:00 en UTC-3 fijo (`ZONA_ARGENTINA`), explícito acá y no tomado de la zona
  horaria del contenedor (que en Elastic Beanstalk es UTC). Argentina no tiene horario de verano,
  así que el desfasaje fijo es correcto todo el año.
- **Una sola ejecución a la vez**: con Postgres se toma `pg_try_advisory_lock` antes de correr. Si
  hay más de una instancia, la que no consigue el candado no hace nada. Si dos instancias corren
  una después de la otra, la segunda no duplica desvíos: `process_hogar` ya es idempotente por
  `(fecha, detector, clave)`. Con SQLite (desarrollo) no hay candado.
- **Desactivar**: `LINEA_BASE_DIARIA_DESACTIVADA=1` (también acepta `true`/`si`), para tests y
  desarrollo local. El CLI `run_baseline_daily.py` sigue sirviendo para correr un día puntual a
  mano.
"""
from __future__ import annotations

import logging
import os
import threading
from datetime import date, datetime, time, timedelta, timezone

from sqlalchemy import text

from .baseline_runner import process_hogar
from .db import SessionLocal, engine
from .models import Hogar

logger = logging.getLogger(__name__)

ZONA_ARGENTINA = timezone(timedelta(hours=-3), "ART")
HORA_EJECUCION = time(3, 0)
# Número arbitrario que identifica este candado entre todos los advisory locks de la base.
CLAVE_CANDADO = 715_202_610
VARIABLE_DESACTIVAR = "LINEA_BASE_DIARIA_DESACTIVADA"


def esta_desactivada() -> bool:
    return os.environ.get(VARIABLE_DESACTIVAR, "").strip().lower() in {"1", "true", "si", "sí", "yes"}


def proxima_ejecucion(ahora: datetime) -> datetime:
    """Próximo 03:00 de Argentina estrictamente posterior a `ahora` (que debe tener zona horaria)."""
    ahora_ar = ahora.astimezone(ZONA_ARGENTINA)
    candidata = datetime.combine(ahora_ar.date(), HORA_EJECUCION, tzinfo=ZONA_ARGENTINA)
    if candidata <= ahora_ar:
        candidata += timedelta(days=1)
    return candidata


def dia_a_evaluar(ahora: datetime) -> date:
    """El día anterior completo, según el calendario de Argentina."""
    return ahora.astimezone(ZONA_ARGENTINA).date() - timedelta(days=1)


def ejecutar_linea_base(fecha: date) -> bool:
    """Evalúa `fecha` en todos los hogares. Devuelve False si otra instancia tenía el candado."""
    usa_candado = engine.dialect.name == "postgresql"
    with engine.connect() as conexion:
        if usa_candado:
            obtenido = conexion.execute(text("SELECT pg_try_advisory_lock(:clave)"), {"clave": CLAVE_CANDADO}).scalar()
            if not obtenido:
                logger.info("linea_base_diaria: otra instancia ya está corriendo, se omite %s", fecha)
                return False
        try:
            db = SessionLocal()
            try:
                for hogar in db.query(Hogar).all():
                    try:
                        process_hogar(db, hogar, fecha)
                    except Exception:
                        # Un hogar con datos raros no debe frenar al resto.
                        db.rollback()
                        logger.exception("linea_base_diaria: falló el hogar %s el %s", hogar.id, fecha)
            finally:
                db.close()
        finally:
            if usa_candado:
                conexion.execute(text("SELECT pg_advisory_unlock(:clave)"), {"clave": CLAVE_CANDADO})
    logger.info("linea_base_diaria: evaluado %s", fecha)
    return True


class LineaBaseDiaria(threading.Thread):
    """Thread de fondo que espera hasta las 03:00 de Argentina y corre `ejecutar_linea_base`."""

    def __init__(self) -> None:
        super().__init__(name="linea-base-diaria", daemon=True)
        self._detener = threading.Event()

    def run(self) -> None:
        while not self._detener.is_set():
            ahora = datetime.now(timezone.utc)
            siguiente = proxima_ejecucion(ahora)
            logger.info("linea_base_diaria: próxima ejecución %s", siguiente.isoformat())
            if self._detener.wait((siguiente - ahora).total_seconds()):
                break
            try:
                ejecutar_linea_base(dia_a_evaluar(datetime.now(timezone.utc)))
            except Exception:
                logger.exception("linea_base_diaria: la ejecución falló, se reintenta mañana")

    def detener(self) -> None:
        self._detener.set()
