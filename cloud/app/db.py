"""Engine/sesión SQLAlchemy. `DATABASE_URL` por variable de entorno: Postgres (RDS) en
producción (`postgresql+psycopg2://usuario:password@host:5432/nombre_db`), SQLite local por
default para desarrollo/pruebas sin provisionar nada (ver `cloud/README.md`, "Correr localmente").

Tipos de columna portables a propósito (ver `models.py`: `JSON` genérico de SQLAlchemy, no
`JSONB` de Postgres) para que el mismo código funcione contra los dos motores sin ramas por
dialecto — importa más poder probar sin Postgres instalado que aprovechar features específicas
de un motor que todavía no está provisionado.
"""
from __future__ import annotations

import os
from collections.abc import Iterator

from sqlalchemy import create_engine
from sqlalchemy.orm import DeclarativeBase, Session, sessionmaker

DATABASE_URL = os.environ.get("DATABASE_URL", "sqlite:///./cloud_dev.db")

_connect_args = {"check_same_thread": False} if DATABASE_URL.startswith("sqlite") else {}
engine = create_engine(DATABASE_URL, connect_args=_connect_args)
SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False)


class Base(DeclarativeBase):
    pass


def get_db() -> Iterator[Session]:
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
