"""Ingestión idempotente: `POST /devices/events` no guarda dos veces un evento con el mismo
`event_id` (lo que pasa cuando la controladora reenvía un lote cuya respuesta se perdió), y la
migración de Alembic que agrega la columna."""
from __future__ import annotations

import os
import sqlite3
import subprocess
import sys
import time
from pathlib import Path

import pytest

from app.models import Evento
from app.routers import devices

_CLOUD = Path(__file__).resolve().parents[1]


def _evento(event_id: str | None, tipo: str = "activity", camara: str = "cam1", **metadata) -> dict:
    evento = {
        "video_id": camara, "event_type": tipo, "video_timestamp_s": 0.0, "wall_clock_ts": time.time(),
        "metadata": metadata,
    }
    if event_id is not None:
        evento["event_id"] = event_id
    return evento


def _enviar(cliente, hogar: dict, eventos: list[dict]) -> dict:
    respuesta = cliente.post("/devices/events", headers={"X-API-Key": hogar["api_key"]}, json=eventos)
    assert respuesta.status_code == 201, respuesta.text
    return respuesta.json()


def _guardados(db, hogar: dict) -> list[str | None]:
    db.expire_all()
    filas = db.query(Evento).filter(Evento.residente_id == hogar["residente_id"]).order_by(Evento.id).all()
    return [f.event_id for f in filas]


@pytest.fixture
def hogar(crear_hogar):
    return crear_hogar("idem")


@pytest.fixture
def avisos(monkeypatch):
    """Los avisos (push) que la ingestión manda por alertas nuevas."""
    enviados = []
    monkeypatch.setattr(devices, "enviar_push_alerta", lambda tokens, **datos: enviados.append(datos["titulo"]))
    return enviados


def test_reenviar_el_mismo_lote_no_duplica(cliente, db, hogar):
    lote = [_evento(f"ev-{i}") for i in range(50)]

    assert _enviar(cliente, hogar, lote) == {"insertados": 50, "duplicados": 0}
    for _ in range(3):
        assert _enviar(cliente, hogar, lote) == {"insertados": 0, "duplicados": 50}

    assert _guardados(db, hogar) == [f"ev-{i}" for i in range(50)]


def test_un_lote_con_eventos_ya_recibidos_y_nuevos_guarda_solo_los_nuevos(cliente, db, hogar):
    _enviar(cliente, hogar, [_evento("a"), _evento("b")])

    assert _enviar(cliente, hogar, [_evento("b"), _evento("c"), _evento("a"), _evento("d")]) == {
        "insertados": 2, "duplicados": 2,
    }
    assert _guardados(db, hogar) == ["a", "b", "c", "d"]


def test_un_evento_repetido_dentro_del_mismo_lote_se_guarda_una_vez(cliente, db, hogar):
    assert _enviar(cliente, hogar, [_evento("x"), _evento("x"), _evento("y")]) == {"insertados": 2, "duplicados": 1}
    assert _guardados(db, hogar) == ["x", "y"]


def test_una_alerta_reenviada_no_vuelve_a_avisar(cliente, db, hogar, avisos):
    caida = _evento("caida-1", tipo="fall", confirmed=True)

    _enviar(cliente, hogar, [caida])
    _enviar(cliente, hogar, [caida])

    assert avisos == ["Posible caída detectada"]
    db.expire_all()
    assert db.query(Evento).filter(Evento.es_alerta.is_(True)).count() == 1


def test_eventos_sin_identificador_se_guardan_siempre(cliente, db, hogar):
    """Una controladora sin actualizar no manda `event_id`: sigue funcionando como antes, sin
    protección contra duplicados."""
    lote = [_evento(None), _evento(None)]

    assert _enviar(cliente, hogar, lote) == {"insertados": 2, "duplicados": 0}
    assert _enviar(cliente, hogar, lote) == {"insertados": 2, "duplicados": 0}
    assert _guardados(db, hogar) == [None, None, None, None]


def test_el_mismo_identificador_en_otro_kit_no_se_considera_repetido(cliente, db, crear_hogar):
    """Los identificadores recibidos se miran por kit: lo que mandó uno no bloquea a otro."""
    uno, otro = crear_hogar("uno"), crear_hogar("otro")

    assert _enviar(cliente, uno, [_evento("compartido")]) == {"insertados": 1, "duplicados": 0}
    assert _enviar(cliente, otro, [_evento("compartido")]) == {"insertados": 1, "duplicados": 0}
    assert _enviar(cliente, otro, [_evento("compartido")]) == {"insertados": 0, "duplicados": 1}

    assert _guardados(db, uno) == ["compartido"]
    assert _guardados(db, otro) == ["compartido"]


def test_la_base_rechaza_dos_filas_con_el_mismo_identificador_en_una_camara(db, hogar):
    """Última barrera, por si dos pedidos iguales llegan a la vez: la restricción única."""
    from datetime import datetime, timezone

    from sqlalchemy.exc import IntegrityError

    def fila() -> Evento:
        ahora = datetime.now(timezone.utc)
        return Evento(
            camara_id=hogar["camara_ids"][0], residente_id=hogar["residente_id"], event_type="activity",
            occurred_at=ahora, synced_at=ahora, metadata_json={}, event_id="igual",
        )

    db.add(fila())
    db.commit()
    db.add(fila())
    with pytest.raises(IntegrityError):
        db.commit()
    db.rollback()


# ── Migración ────────────────────────────────────────────────────────────────

def _alembic(base: Path, *argumentos: str) -> None:
    entorno = {**os.environ, "DATABASE_URL": f"sqlite:///{base.as_posix()}"}
    subprocess.run(
        [sys.executable, "-m", "alembic", *argumentos], cwd=_CLOUD, env=entorno, check=True, capture_output=True,
    )


def test_la_migracion_agrega_la_columna_sin_tocar_los_eventos_que_ya_estaban(tmp_path):
    base = tmp_path / "nube.db"
    _alembic(base, "upgrade", "a3f9c2d17b40")  # el esquema desplegado antes de este cambio
    conexion = sqlite3.connect(base)
    conexion.execute("INSERT INTO hogares (id, nombre, fecha_alta) VALUES (1, 'Casa', '2026-10-01 00:00:00')")
    conexion.execute("INSERT INTO residentes (id, hogar_id, nombre) VALUES (1, 1, 'Rosa')")
    conexion.executemany(
        "INSERT INTO eventos (residente_id, event_type, occurred_at, synced_at, es_alerta, metadata) "
        "VALUES (1, ?, '2026-10-01 10:00:00', '2026-10-01 10:00:05', 0, '{}')",
        [("activity",), ("activity",), ("fall",)],
    )
    conexion.commit()
    conexion.close()

    _alembic(base, "upgrade", "head")

    conexion = sqlite3.connect(base)
    columnas = {fila[1] for fila in conexion.execute("PRAGMA table_info(eventos)")}
    filas = conexion.execute("SELECT event_type, event_id FROM eventos ORDER BY id").fetchall()
    esquema = conexion.execute("SELECT sql FROM sqlite_master WHERE name = 'eventos'").fetchone()[0]
    conexion.close()
    assert "event_id" in columnas
    assert filas == [("activity", None), ("activity", None), ("fall", None)]
    assert "uq_eventos_camara_event_id" in esquema

    _alembic(base, "downgrade", "a3f9c2d17b40")
    conexion = sqlite3.connect(base)
    columnas = {fila[1] for fila in conexion.execute("PRAGMA table_info(eventos)")}
    cantidad = conexion.execute("SELECT COUNT(*) FROM eventos").fetchone()[0]
    conexion.close()
    assert "event_id" not in columnas and cantidad == 3
