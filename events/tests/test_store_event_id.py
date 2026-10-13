"""`event_id`: cada evento guardado en la controladora tiene un identificador único y estable,
que es lo que la nube usa para no guardar dos veces un evento reenviado."""
from __future__ import annotations

import sqlite3

from schema import EventType, MonitoringEvent
from store import EventStore


def _evento(nombre: str = "cam1", event_id: str | None = None) -> MonitoringEvent:
    return MonitoringEvent(
        video_id=nombre, event_type=EventType.ACTIVITY, video_timestamp_s=0.0, wall_clock_ts=1_800_000_000.0,
        event_id=event_id,
    )


def test_cada_evento_guardado_recibe_un_identificador_distinto(tmp_path):
    store = EventStore(tmp_path / "events.db")
    for _ in range(200):
        store.add(_evento())  # 200 eventos idénticos en todo lo demás

    ids = [evento.event_id for _, evento in store.unsynced()]
    store.close()

    assert len(ids) == 200
    assert all(isinstance(i, str) and len(i) == 32 for i in ids)
    assert len(set(ids)) == 200


def test_el_identificador_no_cambia_entre_lecturas_ni_al_reabrir_la_base(tmp_path):
    db_path = tmp_path / "events.db"
    store = EventStore(db_path)
    store.add(_evento())
    store.add(_evento())
    primera = [evento.event_id for _, evento in store.unsynced()]
    segunda = [evento.event_id for _, evento in store.unsynced()]
    store.close()

    reabierto = EventStore(db_path)
    tercera = [evento.event_id for _, evento in reabierto.unsynced()]
    en_all = [evento.event_id for evento in reabierto.all()]
    reabierto.close()

    assert primera == segunda == tercera == en_all


def test_un_evento_que_ya_trae_identificador_lo_conserva(tmp_path):
    store = EventStore(tmp_path / "events.db")
    store.add(_evento(event_id="identificador-propio"))
    (_, guardado), = store.unsynced()
    store.close()

    assert guardado.event_id == "identificador-propio"


def test_una_base_anterior_recibe_identificadores_al_abrirse(tmp_path):
    """Base creada antes de que existiera la columna, con eventos todavía sin enviar: al
    abrirla se agrega la columna y cada fila queda con un identificador propio y estable."""
    db_path = tmp_path / "events.db"
    vieja = sqlite3.connect(db_path)
    vieja.execute(
        "CREATE TABLE events (id INTEGER PRIMARY KEY AUTOINCREMENT, video_id TEXT NOT NULL, "
        "event_type TEXT NOT NULL, video_timestamp_s REAL NOT NULL, wall_clock_ts REAL, "
        "cloud_synced_at REAL, metadata TEXT NOT NULL)"
    )
    vieja.executemany(
        "INSERT INTO events (video_id, event_type, video_timestamp_s, wall_clock_ts, metadata) "
        "VALUES (?, 'activity', 0.0, 1800000000.0, '{}')",
        [("cam1",), ("cam1",), ("cam2",)],
    )
    vieja.commit()
    vieja.close()

    store = EventStore(db_path)
    ids = [evento.event_id for _, evento in store.unsynced()]
    store.close()
    reabierto = EventStore(db_path)
    ids_despues = [evento.event_id for _, evento in reabierto.unsynced()]
    reabierto.close()

    assert len(ids) == 3 and None not in ids and len(set(ids)) == 3
    assert ids_despues == ids
