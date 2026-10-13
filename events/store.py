"""Almacenamiento local de eventos en SQLite."""
from __future__ import annotations

import json
import sqlite3
import time
import uuid
from pathlib import Path

from schema import EventType, MonitoringEvent

DB_PATH = Path(__file__).resolve().parent / "events.db"

_SCHEMA = """
CREATE TABLE IF NOT EXISTS events (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    video_id TEXT NOT NULL,
    event_type TEXT NOT NULL,
    video_timestamp_s REAL NOT NULL,
    wall_clock_ts REAL,
    cloud_synced_at REAL,
    metadata TEXT NOT NULL,
    event_id TEXT
)
"""

# Migraciones para DBs creadas antes de estas columnas (nuevas, nullable).
_MIGRATIONS = (
    ("wall_clock_ts", "ALTER TABLE events ADD COLUMN wall_clock_ts REAL"),
    # `cloud_synced_at`: cuándo `runner/cloud_sync.py` confirmó que este evento ya llegó al
    # backend cloud (RNF6/RNF7) — NULL = todavía no sincronizado. No es parte de `MonitoringEvent`
    # (es contabilidad interna de este store, no del contrato entre vision//vlm//baseline/).
    ("cloud_synced_at", "ALTER TABLE events ADD COLUMN cloud_synced_at REAL"),
    # `event_id`: identificador único del evento (ver `schema.py`). La nube lo usa para no
    # guardar dos veces un evento que la controladora reenvía.
    ("event_id", "ALTER TABLE events ADD COLUMN event_id TEXT"),
)


class EventStore:
    def __init__(self, db_path: Path = DB_PATH):
        self._conn = sqlite3.connect(db_path)
        self._conn.execute(_SCHEMA)
        existing_cols = {row[1] for row in self._conn.execute("PRAGMA table_info(events)")}
        for col, migration in _MIGRATIONS:
            if col not in existing_cols:
                self._conn.execute(migration)
        # Filas guardadas antes de que existiera `event_id` (o por código que escribe sin pasar
        # por `add`): se les asigna uno ahora, una sola vez, para que también queden protegidas
        # si todavía no se enviaron.
        self._conn.execute("UPDATE events SET event_id = lower(hex(randomblob(16))) WHERE event_id IS NULL")
        self._conn.execute("CREATE UNIQUE INDEX IF NOT EXISTS ix_events_event_id ON events (event_id)")
        self._conn.commit()

    def add(self, event: MonitoringEvent) -> None:
        """Guarda el evento. Si no trae `event_id` se le genera uno acá: el identificador nace
        con la fila y no cambia en los reenvíos."""
        self._conn.execute(
            "INSERT INTO events (video_id, event_type, video_timestamp_s, wall_clock_ts, metadata, event_id) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            (
                event.video_id,
                event.event_type.value,
                event.video_timestamp_s,
                event.wall_clock_ts,
                json.dumps(event.metadata),
                event.event_id or uuid.uuid4().hex,
            ),
        )
        self._conn.commit()

    def all(self, video_id: str | None = None) -> list[MonitoringEvent]:
        query = "SELECT video_id, event_type, video_timestamp_s, wall_clock_ts, metadata, event_id FROM events"
        params: tuple = ()
        if video_id is not None:
            query += " WHERE video_id = ?"
            params = (video_id,)
        rows = self._conn.execute(query, params).fetchall()
        return [
            MonitoringEvent(
                video_id=r[0],
                event_type=EventType(r[1]),
                video_timestamp_s=r[2],
                wall_clock_ts=r[3],
                metadata=json.loads(r[4]),
                event_id=r[5],
            )
            for r in rows
        ]

    def unsynced(self, video_id: str | None = None, limit: int = 500) -> list[tuple[int, MonitoringEvent]]:
        """Eventos con `cloud_synced_at IS NULL` — lo que `runner/cloud_sync.py` todavía no mandó
        al backend cloud. Devuelve el `id` de fila junto al evento porque hace falta para
        `mark_synced` después."""
        query = (
            "SELECT id, video_id, event_type, video_timestamp_s, wall_clock_ts, metadata, event_id "
            "FROM events WHERE cloud_synced_at IS NULL"
        )
        params: tuple = ()
        if video_id is not None:
            query += " AND video_id = ?"
            params = (video_id,)
        query += " ORDER BY id ASC LIMIT ?"
        params = params + (limit,)
        rows = self._conn.execute(query, params).fetchall()
        return [
            (
                r[0],
                MonitoringEvent(
                    video_id=r[1], event_type=EventType(r[2]), video_timestamp_s=r[3],
                    wall_clock_ts=r[4], metadata=json.loads(r[5]), event_id=r[6],
                ),
            )
            for r in rows
        ]

    def mark_synced(self, ids: list[int], synced_at: float) -> None:
        if not ids:
            return
        placeholders = ",".join("?" for _ in ids)
        self._conn.execute(
            f"UPDATE events SET cloud_synced_at = ? WHERE id IN ({placeholders})",
            (synced_at, *ids),
        )
        self._conn.commit()

    def discard(self, ids: list[int]) -> int:
        """Borra esas filas, estén sincronizadas o no, y devuelve cuántas borró. Un solo uso:
        eventos que la nube rechazó de forma definitiva porque el hogar se dio de baja (ver
        `runner/cloud_sync.py`). No es retención: para eso está `purge_synced`."""
        if not ids:
            return 0
        placeholders = ",".join("?" for _ in ids)
        cursor = self._conn.execute(f"DELETE FROM events WHERE id IN ({placeholders})", tuple(ids))
        self._conn.commit()
        return cursor.rowcount

    def purge_synced(self, older_than_days: float, now: float | None = None) -> int:
        """Retención local: borra los eventos que la nube ya confirmó (`cloud_synced_at` no nulo)
        hace más de `older_than_days` días y devuelve cuántos borró. Nunca toca un evento con
        `cloud_synced_at IS NULL`: mientras no se confirme el envío, la copia local es la única
        (RNF6/RNF7). El plazo corre desde la sincronización, no desde que ocurrió el evento.
        `now` (epoch Unix, mismo reloj que `mark_synced`) solo se pasa en las pruebas."""
        if older_than_days <= 0:
            raise ValueError("older_than_days tiene que ser mayor que 0")
        if now is None:
            now = time.time()
        cursor = self._conn.execute(
            "DELETE FROM events WHERE cloud_synced_at IS NOT NULL AND cloud_synced_at < ?",
            (now - older_than_days * 86400.0,),
        )
        self._conn.commit()
        return cursor.rowcount

    def close(self) -> None:
        self._conn.close()
