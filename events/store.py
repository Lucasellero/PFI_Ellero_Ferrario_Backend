"""Almacenamiento local de eventos en SQLite."""
from __future__ import annotations

import json
import sqlite3
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
    metadata TEXT NOT NULL
)
"""

# Migraciones para DBs creadas antes de estas columnas (nuevas, nullable).
_MIGRATIONS = (
    ("wall_clock_ts", "ALTER TABLE events ADD COLUMN wall_clock_ts REAL"),
    # `cloud_synced_at`: cuándo `runner/cloud_sync.py` confirmó que este evento ya llegó al
    # backend cloud (RNF6/RNF7) — NULL = todavía no sincronizado. No es parte de `MonitoringEvent`
    # (es contabilidad interna de este store, no del contrato entre vision//vlm//baseline/).
    ("cloud_synced_at", "ALTER TABLE events ADD COLUMN cloud_synced_at REAL"),
)


class EventStore:
    def __init__(self, db_path: Path = DB_PATH):
        self._conn = sqlite3.connect(db_path)
        self._conn.execute(_SCHEMA)
        existing_cols = {row[1] for row in self._conn.execute("PRAGMA table_info(events)")}
        for col, migration in _MIGRATIONS:
            if col not in existing_cols:
                self._conn.execute(migration)
        self._conn.commit()

    def add(self, event: MonitoringEvent) -> None:
        self._conn.execute(
            "INSERT INTO events (video_id, event_type, video_timestamp_s, wall_clock_ts, metadata) "
            "VALUES (?, ?, ?, ?, ?)",
            (
                event.video_id,
                event.event_type.value,
                event.video_timestamp_s,
                event.wall_clock_ts,
                json.dumps(event.metadata),
            ),
        )
        self._conn.commit()

    def all(self, video_id: str | None = None) -> list[MonitoringEvent]:
        query = "SELECT video_id, event_type, video_timestamp_s, wall_clock_ts, metadata FROM events"
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
            )
            for r in rows
        ]

    def unsynced(self, video_id: str | None = None, limit: int = 500) -> list[tuple[int, MonitoringEvent]]:
        """Eventos con `cloud_synced_at IS NULL` — lo que `runner/cloud_sync.py` todavía no mandó
        al backend cloud. Devuelve el `id` de fila junto al evento porque hace falta para
        `mark_synced` después."""
        query = (
            "SELECT id, video_id, event_type, video_timestamp_s, wall_clock_ts, metadata "
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
                    wall_clock_ts=r[4], metadata=json.loads(r[5]),
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

    def close(self) -> None:
        self._conn.close()
