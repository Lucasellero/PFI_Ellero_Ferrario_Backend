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
    metadata TEXT NOT NULL
)
"""


class EventStore:
    def __init__(self, db_path: Path = DB_PATH):
        self._conn = sqlite3.connect(db_path)
        self._conn.execute(_SCHEMA)
        self._conn.commit()

    def add(self, event: MonitoringEvent) -> None:
        self._conn.execute(
            "INSERT INTO events (video_id, event_type, video_timestamp_s, metadata) "
            "VALUES (?, ?, ?, ?)",
            (
                event.video_id,
                event.event_type.value,
                event.video_timestamp_s,
                json.dumps(event.metadata),
            ),
        )
        self._conn.commit()

    def all(self, video_id: str | None = None) -> list[MonitoringEvent]:
        query = "SELECT video_id, event_type, video_timestamp_s, metadata FROM events"
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
                metadata=json.loads(r[3]),
            )
            for r in rows
        ]

    def close(self) -> None:
        self._conn.close()
