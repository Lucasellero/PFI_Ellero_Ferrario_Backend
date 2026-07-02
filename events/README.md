# `events/` — Esquema y almacenamiento de eventos

Este módulo es el **contrato** entre los productores de eventos (`vision/`,
`vlm/`) y el consumidor (`baseline/`). Es intencionalmente el módulo más
chico y estable del repo: cambios acá afectan a todos los demás, así que hay
que coordinar antes de modificarlo.

## Esquema

Ver [`schema.py`](schema.py) — `MonitoringEvent` y `EventType`.

Nota importante: como estamos procesando un dataset de video grabado (no
cámara en vivo), el timestamp del evento es **relativo al video**
(`video_timestamp_s`, segundos desde el inicio del archivo), no un timestamp
de reloj real. Esto cambiará cuando se migre a procesamiento en vivo (fase
futura).

## Almacenamiento

SQLite local (`events.db`, generado en este directorio, gitignored). Ver
[`store.py`](store.py), que expone:

- `EventStore.add(event: MonitoringEvent) -> None`
- `EventStore.all(video_id: str | None = None) -> list[MonitoringEvent]`

No hay servidor/API todavía (fase local): los demás módulos importan
`EventStore` directamente como librería Python, no por HTTP.

## Setup

```
cd events
py -m venv .venv && .venv\Scripts\activate
pip install -r requirements.txt
```
