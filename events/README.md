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
de reloj real.

**Actualización 2026-10-01:** `MonitoringEvent` suma `wall_clock_ts` (epoch
Unix, hora local), el reloj real que faltaba — `None` en modo batch sobre
`videos/` (sigue sin sentido ahí), poblado por `runner/` en modo vivo
(`runner/camera_source.py::Frame.wall_clock_ts`, propagado en
`camera_worker.py`). Es lo que usa `baseline/run_live.py` para saber a qué
día/hora real pertenece cada evento de actividad — ver "Correr en
producción" en `baseline/README.md`. `video_timestamp_s` se mantiene tal
cual para el modo batch, no se reemplaza.

También se sumó `EventType.ROUTINE_DEVIATION`: lo escribe `baseline/
run_live.py` cuando detecta un desvío de rutina (no lo producen `vision/`
ni `vlm/`) — un evento por desvío encontrado, con `metadata.detector`
(`multivariado`/`clase`/`timing`/`tendencia`) y los campos propios de cada
uno. Ver `baseline/src/camera_events.py` y `baseline/run_live.py`.

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
