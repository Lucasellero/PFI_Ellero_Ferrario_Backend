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

**Actualización 2026-10-13:** `MonitoringEvent` suma `event_id` (opcional, `None`
por defecto): un identificador único del evento, UUID en hexadecimal. Quien crea
el evento no tiene que pasarlo: lo asigna `EventStore.add` al guardarlo (columna
`event_id` de `events.db`, con índice único) y lo devuelven `all()` y
`unsynced()`. Viaja a la nube con el evento, y la nube lo usa para no guardar dos
veces un evento que la controladora reenvía (ver "Ingestión idempotente" en
`cloud/README.md`). Una `events.db` creada antes de este cambio recibe la columna
al abrirse, y a las filas que ya tenía se les asigna un identificador en ese
momento, una sola vez. No cambia nada para `vision/`, `vlm/` ni `baseline/`.

## Almacenamiento

SQLite local (`events.db`, generado en este directorio, gitignored). Ver
[`store.py`](store.py), que expone:

- `EventStore.add(event: MonitoringEvent) -> None`
- `EventStore.all(video_id: str | None = None) -> list[MonitoringEvent]`
- `EventStore.unsynced(...)` / `EventStore.mark_synced(ids, synced_at)`: lo que
  usa `runner/cloud_sync.py` para enviar a la nube y anotar qué ya llegó
  (columna `cloud_synced_at`, `NULL` = todavía no enviado).
- `EventStore.purge_synced(older_than_days) -> int`: retención local, ver abajo.
- `EventStore.discard(ids) -> int`: borra esas filas aunque no estén enviadas. Un
  solo uso: eventos que la nube rechazó con 410 porque el hogar se dio de baja
  (ver "Retención local de eventos" en `runner/README.md`).

No hay servidor/API todavía (fase local): los demás módulos importan
`EventStore` directamente como librería Python, no por HTTP.

## Retención local (2026-10-13)

El informe (capítulo 3, Viabilidad Legal) declara que la controladora no
conserva los eventos indefinidamente: una vez que llegaron a la nube, la copia
local se borra a los pocos días. `EventStore.purge_synced(older_than_days)` es
lo que lo implementa. Borra las filas con `cloud_synced_at` no nulo y anterior
a `older_than_days` días, y devuelve cuántas borró.

- **La retención no borra nunca un evento sin sincronizar**, tenga la
  antigüedad que tenga: mientras la nube no confirmó el envío, la copia local es
  la única que existe (RNF6/RNF7). La condición está en el propio `DELETE`
  (`cloud_synced_at IS NOT NULL`), no en quien llama. La única vía por la que un
  evento sin enviar se borra es `discard`, y solo cuando la nube avisa que el
  hogar se dio de baja.
- **El plazo corre desde la sincronización, no desde que ocurrió el evento.**
  Si la controladora estuvo una semana sin conexión, los eventos de esa semana
  se envían al reconectar y recién ahí empiezan a contar sus días.
- Un plazo de 0 o negativo lanza `ValueError` sin borrar nada.
- No cambia el esquema: usa la columna `cloud_synced_at` que ya existía.
  `schema.py` no se tocó.

Este módulo solo ofrece el método. Quién lo llama, cada cuánto y con qué plazo
está en `runner/README.md` ("Retención local de eventos"): lo hace el thread de
sincronización, una vez por hora como mucho, con `retencion_local_dias` de
`cameras.yaml` (default 7). Sin sección `cloud:` nadie lo llama y el store
conserva todo, como antes.

**Qué cambia para quien lee `events.db`:** con la nube configurada, el store
local pasa a tener solo los eventos pendientes de envío y los enviados en los
últimos días. `dashboard/` sigue funcionando igual (muestra los últimos
eventos). `baseline/run_live.py`, en cambio, necesita semanas de historial
local: es una herramienta de desarrollo y no el camino de producción (la línea
base corre en la nube, ver `cloud/README.md`), pero no conviene usarla sobre
una controladora que sincroniza, porque va a ver solo los últimos días.

El archivo `events.db` no se achica al borrar: SQLite reutiliza las páginas
libres para los eventos nuevos, así que deja de crecer, pero no se compacta
(no se corre `VACUUM`).

## Pruebas

```
pip install pytest
pytest events/tests        # desde la raíz del repo
```

`tests/test_store_retencion.py` (2026-10-13) cubre `purge_synced` contra un
SQLite temporal: un evento sin sincronizar nunca se borra, la purga respeta los
días pedidos, el plazo se mide desde la sincronización, el límite exacto no se
borra y un plazo no positivo se rechaza. `tests/test_store_event_id.py`
(2026-10-13) cubre el identificador: 200 eventos idénticos reciben 200
identificadores distintos, no cambia entre lecturas ni al reabrir la base, se
respeta el que ya venga en el evento, y una base anterior al cambio recibe
identificadores al abrirse. La configuración de pytest
(`pytest.ini`) está en la raíz del repo; `pytest` a secas corre también las
pruebas de `runner/`, `dashboard/` y `cloud/`.

## Setup

```
cd events
py -m venv .venv && .venv\Scripts\activate
pip install -r requirements.txt
```
