"""Cliente de sincronización Pi -> nube (RNF6/RNF7 del informe): lee eventos locales no
sincronizados (`events/store.py::EventStore.unsynced`) y los manda en batch a
`POST /devices/events` del backend `cloud/` (ver `cloud/README.md`).

Solo hace conexiones SALIENTES — la Pi nunca necesita exponer un puerto ni tener IP pública,
coherente con la arquitectura de red del informe. Ante un corte de conexión simplemente no marca
nada como sincronizado y lo reintenta en el próximo ciclo; los eventos siguen disponibles
localmente mientras tanto (`events/events.db` es la fuente de verdad hasta que se confirma el
envío). La única respuesta que hace descartar eventos sin enviarlos es un 410: el hogar del kit se
dio de baja (ver `sync_once`).

Deshabilitado por default: sin la sección `cloud:` en `cameras.yaml` no hace nada (instalaciones
sin nube no se ven afectadas). Pensado para correr como thread de fondo dentro de `runner/main.py`
(ver `CloudSyncThread`), con el mismo patrón que `motion_gate`/`overlay_state`.
"""
from __future__ import annotations

import json
import logging
import sys
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path

logger = logging.getLogger(__name__)

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT / "events"))

from store import EventStore  # noqa: E402  (de events/)

from .cloud_state import SOURCE_PATH, STATE_PATH, read_cloud_source, write_cloud_state  # noqa: E402

_BATCH_SIZE = 200
_REQUEST_TIMEOUT_S = 15.0
# Cada cuánto, como mucho, se purgan los eventos ya enviados (ver `CloudSyncThread._purge_if_due`).
_PURGE_INTERVAL_S = 3600.0


def sync_once(store: EventStore, url: str, api_key: str) -> int:
    """Un ciclo de sincronización: manda el próximo lote de eventos pendientes. Devuelve cuántos
    se confirmaron. No lanza excepción ante error de red/HTTP — loguea y deja los eventos como
    pendientes para el próximo ciclo (ver docstring del módulo)."""
    return _sync_lote(store, url, api_key)[0]


def _sync_lote(store: EventStore, url: str, api_key: str, solo_desde_vinculacion: bool = False,
               camaras_locales: frozenset[str] = frozenset(),
               fuente: tuple[str | None, str] | None = None) -> tuple[int, str | None, str]:
    """`sync_once` más el estado de la nube para `runner/cloud_state.py`: (enviados, estado,
    detalle). `estado` es None si no había nada que mandar (no se le preguntó nada a la nube).
    Con `solo_desde_vinculacion`, un 409 (kit sin vincular) marca el lote como enviado sin
    mandarlo, en vez de dejarlo pendiente para cuando se vincule. Los eventos de
    `camaras_locales` (`nube: false` en cameras.yaml) se marcan como enviados sin mandarlos.

    `fuente` = (fuente elegida en el dashboard, cámara del kit): solo se sube esa fuente, con el
    id de la cámara del kit; las demás (y todas, si no hay ninguna elegida) quedan locales."""
    def es_local(video_id: str) -> bool:
        if fuente is not None:
            return video_id != fuente[0]
        return video_id in camaras_locales

    while True:
        pendientes = store.unsynced(limit=_BATCH_SIZE)
        locales = [i for i, ev in pendientes if es_local(ev.video_id)]
        if not locales:
            break
        store.mark_synced(locales, time.time())
    if not pendientes:
        return 0, None, ""

    ids = [i for i, _ in pendientes]
    payload = [json.loads(ev.model_dump_json()) for _, ev in pendientes]
    if fuente is not None:
        for evento in payload:
            evento["video_id"] = fuente[1]

    body = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        url.rstrip("/") + "/devices/events",
        data=body,
        method="POST",
        headers={"Content-Type": "application/json", "X-API-Key": api_key},
    )
    try:
        with urllib.request.urlopen(req, timeout=_REQUEST_TIMEOUT_S) as resp:
            if resp.status >= 300:
                logger.warning("cloud_sync: respuesta inesperada %s", resp.status)
                return 0, "error", f"Respuesta inesperada de la nube: HTTP {resp.status}"
    except urllib.error.HTTPError as exc:
        detail = _detalle_http(exc)
        if exc.code == 410:
            # El hogar de este kit se dio de baja (ver cloud/README.md, "Baja del hogar"): la
            # nube no va a aceptar estos eventos nunca y son de un hogar que pidió borrar sus
            # datos. Se descartan; si quedaran pendientes se subirían al hogar siguiente cuando
            # el kit se vuelva a vincular.
            descartados = store.discard(ids)
            logger.warning(
                "cloud_sync: el hogar de este kit se dio de baja — %d eventos descartados sin enviar",
                descartados,
            )
            return 0, "dado_de_baja", detail
        if exc.code == 409 and solo_desde_vinculacion:
            store.mark_synced(ids, time.time())
            logger.info(
                "cloud_sync: kit sin vincular — %d eventos marcados como enviados sin mandarlos "
                "(solo_desde_vinculacion)", len(ids),
            )
            return 0, "esperando_vinculacion", detail
        # 404 típico: una cámara de este lote no está registrada todavía del lado cloud (ver
        # cloud/README.md, "POST /devices/camaras") — se loguea el detalle para que sea
        # accionable, no se reintenta infinito en silencio.
        logger.warning("cloud_sync: HTTP %s al sincronizar — %s", exc.code, detail)
        estado = {409: "esperando_vinculacion", 401: "api_key_invalida"}.get(exc.code, "error")
        return 0, estado, detail
    except urllib.error.URLError as exc:
        logger.info("cloud_sync: sin conexión al backend cloud (%s) — reintenta en el próximo ciclo", exc.reason)
        return 0, "sin_conexion", str(exc.reason)

    store.mark_synced(ids, time.time())
    logger.info("cloud_sync: %d eventos sincronizados", len(ids))
    return len(ids), "conectado", ""


def _detalle_http(exc: urllib.error.HTTPError) -> str:
    """El `detail` de FastAPI si la respuesta es JSON; si no, el cuerpo tal cual."""
    cuerpo = exc.read().decode(errors="replace")
    try:
        detalle = json.loads(cuerpo).get("detail")
    except (ValueError, AttributeError):
        return cuerpo
    return detalle if isinstance(detalle, str) else cuerpo


def marcar_pendientes_como_enviados(store: EventStore) -> int:
    """Marca como enviados, sin mandarlos, todos los eventos pendientes. Solo para
    `solo_desde_vinculacion`: los eventos viejos de la controladora no llegan a la app."""
    total = 0
    while pendientes := store.unsynced(limit=500):
        store.mark_synced([i for i, _ in pendientes], time.time())
        total += len(pendientes)
    return total


class CloudSyncThread(threading.Thread):
    """Thread de fondo opcional que llama a `sync_once` cada `interval_s`. Arrancado desde
    `runner/main.py` solo si `cameras.yaml` tiene una sección `cloud:` configurada.

    Además aplica la retención local: una vez por hora como mucho borra los eventos que la nube
    confirmó hace más de `retencion_local_dias` (`EventStore.purge_synced`). Como este thread
    solo existe con la sección `cloud:`, en una instalación 100% local no se purga nada."""

    def __init__(self, db_path: Path, url: str, api_key: str, interval_s: float = 30.0,
                 stop_event: threading.Event | None = None, retencion_local_dias: float = 7.0,
                 solo_desde_vinculacion: bool = False, state_path: Path = STATE_PATH,
                 camaras_locales: frozenset[str] = frozenset(), camara_kit: str | None = None,
                 source_path: Path = SOURCE_PATH):
        super().__init__(name="cloud-sync", daemon=True)
        self._db_path = db_path
        self._url = url
        self._api_key = api_key
        self._interval = interval_s
        self._retencion_local_dias = retencion_local_dias
        self._solo_desde_vinculacion = solo_desde_vinculacion
        self._camaras_locales = camaras_locales
        self._camara_kit = camara_kit
        self._source_path = source_path
        self._state_path = state_path
        self._state = {"estado": "iniciando", "detalle": "", "actualizado_at": time.time(),
                       "ultimo_envio_at": None, "enviados": 0}
        self._last_purge: float | None = None
        # OJO: no llamar self._stop — threading.Thread ya usa ese nombre internamente (método
        # privado), pisarlo rompe Thread.join(). Mismo aviso que en camera_worker.py.
        self._stop_event = stop_event or threading.Event()

    def stop(self) -> None:
        self._stop_event.set()

    def _purge_if_due(self, store: EventStore, now: float | None = None) -> int | None:
        """Purga los eventos ya enviados si pasó `_PURGE_INTERVAL_S` desde la última vez (o si es
        la primera del proceso). Devuelve cuántos borró, o None si todavía no tocaba. `now` es
        reloj monotónico y solo se pasa en las pruebas."""
        if now is None:
            now = time.monotonic()
        if self._last_purge is not None and now - self._last_purge < _PURGE_INTERVAL_S:
            return None
        self._last_purge = now
        borrados = store.purge_synced(self._retencion_local_dias)
        if borrados:
            logger.info(
                "cloud_sync: %d eventos ya enviados borrados del store local (retención de %g días)",
                borrados, self._retencion_local_dias,
            )
        return borrados

    def _fuente(self) -> tuple[str | None, str] | None:
        """Se lee en cada ciclo: el botón "App" del dashboard cambia la fuente sin reiniciar."""
        if self._camara_kit is None:
            return None
        return read_cloud_source(self._source_path), self._camara_kit

    def _sync_and_record(self, store: EventStore) -> None:
        enviados, estado, detalle = _sync_lote(
            store, self._url, self._api_key, self._solo_desde_vinculacion, self._camaras_locales,
            fuente=self._fuente(),
        )
        now = time.time()
        self._state["actualizado_at"] = now
        if estado is not None:
            self._state["estado"] = estado
            self._state["detalle"] = detalle
        if enviados:
            self._state["ultimo_envio_at"] = now
            self._state["enviados"] += enviados
        write_cloud_state(self._state, self._state_path)

    def run(self) -> None:
        store = EventStore(self._db_path)
        try:
            write_cloud_state(self._state, self._state_path)
            if self._solo_desde_vinculacion:
                viejos = marcar_pendientes_como_enviados(store)
                if viejos:
                    logger.info(
                        "cloud_sync: %d eventos de antes de arrancar marcados como enviados sin "
                        "mandarlos (solo_desde_vinculacion)", viejos,
                    )
            while not self._stop_event.is_set():
                try:
                    self._sync_and_record(store)
                    self._purge_if_due(store)
                except Exception:
                    logger.exception("cloud_sync: error inesperado en el ciclo de sincronización")
                self._stop_event.wait(self._interval)
        finally:
            store.close()
