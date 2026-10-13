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

_BATCH_SIZE = 200
_REQUEST_TIMEOUT_S = 15.0
# Cada cuánto, como mucho, se purgan los eventos ya enviados (ver `CloudSyncThread._purge_if_due`).
_PURGE_INTERVAL_S = 3600.0


def sync_once(store: EventStore, url: str, api_key: str) -> int:
    """Un ciclo de sincronización: manda el próximo lote de eventos pendientes. Devuelve cuántos
    se confirmaron. No lanza excepción ante error de red/HTTP — loguea y deja los eventos como
    pendientes para el próximo ciclo (ver docstring del módulo)."""
    pendientes = store.unsynced(limit=_BATCH_SIZE)
    if not pendientes:
        return 0

    ids = [i for i, _ in pendientes]
    payload = [json.loads(ev.model_dump_json()) for _, ev in pendientes]

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
                return 0
    except urllib.error.HTTPError as exc:
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
            return 0
        # 404 típico: una cámara de este lote no está registrada todavía del lado cloud (ver
        # cloud/README.md, "POST /devices/camaras") — se loguea el detalle para que sea
        # accionable, no se reintenta infinito en silencio.
        detail = exc.read().decode(errors="replace")
        logger.warning("cloud_sync: HTTP %s al sincronizar — %s", exc.code, detail)
        return 0
    except urllib.error.URLError as exc:
        logger.info("cloud_sync: sin conexión al backend cloud (%s) — reintenta en el próximo ciclo", exc.reason)
        return 0

    store.mark_synced(ids, time.time())
    logger.info("cloud_sync: %d eventos sincronizados", len(ids))
    return len(ids)


class CloudSyncThread(threading.Thread):
    """Thread de fondo opcional que llama a `sync_once` cada `interval_s`. Arrancado desde
    `runner/main.py` solo si `cameras.yaml` tiene una sección `cloud:` configurada.

    Además aplica la retención local: una vez por hora como mucho borra los eventos que la nube
    confirmó hace más de `retencion_local_dias` (`EventStore.purge_synced`). Como este thread
    solo existe con la sección `cloud:`, en una instalación 100% local no se purga nada."""

    def __init__(self, db_path: Path, url: str, api_key: str, interval_s: float = 30.0,
                 stop_event: threading.Event | None = None, retencion_local_dias: float = 7.0):
        super().__init__(name="cloud-sync", daemon=True)
        self._db_path = db_path
        self._url = url
        self._api_key = api_key
        self._interval = interval_s
        self._retencion_local_dias = retencion_local_dias
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

    def run(self) -> None:
        store = EventStore(self._db_path)
        try:
            while not self._stop_event.is_set():
                try:
                    sync_once(store, self._url, self._api_key)
                    self._purge_if_due(store)
                except Exception:
                    logger.exception("cloud_sync: error inesperado en el ciclo de sincronización")
                self._stop_event.wait(self._interval)
        finally:
            store.close()
