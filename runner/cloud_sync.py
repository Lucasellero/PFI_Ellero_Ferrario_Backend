"""Cliente de sincronización Pi -> nube (RNF6/RNF7 del informe): lee eventos locales no
sincronizados (`events/store.py::EventStore.unsynced`) y los manda en batch a
`POST /devices/events` del backend `cloud/` (ver `cloud/README.md`).

Solo hace conexiones SALIENTES — la Pi nunca necesita exponer un puerto ni tener IP pública,
coherente con la arquitectura de red del informe. Ante un corte de conexión simplemente no marca
nada como sincronizado y lo reintenta en el próximo ciclo; los eventos siguen disponibles
localmente mientras tanto (`events/events.db` es la fuente de verdad hasta que se confirma el
envío).

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
    `runner/main.py` solo si `cameras.yaml` tiene una sección `cloud:` configurada."""

    def __init__(self, db_path: Path, url: str, api_key: str, interval_s: float = 30.0,
                 stop_event: threading.Event | None = None):
        super().__init__(name="cloud-sync", daemon=True)
        self._db_path = db_path
        self._url = url
        self._api_key = api_key
        self._interval = interval_s
        self._stop = stop_event or threading.Event()

    def stop(self) -> None:
        self._stop.set()

    def run(self) -> None:
        store = EventStore(self._db_path)
        try:
            while not self._stop.is_set():
                try:
                    sync_once(store, self._url, self._api_key)
                except Exception:
                    logger.exception("cloud_sync: error inesperado en el ciclo de sincronización")
                self._stop.wait(self._interval)
        finally:
            store.close()
