"""Retención local desde el ciclo de sincronización (`runner/cloud_sync.py`) y su configuración."""
from __future__ import annotations

import io
import sys
import textwrap
import threading
import time

import pytest

from runner import cloud_sync
from runner.cloud_sync import CloudSyncThread
from runner.config import load_config
from schema import EventType, MonitoringEvent
from store import EventStore

DIA = 86400.0
URL = "http://127.0.0.1:9"


def _agregar(store: EventStore, nombre: str, *, sincronizado_hace_dias: float | None) -> None:
    store.add(MonitoringEvent(
        video_id=nombre, event_type=EventType.ACTIVITY, video_timestamp_s=0.0, wall_clock_ts=time.time(),
    ))
    if sincronizado_hace_dias is not None:
        (fila_id, _), = store.unsynced(video_id=nombre)
        store.mark_synced([fila_id], time.time() - sincronizado_hace_dias * DIA)


def _nombres(db_path) -> set[str]:
    store = EventStore(db_path)
    try:
        return {e.video_id for e in store.all()}
    finally:
        store.close()


def _escribir_config(tmp_path, seccion_cloud: str = ""):
    ruta = tmp_path / "cameras.yaml"
    ruta.write_text(
        "cameras:\n- id: cam1\n  source: video.mp4\n" + textwrap.dedent(seccion_cloud), encoding="utf-8",
    )
    return ruta


@pytest.fixture
def db_path(tmp_path):
    return tmp_path / "events.db"


@pytest.fixture
def store(db_path):
    s = EventStore(db_path)
    yield s
    s.close()


# ── Configuración (cameras.yaml) ─────────────────────────────────────────────

def test_retencion_local_dias_es_7_por_default(tmp_path):
    config = load_config(_escribir_config(tmp_path, """
        cloud:
          url: http://127.0.0.1:9
          api_key: clave
    """))
    assert config.cloud_sync.retencion_local_dias == 7.0


def test_retencion_local_dias_se_lee_de_la_seccion_cloud(tmp_path):
    config = load_config(_escribir_config(tmp_path, """
        cloud:
          url: http://127.0.0.1:9
          api_key: clave
          retencion_local_dias: 3
    """))
    assert config.cloud_sync.retencion_local_dias == 3.0


def test_sin_seccion_cloud_no_hay_sincronizacion_ni_retencion(tmp_path):
    assert load_config(_escribir_config(tmp_path)).cloud_sync is None


@pytest.mark.parametrize("valor", ["0", "-2"])
def test_un_plazo_no_positivo_en_el_yaml_se_rechaza(tmp_path, valor):
    ruta = _escribir_config(tmp_path, f"""
        cloud:
          url: http://127.0.0.1:9
          api_key: clave
          retencion_local_dias: {valor}
    """)
    with pytest.raises(ValueError):
        load_config(ruta)


# ── Purga desde el thread de sincronización ──────────────────────────────────

def test_la_purga_usa_los_dias_configurados(db_path, store):
    _agregar(store, "enviado_hace_2", sincronizado_hace_dias=2)
    _agregar(store, "enviado_hace_5", sincronizado_hace_dias=5)
    _agregar(store, "pendiente", sincronizado_hace_dias=None)

    hilo = CloudSyncThread(db_path, URL, "clave", retencion_local_dias=3)

    assert hilo._purge_if_due(store, now=0.0) == 1
    assert _nombres(db_path) == {"enviado_hace_2", "pendiente"}


def test_la_purga_corre_una_vez_por_hora_como_mucho(db_path, store):
    hilo = CloudSyncThread(db_path, URL, "clave", retencion_local_dias=7)
    _agregar(store, "viejo_1", sincronizado_hace_dias=30)

    assert hilo._purge_if_due(store, now=1000.0) == 1

    # Aparece otro evento vencido, pero todavía no pasó una hora desde la última purga.
    _agregar(store, "viejo_2", sincronizado_hace_dias=30)
    assert hilo._purge_if_due(store, now=1000.0 + 30) is None
    assert hilo._purge_if_due(store, now=1000.0 + 3599) is None
    assert _nombres(db_path) == {"viejo_2"}

    assert hilo._purge_if_due(store, now=1000.0 + 3600) == 1
    assert _nombres(db_path) == set()


def test_el_thread_purga_en_su_ciclo_y_se_detiene_bien(db_path, store, monkeypatch):
    _agregar(store, "enviado_viejo", sincronizado_hace_dias=30)
    _agregar(store, "enviado_reciente", sincronizado_hace_dias=1)
    llamadas = []
    original = EventStore.purge_synced

    def espia(self, older_than_days, now=None):
        llamadas.append(older_than_days)
        return original(self, older_than_days, now)

    monkeypatch.setattr(EventStore, "purge_synced", espia)
    # Sin eventos pendientes `sync_once` no abre ninguna conexión: la prueba no toca la red.
    monkeypatch.setattr(cloud_sync.urllib.request, "urlopen", lambda *a, **k: pytest.fail("no debe usar la red"))

    hilo = CloudSyncThread(db_path, URL, "clave", interval_s=0.01, retencion_local_dias=7)
    hilo.start()
    limite = time.monotonic() + 5.0
    while "enviado_viejo" in _nombres(db_path) and time.monotonic() < limite:
        time.sleep(0.02)
    time.sleep(0.1)  # varios ciclos más de 10 ms: no tiene que volver a purgar
    hilo.stop()
    hilo.join(timeout=5.0)

    assert not hilo.is_alive()
    assert _nombres(db_path) == {"enviado_reciente"}
    assert llamadas == [7]


def test_la_purga_no_borra_lo_que_no_se_pudo_enviar(db_path, store, monkeypatch):
    """Sin conexión con la nube los eventos quedan pendientes, y pendientes no se borran."""
    _agregar(store, "pendiente_1", sincronizado_hace_dias=None)
    _agregar(store, "pendiente_2", sincronizado_hace_dias=None)

    def sin_conexion(*args, **kwargs):
        raise cloud_sync.urllib.error.URLError("sin red")

    monkeypatch.setattr(cloud_sync.urllib.request, "urlopen", sin_conexion)
    hilo = CloudSyncThread(db_path, URL, "clave", retencion_local_dias=0.0001)

    assert cloud_sync.sync_once(store, URL, "clave") == 0
    assert hilo._purge_if_due(store, now=0.0) == 0
    assert _nombres(db_path) == {"pendiente_1", "pendiente_2"}


# ── Hogar dado de baja en la nube ────────────────────────────────────────────

def _respuesta_http(codigo: int):
    def responder(req, timeout=None):
        raise cloud_sync.urllib.error.HTTPError(req.full_url, codigo, "error", None, io.BytesIO(b'{"detail": "x"}'))
    return responder


def test_si_el_hogar_se_dio_de_baja_los_eventos_se_descartan(db_path, store, monkeypatch):
    """410: la nube no los va a aceptar nunca. Si quedaran pendientes se acumularían sin límite
    y, al volver a vincular el kit, se subirían al hogar nuevo."""
    _agregar(store, "pendiente_1", sincronizado_hace_dias=None)
    _agregar(store, "pendiente_2", sincronizado_hace_dias=None)
    _agregar(store, "enviado_ayer", sincronizado_hace_dias=1)
    monkeypatch.setattr(cloud_sync.urllib.request, "urlopen", _respuesta_http(410))

    assert cloud_sync.sync_once(store, URL, "clave") == 0

    assert _nombres(db_path) == {"enviado_ayer"}
    assert store.unsynced() == []


@pytest.mark.parametrize("codigo", [401, 404, 409, 422, 500, 503])
def test_cualquier_otro_error_de_la_nube_conserva_los_eventos(db_path, store, monkeypatch, codigo):
    """En particular 409 (kit todavía sin vincular): se guardan hasta que se puedan enviar."""
    _agregar(store, "pendiente_1", sincronizado_hace_dias=None)
    _agregar(store, "pendiente_2", sincronizado_hace_dias=None)
    monkeypatch.setattr(cloud_sync.urllib.request, "urlopen", _respuesta_http(codigo))

    assert cloud_sync.sync_once(store, URL, "clave") == 0

    assert {e.video_id for _, e in store.unsynced()} == {"pendiente_1", "pendiente_2"}


# ── Modo 100% local ──────────────────────────────────────────────────────────

def test_sin_seccion_cloud_el_runner_no_purga_nada(tmp_path, db_path, store, camera_worker, monkeypatch):
    """`python -m runner` con un cameras.yaml sin `cloud:`: no arranca el thread de
    sincronización y un evento marcado como enviado hace años sigue en el store."""
    import runner.main as runner_main

    _agregar(store, "enviado_hace_anios", sincronizado_hace_dias=2000)
    _agregar(store, "pendiente", sincronizado_hace_dias=None)

    class CamaraFalsa(threading.Thread):
        def __init__(self, **kwargs):
            super().__init__(daemon=True)

        def run(self) -> None:
            pass

    hilos_creados, purgas = [], []
    monkeypatch.setattr(runner_main, "CameraWorker", CamaraFalsa)
    monkeypatch.setattr(runner_main, "CloudSyncThread", lambda **k: hilos_creados.append(k))
    monkeypatch.setattr(EventStore, "purge_synced", lambda self, *a, **k: purgas.append(a))
    for cargador in ("_load_clip", "_load_skeleton", "_load_fall_temporal"):
        monkeypatch.setattr(runner_main, cargador, lambda log: None)
    monkeypatch.setattr(runner_main, "_DB_PATH", db_path)
    monkeypatch.setattr(runner_main.signal, "signal", lambda *a: None)
    monkeypatch.setattr(runner_main.logging, "basicConfig", lambda **k: None)
    monkeypatch.setattr(sys, "argv", ["runner", "--config", str(_escribir_config(tmp_path))])

    runner_main.main()

    assert hilos_creados == []
    assert purgas == []
    assert _nombres(db_path) == {"enviado_hace_anios", "pendiente"}
