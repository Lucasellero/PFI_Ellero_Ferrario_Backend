"""Vincular la controladora con la app móvil (ver "Vincular con la app móvil" en runner/README.md):
`kit.yaml` generado por `cloud/alta_kit.py --kit-yaml`, el estado de la nube que el runner deja
en `runner_state/cloud.json` para el dashboard, y `solo_desde_vinculacion`.

Las pruebas de sincronización usan el backend `cloud/` real en local, igual que
`test_cloud_sync_corte_reconexion.py`.
"""
from __future__ import annotations

import subprocess
import sys
import threading
import time

import pytest
import yaml

from runner.cloud_state import read_cloud_state
from runner.cloud_sync import _BATCH_SIZE, CloudSyncThread
from runner.config import load_config
from runner.tests.test_cloud_sync_corte_reconexion import _CLOUD, NubeLocal, _estado_local, _generar
from schema import EventType, MonitoringEvent
from store import EventStore

_CAMARAS = "cameras:\n- id: cam1\n  source: '0'\n"


@pytest.fixture(scope="module")
def nube(tmp_path_factory):
    nube = NubeLocal(tmp_path_factory.mktemp("nube"))
    nube.crear_esquema()
    nube.encender()
    yield nube
    nube.apagar()


@pytest.fixture
def db_local(tmp_path):
    return tmp_path / "events.db"


@pytest.fixture
def store(db_local):
    s = EventStore(db_local)
    yield s
    s.close()


def _alta_kit_yaml(nube: NubeLocal, destino, serie: str) -> dict:
    salida = subprocess.run(
        [sys.executable, "alta_kit.py", "--camaras", "1", "--ids", "cam1", "--serie", serie,
         "--kit-yaml", str(destino), "--url", nube.url],
        env=nube._entorno, cwd=_CLOUD, check=True, capture_output=True, encoding="utf-8",
    ).stdout
    kit = yaml.safe_load(destino.read_text())
    assert kit["api_key"] not in salida and kit["codigo_vinculacion"] not in salida
    return {"serie": serie, "codigo": kit["codigo_vinculacion"], "api_key": kit["api_key"]}


def _hilo(db_local, kit_yaml, estado, solo_desde_vinculacion=True) -> CloudSyncThread:
    kit = yaml.safe_load(kit_yaml.read_text())
    return CloudSyncThread(
        db_path=db_local, url=kit["url"], api_key=kit["api_key"], stop_event=threading.Event(),
        solo_desde_vinculacion=solo_desde_vinculacion, state_path=estado,
    )


# ── Configuración ────────────────────────────────────────────────────────────

def test_sin_cloud_ni_kit_yaml_no_hay_sincronizacion(tmp_path):
    (tmp_path / "cameras.yaml").write_text(_CAMARAS)
    assert load_config(tmp_path / "cameras.yaml").cloud_sync is None


def test_kit_yaml_activa_la_sincronizacion(tmp_path):
    (tmp_path / "cameras.yaml").write_text(_CAMARAS)
    (tmp_path / "kit.yaml").write_text(
        "url: http://nube\nnumero_serie: LR-000001\ncodigo_vinculacion: ABCD2345\n"
        "api_key: k\ninterval_s: 10\nsolo_desde_vinculacion: true\n"
    )
    cloud = load_config(tmp_path / "cameras.yaml").cloud_sync
    assert (cloud.url, cloud.api_key, cloud.interval_s, cloud.solo_desde_vinculacion) == ("http://nube", "k", 10.0, True)


def test_la_seccion_cloud_de_cameras_yaml_tiene_prioridad_sobre_kit_yaml(tmp_path):
    (tmp_path / "cameras.yaml").write_text(_CAMARAS + "cloud:\n  url: http://otra\n  api_key: x\n")
    (tmp_path / "kit.yaml").write_text("url: http://nube\napi_key: k\n")
    cloud = load_config(tmp_path / "cameras.yaml").cloud_sync
    assert (cloud.url, cloud.api_key, cloud.solo_desde_vinculacion) == ("http://otra", "x", False)


def test_alta_kit_no_pisa_un_kit_yaml_existente(nube, tmp_path):
    destino = tmp_path / "kit.yaml"
    destino.write_text("api_key: la-del-kit-anterior\n")
    proceso = subprocess.run(
        [sys.executable, "alta_kit.py", "--kit-yaml", str(destino), "--url", nube.url],
        env=nube._entorno, cwd=_CLOUD, capture_output=True, encoding="utf-8",
    )
    assert proceso.returncode != 0
    assert destino.read_text() == "api_key: la-del-kit-anterior\n"


# ── Flujo de la demo, contra la nube real en local ──────────────────────────

def test_demo_de_punta_a_punta(nube, store, db_local, tmp_path):
    kit_yaml = tmp_path / "kit.yaml"
    kit = _alta_kit_yaml(nube, kit_yaml, "LR-910001")
    estado = tmp_path / "cloud.json"

    # Eventos viejos de la controladora (corridas anteriores): no tienen que llegar nunca.
    _generar(store, 5, desde=0)
    hilo = _hilo(db_local, kit_yaml, estado)
    hilo._stop_event.set()           # solo el arranque: marca lo viejo y no entra al ciclo
    hilo.run()
    assert _estado_local(db_local)["pendientes"] == 0
    assert read_cloud_state(estado)["estado"] == "iniciando"

    # Antes de vincular: la nube responde 409 y la caída no queda guardada para después.
    _generar(store, 3, desde=100)
    hilo._sync_and_record(store)
    assert read_cloud_state(estado)["estado"] == "esperando_vinculacion"
    assert _estado_local(db_local)["pendientes"] == 0

    # El familiar escanea el QR y carga al residente.
    nube.vincular(kit, "familiar-demo")
    _generar(store, 4, desde=200)
    hilo._sync_and_record(store)
    leido = read_cloud_state(estado)
    assert (leido["estado"], leido["enviados"]) == ("conectado", 4)
    assert leido["ultimo_envio_at"] is not None
    assert nube.filas_del_kit(kit) == 4

    # Cola vacía: no se le pregunta nada a la nube y el estado se mantiene.
    hilo._sync_and_record(store)
    assert read_cloud_state(estado)["estado"] == "conectado"


def test_sin_solo_desde_vinculacion_lo_de_antes_de_vincular_sigue_pendiente(nube, store, db_local, tmp_path):
    kit_yaml = tmp_path / "kit.yaml"
    _alta_kit_yaml(nube, kit_yaml, "LR-910002")
    estado = tmp_path / "cloud.json"
    _generar(store, 3)
    hilo = _hilo(db_local, kit_yaml, estado, solo_desde_vinculacion=False)
    hilo._sync_and_record(store)
    assert read_cloud_state(estado)["estado"] == "esperando_vinculacion"
    assert _estado_local(db_local)["pendientes"] == 3


def test_api_key_desconocida(nube, store, db_local, tmp_path):
    kit_yaml = tmp_path / "kit.yaml"
    kit_yaml.write_text(f"url: {nube.url}\napi_key: no-existe\n")
    estado = tmp_path / "cloud.json"
    _generar(store, 2)
    _hilo(db_local, kit_yaml, estado)._sync_and_record(store)
    assert read_cloud_state(estado)["estado"] == "api_key_invalida"


def test_nube_inalcanzable(store, db_local, tmp_path):
    kit_yaml = tmp_path / "kit.yaml"
    kit_yaml.write_text("url: http://127.0.0.1:9\napi_key: k\n")
    estado = tmp_path / "cloud.json"
    _generar(store, 2)
    _hilo(db_local, kit_yaml, estado)._sync_and_record(store)
    assert read_cloud_state(estado)["estado"] == "sin_conexion"
    assert _estado_local(db_local)["pendientes"] == 2


def test_las_camaras_con_nube_false_no_suben_ni_traban_la_cola(nube, store, db_local, tmp_path):
    """Varias cámaras de video en la demo con un kit de una sola cámara: los eventos de las que
    tienen `nube: false` no se mandan (la nube los rechazaría con 404 y trabaría el lote)."""
    kit_yaml = tmp_path / "kit.yaml"
    kit = _alta_kit_yaml(nube, kit_yaml, "LR-910003")
    nube.vincular(kit, "familiar-camaras")
    estado = tmp_path / "cloud.json"
    for n in range(_BATCH_SIZE + 5):  # más de un lote entero de la cámara local
        store.add(MonitoringEvent(video_id="video-local", event_type=EventType.ACTIVITY,
                                  video_timestamp_s=float(n), wall_clock_ts=time.time(),
                                  metadata={"activity_label": "Resting", "n": -n}))
    _generar(store, 3, desde=300)
    hilo = _hilo(db_local, kit_yaml, estado)
    hilo._camaras_locales = frozenset({"video-local"})
    hilo._sync_and_record(store)
    assert read_cloud_state(estado)["estado"] == "conectado"
    assert nube.filas_del_kit(kit) == 3
    assert _estado_local(db_local)["pendientes"] == 0


def test_nube_false_en_cameras_yaml(tmp_path):
    (tmp_path / "cameras.yaml").write_text(_CAMARAS + "- id: otra\n  source: '1'\n  nube: false\n")
    camaras = load_config(tmp_path / "cameras.yaml").cameras
    assert [(c.id, c.nube) for c in camaras] == [("cam1", True), ("otra", False)]


def _actividad(store: EventStore, video_id: str, cantidad: int) -> None:
    for n in range(cantidad):
        store.add(MonitoringEvent(video_id=video_id, event_type=EventType.ACTIVITY,
                                  video_timestamp_s=float(n), wall_clock_ts=time.time(),
                                  metadata={"activity_label": "Resting", "n": n}))


def test_fuente_elegida_desde_el_dashboard_se_sube_como_la_camara_del_kit(nube, store, db_local, tmp_path):
    """Botón "App" del dashboard: solo la fuente elegida llega a la nube, con el id de la cámara
    del kit, y el cambio de fuente se toma en el ciclo siguiente sin reiniciar."""
    from runner.cloud_state import write_cloud_source

    kit_yaml = tmp_path / "kit.yaml"
    kit = _alta_kit_yaml(nube, kit_yaml, "LR-910004")
    assert yaml.safe_load(kit_yaml.read_text())["camara"] == "cam1"
    nube.vincular(kit, "familiar-fuente")
    fuente = tmp_path / "cloud_source.json"
    hilo = _hilo(db_local, kit_yaml, tmp_path / "cloud.json")
    hilo._camara_kit, hilo._source_path = "cam1", fuente

    # Sin fuente elegida no se sube nada.
    _actividad(store, "VIDEO CAIDA 1", 2)
    hilo._sync_and_record(store)
    assert nube.filas_del_kit(kit) == 0

    write_cloud_source("VIDEO CAIDA 1", fuente)
    _actividad(store, "VIDEO CAIDA 1", 3)
    _actividad(store, "VIVO", 4)
    hilo._sync_and_record(store)
    assert nube.filas_del_kit(kit) == 3

    write_cloud_source("VIVO", fuente)
    _actividad(store, "VIDEO CAIDA 1", 2)
    _actividad(store, "VIVO", 5)
    hilo._sync_and_record(store)
    assert nube.filas_del_kit(kit) == 8
    assert read_cloud_state(tmp_path / "cloud.json")["estado"] == "conectado"
    assert _estado_local(db_local)["pendientes"] == 0
