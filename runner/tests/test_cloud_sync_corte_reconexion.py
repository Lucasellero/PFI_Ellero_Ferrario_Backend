"""Corte y reconexión con la nube (RNF6 y RNF7): los eventos generados sin conexión quedan en la
base local de la controladora y llegan todos, una sola vez, cuando la nube vuelve.

Son pruebas de punta a punta: el backend `cloud/` real corre en un proceso aparte contra un SQLite
temporal (`backend_nube_local.py`) y `runner/cloud_sync.py` le habla por HTTP, sin simular la red.
"Backend apagado" es el proceso detenido de verdad, con el puerto cerrado.
"""
from __future__ import annotations

import json
import logging
import os
import re
import socket
import sqlite3
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

import pytest

from runner import cloud_sync
from runner.cloud_sync import CloudSyncThread, sync_once
from schema import EventType, MonitoringEvent
from store import EventStore

_RAIZ = Path(__file__).resolve().parents[2]
_CLOUD = _RAIZ / "cloud"
_LANZADOR = Path(__file__).resolve().parent / "backend_nube_local.py"
_ARRANQUE_MAX_S = 60.0


class NubeLocal:
    """El backend `cloud/` en esta máquina: un SQLite propio y un puerto fijo, para poder
    apagarlo y volver a encenderlo en la misma dirección, como le pasa a una controladora."""

    def __init__(self, carpeta: Path):
        self._carpeta = carpeta
        self.db_path = carpeta / "nube.db"
        with socket.socket() as s:
            s.bind(("127.0.0.1", 0))
            self.puerto = s.getsockname()[1]
        self.url = f"http://127.0.0.1:{self.puerto}"
        self._entorno = {
            **os.environ,
            "DATABASE_URL": f"sqlite:///{self.db_path.as_posix()}",
            "PYTHONIOENCODING": "utf-8",
        }
        self._proceso: subprocess.Popen | None = None
        self._series = 0

    def crear_esquema(self) -> None:
        subprocess.run(
            [sys.executable, str(_LANZADOR), "--crear-esquema"],
            env=self._entorno, cwd=_CLOUD, check=True, capture_output=True,
        )

    def alta_kit(self) -> dict:
        """Kit "de fábrica" con una cámara (`cam1`), dado de alta con `cloud/alta_kit.py`."""
        self._series += 1
        serie = f"LR-{900000 + self._series}"
        salida = subprocess.run(
            [sys.executable, "alta_kit.py", "--camaras", "1", "--ids", "cam1", "--serie", serie],
            env=self._entorno, cwd=_CLOUD, check=True, capture_output=True, encoding="utf-8",
        ).stdout
        return {
            "serie": serie,
            "codigo": re.search(r"codigo=(\S+)", salida).group(1),
            "api_key": re.search(r"cloud\.api_key\): (\S+)", salida).group(1),
        }

    def encendida(self) -> bool:
        try:
            with urllib.request.urlopen(self.url + "/health", timeout=2.0) as respuesta:
                return respuesta.status == 200
        except (urllib.error.URLError, OSError):
            return False

    def encender(self) -> None:
        assert self._proceso is None, "la nube ya estaba encendida"
        registro = open(self._carpeta / "backend.log", "ab")
        self._proceso = subprocess.Popen(
            [sys.executable, str(_LANZADOR), "--puerto", str(self.puerto)],
            env=self._entorno, cwd=_CLOUD, stdout=registro, stderr=subprocess.STDOUT,
        )
        registro.close()
        limite = time.monotonic() + _ARRANQUE_MAX_S
        while not self.encendida():
            if self._proceso.poll() is not None or time.monotonic() > limite:
                self.apagar()
                pytest.fail("el backend no arrancó:\n" + (self._carpeta / "backend.log").read_text(errors="replace"))
            time.sleep(0.2)

    def apagar(self) -> None:
        if self._proceso is not None:
            self._proceso.terminate()
            self._proceso.wait(timeout=15.0)
            self._proceso = None
        assert not self.encendida()

    def pedir(self, metodo: str, ruta: str, cuerpo=None, uid: str | None = None):
        pedido = urllib.request.Request(
            self.url + ruta, method=metodo,
            data=None if cuerpo is None else json.dumps(cuerpo).encode(),
            headers={"Content-Type": "application/json", **({"Authorization": f"Bearer {uid}"} if uid else {})},
        )
        with urllib.request.urlopen(pedido, timeout=30.0) as respuesta:
            contenido = respuesta.read()
        return json.loads(contenido) if contenido else None

    def vincular(self, kit: dict, uid: str, *, con_residente: bool = True) -> int:
        """Lo que hace el familiar desde la app: crear su perfil, escanear el QR y cargar al
        residente. Recién con residente la nube acepta eventos de ese kit."""
        self.pedir("POST", "/me", {"nombre": "Familiar de prueba"}, uid)
        hogar_id = self.pedir("POST", "/kits/vincular", {"codigo": kit["codigo"]}, uid)["hogar_id"]
        if con_residente:
            self.cargar_residente(hogar_id, uid)
        return hogar_id

    def cargar_residente(self, hogar_id: int, uid: str) -> None:
        self.pedir("PUT", f"/hogares/{hogar_id}/residente", {"nombre": "Rosa Prueba", "telefono": None}, uid)

    def eventos(self, hogar_id: int, uid: str, ruta: str = "eventos") -> list[dict]:
        return self.pedir("GET", f"/hogares/{hogar_id}/{ruta}?limit=100000", uid=uid)

    def filas_del_kit(self, kit: dict) -> int:
        """Cantidad de eventos de ese kit leída directo del SQLite de la nube, sin pasar por la
        API: sirve también con el backend apagado o con el kit sin vincular."""
        conexion = sqlite3.connect(self.db_path)
        try:
            return conexion.execute(
                "SELECT COUNT(*) FROM eventos e JOIN camaras c ON e.camara_id = c.id "
                "JOIN kits k ON c.kit_id = k.id WHERE k.numero_serie = ?", (kit["serie"],),
            ).fetchone()[0]
        finally:
            conexion.close()


@pytest.fixture(scope="module")
def nube(tmp_path_factory):
    nube = NubeLocal(tmp_path_factory.mktemp("nube"))
    nube.crear_esquema()
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


def _generar(store: EventStore, cantidad: int, desde: int = 0) -> list[int]:
    """Agrega `cantidad` eventos a la base local, numerados en `metadata.n`. Los dos primeros de
    cada tanda son una caída confirmada y una inmovilidad grave (alertas); el resto, actividad."""
    ahora = time.time()
    numeros = list(range(desde, desde + cantidad))
    for posicion, n in enumerate(numeros):
        if posicion == 0:
            tipo, metadata = EventType.FALL, {"confirmed": True, "evidence": "full"}
        elif posicion == 1:
            tipo, metadata = EventType.IMMOBILITY, {"severity": "alta"}
        else:
            tipo, metadata = EventType.ACTIVITY, {"activity_label": "Resting"}
        store.add(MonitoringEvent(
            video_id="cam1", event_type=tipo, video_timestamp_s=float(n),
            wall_clock_ts=ahora - (cantidad - posicion), metadata={**metadata, "n": n},
        ))
    return numeros


def _estado_local(db_local: Path) -> dict:
    conexion = sqlite3.connect(db_local)
    try:
        filas = conexion.execute("SELECT metadata, cloud_synced_at FROM events").fetchall()
    finally:
        conexion.close()
    return {
        "numeros": sorted(json.loads(m)["n"] for m, _ in filas),
        "pendientes": sum(1 for _, enviado in filas if enviado is None),
        "enviados": sum(1 for _, enviado in filas if enviado is not None),
    }


def _sincronizar_hasta_vaciar(store: EventStore, nube: NubeLocal, kit: dict) -> list[int]:
    """Repite `sync_once` (un ciclo del thread) hasta que no envía nada. Devuelve cuántos eventos
    confirmó cada ciclo."""
    ciclos = []
    while len(ciclos) < 50:
        enviados = sync_once(store, nube.url, kit["api_key"])
        if enviados == 0:
            return ciclos
        ciclos.append(enviados)
    pytest.fail("la sincronización no termina")


def _numeros_en_la_nube(nube: NubeLocal, hogar_id: int, uid: str) -> list[int]:
    return sorted(e["metadata"]["n"] for e in nube.eventos(hogar_id, uid))


def test_eventos_generados_sin_nube_llegan_todos_una_sola_vez(nube, store, db_local):
    kit = nube.alta_kit()
    nube.encender()
    hogar_id = nube.vincular(kit, "uid-corte")
    nube.apagar()

    # Sin nube: la controladora sigue generando eventos y los intentos de envío fallan.
    numeros = _generar(store, 60)
    for _ in range(3):
        assert sync_once(store, nube.url, kit["api_key"]) == 0
    assert _estado_local(db_local) == {"numeros": numeros, "pendientes": 60, "enviados": 0}
    assert nube.filas_del_kit(kit) == 0

    # Vuelve la nube: llegan todos, una sola vez, y quedan marcados como enviados.
    nube.encender()
    assert _sincronizar_hasta_vaciar(store, nube, kit) == [60]
    assert _numeros_en_la_nube(nube, hogar_id, "uid-corte") == numeros
    assert nube.filas_del_kit(kit) == 60
    assert _estado_local(db_local) == {"numeros": numeros, "pendientes": 0, "enviados": 60}
    alertas = nube.eventos(hogar_id, "uid-corte", "alertas")
    assert sorted(a["event_type"] for a in alertas) == ["fall", "immobility"]

    # Repetir la sincronización no manda nada de nuevo: no hay duplicados.
    for _ in range(3):
        assert sync_once(store, nube.url, kit["api_key"]) == 0
    assert _numeros_en_la_nube(nube, hogar_id, "uid-corte") == numeros
    assert nube.filas_del_kit(kit) == 60
    nube.apagar()


def test_corte_en_medio_de_la_sincronizacion(nube, store, db_local):
    """La nube se cae entre dos lotes: lo ya confirmado no se reenvía y el resto espera."""
    kit = nube.alta_kit()
    nube.encender()
    hogar_id = nube.vincular(kit, "uid-medio")
    numeros = _generar(store, 450)

    assert sync_once(store, nube.url, kit["api_key"]) == cloud_sync._BATCH_SIZE == 200
    nube.apagar()
    assert sync_once(store, nube.url, kit["api_key"]) == 0
    assert _estado_local(db_local) == {"numeros": numeros, "pendientes": 250, "enviados": 200}

    # Durante el corte se siguen generando eventos.
    numeros += _generar(store, 30, desde=450)
    assert _estado_local(db_local)["pendientes"] == 280

    nube.encender()
    assert _sincronizar_hasta_vaciar(store, nube, kit) == [200, 80]
    assert _numeros_en_la_nube(nube, hogar_id, "uid-medio") == numeros
    assert nube.filas_del_kit(kit) == 480
    assert _estado_local(db_local) == {"numeros": numeros, "pendientes": 0, "enviados": 480}
    nube.apagar()


def test_kit_sin_vincular_conserva_los_eventos_y_los_envia_al_vincularlo(nube, store, db_local, caplog):
    kit = nube.alta_kit()
    nube.encender()
    numeros = _generar(store, 40)

    # Kit recién salido de fábrica: la nube responde 409 y la controladora no pierde nada.
    with caplog.at_level(logging.WARNING, logger="runner.cloud_sync"):
        assert sync_once(store, nube.url, kit["api_key"]) == 0
    assert "HTTP 409" in caplog.text
    assert _estado_local(db_local) == {"numeros": numeros, "pendientes": 40, "enviados": 0}
    assert nube.filas_del_kit(kit) == 0

    # Vinculado pero todavía sin residente: sigue respondiendo 409.
    caplog.clear()
    hogar_id = nube.vincular(kit, "uid-409", con_residente=False)
    with caplog.at_level(logging.WARNING, logger="runner.cloud_sync"):
        assert sync_once(store, nube.url, kit["api_key"]) == 0
    assert "HTTP 409" in caplog.text
    assert _estado_local(db_local) == {"numeros": numeros, "pendientes": 40, "enviados": 0}
    assert nube.filas_del_kit(kit) == 0

    # Con la vinculación completa llegan todos, una sola vez.
    nube.cargar_residente(hogar_id, "uid-409")
    assert _sincronizar_hasta_vaciar(store, nube, kit) == [40]
    assert _numeros_en_la_nube(nube, hogar_id, "uid-409") == numeros
    assert _estado_local(db_local) == {"numeros": numeros, "pendientes": 0, "enviados": 40}
    assert sync_once(store, nube.url, kit["api_key"]) == 0
    assert nube.filas_del_kit(kit) == 40
    nube.apagar()


def test_respuesta_perdida_no_deja_filas_duplicadas(nube, store, db_local, monkeypatch):
    """La nube guarda el lote pero la respuesta no llega a la controladora (corte justo después
    de confirmar): la controladora lo reenvía y la nube reconoce cada evento por su `event_id`."""
    kit = nube.alta_kit()
    nube.encender()
    hogar_id = nube.vincular(kit, "uid-perdida")
    numeros = _generar(store, 250)  # dos lotes: 200 y 50

    real = cloud_sync.urllib.request.urlopen
    respuestas_perdidas = []

    def respuesta_perdida(pedido, timeout=None):
        with real(pedido, timeout=timeout) as respuesta:  # el pedido llega y la nube lo procesa
            respuestas_perdidas.append(respuesta.status)
        raise cloud_sync.urllib.error.URLError("se cortó antes de recibir la respuesta")

    monkeypatch.setattr(cloud_sync.urllib.request, "urlopen", respuesta_perdida)
    for _ in range(2):  # dos intentos seguidos del primer lote, los dos con la respuesta perdida
        assert sync_once(store, nube.url, kit["api_key"]) == 0
    monkeypatch.setattr(cloud_sync.urllib.request, "urlopen", real)

    # La nube ya tiene el primer lote; la controladora cree que no envió nada.
    assert respuestas_perdidas == [201, 201]
    assert nube.filas_del_kit(kit) == 200
    assert _estado_local(db_local) == {"numeros": numeros, "pendientes": 250, "enviados": 0}

    # Vuelve la conexión: reenvía el primer lote (ya recibido) y manda el segundo.
    assert _sincronizar_hasta_vaciar(store, nube, kit) == [200, 50]
    assert _numeros_en_la_nube(nube, hogar_id, "uid-perdida") == numeros
    assert nube.filas_del_kit(kit) == 250
    assert _estado_local(db_local) == {"numeros": numeros, "pendientes": 0, "enviados": 250}
    alertas = nube.eventos(hogar_id, "uid-perdida", "alertas")
    assert sorted(a["event_type"] for a in alertas) == ["fall", "immobility"]
    nube.apagar()


def test_el_thread_de_sincronizacion_se_recupera_solo(nube, store, db_local):
    """El thread que corre dentro del runner, sin que nadie lo reinicie: sigue intentando durante
    el corte y envía lo acumulado apenas la nube responde."""
    kit = nube.alta_kit()
    nube.encender()
    hogar_id = nube.vincular(kit, "uid-thread")
    nube.apagar()

    hilo = CloudSyncThread(db_local, nube.url, kit["api_key"], interval_s=0.2)
    hilo.start()
    try:
        numeros = _generar(store, 25)
        time.sleep(1.5)  # varios ciclos fallidos
        assert _estado_local(db_local) == {"numeros": numeros, "pendientes": 25, "enviados": 0}

        nube.encender()
        limite = time.monotonic() + 30.0
        while _estado_local(db_local)["pendientes"] and time.monotonic() < limite:
            time.sleep(0.1)
        time.sleep(1.0)  # varios ciclos más: no tienen que reenviar nada
    finally:
        hilo.stop()
        hilo.join(timeout=20.0)

    assert not hilo.is_alive()
    assert _estado_local(db_local) == {"numeros": numeros, "pendientes": 0, "enviados": 25}
    assert _numeros_en_la_nube(nube, hogar_id, "uid-thread") == numeros
    assert nube.filas_del_kit(kit) == 25
    nube.apagar()
