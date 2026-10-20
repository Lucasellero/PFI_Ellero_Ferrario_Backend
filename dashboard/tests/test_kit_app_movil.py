"""Panel "App móvil": QR del kit (kit.yaml) y estado de la nube que deja el runner
(runner_state/cloud.json). Ver "Vincular con la app móvil" en runner/README.md."""
from __future__ import annotations

import json
import time

import pytest
from fastapi.testclient import TestClient

from dashboard import api

_KIT = (
    "url: http://nube\nnumero_serie: LR-000042\ncodigo_vinculacion: 7K3P9QXM\n"
    "api_key: secreta-no-mostrar\ninterval_s: 10\nsolo_desde_vinculacion: true\n"
)


@pytest.fixture
def rutas(tmp_path, monkeypatch):
    monkeypatch.setattr(api, "KIT_YAML", tmp_path / "kit.yaml")
    monkeypatch.setattr(api, "CLOUD_STATE_PATH", tmp_path / "cloud.json")
    monkeypatch.setattr(api, "CLOUD_SOURCE_PATH", tmp_path / "cloud_source.json")
    (tmp_path / "cameras.yaml").write_text("cameras:\n- id: VIDEO CAIDA 1\n  source: a.mp4\n- id: VIVO\n  source: b.mp4\n")
    monkeypatch.setattr(api, "CAMERAS_YAML", tmp_path / "cameras.yaml")
    monkeypatch.setattr(api, "DB_PATH", tmp_path / "events.db")
    return tmp_path


@pytest.fixture
def cliente(rutas):
    return TestClient(api.app)


def test_sin_kit_yaml(cliente):
    assert cliente.get("/api/kit").json() == {"configurado": False}
    assert cliente.get("/kit").status_code == 404


def test_kit_con_qr_y_sin_api_key(cliente, rutas):
    (rutas / "kit.yaml").write_text(_KIT)
    respuesta = cliente.get("/api/kit")
    datos = respuesta.json()
    assert datos["configurado"] and datos["numero_serie"] == "LR-000042"
    assert datos["codigo"] == "7K3P-9QXM"
    assert datos["qr_svg"].startswith("<svg")
    assert datos["estado"] is None and datos["estado_vigente"] is False
    assert "secreta-no-mostrar" not in respuesta.text


def test_qr_con_el_formato_que_lee_la_app():
    kit = {"numero_serie": "LR-000042", "codigo_vinculacion": "7K3P9QXM"}
    assert api._qr_kit(kit) == "lares://vincular?serie=LR-000042&codigo=7K3P9QXM"


def test_estado_de_la_nube_vigente_y_viejo(cliente, rutas):
    (rutas / "kit.yaml").write_text(_KIT)
    estado = {"estado": "conectado", "detalle": "", "actualizado_at": time.time(),
              "ultimo_envio_at": time.time(), "enviados": 3}
    (rutas / "cloud.json").write_text(json.dumps(estado))
    datos = cliente.get("/api/kit").json()
    assert datos["estado"]["estado"] == "conectado" and datos["estado_vigente"] is True

    # Runner detenido: el archivo queda, pero sin actualizar.
    estado["actualizado_at"] = time.time() - 600
    (rutas / "cloud.json").write_text(json.dumps(estado))
    assert cliente.get("/api/kit").json()["estado_vigente"] is False


def test_etiqueta_para_imprimir(cliente, rutas):
    (rutas / "kit.yaml").write_text(_KIT)
    respuesta = cliente.get("/kit")
    assert respuesta.status_code == 200
    assert "7K3P-9QXM" in respuesta.text and "LR-000042" in respuesta.text and "<svg" in respuesta.text
    assert "secreta-no-mostrar" not in respuesta.text


def test_elegir_la_fuente_conectada_a_la_app(cliente, rutas):
    (rutas / "kit.yaml").write_text(_KIT)
    assert cliente.post("/api/kit/fuente", json={"cam_id": "VIVO"}).status_code == 400  # sin `camara`

    (rutas / "kit.yaml").write_text(_KIT + "camara: cam1\n")
    assert cliente.get("/api/kit").json()["fuente_nube"] is None
    assert cliente.post("/api/kit/fuente", json={"cam_id": "VIVO"}).json() == {"cam_id": "VIVO"}
    datos = cliente.get("/api/kit").json()
    assert (datos["camara_kit"], datos["fuente_nube"]) == ("cam1", "VIVO")
    assert cliente.post("/api/kit/fuente", json={"cam_id": "no-existe"}).status_code == 404
    assert cliente.post("/api/kit/fuente", json={"cam_id": None}).json() == {"cam_id": None}
    assert cliente.get("/api/kit").json()["fuente_nube"] is None
