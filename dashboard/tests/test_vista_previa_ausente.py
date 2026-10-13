"""El panel tiene que seguir funcionando cuando `runner_state/frames/{cam_id}.jpg` no existe: es lo normal
con el runner detenido o con la cámara dormida (ver "Frame preview" en `runner/README.md`)."""
from __future__ import annotations

import asyncio

import pytest
from fastapi.testclient import TestClient

from dashboard import api


@pytest.fixture
def frames_dir(tmp_path, monkeypatch):
    carpeta = tmp_path / "frames"
    monkeypatch.setattr(api, "FRAMES_DIR", carpeta)
    return carpeta


@pytest.fixture
def cliente(frames_dir, tmp_path, monkeypatch):
    monkeypatch.setattr(api, "DB_PATH", tmp_path / "events.db")
    # Sin `with`: no corre el evento de arranque, así que no queda ninguna tarea de fondo.
    return TestClient(api.app)


def test_frame_de_una_camara_sin_vista_previa_da_404(cliente):
    assert cliente.get("/api/frame/webcam").status_code == 404


def test_la_lista_de_frames_queda_vacia_sin_carpeta_ni_archivos(cliente, frames_dir):
    assert cliente.get("/api/frames/list").json() == {"available": []}
    frames_dir.mkdir()
    assert cliente.get("/api/frames/list").json() == {"available": []}


def test_el_resto_del_panel_responde_sin_vista_previa(cliente):
    assert cliente.get("/").status_code == 200
    assert cliente.get("/api/status").json() == {"running": False, "pid": None}
    assert cliente.get("/api/events").status_code == 200


def test_el_stream_espera_sin_fallar_y_retoma_cuando_vuelve_la_imagen(frames_dir):
    """`/api/frame-stream`: sin archivo no manda nada (ni corta la conexión); cuando la cámara
    despierta y el archivo vuelve a existir, manda ese frame."""
    frames_dir.mkdir()

    async def escenario() -> bytes:
        respuesta = await api.stream_frame("webcam")
        partes = respuesta.body_iterator
        try:
            siguiente = asyncio.ensure_future(partes.__anext__())
            terminadas, _ = await asyncio.wait({siguiente}, timeout=0.3)
            assert not terminadas, "sin archivo no tiene que mandar nada"
            (frames_dir / "webcam.jpg").write_bytes(b"jpeg-de-prueba")
            return await asyncio.wait_for(siguiente, timeout=2.0)
        finally:
            await partes.aclose()

    parte = asyncio.run(escenario())
    assert b"Content-Type: image/jpeg" in parte
    assert b"jpeg-de-prueba" in parte


def test_detener_el_runner_borra_la_vista_previa_de_cada_camara(cliente, frames_dir, tmp_path, monkeypatch):
    """En Windows `terminate()` mata al runner sin que llegue a limpiar: lo hace el panel."""
    frames_dir.mkdir()
    for nombre in ("cam1.jpg", "cam_2.jpg", "cam_2.tmp.jpg", "CAIDA_1.jpg"):
        (frames_dir / nombre).write_bytes(b"x")
    config = tmp_path / "cameras.yaml"
    config.write_text("cameras:\n- id: cam1\n  source: a.mp4\n- id: cam 2\n  source: b.mp4\n", encoding="utf-8")
    monkeypatch.setattr(api, "CAMERAS_YAML", config)

    class RunnerFalso:
        returncode = None

        def terminate(self) -> None:
            pass

        async def wait(self) -> int:
            self.returncode = 0
            return 0

    monkeypatch.setattr(api, "runner_proc", RunnerFalso())

    assert cliente.post("/api/stop").json() == {"status": "stopped"}

    # Queda solo la imagen que no es de ninguna cámara configurada.
    assert sorted(p.name for p in frames_dir.iterdir()) == ["CAIDA_1.jpg"]
    assert cliente.get("/api/frame/cam1").status_code == 404
