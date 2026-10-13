"""Imagen de vista previa (`runner_state/frames/{cam_id}.jpg`): solo existe mientras la cámara se analiza.
Se borra cuando el worker se detiene y cuando la cámara se duerme por el gate de movimiento."""
from __future__ import annotations

import threading
from pathlib import Path

import numpy as np
import pytest

from runner.camera_source import Frame
from runner.config import CameraConfig
from runner.preview_file import delete_preview, preview_paths

from .conftest import DetectorFalso

CAM_ID = "cam prueba/1"  # con caracteres que el nombre de archivo tiene que sanear


class FuenteFalsa:
    """Entrega un frame por segundo. Antes de cada uno anota si la vista previa existe, o sea
    cómo quedó el disco después de procesar el frame anterior."""

    def __init__(self, destino, segundos, al_entregar=None):
        self._destino = destino
        self._segundos = segundos
        self._al_entregar = al_entregar
        self.existia_antes_de: dict[int, bool] = {}

    def frames(self):
        for t in range(self._segundos):
            self.existia_antes_de[t] = self._destino.exists()
            if self._al_entregar is not None:
                self._al_entregar(t)
            yield Frame(image=np.zeros((48, 64, 3), dtype=np.uint8), timestamp_s=float(t))


@pytest.fixture
def entorno(camera_worker, tmp_path, monkeypatch):
    """Worker armado contra carpetas temporales: nunca toca `frames/` ni `runner_state/` reales."""
    frames_dir = tmp_path / "frames"
    monkeypatch.setattr(camera_worker, "_FRAMES_DIR", frames_dir)
    monkeypatch.setattr(camera_worker, "write_motion_state", lambda *a, **k: None)
    monkeypatch.setattr(camera_worker, "read_focused_camera", lambda: None)
    monkeypatch.setattr(camera_worker, "read_overlay_enabled", lambda: False)
    tmp, destino = preview_paths(CAM_ID, frames_dir)

    def crear(fuente, *, gate: bool, movimiento=lambda t: False, persona=lambda t: False,
              ausencia_s: float = 3.0, stop_event=None):
        reloj = {"t": -1}

        class MovimientoFalso:
            def __init__(self, min_area_ratio: float):
                pass

            def detect(self, image) -> bool:
                reloj["t"] += 1
                return movimiento(reloj["t"])

        monkeypatch.setattr(camera_worker, "MotionDetector", MovimientoFalso)
        monkeypatch.setattr(camera_worker, "make_source", lambda *a, **k: fuente)
        monkeypatch.setattr(DetectorFalso, "hay_persona", staticmethod(persona))
        return camera_worker.CameraWorker(
            config=CameraConfig(id=CAM_ID, source="falsa"), db_path=tmp_path / "events.db",
            pose_model="falso.pt", stop_event=stop_event or threading.Event(),
            motion_gate_enabled=gate, motion_absence_timeout_s=ausencia_s,
        )

    return crear, destino, tmp, frames_dir, camera_worker


def test_se_borra_cuando_el_runner_se_detiene(entorno):
    crear, destino, tmp, _, _ = entorno
    stop = threading.Event()
    fuente = FuenteFalsa(destino, segundos=30, al_entregar=lambda t: stop.set() if t == 12 else None)

    crear(fuente, gate=False, persona=lambda t: True, stop_event=stop).run()

    # Mientras corría, la imagen estaba (se guarda 1 de cada 5 frames: el quinto es t=4).
    assert fuente.existia_antes_de[4] is False
    assert fuente.existia_antes_de[5] is True
    assert fuente.existia_antes_de[12] is True
    assert max(fuente.existia_antes_de) == 12  # se detuvo ahí, no siguió leyendo
    assert not destino.exists()
    assert not tmp.exists()


def test_se_borra_cuando_la_fuente_termina_o_falla(entorno):
    crear, destino, _, _, _ = entorno

    def falla(t):
        if t == 8:
            raise RuntimeError("se cortó la cámara")

    fuente = FuenteFalsa(destino, segundos=30, al_entregar=falla)
    crear(fuente, gate=False, persona=lambda t: True).run()

    assert fuente.existia_antes_de[8] is True
    assert not destino.exists()


def test_se_borra_cuando_la_camara_se_duerme_por_el_gate(entorno):
    crear, destino, tmp, _, _ = entorno
    # Movimiento en t=0 (despierta), persona en cuadro hasta t=4, después nadie. Con 3 s de
    # tolerancia, en t=8 lleva más de 3 s sin ver a nadie y se duerme.
    fuente = FuenteFalsa(destino, segundos=40)
    crear(fuente, gate=True, movimiento=lambda t: t == 0, persona=lambda t: t <= 4, ausencia_s=3.0).run()

    assert fuente.existia_antes_de[5] is True   # despierta: se guardó el quinto frame
    assert fuente.existia_antes_de[8] is True   # todavía despierta
    assert fuente.existia_antes_de[9] is False  # se durmió en t=8 y borró la imagen
    # Dormida no se vuelve a escribir, ni en los frames donde antes tocaba guardar.
    assert not any(existia for t, existia in fuente.existia_antes_de.items() if t >= 9)
    assert not destino.exists()
    assert not tmp.exists()


def test_vuelve_a_escribirse_cuando_la_camara_despierta(entorno):
    crear, destino, _, _, _ = entorno
    fuente = FuenteFalsa(destino, segundos=40)
    crear(
        fuente, gate=True, movimiento=lambda t: t in (0, 20),
        persona=lambda t: t <= 4 or 20 <= t <= 30, ausencia_s=3.0,
    ).run()

    assert fuente.existia_antes_de[9] is False   # dormida
    assert fuente.existia_antes_de[20] is False  # sigue dormida hasta el movimiento de t=20
    assert fuente.existia_antes_de[26] is True   # despierta otra vez: se guardó el frame 25
    assert not destino.exists()                  # y al terminar se borra


def test_dormida_no_escribe_ni_con_la_camara_en_foco(entorno, monkeypatch):
    crear, destino, _, _, camera_worker = entorno
    monkeypatch.setattr(camera_worker, "read_focused_camera", lambda: CAM_ID)
    fuente = FuenteFalsa(destino, segundos=20)

    crear(fuente, gate=True, movimiento=lambda t: False).run()

    assert not any(fuente.existia_antes_de.values())
    assert not destino.exists()


def test_al_arrancar_borra_la_imagen_de_una_corrida_anterior(entorno):
    crear, destino, _, frames_dir, _ = entorno
    frames_dir.mkdir()
    destino.write_bytes(b"imagen de una corrida que se corto sin limpiar")
    otra_camara = frames_dir / "otra_camara.jpg"
    otra_camara.write_bytes(b"de otra camara")
    fuente = FuenteFalsa(destino, segundos=3)

    crear(fuente, gate=True, movimiento=lambda t: False).run()

    assert fuente.existia_antes_de[0] is False
    assert otra_camara.exists()  # solo se borra la imagen de esta cámara


def test_la_vista_previa_no_vive_en_la_carpeta_de_imagenes_versionadas():
    """`frames/` tiene imágenes de demo versionadas (entre ellas `webcam.jpg`): una cámara con
    ese `id` no tiene que pisarlas ni borrarlas."""
    from runner.preview_file import FRAMES_DIR

    raiz = Path(__file__).resolve().parents[2]
    assert FRAMES_DIR == raiz / "runner_state" / "frames"
    assert preview_paths("webcam")[1] != raiz / "frames" / "webcam.jpg"


def test_borrar_una_vista_previa_que_no_existe_no_falla(tmp_path):
    delete_preview("no_existe", tmp_path)
    delete_preview("no_existe", tmp_path / "carpeta_que_no_existe")
