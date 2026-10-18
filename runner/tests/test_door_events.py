"""Entradas y salidas de casa por la zona `puerta` (`runner/door_events.py`): la máquina de
estados con trayectorias sintéticas, y el ciclo de la cámara guardando los eventos.

No hay video real de entradas y salidas: acá se prueba la lógica, no cuántas detecta en un hogar.
La opción (`puerta_activada` en `cameras.yaml`) viene encendida y se puede apagar.
"""
from __future__ import annotations

import sys
import threading

import numpy as np
import pytest

from runner.camera_source import Frame
from runner.config import CameraConfig, load_config
from runner.door_events import (
    LABEL_ENTRAR, LABEL_SALIR, DoorEventDetector, posicion_normalizada,
)

PUERTA = [[0.8, 0.0], [1.0, 0.0], [1.0, 1.0], [0.8, 1.0]]  # franja derecha del cuadro
EN_PUERTA = (0.9, 0.5)
EN_SALA = (0.4, 0.5)
AUSENCIA = 10.0


def _recorrer(detector: DoorEventDetector, posicion, hasta: int, desde: int = 0) -> list[tuple[str, float]]:
    """Una muestra por segundo. `posicion(t)` devuelve dónde está la persona o None."""
    eventos = []
    for t in range(desde, hasta):
        eventos += [(e.label, e.timestamp_s) for e in detector.update(float(t), posicion(t), wall_clock_ts=1000.0 + t)]
    return eventos


@pytest.fixture
def detector():
    return DoorEventDetector(PUERTA, ausencia_min_s=AUSENCIA, memoria_puerta_s=2.0)


def test_desaparecer_en_la_puerta_y_no_volver_es_una_salida(detector):
    posicion = lambda t: EN_SALA if t < 5 else EN_PUERTA if t < 8 else None
    # Se confirma al cumplirse la ausencia mínima, con la hora en que se la vio por última vez.
    assert _recorrer(detector, posicion, hasta=17) == []
    assert _recorrer(detector, posicion, hasta=40, desde=17) == [(LABEL_SALIR, 7.0)]


def test_la_salida_lleva_la_hora_de_reloj_de_la_ultima_vez_que_se_la_vio(detector):
    eventos = []
    for t in range(40):
        eventos += detector.update(float(t), EN_PUERTA if t < 8 else None, wall_clock_ts=1000.0 + t)
    assert [(e.label, e.wall_clock_ts) for e in eventos] == [(LABEL_SALIR, 1007.0)]


def test_aparecer_en_la_puerta_despues_de_una_ausencia_es_una_entrada(detector):
    posicion = lambda t: None if t < 30 else EN_PUERTA if t < 33 else EN_SALA
    assert _recorrer(detector, posicion, hasta=60) == [(LABEL_ENTRAR, 30.0)]


def test_salir_y_volver_da_una_salida_y_una_entrada(detector):
    posicion = lambda t: EN_PUERTA if t < 5 else None if t < 50 else EN_PUERTA if t < 52 else EN_SALA
    assert _recorrer(detector, posicion, hasta=80) == [(LABEL_SALIR, 4.0), (LABEL_ENTRAR, 50.0)]


def test_perderse_un_momento_en_la_puerta_no_es_salida_ni_entrada(detector):
    """Oclusión breve o YOLO que pierde a la persona unos segundos: menos que la ausencia mínima."""
    posicion = lambda t: None if 5 <= t < 13 else EN_PUERTA  # se la vuelve a ver a los 9 s
    assert _recorrer(detector, posicion, hasta=60) == []


def test_desaparecer_lejos_de_la_puerta_no_es_una_salida(detector):
    """Salió de cuadro hacia otro ambiente."""
    posicion = lambda t: EN_SALA if t < 8 else None
    assert _recorrer(detector, posicion, hasta=60) == []


def test_pasar_por_la_puerta_y_desaparecer_en_otro_lado_no_es_una_salida(detector):
    """Estuvo en la zona, pero lo último que se supo de ella (más allá de la tolerancia) es que
    estaba en la sala."""
    posicion = lambda t: EN_PUERTA if t < 5 else EN_SALA if t < 10 else None
    assert _recorrer(detector, posicion, hasta=60) == []


def test_reaparecer_lejos_de_la_puerta_no_es_una_entrada(detector):
    posicion = lambda t: None if t < 30 else EN_SALA
    assert _recorrer(detector, posicion, hasta=60) == []


def test_al_arrancar_con_alguien_en_la_puerta_no_se_inventa_una_entrada(detector):
    """No se sabe cuánto hace que no había nadie: la ausencia se cuenta desde la primera muestra."""
    assert _recorrer(detector, lambda t: EN_PUERTA, hasta=5) == []


def test_una_salida_se_informa_una_sola_vez(detector):
    posicion = lambda t: EN_PUERTA if t < 3 else None
    assert _recorrer(detector, posicion, hasta=500) == [(LABEL_SALIR, 2.0)]


def test_sin_muestras_durante_la_ausencia_la_salida_se_confirma_al_volver(detector):
    """Si no llegó ninguna muestra mientras no había nadie (la fuente se cortó), la salida se
    resuelve con la primera muestra posterior, antes de la entrada."""
    eventos = _recorrer(detector, lambda t: EN_PUERTA, hasta=5)
    eventos += [(e.label, e.timestamp_s) for e in detector.update(100.0, EN_PUERTA)]
    assert eventos == [(LABEL_SALIR, 4.0), (LABEL_ENTRAR, 100.0)]


def test_si_el_reloj_vuelve_atras_se_empieza_de_cero(detector):
    """Video en loop: lo visto en la vuelta anterior no cuenta."""
    _recorrer(detector, lambda t: EN_PUERTA, hasta=5)
    assert _recorrer(detector, lambda t: None, hasta=40) == []


def test_la_zona_puede_ser_un_poligono_cualquiera():
    triangulo = DoorEventDetector([[0.0, 0.0], [0.4, 0.0], [0.0, 0.4]], ausencia_min_s=AUSENCIA)
    assert _recorrer(triangulo, lambda t: (0.1, 0.1) if t < 3 else None, hasta=30) == [(LABEL_SALIR, 2.0)]
    otro = DoorEventDetector([[0.0, 0.0], [0.4, 0.0], [0.0, 0.4]], ausencia_min_s=AUSENCIA)
    assert _recorrer(otro, lambda t: (0.35, 0.35) if t < 3 else None, hasta=30) == []  # fuera del triángulo


@pytest.mark.parametrize("poligono, ausencia, memoria", [
    ([[0, 0], [1, 1]], 10.0, 2.0), (PUERTA, 0.0, 2.0), (PUERTA, 10.0, -1.0),
])
def test_configuracion_invalida_se_rechaza(poligono, ausencia, memoria):
    with pytest.raises(ValueError):
        DoorEventDetector(poligono, ausencia_min_s=ausencia, memoria_puerta_s=memoria)


def _esqueleto(x: float, y: float, ancho: int = 64, alto: int = 48):
    kpts = np.zeros((17, 2), dtype=np.float32)
    kpts[11] = kpts[12] = (x * ancho, y * alto)
    return kpts


def test_la_posicion_es_el_punto_medio_de_las_caderas():
    kpts = _esqueleto(0.0, 0.0)
    kpts[11], kpts[12] = (16.0, 24.0), (32.0, 24.0)
    assert posicion_normalizada(kpts, (0, 0, 64, 48), 64, 48) == pytest.approx((0.375, 0.5))


def test_si_yolo_no_ve_las_caderas_se_usa_el_centro_del_recuadro():
    kpts = _esqueleto(0.0, 0.0)  # caderas sin detectar: YOLO las deja en (0, 0)
    conf = np.full(17, 0.9, dtype=np.float32)
    conf[11] = 0.1
    assert posicion_normalizada(kpts, (48, 12, 64, 36), 64, 48, conf) == pytest.approx((0.875, 0.5))
    conf[11] = 0.9
    assert posicion_normalizada(kpts, (48, 12, 64, 36), 64, 48, conf) == pytest.approx((0.0, 0.0))


# ── El ciclo de la cámara ──────────────────────────────────────────────────────────────────


class FuenteFalsa:
    def __init__(self, segundos: int):
        self._segundos = segundos

    def frames(self):
        for t in range(self._segundos):
            yield Frame(image=np.zeros((48, 64, 3), dtype=np.uint8), timestamp_s=float(t), wall_clock_ts=1_700_000_000.0 + t)


@pytest.fixture
def correr(camera_worker, tmp_path, monkeypatch):
    """Corre un `CameraWorker` sobre una fuente falsa y devuelve los eventos que guardó."""
    monkeypatch.setattr(camera_worker, "_FRAMES_DIR", tmp_path / "frames")
    monkeypatch.setattr(camera_worker, "write_motion_state", lambda *a, **k: None)
    monkeypatch.setattr(camera_worker, "read_focused_camera", lambda: None)
    monkeypatch.setattr(camera_worker, "read_overlay_enabled", lambda: False)

    def _correr(posicion, segundos: int, *, zones, gate: bool = False, movimiento=lambda t: False, activada: bool | None = True):
        class DetectorConPosicion:
            def __init__(self, model_path=None):
                self.last_detection = None
                self.last_detection_conf = None
                self.person_present = False

            def process_frame(self, image, timestamp_s):
                punto = posicion(int(timestamp_s))
                self.last_detection = None if punto is None else (_esqueleto(*punto), np.array([0, 0, 64, 48]))
                self.person_present = self.last_detection is not None
                return []

        reloj = {"t": -1}

        class MovimientoFalso:
            def __init__(self, min_area_ratio: float):
                pass

            def detect(self, image) -> bool:
                reloj["t"] += 1
                return movimiento(reloj["t"])

        monkeypatch.setattr(camera_worker, "FallImmobilityDetector", DetectorConPosicion)
        monkeypatch.setattr(camera_worker, "MotionDetector", MovimientoFalso)
        monkeypatch.setattr(camera_worker, "make_source", lambda *a, **k: FuenteFalsa(segundos))
        db_path = tmp_path / "events.db"
        opcion = {} if activada is None else {"puerta_activada": activada}  # None: el default del worker
        camera_worker.CameraWorker(
            config=CameraConfig(id="cam1", source="falsa", zones=zones), db_path=db_path,
            pose_model="falso.pt", stop_event=threading.Event(),
            motion_gate_enabled=gate, motion_absence_timeout_s=3.0,
            puerta_ausencia_min_s=AUSENCIA, puerta_memoria_s=2.0, **opcion,
        ).run()
        store = sys.modules["store"].EventStore(db_path)
        try:
            return store.all()
        finally:
            store.close()

    return _correr


def _va_y_vuelve(t: int):
    if t < 4:
        return EN_SALA
    if t < 7:
        return EN_PUERTA
    if t < 40:
        return None
    return EN_PUERTA if t < 43 else EN_SALA


def test_la_camara_guarda_la_salida_y_la_entrada_como_eventos_de_actividad(correr):
    eventos = correr(_va_y_vuelve, 60, zones={"puerta": PUERTA})

    assert [(e.metadata["activity_label"], e.video_timestamp_s) for e in eventos] == [(LABEL_SALIR, 6.0), (LABEL_ENTRAR, 40.0)]
    assert all(e.event_type.value == "activity" and e.video_id == "cam1" for e in eventos)
    assert all(e.metadata["confirmed"] is True and "ongoing" not in e.metadata for e in eventos)
    # La salida lleva la hora en que se la vio por última vez; la entrada, la de la detección.
    assert [e.wall_clock_ts for e in eventos] == [1_700_000_006.0, 1_700_000_040.0]
    assert all(e.event_id for e in eventos)  # pasan por el store como cualquier otro evento


def test_con_la_camara_dormida_la_ausencia_sigue_corriendo(correr):
    """Con el gate de movimiento la cámara se duerme a los 3 s sin ver a nadie y deja de correr
    YOLO: la salida igual se confirma, y la entrada se detecta cuando el movimiento la despierta."""
    eventos = correr(_va_y_vuelve, 60, zones={"puerta": PUERTA}, gate=True, movimiento=lambda t: t in (0, 40))

    assert [(e.metadata["activity_label"], e.video_timestamp_s) for e in eventos] == [(LABEL_SALIR, 6.0), (LABEL_ENTRAR, 40.0)]


def test_por_defecto_la_camara_emite_entradas_y_salidas_si_tiene_la_zona(correr):
    """La opción viene encendida: alcanza con dibujar la zona `puerta`."""
    eventos = correr(_va_y_vuelve, 60, zones={"puerta": PUERTA}, activada=None)
    assert [e.metadata["activity_label"] for e in eventos] == [LABEL_SALIR, LABEL_ENTRAR]


def test_con_la_opcion_apagada_la_camara_no_emite_aunque_tenga_la_zona(correr):
    assert correr(_va_y_vuelve, 60, zones={"puerta": PUERTA}, activada=False) == []
    assert correr(_va_y_vuelve, 60, zones={"puerta": PUERTA}, activada=False, gate=True, movimiento=lambda t: t in (0, 40)) == []


def test_sin_zona_puerta_la_camara_no_emite_entradas_ni_salidas(correr):
    assert correr(_va_y_vuelve, 60, zones=None) == []
    assert correr(_va_y_vuelve, 60, zones={"mesa": PUERTA}) == []


def test_los_tiempos_de_la_puerta_se_leen_de_cameras_yaml(tmp_path):
    archivo = tmp_path / "cameras.yaml"
    archivo.write_text(
        "cameras:\n- id: living\n  source: '0'\n  zones:\n    puerta: [[0.8, 0.0], [1.0, 0.0], [1.0, 1.0]]\n",
        encoding="utf-8",
    )
    config = load_config(archivo)
    assert config.puerta_activada is True  # encendida salvo que se apague
    assert (config.puerta_ausencia_min_s, config.puerta_memoria_s) == (60.0, 2.0)
    assert config.cameras[0].zones["puerta"] == [[0.8, 0.0], [1.0, 0.0], [1.0, 1.0]]

    archivo.write_text(
        archivo.read_text(encoding="utf-8") + "puerta_activada: false\npuerta_ausencia_min_s: 90\npuerta_memoria_s: 3\n",
        encoding="utf-8",
    )
    config = load_config(archivo)
    assert config.puerta_activada is False
    assert (config.puerta_ausencia_min_s, config.puerta_memoria_s) == (90.0, 3.0)


def test_el_runner_le_pasa_la_opcion_a_cada_camara():
    """`main.py` construye los workers con `puerta_activada=config.puerta_activada`."""
    from pathlib import Path

    fuente = (Path(__file__).resolve().parents[1] / "main.py").read_text(encoding="utf-8")
    assert "puerta_activada=config.puerta_activada" in fuente
