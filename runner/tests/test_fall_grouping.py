"""Agrupado de caídas (`runner/fall_grouping.py`): como mucho una posible caída publicada por
minuto y por cámara, sin tapar nunca una confirmada que llega después de una sospecha."""
from __future__ import annotations

import sys
import threading

import numpy as np
import pytest

from runner.camera_source import Frame
from runner.config import CameraConfig
from runner.fall_grouping import CONFIRMADA, SOSPECHA, AgrupadorDeCaidas

# Las 16 sospechas de la prueba del 2026-10-14 (video P09T05C05, segundo del video).
SOSPECHAS_DE_LA_PRUEBA = [
    75.11, 510.27, 517.62, 583.22, 613.03, 622.36, 667.84, 690.70,
    704.62, 709.19, 713.78, 717.44, 722.11, 725.61, 730.42, 801.39,
]


def _publicadas(eventos: list[tuple[float, str]], ventana_s: float = 60.0) -> list[tuple[float, str]]:
    agrupador = AgrupadorDeCaidas(ventana_s)
    return [(ts, nivel) for ts, nivel in eventos if agrupador.publicar(ts, nivel)]


def test_las_16_sospechas_de_la_prueba_quedan_en_6():
    publicadas = _publicadas([(ts, SOSPECHA) for ts in SOSPECHAS_DE_LA_PRUEBA])
    assert [ts for ts, _ in publicadas] == [75.11, 510.27, 583.22, 667.84, 730.42, 801.39]
    # Nunca dos en menos de un minuto.
    tiempos = [ts for ts, _ in publicadas]
    assert all(b - a >= 60.0 for a, b in zip(tiempos, tiempos[1:]))


def test_la_ventana_se_cuenta_desde_la_ultima_publicada():
    """Una racha que dura varios minutos publica una por minuto, no una sola."""
    racha = [(float(t), SOSPECHA) for t in range(0, 181, 10)]
    assert [ts for ts, _ in _publicadas(racha)] == [0.0, 60.0, 120.0, 180.0]


def test_una_confirmada_despues_de_una_sospecha_se_publica_siempre():
    eventos = [(0.0, SOSPECHA), (5.0, SOSPECHA), (10.0, CONFIRMADA), (20.0, SOSPECHA), (30.0, CONFIRMADA)]
    assert _publicadas(eventos) == [(0.0, SOSPECHA), (10.0, CONFIRMADA)]


def test_una_sospecha_despues_de_una_confirmada_no_se_publica_dentro_del_minuto():
    assert _publicadas([(0.0, CONFIRMADA), (30.0, SOSPECHA), (61.0, SOSPECHA)]) == [(0.0, CONFIRMADA), (61.0, SOSPECHA)]


def test_si_el_video_vuelve_a_empezar_no_se_arrastra_la_ventana():
    assert _publicadas([(500.0, SOSPECHA), (3.0, SOSPECHA)]) == [(500.0, SOSPECHA), (3.0, SOSPECHA)]


def test_valores_invalidos():
    with pytest.raises(ValueError):
        AgrupadorDeCaidas(-1.0)
    with pytest.raises(ValueError):
        AgrupadorDeCaidas().publicar(0.0, "alta")


# ── En el ciclo de la cámara ───────────────────────────────────────────────────────────────


class _FuenteFalsa:
    def __init__(self, segundos: int):
        self._segundos = segundos

    def frames(self):
        for t in range(self._segundos):
            yield Frame(image=np.zeros((48, 64, 3), dtype=np.uint8), timestamp_s=float(t), wall_clock_ts=1_700_000_000.0 + t)


@pytest.fixture
def caidas_guardadas(camera_worker, tmp_path, monkeypatch):
    """Corre un `CameraWorker` cuya heurística dispara una caída en cada segundo de `disparos`
    (segundo -> evidencia) y devuelve las caídas que guardó. Sin temporal ni CLIP: las de
    evidencia completa se confirman por defecto."""
    monkeypatch.setattr(camera_worker, "_FRAMES_DIR", tmp_path / "frames")
    monkeypatch.setattr(camera_worker, "write_motion_state", lambda *a, **k: None)
    monkeypatch.setattr(camera_worker, "read_focused_camera", lambda: None)
    monkeypatch.setattr(camera_worker, "read_overlay_enabled", lambda: False)

    def _correr(disparos: dict[int, str], segundos: int):
        class DetectorQueDispara:
            def __init__(self, model_path=None):
                self.last_detection = None
                self.last_detection_conf = None

            def process_frame(self, image, timestamp_s):
                evidencia = disparos.get(int(timestamp_s))
                if evidencia is None:
                    return []
                return [{"event_type": "fall", "video_timestamp_s": timestamp_s, "metadata": {"evidence": evidencia}}]

        monkeypatch.setattr(camera_worker, "FallImmobilityDetector", DetectorQueDispara)
        monkeypatch.setattr(camera_worker, "make_source", lambda *a, **k: _FuenteFalsa(segundos))
        db_path = tmp_path / "events.db"
        camera_worker.CameraWorker(
            config=CameraConfig(id="cam1", source="falsa"), db_path=db_path,
            pose_model="falso.pt", stop_event=threading.Event(), motion_gate_enabled=False,
        ).run()
        store = sys.modules["store"].EventStore(db_path)
        try:
            return [(e.video_timestamp_s, e.metadata.get("evidence")) for e in store.all() if e.event_type.value == "fall"]
        finally:
            store.close()

    return _correr


def test_la_camara_guarda_una_sospecha_por_minuto(caidas_guardadas):
    disparos = {t: "partial" for t in (10, 15, 20, 30, 69, 75, 140)}
    assert caidas_guardadas(disparos, 150) == [(10.0, "partial"), (75.0, "partial"), (140.0, "partial")]


def test_la_camara_no_tapa_una_caida_confirmada_por_una_sospecha_previa(caidas_guardadas):
    disparos = {10: "partial", 12: "partial", 20: "full", 25: "full", 40: "partial"}
    assert caidas_guardadas(disparos, 60) == [(10.0, "partial"), (20.0, "full")]
