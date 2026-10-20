"""Modo demo (`playback_id` en `VideoFileLoop`, ver runner/playback_state.py): el video arranca en
pausa, se pausa desde el dashboard, el tiempo en pausa no cuenta para los timestamps, y al terminar queda en
pausa hasta que se le da play, que lo arranca de nuevo desde el principio."""
from __future__ import annotations

import threading
import time

import cv2
import numpy as np
import pytest

from runner import playback_state
from runner.camera_source import VideoFileLoop

CAM_ID = "demo/1"
FPS = 25
N_FRAMES = 25  # 1 s de video


@pytest.fixture
def video(tmp_path, monkeypatch):
    monkeypatch.setattr(playback_state, "STATE_DIR", tmp_path / "playback")
    path = tmp_path / "clip.avi"
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), FPS, (32, 24))
    for i in range(N_FRAMES):
        writer.write(np.full((24, 32, 3), i * 10 % 256, dtype=np.uint8))
    writer.release()
    return str(path)


class Lector:
    """Consume `frames()` en un thread y anota cada timestamp."""

    def __init__(self, source):
        self.timestamps: list[float] = []
        self.restarted: list[bool] = []
        self._source = source
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def _run(self):
        for frame in self._source.frames():
            self.timestamps.append(frame.timestamp_s)
            self.restarted.append(frame.restarted)

    def join(self, timeout=3.0):
        self._thread.join(timeout)
        return not self._thread.is_alive()


def esperar(cond, timeout=3.0):
    limite = time.monotonic() + timeout
    while time.monotonic() < limite:
        if cond():
            return True
        time.sleep(0.02)
    return False


def status():
    return playback_state.read_all_status().get(CAM_ID)


def dar_play():
    """El video arranca en pausa: espera a que el runner lo publique y le da play."""
    assert esperar(lambda: status() is not None and status()["paused"])
    playback_state.write_paused(CAM_ID, False)


def test_al_terminar_queda_en_pausa_y_play_lo_reinicia(video):
    stop = threading.Event()
    lector = Lector(VideoFileLoop(video, realtime=False, loop=True, playback_id=CAM_ID, stop_event=stop))
    dar_play()

    assert esperar(lambda: status() is not None and status()["ended"])
    assert len(lector.timestamps) == N_FRAMES  # una sola pasada aunque loop=True
    assert playback_state.read_paused(CAM_ID)
    time.sleep(0.3)
    assert len(lector.timestamps) == N_FRAMES  # sigue quieto en el final

    playback_state.write_paused(CAM_ID, False)
    assert esperar(lambda: len(lector.timestamps) == 2 * N_FRAMES)
    # Los timestamps siguen creciendo en la segunda pasada (no vuelven a 0).
    assert lector.timestamps == sorted(lector.timestamps)
    # Por eso el primer frame de la segunda pasada viene marcado: el worker reinicia ahí el estado
    # de detección (si no, la caída del segundo play caía dentro del minuto del agrupador).
    assert [i for i, r in enumerate(lector.restarted) if r] == [N_FRAMES]

    stop.set()
    assert lector.join()
    assert status() is None  # al cortar se borra el estado


def test_la_pausa_no_cuenta_como_tiempo_transcurrido(video):
    stop = threading.Event()
    lector = Lector(VideoFileLoop(video, realtime=True, playback_id=CAM_ID, stop_event=stop))
    dar_play()

    assert esperar(lambda: len(lector.timestamps) >= 5)
    playback_state.write_paused(CAM_ID, True)
    assert esperar(lambda: status() is not None and status()["paused"])
    vistos = len(lector.timestamps)
    time.sleep(1.0)
    assert len(lector.timestamps) <= vistos + 1  # pausado no entrega frames

    playback_state.write_paused(CAM_ID, False)
    assert esperar(lambda: status() is not None and status()["ended"])
    # 1 s de video con 1 s de pausa en el medio: los timestamps cubren ~1 s, no ~2 s.
    assert lector.timestamps[-1] - lector.timestamps[0] < 1.4
    # Y casi no se descartaron frames por "atraso" al reanudar.
    assert len(lector.timestamps) >= N_FRAMES - 3

    stop.set()
    assert lector.join()


def test_cortar_estando_en_pausa(video):
    stop = threading.Event()
    lector = Lector(VideoFileLoop(video, realtime=True, playback_id=CAM_ID, stop_event=stop))
    dar_play()
    assert esperar(lambda: len(lector.timestamps) >= 1)
    playback_state.write_paused(CAM_ID, True)
    assert esperar(lambda: status() is not None and status()["paused"])
    stop.set()
    assert lector.join(timeout=1.0)


def test_arranca_en_pausa_aunque_haya_quedado_reproduciendo(video):
    playback_state.write_paused(CAM_ID, False)  # corrida anterior
    stop = threading.Event()
    lector = Lector(VideoFileLoop(video, realtime=False, playback_id=CAM_ID, stop_event=stop))
    assert esperar(lambda: status() is not None)
    assert status()["paused"] and not status()["ended"] and status()["position_s"] == 0
    time.sleep(0.3)
    assert lector.timestamps == []  # no entrega nada hasta el play

    playback_state.write_paused(CAM_ID, False)
    assert esperar(lambda: len(lector.timestamps) == N_FRAMES)
    stop.set()
    assert lector.join()


def test_sin_modo_demo_no_toca_el_estado(video):
    lector = Lector(VideoFileLoop(video, realtime=False, loop=False))
    assert lector.join()
    assert len(lector.timestamps) == N_FRAMES
    assert not playback_state.STATE_DIR.exists()
