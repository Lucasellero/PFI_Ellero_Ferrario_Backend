"""Las pruebas de `runner/` no cargan YOLO: `vision/src/pose_pipeline.py` importa `ultralytics`
(y con él PyTorch), así que se reemplaza por un módulo falso antes de importar
`runner.camera_worker`. Lo que se prueba acá es el ciclo de vida de los archivos, no la detección.
"""
from __future__ import annotations

import sys
import types

import pytest


class DetectorFalso:
    """Reemplazo de `FallImmobilityDetector`: `hay_persona(ts)` decide si "ve" a alguien."""

    hay_persona = staticmethod(lambda ts: False)

    def __init__(self, model_path: str | None = None):
        self.last_detection = None
        self.last_detection_conf = None

    def process_frame(self, image, timestamp_s: float) -> list[dict]:
        self.last_detection = ("kpts", "box") if type(self).hay_persona(timestamp_s) else None
        return []


@pytest.fixture
def camera_worker(monkeypatch):
    """`runner.camera_worker` importado con `vision/` reemplazado por `DetectorFalso`."""
    paquete = types.ModuleType("src")
    paquete.__path__ = []
    falso = types.ModuleType("src.pose_pipeline")
    falso.FallImmobilityDetector = DetectorFalso
    falso.build_pose_sample = lambda *a, **k: None
    falso.draw_pose = lambda image, kpts, box: image
    monkeypatch.setitem(sys.modules, "src", paquete)
    monkeypatch.setitem(sys.modules, "src.pose_pipeline", falso)
    monkeypatch.delitem(sys.modules, "runner.camera_worker", raising=False)
    monkeypatch.delitem(sys.modules, "runner.main", raising=False)

    import runner.camera_worker as modulo

    monkeypatch.setattr(DetectorFalso, "hay_persona", staticmethod(lambda ts: False))
    yield modulo
    sys.modules.pop("runner.camera_worker", None)
    sys.modules.pop("runner.main", None)
