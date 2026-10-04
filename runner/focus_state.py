"""Estado compartido (vía archivo) para indicarle a runner/ qué cámara está en
vista única en el dashboard.

Mientras una cámara está "enfocada", su CameraWorker escribe el preview
(frames/<id>.jpg) en cada frame procesado en vez de 1 de cada
_FRAME_PREVIEW_EVERY_N (ver camera_worker.py) — la pose ya corre sobre todos
los frames mientras la cámara está despierta, así que esto no agrega costo de
inferencia, solo de encode+escritura del JPEG. Eso permite que la vista única
del dashboard muestre un stream fluido con overlay de YOLO sin levantar un
segundo modelo/segunda decodificación del lado del dashboard — mismo patrón
de archivo compartido que overlay_state.py, porque dashboard/ (FastAPI) y
runner/ (subproceso) corren en procesos separados.
"""
from __future__ import annotations

import json
from pathlib import Path

from .atomic_file import replace_atomic

_ROOT = Path(__file__).resolve().parents[1]
FOCUS_STATE_PATH = _ROOT / "runner_state" / "focus.json"


def read_focused_camera() -> str | None:
    """Id de la cámara en vista única, o None si no hay ninguna (vista de grilla)."""
    try:
        data = json.loads(FOCUS_STATE_PATH.read_text())
        cam_id = data.get("cam_id")
        return cam_id if cam_id else None
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        return None


def write_focused_camera(cam_id: str | None) -> None:
    FOCUS_STATE_PATH.parent.mkdir(exist_ok=True)
    tmp = FOCUS_STATE_PATH.with_suffix(".tmp.json")
    tmp.write_text(json.dumps({"cam_id": cam_id}))
    replace_atomic(tmp, FOCUS_STATE_PATH)
