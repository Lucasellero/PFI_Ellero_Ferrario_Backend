"""Estado compartido (vía archivo) para mostrar/ocultar el overlay de pose YOLO.

El dashboard (proceso FastAPI) y el runner (subproceso con un thread por
cámara) corren en procesos separados, así que la forma más simple de que un
toggle en la UI afecte a todas las cámaras ya en ejecución es a través de un
archivo que ambos procesos leen/escriben.
"""
from __future__ import annotations

import json
from pathlib import Path

from .atomic_file import replace_atomic

_ROOT = Path(__file__).resolve().parents[1]
OVERLAY_STATE_PATH = _ROOT / "runner_state" / "overlay.json"


def read_overlay_enabled() -> bool:
    """True si hay que dibujar el bbox + keypoints en los previews. Default: True."""
    try:
        data = json.loads(OVERLAY_STATE_PATH.read_text())
        return bool(data.get("show_overlay", True))
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        return True


def write_overlay_enabled(enabled: bool) -> None:
    OVERLAY_STATE_PATH.parent.mkdir(exist_ok=True)
    tmp = OVERLAY_STATE_PATH.with_suffix(".tmp.json")
    tmp.write_text(json.dumps({"show_overlay": bool(enabled)}))
    replace_atomic(tmp, OVERLAY_STATE_PATH)
