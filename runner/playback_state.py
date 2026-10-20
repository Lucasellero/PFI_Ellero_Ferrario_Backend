"""Play/pausa de los videos en modo demo, compartido (vía archivo) entre dashboard/ y runner/.

Solo se usa con `demo: true` en cameras.yaml (ver "Modo demo" en README.md): las cámaras cuya
fuente es un archivo arrancan en pausa, se reproducen una sola vez, se pueden pausar desde el panel, y al terminar
quedan en el último frame hasta que se les vuelve a dar play (arrancan de nuevo desde el
principio). Mismo patrón de archivo compartido que overlay_state.py y motion_state.py, porque
dashboard/ (FastAPI) y runner/ (subproceso) corren en procesos separados.

Dos archivos por cámara en runner_state/playback/, para que nunca escriban los dos procesos el
mismo archivo a la vez:
  {id}.control.json — lo escribe el dashboard (botón play/pausa): {"paused": bool}. El runner
                      también lo escribe al arrancar (cada video empieza en pausa, esperando el play)
                      y al llegar al final del video (queda en pausa).
  {id}.status.json  — lo escribe el runner: posición, duración, si está pausado o terminó.
                      No existe si la cámara no está en modo demo o el runner está detenido.
"""
from __future__ import annotations

import json
import re
import time
from pathlib import Path

from .atomic_file import replace_atomic

_ROOT = Path(__file__).resolve().parents[1]
STATE_DIR = _ROOT / "runner_state" / "playback"


def _safe_name(cam_id: str) -> str:
    return re.sub(r"[^a-zA-Z0-9_-]", "_", cam_id)


def _write_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(data))
    replace_atomic(tmp, path)


def _control_path(cam_id: str) -> Path:
    return STATE_DIR / f"{_safe_name(cam_id)}.control.json"


def _status_path(cam_id: str) -> Path:
    return STATE_DIR / f"{_safe_name(cam_id)}.status.json"


def write_paused(cam_id: str, paused: bool) -> None:
    _write_json(_control_path(cam_id), {"cam_id": cam_id, "paused": bool(paused)})


def read_paused(cam_id: str) -> bool:
    """True si el video de esa cámara tiene que estar en pausa. Sin archivo: reproduciendo."""
    try:
        return bool(json.loads(_control_path(cam_id).read_text()).get("paused", False))
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        return False


def write_status(cam_id: str, *, paused: bool, ended: bool, position_s: float, duration_s: float) -> None:
    _write_json(_status_path(cam_id), {
        "cam_id": cam_id,
        "paused": bool(paused),
        "ended": bool(ended),
        "position_s": float(position_s),
        "duration_s": float(duration_s),
        "updated_at": time.time(),
    })


def delete_status(cam_id: str) -> None:
    try:
        _status_path(cam_id).unlink(missing_ok=True)
    except OSError:
        pass


def read_all_status() -> dict[str, dict]:
    """{cam_id: {paused, ended, position_s, duration_s, updated_at}} de las cámaras que hoy se
    están reproduciendo en modo demo. Las demás no aparecen."""
    if not STATE_DIR.exists():
        return {}
    result: dict[str, dict] = {}
    for f in STATE_DIR.glob("*.status.json"):
        try:
            data = json.loads(f.read_text())
        except (json.JSONDecodeError, OSError):
            continue
        cam_id = data.pop("cam_id", None)
        if cam_id:
            result[cam_id] = data
    return result
