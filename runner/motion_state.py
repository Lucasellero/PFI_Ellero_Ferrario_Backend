"""Estado del gate de movimiento por cámara (ver motion_gate.py), expuesto
vía archivo para que `dashboard/` (proceso separado del runner, ver
overlay_state.py para el mismo patrón) lo muestre en el panel.

Un archivo JSON por cámara en runner_state/motion/, nombrado con el mismo id
sanitizado que ya usa camera_worker.py para el frame preview
(frames/{id}.jpg) — evita que los threads de distintas cámaras compitan por
escribir un único archivo compartido. El id original (sin sanitizar) queda
también dentro del contenido del archivo, para que el lector no dependa de
poder revertir la sanitización.
"""
from __future__ import annotations

import json
import re
import time
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
_STATE_DIR = _ROOT / "runner_state" / "motion"


def _safe_name(cam_id: str) -> str:
    return re.sub(r'[^a-zA-Z0-9_-]', '_', cam_id)


def write_motion_state(cam_id: str, awake: bool) -> None:
    """`awake`: si el gate de movimiento tiene YOLO/CLIP corriendo para esta
    cámara ahora mismo (por movimiento reciente o presencia confirmada por
    YOLO — ver docstring de motion_gate.py). No se llama si el gate está
    desactivado para esa cámara: la ausencia de archivo es la forma en que
    el dashboard distingue "gate desactivado" de "cámara dormida"."""
    _STATE_DIR.mkdir(parents=True, exist_ok=True)
    path = _STATE_DIR / f"{_safe_name(cam_id)}.json"
    tmp = path.with_suffix(".tmp.json")
    tmp.write_text(json.dumps({
        "cam_id": cam_id,
        "awake": bool(awake),
        "updated_at": time.time(),
    }))
    tmp.replace(path)


def read_all_motion_state() -> dict[str, dict]:
    """Devuelve {cam_id: {"awake": bool, "updated_at": float}} para todas las
    cámaras que tienen el gate de movimiento activo y ya escribieron estado
    al menos una vez. Cámaras con el gate desactivado, o el runner sin
    arrancar todavía, simplemente no aparecen."""
    if not _STATE_DIR.exists():
        return {}
    result: dict[str, dict] = {}
    for f in _STATE_DIR.glob("*.json"):
        if f.name.endswith(".tmp.json"):
            continue
        try:
            data = json.loads(f.read_text())
        except (json.JSONDecodeError, OSError):
            continue
        cam_id = data.get("cam_id")
        if cam_id:
            result[cam_id] = {"awake": bool(data.get("awake")), "updated_at": data.get("updated_at")}
    return result
