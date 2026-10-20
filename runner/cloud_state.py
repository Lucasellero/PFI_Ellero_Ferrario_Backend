"""Estado de la sincronización con la nube, compartido (vía archivo) entre runner/ y dashboard/.

Lo escribe `runner/cloud_sync.py::CloudSyncThread` después de cada intento y lo lee el
dashboard para el panel "App móvil" (ver dashboard/README.md). Mismo patrón de archivo
compartido que overlay_state.py y motion_state.py, porque dashboard/ (FastAPI) y runner/
(subproceso) corren en procesos separados. Un solo escritor: el thread de sincronización.

`runner_state/cloud.json`:
  estado          — "iniciando" | "conectado" | "esperando_vinculacion" | "dado_de_baja" |
                    "sin_conexion" | "api_key_invalida" | "error"
  detalle         — texto de la nube o del error de red (vacío si no hay nada que contar)
  actualizado_at  — último intento de sincronizar (epoch, segundos)
  ultimo_envio_at — último lote que la nube aceptó, o None
  enviados        — eventos que la nube aceptó desde que arrancó el runner

El estado solo cambia cuando hay eventos para mandar: con la cola vacía la controladora no le
pregunta nada a la nube y el último estado conocido se mantiene.

`runner_state/cloud_source.json` (lo escribe el dashboard, botón "App" de cada fuente): qué
fuente de cameras.yaml es la cámara del kit. Ver `runner/cloud_sync.py::_sync_lote`.
"""
from __future__ import annotations

import json
from pathlib import Path

from .atomic_file import replace_atomic

_ROOT = Path(__file__).resolve().parents[1]
STATE_PATH = _ROOT / "runner_state" / "cloud.json"
SOURCE_PATH = _ROOT / "runner_state" / "cloud_source.json"


def write_cloud_state(state: dict, path: Path = STATE_PATH) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(state))
    replace_atomic(tmp, path)


def read_cloud_state(path: Path = STATE_PATH) -> dict | None:
    """El último estado escrito, o None si el runner nunca sincronizó (o sin nube configurada)."""
    try:
        return json.loads(path.read_text())
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        return None


def delete_cloud_state(path: Path = STATE_PATH) -> None:
    path.unlink(missing_ok=True)


def write_cloud_source(cam_id: str | None, path: Path = SOURCE_PATH) -> None:
    """Elige la fuente de cameras.yaml que se sube a la nube como la cámara del kit (None: ninguna)."""
    write_cloud_state({"cam_id": cam_id}, path)


def read_cloud_source(path: Path = SOURCE_PATH) -> str | None:
    data = read_cloud_state(path)
    return data.get("cam_id") if data else None
