"""Imagen de vista previa de cada cámara (`runner_state/frames/{cam_id}.jpg`).

Es la única imagen del hogar que la controladora deja en disco, así que solo existe mientras la
cámara se está analizando: `camera_worker.py` la escribe con la cámara despierta y la borra al
dormirse por el gate de movimiento y al detenerse. `dashboard/` la borra también al detener el
runner, porque en Windows `terminate()` mata el proceso sin darle tiempo a limpiar.

Vive en `runner_state/` (ignorada por git, igual que el resto del estado que comparten runner y
dashboard) y no en `frames/`: ahí están las imágenes de demo versionadas, y una cámara cuyo `id`
coincidiera con una de ellas (`webcam`) la pisaba y la borraba.
"""
from __future__ import annotations

import re
import time
from pathlib import Path

FRAMES_DIR = Path(__file__).resolve().parents[1] / "runner_state" / "frames"

_RETRIES = 5
_RETRY_DELAY_S = 0.005


def preview_name(cam_id: str) -> str:
    """Nombre de archivo seguro para el id de una cámara."""
    return re.sub(r"[^a-zA-Z0-9_-]", "_", cam_id)


def preview_paths(cam_id: str, frames_dir: Path | None = None) -> tuple[Path, Path]:
    """(temporal, definitivo): se escribe el temporal y se mueve sobre el definitivo."""
    base = frames_dir if frames_dir is not None else FRAMES_DIR
    name = preview_name(cam_id)
    return base / f"{name}.tmp.jpg", base / f"{name}.jpg"


def delete_preview(cam_id: str, frames_dir: Path | None = None) -> None:
    """Borra la vista previa de esa cámara, y el temporal si quedó uno a medio escribir. No falla
    si no existe. En Windows el archivo puede estar abierto por el dashboard en ese instante
    (misma carrera que describe `atomic_file.py`): se reintenta unas pocas veces."""
    for path in preview_paths(cam_id, frames_dir):
        for _ in range(_RETRIES):
            try:
                path.unlink(missing_ok=True)
                break
            except PermissionError:
                time.sleep(_RETRY_DELAY_S)
            except OSError:
                break
