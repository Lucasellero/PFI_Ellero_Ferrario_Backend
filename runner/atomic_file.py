"""Reemplazo atómico de archivos de estado compartidos entre runner/ y dashboard/.

En Windows, `Path.replace()` falla con PermissionError si el destino está
abierto por otro proceso en ese instante (Python abre sin FILE_SHARE_DELETE) —
y los archivos de runner_state/ y frames/ los lee el dashboard mientras el
runner los pisa (y al revés con focus.json/overlay.json). Es una carrera
esperable, no un error: se reintenta unas pocas veces y, si sigue ocupado, se
descarta esa escritura (la próxima la pisa igual).
"""
from __future__ import annotations

import time
from pathlib import Path

_RETRIES = 5
_RETRY_DELAY_S = 0.005


def replace_atomic(tmp: Path, dest: Path) -> bool:
    """Mueve `tmp` sobre `dest`. False si el destino quedó ocupado tras los reintentos."""
    for _ in range(_RETRIES):
        try:
            tmp.replace(dest)
            return True
        except PermissionError:
            time.sleep(_RETRY_DELAY_S)
    return False
