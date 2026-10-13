"""Los scripts de `baseline/` se importan como cuando se corren desde esa carpeta
(`from src.camera_events import ...`, `from cv_grid import ...`). `src` es también el nombre del
paquete de `vision/` que usa `runner/`, así que cada prueba lo importa de cero con `baseline/`
primero en el path y lo saca al terminar.
"""
from __future__ import annotations

import importlib
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

BASELINE_DIR = Path(__file__).resolve().parents[1]
_SCRIPTS = (
    "cv_grid", "evaluate_classifier_noise", "evaluate_deviations", "evaluate_production_vocab",
    "inject_synthetic_deviations",
)


def _olvidar_modulos() -> None:
    for nombre in list(sys.modules):
        if nombre == "src" or nombre.startswith("src.") or nombre in _SCRIPTS:
            del sys.modules[nombre]


@pytest.fixture
def baseline_modulo(monkeypatch):
    """`baseline_modulo("src.camera_events")` -> el módulo de `baseline/`, recién importado."""
    _olvidar_modulos()
    monkeypatch.syspath_prepend(str(BASELINE_DIR))
    yield importlib.import_module
    _olvidar_modulos()


@pytest.fixture
def epv(baseline_modulo):
    return baseline_modulo("evaluate_production_vocab")


# Rutina de un día normal del hogar de prueba: (episodios, minutos por episodio).
RUTINA = {
    "comer": (3, 20.0), "beber": (2, 2.0), "medicacion": (1, 1.0),
    "cocinar": (2, 25.0), "lavar_platos": (1, 10.0),
    "descansar": (2, 30.0), "ver_tv": (1, 60.0),
    "leer": (1, 30.0), "trabajo": (1, 45.0), "telefono": (1, 5.0),
    "salir_casa": (1, 90.0), "entrar_casa": (1, 1.0),
    "dormir": (1, 420.0),
}


@pytest.fixture
def hogares(baseline_modulo):
    """Vector de día de dos hogares regulares de 90 días (con un poco de variación en los
    minutos) y uno de 30 días, que no llega al historial mínimo para evaluarse."""
    canonicas = baseline_modulo("src.features").CANONICAL_CLASSES
    rng = np.random.default_rng(0)
    filas = []
    for hogar, dias in (("hA", 90), ("hB", 90), ("hCorto", 30)):
        for fecha in pd.date_range("2011-01-03", periods=dias).date:
            fila = {"hogar": hogar, "fecha": fecha}
            for clase in canonicas:
                n, minutos = RUTINA.get(clase, (0, 0.0))
                fila[f"{clase}_n"] = n
                fila[f"{clase}_min"] = n * minutos * float(rng.uniform(0.9, 1.1))
            fila["cobertura_horas"] = 12.0
            fila["cobertura_baja"] = False
            filas.append(fila)
    return pd.DataFrame(filas).set_index(["hogar", "fecha"]).sort_index()
