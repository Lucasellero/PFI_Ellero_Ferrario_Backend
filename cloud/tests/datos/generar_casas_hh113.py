"""Regenera `casas_hh113_60dias.csv`, el historial que usa `test_linea_base_produccion.py`.

Toma 60 días seguidos del hogar `hh113` de CASAS (`dataset_events/events/casas_events.parquet`,
ver `dataset_events/README.md` para descargarlo y armarlo) y deja solo las actividades que la
cámara puede clasificar, ya traducidas a la categoría del clasificador (`activity_label`) con el
mismo mapeo que `baseline/evaluate_classifier_noise.py`. Las horas son la hora local del hogar,
tal como vienen en CASAS.

    python cloud/tests/datos/generar_casas_hh113.py

Fuente: CASAS Smart Home, `labeled_data.zip` (DOI 10.5281/zenodo.15708568), CC BY 4.0.
Cook, Crandall, Thomas y Krishnan (2013), "CASAS: A smart home in a box", IEEE Computer 46(7).
"""
from __future__ import annotations

import sys
from datetime import date, timedelta
from pathlib import Path

import pandas as pd

_RAIZ = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(_RAIZ / "baseline"))

from evaluate_classifier_noise import CANONICAL_TO_CATEGORY  # noqa: E402

HOGAR = "hh113"
PRIMER_DIA = date(2011, 9, 5)
DIAS = 60
SALIDA = Path(__file__).resolve().parent / "casas_hh113_60dias.csv"


def main() -> None:
    df = pd.read_parquet(
        _RAIZ / "dataset_events" / "events" / "casas_events.parquet",
        columns=["hogar", "actividad", "fecha", "inicio", "duracion_s"],
    )
    ultimo_dia = PRIMER_DIA + timedelta(days=DIAS - 1)
    df = df[(df["hogar"].astype(str) == HOGAR) & (df["fecha"] >= PRIMER_DIA) & (df["fecha"] <= ultimo_dia)]
    df = df.assign(actividad=df["actividad"].astype(str))
    df = df[df["actividad"].isin(CANONICAL_TO_CATEGORY)].sort_values("inicio")
    salida = pd.DataFrame({
        "inicio": df["inicio"].dt.strftime("%Y-%m-%dT%H:%M:%S"),
        "duracion_s": df["duracion_s"].round().astype(int),
        "actividad_casas": df["actividad"],
        "activity_label": df["actividad"].map(CANONICAL_TO_CATEGORY),
    })
    salida.to_csv(SALIDA, index=False, lineterminator="\n")
    print(f"{SALIDA}: {len(salida)} actividades, {PRIMER_DIA} a {ultimo_dia}")


if __name__ == "__main__":
    main()
