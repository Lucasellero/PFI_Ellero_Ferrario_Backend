"""Vectores de "día" por residente: insumo de `dynamic_clustering.py`.

Para cada `(hogar, fecha)` se arma un vector con, por cada clase canónica de actividad
(`dataset_events.activity_map.CANONICAL_CLASSES`, 19 clases), la duración total en minutos
y la cantidad de eventos de ese día — 38 dims en total. Es la versión "vector de rutina diaria"
del mismo evento que ya produce el ETL de CASAS (`dataset_events/build_table.py`), pensada para
alimentar el clustering dinámico de `dynamic_clustering.py`.

Nota: la duración de un evento se atribuye entero al día de su `inicio` (columna `fecha` de
`casas_events.parquet`), aunque el evento cruce la medianoche (ej. dormir de 23:40 a 07:00) —
mismo criterio simple que usa el resto del proyecto para no sobre-ingenierizar antes de medir si
hace falta más precisión.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

_DATASET_EVENTS_DIR = Path(__file__).resolve().parents[2] / "dataset_events"
if str(_DATASET_EVENTS_DIR) not in sys.path:
    sys.path.insert(0, str(_DATASET_EVENTS_DIR))

from activity_map import CANONICAL_CLASSES, COARSE, COARSE_CLASSES  # noqa: E402

# Por debajo de esto, un día se considera de cobertura insuficiente para representar la rutina
# real de esa fecha (poco tiempo etiquetado) — se marca, no se descarta acá (lo filtra quien
# consume el vector, ej. `dynamic_clustering.score_days`).
COBERTURA_MIN_HORAS = 4.0


def feature_columns() -> list[str]:
    """Nombres de las 38 columnas de features (sin `cobertura_horas`/`cobertura_baja`)."""
    cols = []
    for cls in CANONICAL_CLASSES:
        cols.append(f"{cls}_min")
        cols.append(f"{cls}_n")
    return cols


def build_day_vectors(events_parquet: Path, homes: list[str] | None = None) -> pd.DataFrame:
    """Arma el vector de 38 dims por `(hogar, fecha)` a partir de `casas_events.parquet`.

    Devuelve un DataFrame con MultiIndex `(hogar, fecha)`, ordenado cronológicamente dentro de
    cada hogar, con columnas `<clase>_min`, `<clase>_n` por cada una de las 19 clases canónicas,
    más `cobertura_horas` (float) y `cobertura_baja` (bool, ver `COBERTURA_MIN_HORAS`).
    """
    df = pd.read_parquet(events_parquet, columns=["hogar", "actividad", "fecha", "duracion_min"])
    if homes is not None:
        df = df[df["hogar"].isin(homes)]
    if df.empty:
        raise ValueError(f"Sin eventos para homes={homes!r} en {events_parquet}")

    # cast a str: evita que unstack() traiga columnas de categorías no observadas en el subset
    df = df.assign(hogar=df["hogar"].astype(str), actividad=df["actividad"].astype(str))

    grouped = df.groupby(["hogar", "fecha", "actividad"], observed=True)["duracion_min"]
    dur = grouped.sum().unstack("actividad", fill_value=0.0)
    cnt = grouped.count().unstack("actividad", fill_value=0)

    for cls in CANONICAL_CLASSES:
        if cls not in dur.columns:
            dur[cls] = 0.0
        if cls not in cnt.columns:
            cnt[cls] = 0

    dur = dur[CANONICAL_CLASSES].add_suffix("_min").astype("float64")
    cnt = cnt[CANONICAL_CLASSES].add_suffix("_n").astype("int32")

    cobertura_horas = df.groupby(["hogar", "fecha"], observed=True)["duracion_min"].sum() / 60.0
    cobertura_horas.name = "cobertura_horas"

    out = pd.concat([dur, cnt], axis=1)
    out["cobertura_horas"] = cobertura_horas
    out["cobertura_baja"] = out["cobertura_horas"] < COBERTURA_MIN_HORAS
    out.index = out.index.set_names(["hogar", "fecha"])
    out = out.sort_index()
    return out


def add_camera_columns(day_vectors: pd.DataFrame) -> pd.DataFrame:
    """Agrega al vector-día las columnas `cam_<clase_gruesa>_min`/`_n` (prefijo `cam_` para no
    pisar ninguna columna canónica existente, ver `dataset_events.activity_map.COARSE`) — la
    versión "qué puede dar una cámara única" de las 19 columnas canónicas, sumando las clases
    canónicas que caen en cada clase gruesa (ej. `comer`/`beber`/`medicacion` se reparten entre
    `cam_comida_mesa` y `cam_comida`, ver la nota en `activity_map.COARSE`).

    No reemplaza las columnas canónicas — quedan las dos disponibles en el mismo DataFrame, para
    poder comparar "desvíos detectables con sensor" (columnas canónicas) vs. "solo con cámara"
    (columnas `cam_*`) sin tener que rearmar el vector dos veces.
    """
    out = day_vectors.copy()
    for coarse_cls in COARSE_CLASSES:
        canonicas = [c for c, g in COARSE.items() if g == coarse_cls]
        min_cols = [f"{c}_min" for c in canonicas if f"{c}_min" in out.columns]
        n_cols = [f"{c}_n" for c in canonicas if f"{c}_n" in out.columns]
        if min_cols:
            out[f"cam_{coarse_cls}_min"] = out[min_cols].sum(axis=1)
        if n_cols:
            out[f"cam_{coarse_cls}_n"] = out[n_cols].sum(axis=1)
    return out
