"""Genera un set de evaluación etiquetado inyectando desvíos sintéticos sobre secuencias reales
de CASAS — no hay ground truth real de "día anómalo" en el dataset, así que se mide igual que el
resto del proyecto mide sin dataset etiquetado de antemano (ver hard-negative mining en
`fusion/extract_falls.py`, grid search en `vision/recalibrate_fall_thresholds.py`): perturbar
copias de datos reales de forma controlada y medir recall/FP contra esas perturbaciones.

Escenarios inyectados (uno por residente y por tipo, sobre días reales elegidos al azar). Los
primeros 3 usan columnas **canónicas** (19 clases, requieren sensor dedicado — ver "Vocabulario de
cámara" en README); el último usa la columna **`cam_comida`** (`features.add_camera_columns()`),
la versión realista de lo que una cámara sin zona puede dar:
  - `sin_medicacion`: un día con medicación anotada, puesto a 0 (min y n). Solo detectable con
    sensor — la cámara no distingue medicación de comer/beber.
  - `bano_nocturno_excesivo`: un día con `bano_nocturno_n` subido bien por encima de lo habitual
    de ese residente.
  - `semana_sin_salir`: 7 días consecutivos con `salir_casa`/`entrar_casa` puestos a 0 — esta sí es
    alcanzable por cámara (`salir_casa`/`entrar_casa` son eventos propios en el vocabulario de 9
    clases, ver `dataset_events/activity_map.py`).
  - `comida_camara_ausente`: un día con `comer`/`beber`/`medicacion` (las 3 canónicas) puestas a
    0 — la versión "cámara realista" de `sin_medicacion`: detecta que no hubo ningún registro de
    comida/bebida/medicación, sin poder decir cuál de las tres faltó. Se perturban las columnas
    canónicas, no las `cam_*` directamente — `add_camera_columns()` se recalcula al final, ya con
    todo perturbado, para que `cam_comida`/`cam_comida_mesa` reflejen el estado final.

Además se guarda una muestra de días reales sin tocar (`real_sin_tocar`) por residente, como
negativos para medir falsos positivos.

Uso:
  py -3 inject_synthetic_deviations.py
  py -3 inject_synthetic_deviations.py --homes hh113 hh115 --seed 1
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from src.features import add_camera_columns, build_day_vectors

EVENTS_PARQUET = Path(__file__).resolve().parents[1] / "dataset_events" / "events" / "casas_events.parquet"


def _all_homes(events_parquet: Path) -> list[str]:
    """Todos los hogares presentes en el parquet — default 2026-09-05 (antes eran solo los 7
    'recomendados' por dataset_events/README.md; se amplió a los 80 tras encontrar que dejaba
    afuera hogares con volumen de sobra, ver "Cross-validation por hogar" en README). `inject()`
    ya descarta los que no llegan a `MIN_DIAS_POR_HOGAR` de historial (quedan ~21 de 80)."""
    return sorted(pd.read_parquet(events_parquet, columns=["hogar"])["hogar"].unique().tolist())
MIN_DIAS_POR_HOGAR = 60  # necesita historial mínimo para que evaluar tenga sentido
N_NEGATIVOS_POR_HOGAR = 20


def _inject_one_home(day_vectors: pd.DataFrame, hogar: str, rng: np.random.Generator) -> list[dict]:
    """Perturba `day_vectors` (in-place) para `hogar`, devuelve las etiquetas generadas."""
    g = day_vectors.xs(hogar, level="hogar")
    good = g[~g["cobertura_baja"]]
    fechas = good.index.to_list()
    if len(fechas) < MIN_DIAS_POR_HOGAR:
        return []

    labels: list[dict] = []

    # --- sin_medicacion ---
    candidatos = [f for f in fechas if good.loc[f, "medicacion_min"] > 0]
    if candidatos:
        f = candidatos[rng.integers(len(candidatos))]
        day_vectors.loc[(hogar, f), "medicacion_min"] = 0.0
        day_vectors.loc[(hogar, f), "medicacion_n"] = 0
        labels.append({"hogar": hogar, "fecha": f, "escenario": "sin_medicacion", "es_positivo": True})

    # --- bano_nocturno_excesivo ---
    base_bn_n = good["bano_nocturno_n"]
    mediana_bn = base_bn_n.median()
    candidatos = [f for f in fechas if base_bn_n.loc[f] <= mediana_bn]
    if candidatos:
        f = candidatos[rng.integers(len(candidatos))]
        p95 = base_bn_n.quantile(0.95)
        nuevo_n = max(4, int(p95) + 2)
        dur_media_evento = good["bano_nocturno_min"].replace(0, np.nan).median()
        dur_media_evento = 5.0 if pd.isna(dur_media_evento) else float(dur_media_evento)
        day_vectors.loc[(hogar, f), "bano_nocturno_n"] = nuevo_n
        day_vectors.loc[(hogar, f), "bano_nocturno_min"] = nuevo_n * dur_media_evento
        labels.append({"hogar": hogar, "fecha": f, "escenario": "bano_nocturno_excesivo", "es_positivo": True})

    # --- comida_salteada: se saltó una comida completa (ej. no almorzó), no todo el día ---
    base_comer_n = good["comer_n"]
    mediana_comer = base_comer_n.median()
    candidatos = [f for f in fechas if base_comer_n.loc[f] >= mediana_comer and base_comer_n.loc[f] >= 2]
    if candidatos:
        f = candidatos[rng.integers(len(candidatos))]
        n_actual = int(good.loc[f, "comer_n"])
        n_nuevo = max(1, n_actual - max(1, int(mediana_comer * 0.4)))
        dur_actual = good.loc[f, "comer_min"]
        dur_por_evento = dur_actual / n_actual if n_actual > 0 else dur_actual
        day_vectors.loc[(hogar, f), "comer_n"] = n_nuevo
        day_vectors.loc[(hogar, f), "comer_min"] = round(dur_por_evento * n_nuevo, 2)
        labels.append({"hogar": hogar, "fecha": f, "escenario": "comida_salteada", "es_positivo": True})

    # --- comida_camara_ausente: versión "cámara realista" de sin_medicacion — se perturban las
    # 3 columnas canónicas que hoy se reparten entre `cam_comida_mesa` (comer) y `cam_comida`
    # (beber/medicacion, ver activity_map.COARSE); `add_camera_columns()` se recalcula después de
    # perturbar (en main()), así que no hace falta tocar las columnas `cam_*` acá directamente.
    total_comida_dia = good["comer_n"] + good["beber_n"] + good["medicacion_n"]
    candidatos = [f for f in fechas if total_comida_dia.loc[f] >= max(total_comida_dia.median(), 1)]
    if candidatos:
        f = candidatos[rng.integers(len(candidatos))]
        for cls in ("comer", "beber", "medicacion"):
            day_vectors.loc[(hogar, f), f"{cls}_n"] = 0
            day_vectors.loc[(hogar, f), f"{cls}_min"] = 0.0
        labels.append({"hogar": hogar, "fecha": f, "escenario": "comida_camara_ausente", "es_positivo": True})

    # --- semana_sin_salir ---
    if len(fechas) >= 30:
        start_i = int(rng.integers(20, len(fechas) - 10))
        semana = fechas[start_i : start_i + 7]
        for f in semana:
            day_vectors.loc[(hogar, f), "salir_casa_min"] = 0.0
            day_vectors.loc[(hogar, f), "salir_casa_n"] = 0
            day_vectors.loc[(hogar, f), "entrar_casa_min"] = 0.0
            day_vectors.loc[(hogar, f), "entrar_casa_n"] = 0
            labels.append({"hogar": hogar, "fecha": f, "escenario": "semana_sin_salir", "es_positivo": True})

    # --- negativos: días reales sin tocar ---
    tocados = {l["fecha"] for l in labels}
    negativos_candidatos = [f for f in fechas if f not in tocados]
    n_neg = min(N_NEGATIVOS_POR_HOGAR, len(negativos_candidatos))
    if n_neg:
        idx = rng.choice(len(negativos_candidatos), size=n_neg, replace=False)
        for i in idx:
            labels.append(
                {"hogar": hogar, "fecha": negativos_candidatos[i], "escenario": "real_sin_tocar", "es_positivo": False}
            )

    return labels


def inject(day_vectors: pd.DataFrame, homes: list[str], seed: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(seed)
    perturbed = day_vectors.copy()
    homes_presentes = set(day_vectors.index.get_level_values("hogar"))
    all_labels: list[dict] = []
    for hogar in homes:
        if hogar not in homes_presentes:
            print(f"  ! {hogar} no está en day_vectors, se omite")
            continue
        all_labels.extend(_inject_one_home(perturbed, hogar, rng))
    return perturbed, pd.DataFrame(all_labels)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--homes", nargs="*", default=None, help="default: todos los hogares del parquet")
    ap.add_argument("--events-parquet", type=Path, default=EVENTS_PARQUET)
    ap.add_argument("--out-dir", type=Path, default=Path(__file__).resolve().parent / "synthetic_eval")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    if args.homes is None:
        args.homes = _all_homes(args.events_parquet)

    print(f"Construyendo vectores-día reales para {len(args.homes)} hogares ...")
    day_vectors = build_day_vectors(args.events_parquet, homes=args.homes)
    day_vectors = add_camera_columns(day_vectors)  # necesario para elegir candidatos de algunos escenarios
    print(f"  {len(day_vectors)} días")

    print("Inyectando desvíos sintéticos ...")
    perturbed, labels = inject(day_vectors, args.homes, seed=args.seed)
    # recalcular cam_* DESPUÉS de perturbar: las columnas cam_* deben reflejar el estado final de
    # las canónicas, no el que tenían antes de inyectar (si no, un escenario que perturba una
    # canónica —ej. semana_sin_salir sobre salir_casa— no se vería reflejado en su columna cam_*).
    perturbed = add_camera_columns(perturbed)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    perturbed.to_parquet(args.out_dir / "day_vectors_perturbed.parquet")
    labels.to_parquet(args.out_dir / "labels.parquet")

    print(f"\n  {args.out_dir / 'day_vectors_perturbed.parquet'}  ({len(perturbed)} días)")
    print(f"  {args.out_dir / 'labels.parquet'}  ({len(labels)} etiquetas)")
    print()
    print(labels["escenario"].value_counts().to_string())


if __name__ == "__main__":
    main()
