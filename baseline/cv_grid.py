"""La misma validación cruzada por hogar de `evaluate_deviations.cross_validate`, sin repetir
trabajo: cada candidato del barrido se evalúa una sola vez sobre todos los hogares y después se
arman las particiones con ese resultado ya guardado.

Da exactamente lo mismo que `cross_validate` porque los detectores tratan cada hogar por separado
(`score_class_deviations` y `score_days` recorren hogar por hogar, sin compartir nada entre
ellos): que un candidato marque o no un día de un hogar no depende de qué otros hogares estén en
la partición. `cross_validate` recalcula cada candidato en cada partición (5 veces); acá se
calcula una. `verify_cv_grid.py` comprueba la igualdad contra `cross_validate`.

Sirve además para comparar los dos barridos de `cross_validate` sin volver a calcular nada: el
resultado guardado tiene todos los candidatos, y al armar las particiones se elige entre los de
umbral (el default de `--cv 5`) o se suman los binomiales (`--binomial`). Ver "Actualización
2026-10-13" y "Actualización 2026-10-13 (2)" en README.md.
"""
from __future__ import annotations

import pickle
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd

from evaluate_deviations import _grid_configs
from src.class_deviations import CAMERA_TRACKED_CLASSES, ClassDeviationConfig, score_class_deviations
from src.dynamic_clustering import DriftConfig, score_days

# Los dos barridos de `evaluate_deviations.cross_validate` (ver `_grid_configs`).
# - "umbral": el default de `--cv 5`, el barrido del 2026-09-20. Da los números citados:
#   72,8 %/13,1 % (sensor) y 69,5 %/13,8 % (cámara).
# - "con_binomial": `--cv 5 --binomial`, el del experimento del 2026-10-08. Da 71,0 %/14,2 %
#   (sensor) y 67,3 %/16,0 % (cámara).
CRITERIOS: dict[str, dict] = {
    "umbral": {"binomial": False},
    "con_binomial": {"binomial": True},
}


def evaluate_grid(
    perturbed: pd.DataFrame, labels: pd.DataFrame, camera: bool = False, min_window_days: int | None = None,
) -> dict:
    """Evalúa una vez todos los candidatos (los de umbral y los binomiales, `_grid_configs(...,
    binomial=True)`); cuáles compiten se decide después, en `cross_validate_from_grid`. Devuelve las filas evaluadas
    (`filas`: una por etiqueta; en modo sensor, solo las que tienen score multivariado, igual que
    `evaluate_class_config`), los candidatos (`configs`) y, por candidato y fila, si el día quedó
    marcado (`hits`: detector por clase solo en modo cámara, combinado con el multivariado en
    modo sensor)."""
    homes = sorted(labels["hogar"].unique())
    sub = perturbed[perturbed.index.get_level_values("hogar").isin(homes)]
    exclude = {(r.hogar, r.fecha) for r in labels.itertuples() if r.es_positivo}

    if camera:
        filas = labels.reset_index(drop=True).copy()
        multivariado = np.zeros(len(filas), dtype=bool)
    else:
        drift = DriftConfig() if min_window_days is None else DriftConfig(min_window_days=min_window_days)
        scored = score_days(sub, drift, exclude_from_reference=exclude)
        filas = labels.merge(scored[["hogar", "fecha", "es_desvio"]], on=["hogar", "fecha"], how="inner")
        multivariado = filas.pop("es_desvio").to_numpy(dtype=bool)

    claves = list(zip(filas["hogar"], filas["fecha"]))
    configs = list(_grid_configs(min_window_days, binomial=True))
    hits = np.zeros((len(configs), len(filas)), dtype=bool)
    for i, cfg in enumerate(configs):
        if camera:
            class_hits = score_class_deviations(sub, tracked_classes=CAMERA_TRACKED_CLASSES, config=cfg)
        else:
            class_hits = score_class_deviations(sub, config=cfg, exclude_from_reference=exclude)
        dias = set(zip(class_hits["hogar"], class_hits["fecha"])) if len(class_hits) else set()
        hits[i] = [clave in dias for clave in claves]
    hits |= multivariado

    return {
        "filas": filas, "configs": [asdict(c) for c in configs], "hits": hits,
        "homes": homes, "etiquetas": labels[["hogar", "es_positivo"]].reset_index(drop=True),
    }


def _orden(configs: list[dict], binomial: bool) -> list[int]:
    """Índices de los candidatos que compiten, en el orden en que los recorre `cross_validate`.
    Con empate de recall gana el primero, así que el orden importa. `configs` está guardado en
    el orden del barrido con binomial; el de umbral se arma con el orden de `_grid_configs()`."""
    if binomial:
        return list(range(len(configs)))

    def _clave(c: dict) -> tuple:
        return (
            c["frac_historica_min"], c["excess_k"], c["dias_consecutivos_ausencia"],
            c["separar_fin_de_semana"], c["dias_consecutivos_exceso_deficit"], c["ausencia_requiere_racha"],
        )

    posicion = {_clave(asdict(c)): n for n, c in enumerate(_grid_configs())}
    umbral = [i for i, c in enumerate(configs) if c["ausencia_metodo"] == "umbral"]
    return sorted(umbral, key=lambda i: posicion[_clave(configs[i])])


def cross_validate_from_grid(
    grid: dict,
    n_folds: int = 5,
    seed: int = 0,
    max_fp: float = 0.15,
    binomial: bool = False,
) -> tuple[pd.DataFrame, dict]:
    """Mismas particiones, mismo criterio de selección y mismo resumen que
    `evaluate_deviations.cross_validate`, a partir de un resultado de `evaluate_grid`. `binomial`
    significa lo mismo que ahí: `False` (default) elige entre los candidatos de umbral."""
    filas, configs, hits, homes = grid["filas"], grid["configs"], grid["hits"], grid["homes"]
    rng = np.random.default_rng(seed)
    homes_shuffled = list(homes)
    rng.shuffle(homes_shuffled)
    folds = np.array_split(homes_shuffled, n_folds)

    candidatos = _orden(configs, binomial)
    hogar = filas["hogar"].to_numpy()
    es_pos = filas["es_positivo"].to_numpy(dtype=bool)
    etiquetas = grid["etiquetas"]

    def _tasas(i: int, mascara: np.ndarray) -> tuple[float, float]:
        pos, neg = hits[i][mascara & es_pos], hits[i][mascara & ~es_pos]
        return (float(neg.mean()) if len(neg) else float("nan"), float(pos.mean()) if len(pos) else float("nan"))

    fold_rows = []
    pooled_hit = np.zeros(len(filas), dtype=bool)
    for n, test_homes in enumerate(folds):
        en_test = np.isin(hogar, test_homes.tolist())
        resultados = [(*_tasas(i, ~en_test), i) for i in candidatos]
        ok = [g for g in resultados if g[0] <= max_fp]
        _, _, mejor = max(ok if ok else resultados, key=lambda g: g[1])

        fp_test, recall_test = _tasas(mejor, en_test)
        pooled_hit[en_test] = hits[mejor][en_test]
        cfg = configs[mejor]
        fold_rows.append({
            "fold": n, "n_test_homes": len(test_homes),
            **{k: cfg[k] for k in (
                "ausencia_metodo", "frac_historica_min", "ausencia_alpha", "excess_k", "dias_consecutivos_ausencia",
                "separar_fin_de_semana", "dias_consecutivos_exceso_deficit", "ausencia_requiere_racha",
            )},
            "recall_test": recall_test, "fp_rate_test": fp_test,
            # como en `cross_validate`: todas las etiquetas de la partición, tengan score o no
            "n_pos_test": int(etiquetas.loc[etiquetas["hogar"].isin(test_homes.tolist()), "es_positivo"].sum()),
            "n_neg_test": int((~etiquetas.loc[etiquetas["hogar"].isin(test_homes.tolist()), "es_positivo"]).sum()),
        })

    escenarios = filas.loc[es_pos, "escenario"]
    por_escenario = pd.Series(pooled_hit[es_pos], index=escenarios.to_numpy()).groupby(level=0)
    pooled = {
        "recall_out_of_fold": float(pooled_hit[es_pos].mean()),
        "fp_rate_out_of_fold": float(pooled_hit[~es_pos].mean()),
        "n_pos": int(es_pos.sum()),
        "n_neg": int((~es_pos).sum()),
        "detectados": int(pooled_hit[es_pos].sum()),
        "falsos_positivos": int(pooled_hit[~es_pos].sum()),
        "sin_evaluar": len(etiquetas) - len(filas),
        "recall_por_escenario": por_escenario.mean().to_dict(),
        "n_por_escenario": por_escenario.size().to_dict(),
    }
    return pd.DataFrame(fold_rows), pooled


def cached_grid(path: Path, compute) -> dict:
    """Devuelve el barrido guardado en `path`; si no existe, lo calcula con `compute()` y lo
    guarda. Borrar `synthetic_eval/cv_grid_*.pkl` después de regenerar el set de desvíos."""
    if path.exists():
        with path.open("rb") as f:
            return pickle.load(f)
    grid = compute()
    with path.open("wb") as f:
        pickle.dump(grid, f)
    return grid
