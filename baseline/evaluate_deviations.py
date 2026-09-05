"""Mide recall/FP de los detectores de día (`dynamic_clustering.py` + `class_deviations.py`)
contra el set de desvíos sintéticos (`inject_synthetic_deviations.py`) — mismo framing recall/FP
que ya usan `vision/evaluate_falls.py` y `fusion/train_fall_temporal.py`.

`DriftConfig` (multivariado) ya se calibró en una sesión anterior (defaults del dataclass,
`window_days=30, k=2, threshold_z=2.5`) y no se vuelve a barrer acá por defecto — recalcularlo
para cada combinación de `ClassDeviationConfig` sería recomputar KMeans miles de veces sin
necesidad, porque el detector por clase no depende de él. Se corre UNA vez y se barre
`ClassDeviationConfig` (que es barato) sobre ese mismo resultado. Pasar `--full-grid` para barrer
también `(window_days, k, threshold_z)` como en la calibración original (lento).

Uso:
  py -3 inject_synthetic_deviations.py     # generar el set primero (una vez)
  py -3 evaluate_deviations.py
  py -3 evaluate_deviations.py --full-grid  # barrido completo, incluye DriftConfig (lento)
  py -3 evaluate_deviations.py --camera     # versión "sin sensor dedicado"
  py -3 evaluate_deviations.py --cv 5               # cross-validation por hogar, modo sensor
  py -3 evaluate_deviations.py --cv 5 --camera       # ídem, modo cámara
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from src.class_deviations import CAMERA_TRACKED_CLASSES, ClassDeviationConfig, TRACKED_CLASSES, score_class_deviations
from src.dynamic_clustering import DriftConfig, score_days

DEFAULT_DIR = Path(__file__).resolve().parent / "synthetic_eval"

WINDOW_DAYS_GRID = (14, 21, 30)
K_GRID = (2, 3, 4)
THRESHOLD_GRID = (2.0, 2.5, 3.0, 3.5)

FRAC_HISTORICA_GRID = (0.6, 0.7, 0.8, 0.9)
EXCESS_K_GRID = (2.5, 3.0, 4.0, 5.0)
DIAS_CONSECUTIVOS_GRID = (2, 3)


def load_synthetic(out_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    perturbed = pd.read_parquet(out_dir / "day_vectors_perturbed.parquet")
    labels = pd.read_parquet(out_dir / "labels.parquet")
    return perturbed, labels


def _rates(df: pd.DataFrame, col: str) -> dict:
    pos = df[df["es_positivo"]]
    neg = df[~df["es_positivo"]]
    recall = float(pos[col].mean()) if len(pos) else float("nan")
    fp_rate = float(neg[col].mean()) if len(neg) else float("nan")
    por_escenario = pos.groupby("escenario")[col].mean() if len(pos) else pd.Series(dtype=float)
    return {"recall": recall, "fp_rate": fp_rate, **{f"recall_{esc}": float(v) for esc, v in por_escenario.items()}}


def evaluate_class_config(
    perturbed: pd.DataFrame,
    labels: pd.DataFrame,
    scored_multivariado: pd.DataFrame,
    class_config: ClassDeviationConfig,
    return_df: bool = False,
):
    """Combina un resultado multivariado YA CALCULADO con un `ClassDeviationConfig` a evaluar."""
    exclude = {(row.hogar, row.fecha) for row in labels.itertuples() if row.es_positivo}
    merged = labels.merge(scored_multivariado, on=["hogar", "fecha"], how="inner")

    class_hits = score_class_deviations(perturbed, config=class_config, exclude_from_reference=exclude)
    dias_con_hit_clase = set(zip(class_hits["hogar"], class_hits["fecha"])) if len(class_hits) else set()
    merged["es_desvio_clase"] = [(h, f) in dias_con_hit_clase for h, f in zip(merged["hogar"], merged["fecha"])]
    merged["es_desvio_combinado"] = merged["es_desvio"] | merged["es_desvio_clase"]

    result = {
        "frac_historica_min": class_config.frac_historica_min,
        "excess_k": class_config.excess_k,
        "dias_consecutivos_ausencia": class_config.dias_consecutivos_ausencia,
        "n_pos": int(merged["es_positivo"].sum()),
        "n_neg": int((~merged["es_positivo"]).sum()),
        "n_sin_score": len(labels) - len(merged),
    }
    for col, prefix in (("es_desvio", "multivariado"), ("es_desvio_clase", "por_clase"), ("es_desvio_combinado", "combinado")):
        for k_, v_ in _rates(merged, col).items():
            result[f"{prefix}_{k_}"] = v_
    return (result, merged) if return_df else result


def evaluate_camera(perturbed: pd.DataFrame, labels: pd.DataFrame, class_config: ClassDeviationConfig, return_df: bool = False):
    """Evalúa SOLO el detector por clase, con `CAMERA_TRACKED_CLASSES` (columnas `cam_*`) — la
    versión "sin sensor dedicado". No corre el multivariado: `dynamic_clustering.py` trabaja sobre
    el vector canónico de 38 dims (con `medicacion` propia), no tiene todavía un equivalente sobre
    columnas `cam_*` — queda pendiente si se retoma esta vía (ver README).
    """
    class_hits = score_class_deviations(perturbed, tracked_classes=CAMERA_TRACKED_CLASSES, config=class_config)
    hits = set(zip(class_hits["hogar"], class_hits["fecha"])) if len(class_hits) else set()
    labels = labels.copy()
    labels["hit"] = [(h, f) in hits for h, f in zip(labels["hogar"], labels["fecha"])]
    result = {
        "frac_historica_min": class_config.frac_historica_min,
        "excess_k": class_config.excess_k,
        "dias_consecutivos_ausencia": class_config.dias_consecutivos_ausencia,
    }
    result.update({f"camara_{k}": v for k, v in _rates(labels, "hit").items()})
    return (result, labels) if return_df else result


def cross_validate(
    perturbed: pd.DataFrame,
    labels: pd.DataFrame,
    n_folds: int = 5,
    camera: bool = False,
    seed: int = 0,
    max_fp: float = 0.15,
) -> tuple[pd.DataFrame, dict]:
    """K-fold por HOGAR (no por día — mismo criterio que `GroupKFold` por video en
    `vision/`/`fusion/`): en cada fold, la grilla de `ClassDeviationConfig` se elige mirando solo
    los hogares de entrenamiento, y se mide recall/FP solo sobre los hogares de test (que el
    detector nunca usó para elegir parámetros) — evita el sesgo optimista de calibrar y medir
    sobre los mismos datos. Al final se agrupan las predicciones de TODOS los folds de test
    (pooled, no promedio simple de tasas por fold — más robusto con folds de tamaño desigual) para
    un recall/FP "out-of-fold" único.
    """
    homes = sorted(labels["hogar"].unique())
    rng = np.random.default_rng(seed)
    homes_shuffled = list(homes)
    rng.shuffle(homes_shuffled)
    folds = np.array_split(homes_shuffled, n_folds)

    scored_all = None
    if not camera:
        exclude_all = {(r.hogar, r.fecha) for r in labels.itertuples() if r.es_positivo}
        scored_all = score_days(perturbed, DriftConfig(), exclude_from_reference=exclude_all)

    def _eval(perturbed_sub, labels_sub, cfg):
        if camera:
            return evaluate_camera(perturbed_sub, labels_sub, cfg, return_df=True)
        scored_sub = scored_all[scored_all["hogar"].isin(labels_sub["hogar"].unique())]
        return evaluate_class_config(perturbed_sub, labels_sub, scored_sub, cfg, return_df=True)

    recall_key = "camara_recall" if camera else "combinado_recall"
    fp_key = "camara_fp_rate" if camera else "combinado_fp_rate"
    hit_col = "hit" if camera else "es_desvio_combinado"

    fold_rows = []
    pooled_frames = []
    for i, test_homes in enumerate(folds):
        test_homes = set(test_homes.tolist())
        train_homes = [h for h in homes if h not in test_homes]

        train_labels = labels[labels["hogar"].isin(train_homes)]
        train_perturbed = perturbed[perturbed.index.get_level_values("hogar").isin(train_homes)]

        grid_results = []
        for frac in FRAC_HISTORICA_GRID:
            for excess_k in EXCESS_K_GRID:
                for dias_cons in DIAS_CONSECUTIVOS_GRID:
                    cfg = ClassDeviationConfig(frac_historica_min=frac, excess_k=excess_k, dias_consecutivos_ausencia=dias_cons)
                    r, _ = _eval(train_perturbed, train_labels, cfg)
                    grid_results.append((r[fp_key], r[recall_key], cfg))

        ok = [g for g in grid_results if g[0] <= max_fp]
        pool = ok if ok else grid_results
        _, _, best_cfg = max(pool, key=lambda g: g[1])

        test_labels = labels[labels["hogar"].isin(test_homes)]
        test_perturbed = perturbed[perturbed.index.get_level_values("hogar").isin(test_homes)]
        r_test, df_test = _eval(test_perturbed, test_labels, best_cfg)
        df_test = df_test.copy()
        df_test["fold"] = i
        pooled_frames.append(df_test)

        fold_rows.append(
            {
                "fold": i,
                "n_test_homes": len(test_homes),
                "frac_historica_min": best_cfg.frac_historica_min,
                "excess_k": best_cfg.excess_k,
                "dias_consecutivos_ausencia": best_cfg.dias_consecutivos_ausencia,
                "recall_test": r_test[recall_key],
                "fp_rate_test": r_test[fp_key],
                "n_pos_test": int(test_labels["es_positivo"].sum()),
                "n_neg_test": int((~test_labels["es_positivo"]).sum()),
            }
        )

    pooled = pd.concat(pooled_frames, ignore_index=True)
    pos = pooled[pooled["es_positivo"]]
    neg = pooled[~pooled["es_positivo"]]
    pooled_summary = {
        "recall_out_of_fold": float(pos[hit_col].mean()) if len(pos) else float("nan"),
        "fp_rate_out_of_fold": float(neg[hit_col].mean()) if len(neg) else float("nan"),
        "n_pos": int(len(pos)),
        "n_neg": int(len(neg)),
        "recall_por_escenario": pos.groupby("escenario")[hit_col].mean().to_dict() if len(pos) else {},
    }
    return pd.DataFrame(fold_rows), pooled_summary


def evaluate_full(perturbed: pd.DataFrame, labels: pd.DataFrame, config: DriftConfig, class_config: ClassDeviationConfig) -> dict:
    """Recalcula también el multivariado — usado solo por `--full-grid`."""
    exclude = {(row.hogar, row.fecha) for row in labels.itertuples() if row.es_positivo}
    scored = score_days(perturbed, config, exclude_from_reference=exclude)
    result = evaluate_class_config(perturbed, labels, scored, class_config)
    result.update({"window_days": config.window_days, "k": config.k, "threshold_z": config.threshold_z})
    return result


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dir", type=Path, default=DEFAULT_DIR)
    ap.add_argument("--full-grid", action="store_true", help="barrer también DriftConfig (lento)")
    ap.add_argument(
        "--camera",
        action="store_true",
        help="evalúa la versión 'cámara realista' (CAMERA_TRACKED_CLASSES, columnas cam_*) en vez de la de sensor",
    )
    ap.add_argument(
        "--cv",
        type=int,
        default=0,
        metavar="N_FOLDS",
        help="corre cross-validation por hogar (N folds) en vez del barrido simple — elige parámetros con unos "
        "hogares y mide con otros que el detector nunca vio (combinable con --camera)",
    )
    ap.add_argument(
        "--max-fp",
        type=float,
        default=0.15,
        help="FP máximo aceptable al elegir la mejor combinación (sesgo del proyecto: preferir menos FP sobre más recall)",
    )
    args = ap.parse_args()

    if not (args.dir / "day_vectors_perturbed.parquet").exists():
        raise SystemExit(f"No hay {args.dir} — correr primero `py -3 inject_synthetic_deviations.py`")

    perturbed, labels = load_synthetic(args.dir)
    n_homes = labels["hogar"].nunique()
    print(f"Cargado: {len(perturbed)} días, {len(labels)} etiquetas ({labels['es_positivo'].sum()} positivas), {n_homes} hogares")

    if args.cv:
        modo = "CÁMARA" if args.camera else "SENSOR"
        print(f"Cross-validation por hogar, {args.cv} folds, modo {modo} ...")
        fold_df, pooled = cross_validate(perturbed, labels, n_folds=args.cv, camera=args.camera, max_fp=args.max_fp)
        pd.set_option("display.width", 200)
        print()
        print(fold_df.to_string(index=False))
        print("\nParámetros elegidos por fold (deberían ser parecidos si la calibración es estable):")
        print(fold_df[["fold", "frac_historica_min", "excess_k", "dias_consecutivos_ausencia"]].to_string(index=False))
        print(f"\nOut-of-fold (pooled, todos los hogares evaluados solo con hogares que NO vieron sus parámetros):")
        print(f"  recall = {pooled['recall_out_of_fold']:.1%}   fp_rate = {pooled['fp_rate_out_of_fold']:.1%}")
        print(f"  n_pos = {pooled['n_pos']}   n_neg = {pooled['n_neg']}")
        print("  por escenario:")
        for esc, v in pooled["recall_por_escenario"].items():
            print(f"    {esc}: {v:.1%}")
        return

    if args.camera:
        print(f"Modo CÁMARA — clases trackeadas: {CAMERA_TRACKED_CLASSES}")
        results = []
        total = len(FRAC_HISTORICA_GRID) * len(EXCESS_K_GRID) * len(DIAS_CONSECUTIVOS_GRID)
        n = 0
        for frac in FRAC_HISTORICA_GRID:
            for excess_k in EXCESS_K_GRID:
                for dias_cons in DIAS_CONSECUTIVOS_GRID:
                    n += 1
                    print(f"  [{n}/{total}] frac={frac} excess_k={excess_k} dias_cons={dias_cons} ...", flush=True)
                    cfg = ClassDeviationConfig(frac_historica_min=frac, excess_k=excess_k, dias_consecutivos_ausencia=dias_cons)
                    results.append(evaluate_camera(perturbed, labels, cfg))

        df = pd.DataFrame(results).sort_values(["camara_fp_rate", "camara_recall"], ascending=[True, False])
        pd.set_option("display.width", 200)
        print()
        print(df.to_string(index=False))
        ok = df[df["camara_fp_rate"] <= args.max_fp]
        best = (ok if len(ok) else df).sort_values("camara_recall", ascending=False).iloc[0]
        print(f"\nMejor combinación modo cámara (FP <= {args.max_fp} si es posible):")
        print(best.to_string())
        out_path = args.dir / "grid_results_camera.csv"
        df.to_csv(out_path, index=False)
        print(f"\nGuardado: {out_path}")
        return

    print(f"Clases trackeadas por class_deviations: {TRACKED_CLASSES}")

    results = []
    if args.full_grid:
        total = len(WINDOW_DAYS_GRID) * len(K_GRID) * len(THRESHOLD_GRID) * len(FRAC_HISTORICA_GRID) * len(EXCESS_K_GRID) * len(DIAS_CONSECUTIVOS_GRID)
        n = 0
        for window_days in WINDOW_DAYS_GRID:
            for k in K_GRID:
                for threshold_z in THRESHOLD_GRID:
                    for frac in FRAC_HISTORICA_GRID:
                        for excess_k in EXCESS_K_GRID:
                            for dias_cons in DIAS_CONSECUTIVOS_GRID:
                                n += 1
                                print(f"  [{n}/{total}] ...", flush=True)
                                results.append(
                                    evaluate_full(
                                        perturbed,
                                        labels,
                                        DriftConfig(window_days=window_days, k=k, threshold_z=threshold_z),
                                        ClassDeviationConfig(frac_historica_min=frac, excess_k=excess_k, dias_consecutivos_ausencia=dias_cons),
                                    )
                                )
    else:
        drift_config = DriftConfig()  # ya calibrado, ver docstring
        exclude = {(row.hogar, row.fecha) for row in labels.itertuples() if row.es_positivo}
        print(f"Multivariado fijo en window_days={drift_config.window_days} k={drift_config.k} threshold_z={drift_config.threshold_z} (calculando una vez) ...")
        scored = score_days(perturbed, drift_config, exclude_from_reference=exclude)

        total = len(FRAC_HISTORICA_GRID) * len(EXCESS_K_GRID) * len(DIAS_CONSECUTIVOS_GRID)
        n = 0
        for frac in FRAC_HISTORICA_GRID:
            for excess_k in EXCESS_K_GRID:
                for dias_cons in DIAS_CONSECUTIVOS_GRID:
                    n += 1
                    print(f"  [{n}/{total}] frac={frac} excess_k={excess_k} dias_cons={dias_cons} ...", flush=True)
                    cfg = ClassDeviationConfig(frac_historica_min=frac, excess_k=excess_k, dias_consecutivos_ausencia=dias_cons)
                    r = evaluate_class_config(perturbed, labels, scored, cfg)
                    r.update({"window_days": drift_config.window_days, "k": drift_config.k, "threshold_z": drift_config.threshold_z})
                    results.append(r)

    df = pd.DataFrame(results).sort_values(["combinado_fp_rate", "combinado_recall"], ascending=[True, False])
    pd.set_option("display.width", 220)
    pd.set_option("display.max_columns", 30)
    print()
    resumen_cols = [
        "window_days", "k", "threshold_z", "frac_historica_min", "excess_k", "dias_consecutivos_ausencia",
        "n_pos", "n_neg",
        "multivariado_recall", "multivariado_fp_rate",
        "por_clase_recall", "por_clase_fp_rate",
        "combinado_recall", "combinado_fp_rate",
    ]
    print(df[resumen_cols].to_string(index=False))

    ok = df[df["combinado_fp_rate"] <= args.max_fp]
    best = (ok if len(ok) else df).sort_values("combinado_recall", ascending=False).iloc[0]
    print(f"\nMejor combinación por score combinado (FP <= {args.max_fp} si es posible):")
    print(best.to_string())

    out_path = args.dir / "grid_results.csv"
    df.to_csv(out_path, index=False)
    print(f"\nGuardado: {out_path}")


if __name__ == "__main__":
    main()
