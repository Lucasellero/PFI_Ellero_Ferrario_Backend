"""Mide recall/FP de `src/trend_deviations.py` contra el set de tendencias sintéticas
(`inject_gradual_trend.py`) — mismo framing recall/FP que el resto de `baseline/`.

Uso:
  py -3 inject_gradual_trend.py       # generar el set primero (una vez)
  py -3 evaluate_trend_deviations.py
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from src.trend_deviations import DEFAULT_TRACKED_CLASSES, TrendConfig, score_trend_deviations

DEFAULT_DIR = Path(__file__).resolve().parent / "synthetic_eval"

N_RECIENTES_GRID = (2, 3, 4)
N_BASE_GRID = (4, 6, 8)
THRESHOLD_GRID = (2.0, 2.5, 3.0, 3.5)


def load_synthetic(out_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame, set]:
    perturbed = pd.read_parquet(out_dir / "day_vectors_trend_perturbed.parquet")
    labels = pd.read_parquet(out_dir / "labels_trend.parquet")
    excl_df = pd.read_parquet(out_dir / "excluded_days_trend.parquet")
    excluidos = set(zip(excl_df["hogar"], excl_df["fecha"]))
    return perturbed, labels, excluidos


def evaluate(perturbed: pd.DataFrame, labels: pd.DataFrame, excluidos: set, config: TrendConfig) -> dict:
    scored = score_trend_deviations(
        perturbed, tracked_classes=DEFAULT_TRACKED_CLASSES, config=config, exclude_from_reference=excluidos
    )
    hits = set(zip(scored["hogar"], scored["semana_fin"], scored["clase"])) if len(scored) else set()

    labels = labels.copy()
    labels["hit"] = [
        (h, f, c) in hits for h, f, c in zip(labels["hogar"], labels["semana_fin"], labels["clase"])
    ]
    pos = labels[labels["es_positivo"]]
    neg = labels[~labels["es_positivo"]]
    return {
        "n_semanas_recientes": config.n_semanas_recientes,
        "n_semanas_base": config.n_semanas_base,
        "threshold_z": config.threshold_z,
        "n_pos": len(pos),
        "n_neg": len(neg),
        "recall": float(pos["hit"].mean()) if len(pos) else float("nan"),
        "fp_rate": float(neg["hit"].mean()) if len(neg) else float("nan"),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dir", type=Path, default=DEFAULT_DIR)
    ap.add_argument("--max-fp", type=float, default=0.15)
    args = ap.parse_args()

    if not (args.dir / "day_vectors_trend_perturbed.parquet").exists():
        raise SystemExit(f"No hay {args.dir} — correr primero `py -3 inject_gradual_trend.py`")

    perturbed, labels, excluidos = load_synthetic(args.dir)
    print(f"Cargado: {len(perturbed)} días, {len(labels)} etiquetas ({labels['es_positivo'].sum()} positivas)")

    # nota: el set inyectado se armó con TrendConfig() default (n_recientes=3, n_base=6) — barrer
    # otras combinaciones acá mide si el DETECTOR generaliza a otra ventana sobre el MISMO bloque
    # perturbado, no una recalibración honesta del tamaño de bloque en sí (eso requeriría volver a
    # inyectar por cada combinación). Sirve para elegir threshold_z; window sizes se dejan fijos.
    results = []
    for n_recientes in N_RECIENTES_GRID:
        for n_base in N_BASE_GRID:
            for threshold_z in THRESHOLD_GRID:
                cfg = TrendConfig(n_semanas_recientes=n_recientes, n_semanas_base=n_base, threshold_z=threshold_z)
                results.append(evaluate(perturbed, labels, excluidos, cfg))

    df = pd.DataFrame(results).sort_values(["fp_rate", "recall"], ascending=[True, False])
    pd.set_option("display.width", 160)
    print(df.to_string(index=False))

    ok = df[df["fp_rate"] <= args.max_fp]
    best = (ok if len(ok) else df).sort_values("recall", ascending=False).iloc[0]
    print(f"\nMejor combinación (FP <= {args.max_fp} si es posible):")
    print(best.to_string())

    out_path = args.dir / "grid_results_trend.csv"
    df.to_csv(out_path, index=False)
    print(f"\nGuardado: {out_path}")


if __name__ == "__main__":
    main()
