"""Mide cuántos días de historial hacen falta: repite la validación cruzada por hogar variando el
período de aprendizaje (`min_window_days`), en modo sensor y en modo cámara. Ver "Actualización
2026-10-13" en README.md.

Equivale a correr, para cada valor y cada modo,
`py -3 evaluate_deviations.py --cv 5 [--camera] --min-window-days N`; acá se calcula con
`cv_grid.py` (mismo resultado, sin repetir el barrido en cada partición) y se informa además el
barrido con candidatos binomiales (`--binomial`), que no es el de los números citados.

Uso:
  py -3 inject_synthetic_deviations.py          # generar el set primero (una vez)
  py -3 evaluate_learning_period.py             # 7, 14, 21 y 30 días, los dos modos, en paralelo
  py -3 evaluate_learning_period.py --days 7 10 14 --workers 6
"""
from __future__ import annotations

import os

os.environ.setdefault("OMP_NUM_THREADS", "1")  # KMeans de a un hilo: mucho más rápido en ventanas chicas

import argparse
import json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import pandas as pd

from cv_grid import CRITERIOS, cached_grid, cross_validate_from_grid, evaluate_grid
from evaluate_deviations import DEFAULT_DIR, load_synthetic


def _run(job: tuple[int, bool, str]) -> tuple[int, str, dict]:
    min_window_days, camera, out_dir = job
    modo = "camara" if camera else "sensor"

    def _compute() -> dict:
        perturbed, labels = load_synthetic(Path(out_dir))
        return evaluate_grid(perturbed, labels, camera=camera, min_window_days=min_window_days)

    return min_window_days, modo, cached_grid(Path(out_dir) / f"cv_grid_{modo}_mwd{min_window_days}.pkl", _compute)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dir", type=Path, default=DEFAULT_DIR)
    ap.add_argument("--days", type=int, nargs="+", default=[7, 14, 21, 30], help="valores de min_window_days")
    ap.add_argument("--cv", type=int, default=5, metavar="N_FOLDS")
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()

    if not (args.dir / "day_vectors_perturbed.parquet").exists():
        raise SystemExit(f"No hay {args.dir} — correr primero `py -3 inject_synthetic_deviations.py`")

    jobs = [(d, camera, str(args.dir)) for camera in (False, True) for d in args.days]
    print(f"min_window_days en {args.days}, modos sensor y cámara (barridos guardados en {args.dir}/cv_grid_*.pkl) ...", flush=True)
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        grids = list(pool.map(_run, jobs))

    results = []
    pd.set_option("display.width", 240)
    for criterio, opciones in CRITERIOS.items():
        filas = []
        for min_window_days, modo, grid in grids:
            fold_df, p = cross_validate_from_grid(grid, n_folds=args.cv, **opciones)
            results.append({
                "criterio": criterio, "modo": modo, "min_window_days": min_window_days,
                "pooled": p, "folds": fold_df.to_dict(orient="records"),
            })
            filas.append({
                "modo": modo, "min_window_days": min_window_days,
                "recall_%": round(100 * p["recall_out_of_fold"], 1), "fp_%": round(100 * p["fp_rate_out_of_fold"], 1),
                "detectados": f"{p['detectados']}/{p['n_pos']}", "fp": f"{p['falsos_positivos']}/{p['n_neg']}",
                "sin_evaluar": p["sin_evaluar"],
                **{esc: f"{100 * v:.1f} ({p['n_por_escenario'][esc]})" for esc, v in sorted(p["recall_por_escenario"].items())},
            })
        print(f"\nCriterio de selección: {criterio}  (validación cruzada por hogar, {args.cv} particiones, out-of-fold; entre paréntesis, días evaluados)")
        print(pd.DataFrame(filas).to_string(index=False))

    out_path = args.dir / "learning_period_results.json"
    out_path.write_text(json.dumps(results, indent=2, default=str), encoding="utf-8")
    print(f"\nGuardado: {out_path}")


if __name__ == "__main__":
    main()
