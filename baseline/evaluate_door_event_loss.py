"""Análisis de sensibilidad: qué pasa con la detección en modo cámara si la cámara pierde una
parte de las entradas y salidas de casa.

`evaluate_classifier_noise.py` no les aplica error a `salir_casa`/`entrar_casa` porque no existe
un detector de puerta medido. Acá no se usa un error medido sino tres niveles supuestos: cada
evento de `salir_casa` y de `entrar_casa` se pierde, de forma independiente, con probabilidad
10 %, 20 % o 30 %. El resto del vector de día queda limpio y el protocolo es el de siempre: mismo
set de desvíos sintéticos, misma validación cruzada por hogar en modo cámara (`cv_grid.py`,
barrido por defecto de `evaluate_deviations.py --cv 5`).

La pérdida se aplica sobre el vector de día ya inyectado, clase por clase y día por día
(`n` eventos quedan en Binomial(n, 1 - p)): es lo mismo que sortear evento por evento, porque la
inyección nunca deja un día a medias en estas dos clases (o no lo toca, o lo deja en cero).
Solo modela eventos perdidos, no entradas o salidas inventadas.

Uso:
  py -3 inject_synthetic_deviations.py          # generar el set primero (una vez)
  py -3 evaluate_door_event_loss.py             # 10, 20 y 30 %, 3 semillas cada uno
  py -3 evaluate_door_event_loss.py --loss 0.1 0.5 --seeds 0 1 2 3 4
"""
from __future__ import annotations

import os

os.environ.setdefault("OMP_NUM_THREADS", "1")

import argparse
import json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

from cv_grid import cached_grid, cross_validate_from_grid, evaluate_grid
from evaluate_deviations import DEFAULT_DIR, load_synthetic
from src.features import add_camera_columns

DOOR_CLASSES = ("salir_casa", "entrar_casa")


def drop_door_events(perturbed: pd.DataFrame, homes: list[str], loss: float, seed: int) -> tuple[pd.DataFrame, dict]:
    """Copia de `perturbed` donde cada evento de `DOOR_CLASSES` de los hogares `homes` se perdió
    con probabilidad `loss`, con las columnas `cam_*` recalculadas. Devuelve además cuántos
    eventos había y cuántos quedaron."""
    rng = np.random.default_rng(seed)
    out = perturbed.copy()
    filas = out.index.get_level_values("hogar").isin(homes)
    resumen = {}
    for cls in DOOR_CLASSES:
        antes = out.loc[filas, f"{cls}_n"].to_numpy(dtype=int)
        despues = rng.binomial(antes, 1.0 - loss)
        proporcion = np.divide(despues, antes, out=np.ones(len(antes)), where=antes > 0)
        out.loc[filas, f"{cls}_n"] = despues.astype(out[f"{cls}_n"].dtype)
        out.loc[filas, f"{cls}_min"] = out.loc[filas, f"{cls}_min"].to_numpy() * proporcion
        resumen[cls] = {
            "eventos": int(antes.sum()), "quedan": int(despues.sum()),
            "dias_con_eventos": int((antes > 0).sum()), "dias_que_quedan_en_cero": int(((antes > 0) & (despues == 0)).sum()),
        }
    return add_camera_columns(out), resumen


def _run(job: tuple[float, int | None, str]) -> tuple[float, int | None, dict]:
    loss, seed, out_dir = job

    def _compute() -> dict:
        perturbed, labels = load_synthetic(Path(out_dir))
        perdida = None
        if loss > 0:
            perturbed, perdida = drop_door_events(perturbed, sorted(labels["hogar"].unique()), loss, seed)
        grid = evaluate_grid(perturbed, labels, camera=True)
        grid["perdida"] = perdida
        return grid

    sufijo = "" if loss == 0 else f"_puerta{round(loss * 100)}_seed{seed}"
    return loss, seed, cached_grid(Path(out_dir) / f"cv_grid_camara_mwd7{sufijo}.pkl", _compute)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dir", type=Path, default=DEFAULT_DIR)
    ap.add_argument("--loss", type=float, nargs="+", default=[0.1, 0.2, 0.3], help="fracción de eventos de puerta perdidos")
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--cv", type=int, default=5, metavar="N_FOLDS")
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()

    if not (args.dir / "day_vectors_perturbed.parquet").exists():
        raise SystemExit(f"No hay {args.dir} — correr primero `py -3 inject_synthetic_deviations.py`")

    jobs = [(0.0, None, str(args.dir))] + [(p, s, str(args.dir)) for p in args.loss for s in args.seeds]
    print(f"Modo cámara, pérdida de eventos de puerta {args.loss}, semillas {args.seeds} ...", flush=True)
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        grids = list(pool.map(_run, jobs))

    results, filas = [], []
    for loss, seed, grid in grids:
        fold_df, p = cross_validate_from_grid(grid, n_folds=args.cv)
        results.append({"perdida": loss, "seed": seed, "eventos": grid.get("perdida"), "pooled": p, "folds": fold_df.to_dict(orient="records")})
        n = p["n_por_escenario"]["semana_sin_salir"]
        filas.append({
            "perdida_%": round(loss * 100), "seed": "-" if seed is None else seed,
            "semana_sin_salir_%": round(100 * p["recall_por_escenario"]["semana_sin_salir"], 1),
            "semana_sin_salir": f"{round(p['recall_por_escenario']['semana_sin_salir'] * n)}/{n}",
            "recall_%": round(100 * p["recall_out_of_fold"], 1), "detectados": f"{p['detectados']}/{p['n_pos']}",
            "fp_%": round(100 * p["fp_rate_out_of_fold"], 1), "fp": f"{p['falsos_positivos']}/{p['n_neg']}",
            **{esc: round(100 * v, 1) for esc, v in sorted(p["recall_por_escenario"].items()) if esc != "semana_sin_salir"},
        })
    pd.set_option("display.width", 240)
    print()
    print(pd.DataFrame(filas).to_string(index=False))
    print("\nEventos de puerta (por nivel y semilla):")
    for loss, seed, grid in grids:
        if grid.get("perdida"):
            print(f"  {round(loss * 100)} %, seed {seed}: {grid['perdida']}")

    out_path = args.dir / "door_event_loss_results.json"
    out_path.write_text(json.dumps(results, indent=2, default=str), encoding="utf-8")
    print(f"\nGuardado: {out_path}")


if __name__ == "__main__":
    main()
