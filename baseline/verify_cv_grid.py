"""Comprueba que `cv_grid.cross_validate_from_grid` da lo mismo que
`evaluate_deviations.cross_validate`, partición por partición, en modo sensor y en modo cámara.

Para que tarde un par de minutos y no una hora usa un barrido reducido, con el barrido por defecto
(umbral) y con `binomial=True`; el camino de código es el mismo que con el barrido completo.

Con `--completo` corre además `cross_validate` con el barrido completo por defecto, en los dos
modos, y exige que dé los números citados: 72,8 %/13,1 % (sensor) y 69,5 %/13,8 % (cámara).

Uso:
  py -3 verify_cv_grid.py
  py -3 verify_cv_grid.py --completo
"""
from __future__ import annotations

import argparse
import os
from concurrent.futures import ProcessPoolExecutor

os.environ.setdefault("OMP_NUM_THREADS", "1")  # KMeans de a un hilo: mucho más rápido en ventanas chicas

import evaluate_deviations as ed
from cv_grid import cross_validate_from_grid, evaluate_grid

# Números citados (2026-09-20), set de desvíos con semilla 0: (detectados, positivos, falsos
# positivos, negativos) -> 72,8 %/13,1 % y 69,5 %/13,8 %.
REFERENCIA = {"sensor": (158, 217, 49, 373), "camara": (153, 220, 58, 420)}

COLUMNAS = [
    "fold", "ausencia_metodo", "frac_historica_min", "ausencia_alpha", "excess_k", "dias_consecutivos_ausencia",
    "separar_fin_de_semana", "dias_consecutivos_exceso_deficit", "ausencia_requiere_racha",
    "recall_test", "fp_rate_test", "n_pos_test", "n_neg_test",
]


def main() -> None:
    ed.FRAC_HISTORICA_GRID = (0.6, 0.8)
    ed.EXCESS_K_GRID = (3.0, 4.0)
    ed.DIAS_CONSECUTIVOS_GRID = (2,)
    ed.SEPARAR_FINDE_GRID = (True, False)
    ed.DIAS_CONSECUTIVOS_EXCESO_DEFICIT_GRID = (1,)
    ed.AUSENCIA_REQUIERE_RACHA_GRID = (False,)
    ed.AUSENCIA_ALPHA_GRID = (0.05,)

    perturbed, labels = ed.load_synthetic(ed.DEFAULT_DIR)
    todo_igual = True
    casos = [(camera, mwd, binomial) for binomial in (False, True) for camera, mwd in ((True, None), (False, None), (True, 21), (False, 21))]
    for camera, min_window_days, binomial in casos:
        folds_a, pooled_a = ed.cross_validate(perturbed, labels, camera=camera, min_window_days=min_window_days, binomial=binomial)
        grid = evaluate_grid(perturbed, labels, camera=camera, min_window_days=min_window_days)
        folds_b, pooled_b = cross_validate_from_grid(grid, binomial=binomial)
        igual = (
            folds_a[COLUMNAS].to_dict(orient="records") == folds_b[COLUMNAS].to_dict(orient="records")
            and all(pooled_a[k] == pooled_b[k] for k in pooled_a)
        )
        todo_igual &= igual
        print(
            f"{'cámara' if camera else 'sensor'}, min_window_days={min_window_days or 7}, "
            f"{'umbral + binomial' if binomial else 'umbral (default)'}: "
            f"cross_validate {pooled_a['recall_out_of_fold']:.4f}/{pooled_a['fp_rate_out_of_fold']:.4f}  "
            f"cv_grid {pooled_b['recall_out_of_fold']:.4f}/{pooled_b['fp_rate_out_of_fold']:.4f}  "
            f"{'IGUAL' if igual else 'DISTINTO'}",
            flush=True,
        )
    if not todo_igual:
        raise SystemExit("cv_grid NO reproduce cross_validate")
    print(f"cv_grid reproduce cross_validate en los {len(casos)} casos.")


def _referencia_un_modo(camera: bool) -> tuple[bool, dict]:
    perturbed, labels = ed.load_synthetic(ed.DEFAULT_DIR)
    return camera, ed.cross_validate(perturbed, labels, camera=camera)[1]


def referencia() -> None:
    """Corre `cross_validate` con el barrido completo por defecto, en los dos modos (en paralelo),
    y exige los números citados."""
    ok = True
    with ProcessPoolExecutor(max_workers=2) as pool:
        for camera, p in pool.map(_referencia_un_modo, (False, True)):
            esperado = REFERENCIA["camara" if camera else "sensor"]
            pos = round(p["recall_out_of_fold"] * p["n_pos"])
            neg = round(p["fp_rate_out_of_fold"] * p["n_neg"])
            obtenido = (pos, p["n_pos"], neg, p["n_neg"])
            ok &= obtenido == esperado
            print(
                f"{'cámara' if camera else 'sensor'}, --cv 5 por defecto: {p['recall_out_of_fold']:.1%} / "
                f"{p['fp_rate_out_of_fold']:.1%}  ({pos} de {p['n_pos']}, {neg} de {p['n_neg']})  "
                f"{'COINCIDE' if obtenido == esperado else f'ESPERADO {esperado}'}",
                flush=True,
            )
    if not ok:
        raise SystemExit("el barrido por defecto NO da los números citados")
    print("El barrido por defecto da 72,8 %/13,1 % (sensor) y 69,5 %/13,8 % (cámara).", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument(
        "--completo", action="store_true",
        help="además, correr el barrido completo por defecto y exigir los números citados (unos 20 minutos)",
    )
    args = ap.parse_args()
    if args.completo:
        referencia()  # antes de `main()`, que achica el barrido
    main()
