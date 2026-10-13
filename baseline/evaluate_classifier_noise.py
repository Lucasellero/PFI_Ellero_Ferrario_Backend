"""Mide cuánto se degrada la detección de desvíos en modo cámara cuando las etiquetas de actividad
traen el error real del clasificador de la cámara, en vez de las etiquetas limpias de CASAS.

Todo lo demás es el protocolo ya validado: mismo set de desvíos sintéticos
(`inject_synthetic_deviations.py`), misma validación cruzada por hogar
(la de `evaluate_deviations.cross_validate(camera=True)`, calculada con `cv_grid.py`). Lo único
que cambia es que, antes de armar las columnas `cam_*` del vector de día, cada evento de CASAS
que la cámara podría clasificar se re-etiqueta al azar según la matriz de confusión medida del
clasificador (`vlm/README.md`, "Matriz de confusión de esta corrida", 2026-09-20, 6 categorías).

Cómo se contamina (ver "Actualización 2026-10-13" en README.md para el detalle y los supuestos):
  1. Cada clase canónica de CASAS se asocia a la categoría del clasificador que la cámara vería
     (`CANONICAL_TO_CATEGORY`). Las que no tienen equivalente quedan sin ruido.
  2. Por cada evento con equivalente se sortea la categoría predicha con la fila de la matriz que
     corresponde a su categoría real.
  3. Si la predicción acierta, el evento queda en su clase de cámara de siempre
     (`activity_map.COARSE`). Si falla, pasa a la clase de cámara de la categoría predicha
     (`camera_events.CAMERA_ACTIVITY_MAP`, el mismo mapeo que usa producción).
  4. El ruido se aplica sobre los eventos que quedan después de inyectar el desvío: un día sin
     ninguna comida real puede igual mostrar una "comida" si el clasificador confunde otra
     actividad de ese día con comer.

Uso:
  py -3 inject_synthetic_deviations.py          # generar el set primero (una vez)
  py -3 evaluate_classifier_noise.py            # sin ruido + 3 semillas de ruido, en paralelo
  py -3 evaluate_classifier_noise.py --seeds 0 1 2 3 4 --workers 6
"""
from __future__ import annotations

import argparse
import json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

from cv_grid import CRITERIOS, cached_grid, cross_validate_from_grid, evaluate_grid
from evaluate_deviations import DEFAULT_DIR, load_synthetic
from inject_synthetic_deviations import EVENTS_PARQUET
from src.camera_events import CAMERA_ACTIVITY_MAP
from src.features import COARSE, add_camera_columns

# Orden de filas y columnas de la matriz (real \ predicho), tal como está en vlm/README.md.
CATEGORIES = ("Cooking", "EatingDrinkingMedication", "Moving", "Phone", "ReadingScreen", "Resting")

# Porcentajes publicados, redondeados a entero: cada fila se normaliza para que sume 1 (cuatro
# de las seis suman 101 por el redondeo).
CONFUSION_PCT = np.array([
    [80, 11, 5, 4, 1, 0],    # Cooking
    [11, 57, 8, 8, 8, 9],    # EatingDrinkingMedication
    [5, 4, 71, 9, 6, 6],     # Moving
    [3, 15, 11, 59, 9, 4],   # Phone
    [1, 9, 10, 3, 64, 13],   # ReadingScreen
    [0, 9, 9, 6, 9, 67],     # Resting
], dtype=float)
CONFUSION = CONFUSION_PCT / CONFUSION_PCT.sum(axis=1, keepdims=True)

# Clase canónica de CASAS -> categoría del clasificador que le correspondería (según las clases
# finas de cada categoría en vlm/src/clip_classifier.py::COARSE_CATEGORIES). Las clases canónicas
# que no figuran acá no tienen equivalente en el clasificador y quedan sin ruido.
CANONICAL_TO_CATEGORY: dict[str, str] = {
    "cocinar": "Cooking",
    "lavar_platos": "Cooking",                 # Cook.Cleandishes
    "comer": "EatingDrinkingMedication",       # Eat.Attable / Eat.Snack
    "beber": "EatingDrinkingMedication",       # Drink.*
    "medicacion": "EatingDrinkingMedication",  # Takepills
    "descansar": "Resting",
    "ver_tv": "Resting",                       # WatchTV está dentro de Resting
    "leer": "ReadingScreen",                   # Readbook
    "trabajo": "ReadingScreen",                # Uselaptop / Usetablet
    "telefono": "Phone",                       # Usetelephone
}


def load_events(events_parquet: Path, homes: list[str]) -> pd.DataFrame:
    df = pd.read_parquet(events_parquet, columns=["hogar", "actividad", "fecha", "inicio"])
    df = df.assign(hogar=df["hogar"].astype(str), actividad=df["actividad"].astype(str))
    df = df[df["hogar"].isin(homes)]
    return df.sort_values(["hogar", "inicio", "actividad"], kind="stable").reset_index(drop=True)


def surviving_events(events: pd.DataFrame, perturbed: pd.DataFrame) -> pd.DataFrame:
    """Los eventos con equivalente en el clasificador que siguen existiendo después de inyectar
    los desvíos. La inyección trabaja sobre el vector de día (pone `comer_n` en 0, o lo baja);
    acá se lleva eso mismo a nivel de evento: de cada `(hogar, fecha, clase)` se conservan los
    primeros `<clase>_n` eventos del día. Verifica que en ningún día queden menos eventos que los
    que dice el vector (si pasara, el vector y el parquet de eventos no serían del mismo ETL)."""
    ev = events[events["actividad"].isin(CANONICAL_TO_CATEGORY)].copy()
    ev["orden"] = ev.groupby(["hogar", "fecha", "actividad"]).cumcount()

    n_cols = [f"{c}_n" for c in CANONICAL_TO_CATEGORY]
    en_eventos = perturbed.index.get_level_values("hogar").isin(events["hogar"].unique())
    objetivo = perturbed.loc[en_eventos, n_cols].rename(columns=lambda c: c[: -len("_n")])
    objetivo.columns.name = "actividad"
    objetivo = objetivo.stack().rename("n_objetivo").reset_index()

    ev = ev.merge(objetivo, on=["hogar", "fecha", "actividad"], how="left", validate="many_to_one")
    assert ev["n_objetivo"].notna().all(), "hay eventos de un día que no está en el vector de día"
    ev = ev[ev["orden"] < ev["n_objetivo"]]

    quedan = ev.groupby(["hogar", "fecha", "actividad"]).size().rename("n")
    comparado = objetivo.merge(quedan.reset_index(), on=["hogar", "fecha", "actividad"], how="left").fillna({"n": 0})
    assert (comparado["n"] == comparado["n_objetivo"]).all(), "el vector de día no coincide con los eventos"
    return ev.drop(columns=["orden", "n_objetivo"])


def contaminate(perturbed: pd.DataFrame, events: pd.DataFrame, seed: int) -> tuple[pd.DataFrame, dict]:
    """Devuelve una copia de `perturbed` con las columnas `cam_*` recalculadas bajo ruido del
    clasificador, y un resumen de cuánto se contaminó. Las columnas canónicas no se tocan. Solo
    se contaminan los hogares presentes en `events` (los que tienen etiquetas de evaluación)."""
    rng = np.random.default_rng(seed)
    ev = surviving_events(events, perturbed)

    real = ev["actividad"].map(CANONICAL_TO_CATEGORY).to_numpy()
    predicha = np.empty(len(ev), dtype=object)
    for i, categoria in enumerate(CATEGORIES):
        mascara = real == categoria
        predicha[mascara] = rng.choice(CATEGORIES, size=int(mascara.sum()), p=CONFUSION[i])
    ev["clase_limpia"] = ev["actividad"].map(COARSE)
    ev["clase_ruido"] = np.where(real == predicha, ev["clase_limpia"], pd.Series(predicha).map(CAMERA_ACTIVITY_MAP).to_numpy())

    out = add_camera_columns(perturbed)
    out["cam_moverse_n"] = 0
    out["cam_moverse_min"] = 0.0

    # Minutos de un evento movido: el promedio por evento de su clase ese día (el vector de día
    # ya perturbado es la referencia; la evaluación en modo cámara solo usa los conteos `_n`).
    movidos = ev[ev["clase_limpia"] != ev["clase_ruido"]]
    por_dia_clase = movidos.groupby(["hogar", "fecha", "actividad", "clase_limpia", "clase_ruido"]).size().rename("n").reset_index()
    for fila in por_dia_clase.itertuples():
        clave = (fila.hogar, fila.fecha)
        minutos = fila.n * perturbed.at[clave, f"{fila.actividad}_min"] / perturbed.at[clave, f"{fila.actividad}_n"]
        out.at[clave, f"cam_{fila.clase_limpia}_n"] -= fila.n
        out.at[clave, f"cam_{fila.clase_limpia}_min"] -= minutos
        out.at[clave, f"cam_{fila.clase_ruido}_n"] += fila.n
        out.at[clave, f"cam_{fila.clase_ruido}_min"] += minutos

    n_cam = [c for c in out.columns if c.startswith("cam_") and c.endswith("_n")]
    assert (out[n_cam] >= 0).all().all()

    total_eventos = int(events.shape[0])
    resumen = {
        "eventos_totales": total_eventos,
        "eventos_con_equivalente": int(len(ev)),
        "eventos_reetiquetados": int(len(movidos)),
        "frac_con_equivalente": len(ev) / total_eventos,
        "frac_reetiquetados_sobre_equivalentes": len(movidos) / len(ev),
        "falsas_comidas_agregadas": int((movidos["clase_ruido"] == "comida").sum()),
        "comidas_reales_perdidas": int(movidos["clase_limpia"].isin(["comida", "comida_mesa"]).sum()),
    }
    return out, resumen


def _run(job: tuple[str, int | None, str, str]) -> tuple[str, int | None, dict]:
    nombre, seed, out_dir, events_parquet = job

    def _compute() -> dict:
        perturbed, labels = load_synthetic(Path(out_dir))
        resumen_ruido = None
        if seed is not None:
            events = load_events(Path(events_parquet), sorted(labels["hogar"].unique()))
            perturbed, resumen_ruido = contaminate(perturbed, events, seed)
        grid = evaluate_grid(perturbed, labels, camera=True)
        grid["ruido"] = resumen_ruido
        return grid

    sufijo = "" if seed is None else f"_ruido_seed{seed}"
    return nombre, seed, cached_grid(Path(out_dir) / f"cv_grid_camara_mwd7{sufijo}.pkl", _compute)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dir", type=Path, default=DEFAULT_DIR)
    ap.add_argument("--events-parquet", type=Path, default=EVENTS_PARQUET)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2], help="semillas del ruido")
    ap.add_argument("--cv", type=int, default=5, metavar="N_FOLDS")
    ap.add_argument("--workers", type=int, default=4)
    args = ap.parse_args()

    if not (args.dir / "day_vectors_perturbed.parquet").exists():
        raise SystemExit(f"No hay {args.dir} — correr primero `py -3 inject_synthetic_deviations.py`")

    jobs = [("sin_ruido", None, str(args.dir), str(args.events_parquet))]
    jobs += [(f"ruido_seed{s}", s, str(args.dir), str(args.events_parquet)) for s in args.seeds]
    print(f"Modo cámara: sin ruido + {len(args.seeds)} semillas de ruido (barridos guardados en {args.dir}/cv_grid_*.pkl) ...", flush=True)
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        grids = list(pool.map(_run, jobs))

    results = []
    pd.set_option("display.width", 220)
    for criterio, opciones in CRITERIOS.items():
        filas = []
        for nombre, seed, grid in grids:
            fold_df, pooled = cross_validate_from_grid(grid, n_folds=args.cv, **opciones)
            results.append({
                "criterio": criterio, "nombre": nombre, "seed": seed, "ruido": grid["ruido"],
                "pooled": pooled, "folds": fold_df.to_dict(orient="records"),
            })
            filas.append({
                "corrida": nombre, "recall": pooled["recall_out_of_fold"], "fp_rate": pooled["fp_rate_out_of_fold"],
                **dict(sorted(pooled["recall_por_escenario"].items())),
            })
        tabla = pd.DataFrame(filas).set_index("corrida") * 100
        print(f"\nCriterio de selección: {criterio}  (validación cruzada por hogar, {args.cv} particiones, % out-of-fold)")
        print(tabla.round(1).to_string())
        con_ruido = tabla[tabla.index != "sin_ruido"]
        print("  con ruido, mín – máx entre semillas:  " + "   ".join(
            f"{col} {con_ruido[col].min():.1f}–{con_ruido[col].max():.1f}" for col in con_ruido.columns))

    pooled = results[0]["pooled"]
    print(f"\nn_pos = {pooled['n_pos']}   n_neg = {pooled['n_neg']}   por escenario: {pooled['n_por_escenario']}")
    print("\nCuánto se contaminó (por semilla):")
    for _, seed, grid in grids:
        if grid["ruido"]:
            print(f"  seed {seed}: {grid['ruido']}")

    out_path = args.dir / "classifier_noise_results.json"
    out_path.write_text(json.dumps(results, indent=2, default=str), encoding="utf-8")
    print(f"\nGuardado: {out_path}")


if __name__ == "__main__":
    main()
