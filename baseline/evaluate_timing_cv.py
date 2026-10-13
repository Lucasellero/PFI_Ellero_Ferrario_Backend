"""Validación cruzada por hogar del detector de horario (`src/timing_deviations.py`): mide si
acotar las clases que sigue y calibrarlo por clase baja los falsos positivos sin perder detección
frente a la configuración actual (todas las clases, un solo `TimingConfig()`).

Mismo banco de pruebas que `evaluate_timing_deviations.py` (a 3 eventos reales por hogar y clase
se les corre el horario 3 horas; 8 eventos sin tocar son los negativos), con tres diferencias:
todos los hogares de CASAS con historial suficiente en vez de 7, tres semillas de inyección
juntas en vez de una, y los parámetros se eligen con unos hogares y se miden con otros.

Dos vocabularios:
  - `camara`: las clases que el proceso diario sigue en producción (`cam_cocina`, `cam_comida`,
    `cam_descanso_sofa`, `cam_ocio_sentado`), armadas desde CASAS con el mismo mapeo que
    `evaluate_classifier_noise.py`. `cam_moverse` no tiene equivalente en CASAS: no se puede medir.
  - `sensor`: `DEFAULT_TRACKED_CLASSES` (clases canónicas de CASAS).

Variantes que se comparan, todas medidas sobre las mismas etiquetas:
  - `actual`: todas las clases con `TimingConfig()`. No elige nada, no necesita particiones.
  - `calibrado`: todas las clases, cada una con los parámetros elegidos en los hogares de
    entrenamiento (mayor detección con falsos positivos <= `MAX_FP_CLASE`; si ninguno llega, el
    de menos falsos positivos).
  - `acotado`: como `calibrado`, pero una clase solo se sigue si en entrenamiento llega a
    falsos positivos <= `MAX_FP_CLASE` con detección >= `MIN_RECALL_CLASE`. Las demás se dejan de
    seguir y sus desvíos cuentan como no detectados.
  - `misma_deteccion`: todas las clases, cada una con los parámetros de menos falsos positivos
    entre los que en entrenamiento detectan al menos lo mismo que `TimingConfig()`. Es la prueba
    directa de "bajar falsos positivos sin perder detección".

Además de la tasa por etiqueta se mide la tasa de "falsa alarma por día": sobre días reales sin
tocar, cuántos reciben al menos una alerta de horario de cualquiera de las clases seguidas. Es lo
que ve un familiar.

Uso:
  py -3 evaluate_timing_cv.py
  py -3 evaluate_timing_cv.py --vocab camara --workers 6
"""
from __future__ import annotations

import os

os.environ.setdefault("OMP_NUM_THREADS", "1")  # KMeans de a un hilo: mucho más rápido en ventanas chicas

import argparse
import json
import pickle
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

from evaluate_classifier_noise import CANONICAL_TO_CATEGORY
from evaluate_deviations import DEFAULT_DIR, load_synthetic
from evaluate_timing_deviations import EVENTS_PARQUET, N_NEGATIVOS_POR_CLASE_HOGAR, N_POR_CLASE_HOGAR, OFFSET_MIN_MINUTOS
from src.camera_events import CAMERA_ACTIVITY_MAP
from src.timing_deviations import DEFAULT_TRACKED_CLASSES, TimingConfig, timing_statistics

WINDOW_EVENTS_GRID = (40, 60, 90)
K_TIME_SLOTS_GRID = (1, 2, 3)
EXCESS_K_GRID = (2.0, 2.5, 3.0, 3.5, 4.0, 5.0)
SEEDS = (0, 1, 2)
N_DIAS_CONTROL_POR_HOGAR = 5

# Criterio para seguir una clase y para elegir sus parámetros. Fijado antes de medir, con el
# sesgo del proyecto: mejor callar que avisar mal.
MAX_FP_CLASE = 0.05
MIN_RECALL_CLASE = 0.30

CAMERA_CLASSES = ("cam_cocina", "cam_comida", "cam_descanso_sofa", "cam_ocio_sentado")
VOCABULARIOS = {"camara": CAMERA_CLASSES, "sensor": DEFAULT_TRACKED_CLASSES}


def load_events(vocab: str, homes: list[str]) -> pd.DataFrame:
    ev = pd.read_parquet(EVENTS_PARQUET, columns=["hogar", "actividad", "fecha", "inicio", "minuto_del_dia"])
    ev = ev.assign(hogar=ev["hogar"].astype(str), actividad=ev["actividad"].astype(str))
    ev = ev[ev["hogar"].isin(homes)]
    if vocab == "camara":
        clase = ev["actividad"].map(CANONICAL_TO_CATEGORY).map(CAMERA_ACTIVITY_MAP)
        ev = ev.assign(actividad="cam_" + clase)[clase.notna()]
    ev = ev[ev["actividad"].isin(VOCABULARIOS[vocab])]
    return ev.sort_values(["hogar", "inicio", "actividad"], kind="stable").reset_index(drop=True)


def inject(events: pd.DataFrame, seed: int) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Misma inyección que `evaluate_timing_deviations.inject`. Devuelve los eventos perturbados,
    las etiquetas por `(hogar, fecha, actividad)` y los días de control (reales, sin ningún evento
    corrido en ese hogar ese día)."""
    rng = np.random.default_rng(seed)
    perturbed = events.copy()
    labels: list[dict] = []
    for (hogar, actividad), g in events.groupby(["hogar", "actividad"], observed=True):
        if len(g) < 30:
            continue
        idx_pool = g.index.to_numpy().copy()
        rng.shuffle(idx_pool)
        pos_idx = idx_pool[:N_POR_CLASE_HOGAR]
        neg_idx = idx_pool[N_POR_CLASE_HOGAR:N_POR_CLASE_HOGAR + N_NEGATIVOS_POR_CLASE_HOGAR]
        for ix in pos_idx:
            signo = 1 if rng.random() < 0.5 else -1
            perturbed.loc[ix, "minuto_del_dia"] = (events.loc[ix, "minuto_del_dia"] + signo * OFFSET_MIN_MINUTOS) % 1440
            labels.append({"hogar": hogar, "fecha": events.loc[ix, "fecha"], "actividad": actividad, "es_positivo": True})
        for ix in neg_idx:
            labels.append({"hogar": hogar, "fecha": events.loc[ix, "fecha"], "actividad": actividad, "es_positivo": False})
    labels_df = pd.DataFrame(labels).drop_duplicates(["hogar", "fecha", "actividad"], keep="first").reset_index(drop=True)

    con_positivo = set(zip(labels_df.loc[labels_df["es_positivo"], "hogar"], labels_df.loc[labels_df["es_positivo"], "fecha"]))
    control: list[dict] = []
    for hogar, g in events.groupby("hogar"):
        fechas = sorted(f for f in g["fecha"].unique() if (hogar, f) not in con_positivo)
        fechas = fechas[len(fechas) // 4:]  # con historial previo: los primeros días casi no se evalúan
        elegidas = rng.choice(len(fechas), size=min(N_DIAS_CONTROL_POR_HOGAR, len(fechas)), replace=False)
        control.extend({"hogar": hogar, "fecha": fechas[i]} for i in elegidas)
    return perturbed, labels_df, pd.DataFrame(control)


def _run(job: tuple[str, int, int, int, str]) -> dict:
    """Estadísticas de horario (sin umbral) de una combinación `(window_events, k_time_slots)`:
    por clase, solo los eventos de los días con etiqueta de esa clase y de los días de control."""
    vocab, seed, window_events, k_time_slots, out_dir = job
    path = Path(out_dir) / f"timing_cv_{vocab}_seed{seed}_w{window_events}_k{k_time_slots}.pkl"
    if path.exists():
        with path.open("rb") as f:
            return pickle.load(f)

    _, etiquetas_dia = load_synthetic(Path(out_dir))
    events = load_events(vocab, sorted(etiquetas_dia["hogar"].unique()))
    perturbed, labels, control = inject(events, seed)
    exclude = set(zip(labels.loc[labels["es_positivo"], "hogar"], labels.loc[labels["es_positivo"], "fecha"]))
    dias_control = set(zip(control["hogar"], control["fecha"]))
    cfg = TimingConfig(window_events=window_events, k_time_slots=k_time_slots)

    partes = []
    for clase in VOCABULARIOS[vocab]:
        de_la_clase = labels[labels["actividad"] == clase]
        dias = set(zip(de_la_clase["hogar"], de_la_clase["fecha"])) | dias_control
        partes.append(timing_statistics(perturbed, (clase,), cfg, exclude_from_reference=exclude, evaluate_only=dias))
    result = {
        "vocab": vocab, "seed": seed, "window_events": window_events, "k_time_slots": k_time_slots,
        "labels": labels, "control": control, "stats": pd.concat(partes, ignore_index=True),
    }
    with path.open("wb") as f:
        pickle.dump(result, f)
    return result


def build_hits(results: list[dict]) -> tuple[pd.DataFrame, pd.DataFrame, dict, dict]:
    """Junta las semillas. Devuelve las etiquetas, los días de control y, por candidato
    `(window_events, k_time_slots, excess_k)`, si cada etiqueta quedó marcada (`hits`) y qué
    clases quedaron marcadas en cada día de control (`alarmas`: candidato -> clase -> vector)."""
    seeds = sorted({r["seed"] for r in results})
    labels = pd.concat([r["labels"].assign(seed=r["seed"]) for r in results if (r["window_events"], r["k_time_slots"]) == (60, 2)], ignore_index=True)
    control = pd.concat([r["control"].assign(seed=r["seed"]) for r in results if (r["window_events"], r["k_time_slots"]) == (60, 2)], ignore_index=True)
    claves_label = list(zip(labels["seed"], labels["hogar"], labels["fecha"], labels["actividad"]))
    claves_control = list(zip(control["seed"], control["hogar"], control["fecha"]))
    clases = sorted(labels["actividad"].unique())

    hits: dict[tuple, np.ndarray] = {}
    alarmas: dict[tuple, dict[str, np.ndarray]] = {}
    for window_events in WINDOW_EVENTS_GRID:
        for k_time_slots in K_TIME_SLOTS_GRID:
            stats = pd.concat([
                r["stats"].assign(seed=r["seed"]) for r in results
                if (r["window_events"], r["k_time_slots"]) == (window_events, k_time_slots)
            ], ignore_index=True)
            assert sorted(stats["seed"].unique()) == seeds
            for excess_k in EXCESS_K_GRID:
                marcados = stats[stats["delta_min"] > stats["mediana_delta_min"] + excess_k * stats["sigma_min"]]
                dias_clase = set(zip(marcados["seed"], marcados["hogar"], marcados["fecha"], marcados["actividad"]))
                candidato = (window_events, k_time_slots, excess_k)
                hits[candidato] = np.array([c in dias_clase for c in claves_label])
                alarmas[candidato] = {
                    clase: np.array([(s, h, f, clase) in dias_clase for s, h, f in claves_control]) for clase in clases
                }
    return labels, control, hits, alarmas


def _tasas(hit: np.ndarray, es_pos: np.ndarray, mascara: np.ndarray) -> tuple[float, float]:
    pos, neg = hit[mascara & es_pos], hit[mascara & ~es_pos]
    return (float(pos.mean()) if len(pos) else float("nan"), float(neg.mean()) if len(neg) else float("nan"))


def _elegir(hits: dict, es_pos: np.ndarray, mascara: np.ndarray) -> tuple[tuple, float, float]:
    """El candidato de mayor detección con falsos positivos <= `MAX_FP_CLASE` sobre `mascara`; si
    ninguno llega, el de menos falsos positivos. Empates: el primero del barrido."""
    medidos = [(c, *_tasas(h, es_pos, mascara)) for c, h in hits.items()]
    ok = [m for m in medidos if m[2] <= MAX_FP_CLASE]
    if ok:
        return max(ok, key=lambda m: m[1])
    return min(medidos, key=lambda m: (m[2], -m[1]))


def _elegir_sin_perder(hits: dict, es_pos: np.ndarray, mascara: np.ndarray, actual: tuple) -> tuple[tuple, float, float]:
    """El candidato de menos falsos positivos entre los que detectan al menos lo mismo que
    `actual` sobre `mascara` (el propio `actual` siempre califica). Empates: mayor detección."""
    recall_actual, _ = _tasas(hits[actual], es_pos, mascara)
    medidos = [(c, *_tasas(h, es_pos, mascara)) for c, h in hits.items()]
    return min((m for m in medidos if m[1] >= recall_actual), key=lambda m: (m[2], -m[1]))


def _resumen(labels: pd.DataFrame, hit: np.ndarray, alarma_dia: np.ndarray, clases_seguidas: dict | None = None) -> dict:
    es_pos = labels["es_positivo"].to_numpy(dtype=bool)
    todas = np.ones(len(labels), dtype=bool)
    recall, fp = _tasas(hit, es_pos, todas)
    por_clase = {}
    for clase in sorted(labels["actividad"].unique()):
        de_la_clase = (labels["actividad"] == clase).to_numpy()
        r, f = _tasas(hit, es_pos, de_la_clase)
        por_clase[clase] = {
            "recall": r, "fp_rate": f, "detectados": int(hit[de_la_clase & es_pos].sum()), "n_pos": int((de_la_clase & es_pos).sum()),
            "fp": int(hit[de_la_clase & ~es_pos].sum()), "n_neg": int((de_la_clase & ~es_pos).sum()),
        }
    return {
        "recall": recall, "fp_rate": fp,
        "detectados": int(hit[es_pos].sum()), "n_pos": int(es_pos.sum()),
        "fp": int(hit[~es_pos].sum()), "n_neg": int((~es_pos).sum()),
        "dias_control_con_alarma": int(alarma_dia.sum()), "n_dias_control": int(len(alarma_dia)),
        "falsa_alarma_por_dia": float(alarma_dia.mean()),
        "por_clase": por_clase, "clases_seguidas": clases_seguidas,
    }


def cross_validate(labels: pd.DataFrame, control: pd.DataFrame, hits: dict, alarmas: dict, n_folds: int = 5, seed: int = 0) -> dict:
    homes = sorted(labels["hogar"].unique())
    rng = np.random.default_rng(seed)
    homes_shuffled = list(homes)
    rng.shuffle(homes_shuffled)
    folds = np.array_split(homes_shuffled, n_folds)

    clases = sorted(labels["actividad"].unique())
    es_pos = labels["es_positivo"].to_numpy(dtype=bool)
    hogar_label, clase_label = labels["hogar"].to_numpy(), labels["actividad"].to_numpy()
    hogar_control = control["hogar"].to_numpy()

    actual = (60, 2, 3.0)  # TimingConfig()
    hit_actual = hits[actual]
    alarma_actual = np.any([alarmas[actual][c] for c in clases], axis=0)

    variantes = {
        v: {"hit": np.zeros(len(labels), dtype=bool), "alarma": np.zeros(len(control), dtype=bool), "elegido": []}
        for v in ("calibrado", "acotado", "misma_deteccion")
    }
    for n, test_homes in enumerate(folds):
        en_test = np.isin(hogar_label, test_homes.tolist())
        control_test = np.isin(hogar_control, test_homes.tolist())
        for clase in clases:
            de_la_clase = clase_label == clase
            candidato, recall_train, fp_train = _elegir(hits, es_pos, de_la_clase & ~en_test)
            seguir = fp_train <= MAX_FP_CLASE and recall_train >= MIN_RECALL_CLASE
            sin_perder = _elegir_sin_perder(hits, es_pos, de_la_clase & ~en_test, actual)
            elecciones = (
                ("calibrado", True, candidato, recall_train, fp_train),
                ("acotado", seguir, candidato, recall_train, fp_train),
                ("misma_deteccion", True, *sin_perder),
            )
            for variante, usar, candidato, recall_train, fp_train in elecciones:
                variantes[variante]["elegido"].append({
                    "fold": n, "clase": clase, "seguida": bool(usar), "window_events": candidato[0],
                    "k_time_slots": candidato[1], "excess_k": candidato[2], "recall_train": recall_train, "fp_train": fp_train,
                })
                if usar:
                    variantes[variante]["hit"][en_test & de_la_clase] = hits[candidato][en_test & de_la_clase]
                    variantes[variante]["alarma"][control_test] |= alarmas[candidato][clase][control_test]

    # Lo que se adoptaría: el mismo criterio aplicado a todos los hogares juntos.
    todos = np.ones(len(labels), dtype=bool)
    final = {}
    for clase in clases:
        candidato, recall, fp = _elegir(hits, es_pos, (clase_label == clase) & todos)
        final[clase] = {
            "seguida": bool(fp <= MAX_FP_CLASE and recall >= MIN_RECALL_CLASE), "window_events": candidato[0],
            "k_time_slots": candidato[1], "excess_k": candidato[2], "recall": recall, "fp_rate": fp,
        }

    return {
        "actual": _resumen(labels, hit_actual, alarma_actual),
        **{v: {**_resumen(labels, d["hit"], d["alarma"]), "elegido_por_fold": d["elegido"]} for v, d in variantes.items()},
        "con_todos_los_hogares": final,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dir", type=Path, default=DEFAULT_DIR)
    ap.add_argument("--vocab", nargs="+", choices=sorted(VOCABULARIOS), default=sorted(VOCABULARIOS))
    ap.add_argument("--cv", type=int, default=5, metavar="N_FOLDS")
    ap.add_argument("--workers", type=int, default=9)
    args = ap.parse_args()

    jobs = [
        (vocab, seed, w, k, str(args.dir))
        for vocab in args.vocab for seed in SEEDS for w in WINDOW_EVENTS_GRID for k in K_TIME_SLOTS_GRID
    ]
    print(f"Detector de horario, vocabularios {args.vocab}: {len(jobs)} clusterings (guardados en {args.dir}/timing_cv_*.pkl) ...", flush=True)
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        results = list(pool.map(_run, jobs))

    salida = {}
    pd.set_option("display.width", 220)
    for vocab in args.vocab:
        labels, control, hits, alarmas = build_hits([r for r in results if r["vocab"] == vocab])
        cv = cross_validate(labels, control, hits, alarmas, n_folds=args.cv)
        salida[vocab] = cv
        print(f"\n=== Vocabulario {vocab}: {labels['hogar'].nunique()} hogares, {int(labels['es_positivo'].sum())} positivos, "
              f"{int((~labels['es_positivo']).sum())} negativos, {len(control)} días de control ===")
        filas = []
        for variante in ("actual", "misma_deteccion", "calibrado", "acotado"):
            r = cv[variante]
            filas.append({
                "variante": variante, "recall_%": round(100 * r["recall"], 1), "detectados": f"{r['detectados']}/{r['n_pos']}",
                "fp_%": round(100 * r["fp_rate"], 1), "fp": f"{r['fp']}/{r['n_neg']}",
                "falsa_alarma_dia_%": round(100 * r["falsa_alarma_por_dia"], 1),
                "dias": f"{r['dias_control_con_alarma']}/{r['n_dias_control']}",
                **{c: f"{100 * d['recall']:.0f}/{100 * d['fp_rate']:.0f}" for c, d in r["por_clase"].items()},
            })
        print(pd.DataFrame(filas).to_string(index=False))
        print("(por clase: detección % / falsos positivos %)")
        elegido = pd.DataFrame(cv["acotado"]["elegido_por_fold"])
        print("\nElegido en cada partición (con los hogares de entrenamiento):")
        print(elegido.round(3).to_string(index=False))
        print("\nCon el mismo criterio sobre todos los hogares:")
        print(pd.DataFrame(cv["con_todos_los_hogares"]).T.to_string())
        sin_perder = pd.DataFrame(cv["misma_deteccion"]["elegido_por_fold"])
        print("\nElegido por `misma_deteccion` en cada partición:")
        print(sin_perder.drop(columns=["seguida"]).round(3).to_string(index=False))

    out_path = args.dir / "timing_cv_results.json"
    out_path.write_text(json.dumps(salida, indent=2, default=str), encoding="utf-8")
    print(f"\nGuardado: {out_path}")


if __name__ == "__main__":
    main()
