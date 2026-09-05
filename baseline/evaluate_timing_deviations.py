"""Valida `src/timing_deviations.py` (desvíos de horario habitual, ej. "comer 3h más tarde de lo
usual") con el mismo método que el resto de `baseline/`: como no hay ningún evento real etiquetado
como "esto pasó a una hora rara", se perturba una copia de eventos reales (se les corre el
horario) y se mide si el detector se da cuenta — un banco de pruebas, no la lógica de producción.
En producción real (`score_timing_deviations` corrida sobre eventos sin tocar) no se perturba nada.

Uso:
  py -3 evaluate_timing_deviations.py
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from src.timing_deviations import DEFAULT_TRACKED_CLASSES, TimingConfig, score_timing_deviations

EVENTS_PARQUET = Path(__file__).resolve().parent.parent / "dataset_events" / "events" / "casas_events.parquet"
DEFAULT_HOMES = ["hh113", "hh115", "hh117", "hh120", "hh109", "hh101", "mn83"]
N_POR_CLASE_HOGAR = 3  # eventos a perturbar por (hogar, clase)
N_NEGATIVOS_POR_CLASE_HOGAR = 8
OFFSET_MIN_MINUTOS = 180  # cuánto se corre el horario del evento perturbado (3 h)


def inject(events: pd.DataFrame, seed: int = 0) -> tuple[pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(seed)
    perturbed = events.copy()
    labels: list[dict] = []

    for (hogar, actividad), g in events.groupby(["hogar", "actividad"], observed=True):
        if actividad not in DEFAULT_TRACKED_CLASSES or len(g) < 30:
            continue
        idx_pool = g.index.to_numpy().copy()
        rng.shuffle(idx_pool)
        n_pos = min(N_POR_CLASE_HOGAR, len(idx_pool))
        pos_idx = idx_pool[:n_pos]
        resto = idx_pool[n_pos:]
        n_neg = min(N_NEGATIVOS_POR_CLASE_HOGAR, len(resto))
        neg_idx = resto[:n_neg]

        for ix in pos_idx:
            fila = events.loc[ix]
            signo = 1 if rng.random() < 0.5 else -1
            nuevo_min = (fila["minuto_del_dia"] + signo * OFFSET_MIN_MINUTOS) % 1440
            perturbed.loc[ix, "minuto_del_dia"] = nuevo_min
            labels.append(
                {"hogar": hogar, "fecha": fila["fecha"], "actividad": actividad, "es_positivo": True}
            )
        for ix in neg_idx:
            fila = events.loc[ix]
            labels.append(
                {"hogar": hogar, "fecha": fila["fecha"], "actividad": actividad, "es_positivo": False}
            )

    return perturbed, pd.DataFrame(labels)


def evaluate(perturbed: pd.DataFrame, labels: pd.DataFrame, config: TimingConfig) -> dict:
    exclude = {(r.hogar, r.fecha) for r in labels.itertuples() if r.es_positivo}
    scored = score_timing_deviations(perturbed, config=config, exclude_from_reference=exclude)
    hits = set(zip(scored["hogar"], scored["fecha"], scored["actividad"])) if len(scored) else set()

    labels = labels.copy()
    labels["hit"] = [
        (h, f, a) in hits for h, f, a in zip(labels["hogar"], labels["fecha"], labels["actividad"])
    ]
    pos = labels[labels["es_positivo"]]
    neg = labels[~labels["es_positivo"]]
    return {
        "window_events": config.window_events,
        "k_time_slots": config.k_time_slots,
        "excess_k": config.excess_k,
        "n_pos": len(pos),
        "n_neg": len(neg),
        "recall": float(pos["hit"].mean()) if len(pos) else float("nan"),
        "fp_rate": float(neg["hit"].mean()) if len(neg) else float("nan"),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--homes", nargs="*", default=DEFAULT_HOMES)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    events = pd.read_parquet(
        EVENTS_PARQUET, columns=["hogar", "actividad", "fecha", "inicio", "minuto_del_dia"]
    )
    events = events[events["hogar"].isin(args.homes)].copy()
    events["hogar"] = events["hogar"].astype(str)
    events["actividad"] = events["actividad"].astype(str)

    perturbed, labels = inject(events, seed=args.seed)
    print(f"Etiquetas: {len(labels)} ({labels['es_positivo'].sum()} positivas, corridas {OFFSET_MIN_MINUTOS} min)")

    results = []
    for window_events in (40, 60, 90):
        for k_time_slots in (1, 2, 3):
            for excess_k in (2.0, 2.5, 3.0):
                cfg = TimingConfig(window_events=window_events, k_time_slots=k_time_slots, excess_k=excess_k)
                results.append(evaluate(perturbed, labels, cfg))

    df = pd.DataFrame(results).sort_values(["fp_rate", "recall"], ascending=[True, False])
    pd.set_option("display.width", 160)
    print(df.to_string(index=False))

    ok = df[df["fp_rate"] <= 0.15]
    best = (ok if len(ok) else df).sort_values("recall", ascending=False).iloc[0]
    print("\nMejor combinación (FP <= 0.15 si es posible):")
    print(best.to_string())


if __name__ == "__main__":
    main()
