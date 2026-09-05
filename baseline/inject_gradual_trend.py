"""Valida `src/trend_deviations.py` (tendencia gradual, ej. "cada semana sale un poco menos de
casa, sostenido varias semanas") con el mismo método que el resto de `baseline/`: perturbar una
copia de datos reales de forma **sostenida sobre un bloque de semanas completo** (no un solo día
suelto — eso ya lo prueban los otros scripts) y medir si el detector se da cuenta. En producción
real no se perturba nada, el detector solo lee lo que pasó.

Solo tiene sentido sobre hogares con historial suficiente para separar un bloque base de varias
semanas + uno reciente (ver `baseline/README.md`): `hh113` (~70 semanas), `hh115` (~43),
`hh117` (~34). Los otros hogares recomendados por `dataset_events/README.md` solo tienen ~9
semanas — insuficiente acá, aunque sí alcance para los otros detectores día a día.

Uso:
  py -3 inject_gradual_trend.py
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from src.features import build_day_vectors
from src.trend_deviations import DEFAULT_TRACKED_CLASSES, TrendConfig, _build_weekly

EVENTS_PARQUET = Path(__file__).resolve().parent.parent / "dataset_events" / "events" / "casas_events.parquet"
HOMES_CON_HISTORIAL = ["hh113", "hh115", "hh117"]
FACTOR_BAJA = 0.4  # reduce al 40% del nivel habitual, sostenido en el bloque
FACTOR_SUBE = 2.0  # duplica el nivel habitual, sostenido en el bloque
MARGEN_NEGATIVO_SEMANAS = 2  # separación mínima entre el punto negativo y el bloque perturbado


def _dias_de_semana(semana_inicio) -> list:
    return list(pd.date_range(semana_inicio, periods=7, freq="D").date)


def _perturbar_bloque(day_vectors: pd.DataFrame, hogar: str, clase: str, semanas_inicio: list, factor: float) -> set:
    tocados = set()
    for semana_inicio in semanas_inicio:
        for f in _dias_de_semana(semana_inicio):
            key = (hogar, f)
            if key not in day_vectors.index:
                continue
            day_vectors.loc[key, f"{clase}_min"] = day_vectors.loc[key, f"{clase}_min"] * factor
            n_actual = day_vectors.loc[key, f"{clase}_n"]
            day_vectors.loc[key, f"{clase}_n"] = int(round(n_actual * factor))
            tocados.add(key)
    return tocados


def inject(
    day_vectors: pd.DataFrame,
    homes: list[str],
    tracked_classes: tuple[str, ...],
    config: TrendConfig,
    seed: int = 0,
) -> tuple[pd.DataFrame, pd.DataFrame, set]:
    rng = np.random.default_rng(seed)
    perturbed = day_vectors.copy()
    labels: list[dict] = []
    excluidos: set = set()

    weekly_real = _build_weekly(day_vectors, tracked_classes, config)
    n_needed = config.n_semanas_recientes + config.n_semanas_base

    for hogar in homes:
        for clase in tracked_classes:
            g = weekly_real[weekly_real["hogar"] == hogar].sort_values("semana_inicio").reset_index(drop=True)
            if len(g) < n_needed + 2 * MARGEN_NEGATIVO_SEMANAS + 2:
                print(f"  ! {hogar}/{clase}: solo {len(g)} semanas válidas, insuficiente, se omite")
                continue

            i_eval = int(rng.integers(n_needed, len(g) - 1))
            semanas_bloque = g.loc[i_eval - config.n_semanas_recientes : i_eval - 1, "semana_inicio"].to_list()

            factor = FACTOR_BAJA if rng.random() < 0.5 else FACTOR_SUBE
            tocados = _perturbar_bloque(perturbed, hogar, clase, semanas_bloque, factor)
            excluidos |= tocados

            labels.append(
                {
                    "hogar": hogar,
                    "clase": clase,
                    "semana_fin": g.loc[i_eval - 1, "semana_inicio"],
                    "factor": factor,
                    "es_positivo": True,
                }
            )

            candidatos_neg = [
                j
                for j in range(n_needed, len(g))
                if not (i_eval - config.n_semanas_recientes - MARGEN_NEGATIVO_SEMANAS <= j <= i_eval + MARGEN_NEGATIVO_SEMANAS)
            ]
            if candidatos_neg:
                j_neg = candidatos_neg[rng.integers(len(candidatos_neg))]
                labels.append(
                    {
                        "hogar": hogar,
                        "clase": clase,
                        "semana_fin": g.loc[j_neg - 1, "semana_inicio"],
                        "factor": 1.0,
                        "es_positivo": False,
                    }
                )

    return perturbed, pd.DataFrame(labels), excluidos


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--homes", nargs="*", default=HOMES_CON_HISTORIAL)
    ap.add_argument("--out-dir", type=Path, default=Path(__file__).resolve().parent / "synthetic_eval")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    print(f"Construyendo vectores-día reales para {args.homes} ...")
    day_vectors = build_day_vectors(EVENTS_PARQUET, homes=args.homes)

    config = TrendConfig()
    print("Inyectando tendencias graduales sostenidas ...")
    perturbed, labels, excluidos = inject(day_vectors, args.homes, DEFAULT_TRACKED_CLASSES, config, seed=args.seed)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    perturbed.to_parquet(args.out_dir / "day_vectors_trend_perturbed.parquet")
    labels.to_parquet(args.out_dir / "labels_trend.parquet")
    excl_df = pd.DataFrame(sorted(excluidos), columns=["hogar", "fecha"])
    excl_df.to_parquet(args.out_dir / "excluded_days_trend.parquet")

    print(f"\n  {args.out_dir / 'day_vectors_trend_perturbed.parquet'}  ({len(perturbed)} días)")
    print(f"  {args.out_dir / 'labels_trend.parquet'}  ({len(labels)} etiquetas)")
    print(f"  {args.out_dir / 'excluded_days_trend.parquet'}  ({len(excl_df)} días excluidos de referencia)")
    print()
    print(labels.groupby(["es_positivo"]).size().to_string())


if __name__ == "__main__":
    main()
