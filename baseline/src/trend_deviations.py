"""Tendencia gradual — detecta un cambio **lento y sostenido a lo largo de varias semanas**, no
un día raro puntual (eso ya lo cubren `dynamic_clustering.py` y `class_deviations.py`).

El informe (Sec. 2.1.2) enmarca justamente esto como el caso clínico más relevante del deterioro
progresivo: no "hoy salió menos de casa" sino "cada semana sale un poco menos, sostenido durante
varias semanas seguidas". Un detector día-a-día no lo ve bien porque cada día individual dentro de
esa pendiente puede estar apenas fuera de rango — la señal se acumula recién al mirar bloques de
semanas.

**Diseño**: se agregan los vectores-día de `features.build_day_vectors()` a nivel semanal (ISO),
y por cada `(hogar, clase)` se compara la mediana de un bloque de `n_semanas_recientes` semanas
contra la mediana de las `n_semanas_base` semanas inmediatamente anteriores — mismo criterio
robusto (mediana + MAD) que ya usa `class_deviations.py`, aplicado a nivel de semana en vez de
día. No se asume de antemano qué dirección (`sube`/`baja`) es clínicamente mala para cada clase —
eso es una decisión de la capa de mensajes al usuario final, no de este detector.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

_MAD_TO_STD = 1.4826  # misma constante que class_deviations.py / timing_deviations.py

DEFAULT_TRACKED_CLASSES: tuple[str, ...] = ("salir_casa", "visitas", "dormir", "descansar")


@dataclass
class TrendConfig:
    # Defaults calibrados con `evaluate_trend_deviations.py` (ver "Estado" en README) — muestra
    # chica (12 positivos/12 negativos, solo 3 hogares con historial suficiente), no confiar en
    # el número como una calibración robusta, sí como punto de partida razonable.
    n_semanas_recientes: int = 3
    n_semanas_base: int = 8
    min_dias_validos_por_semana: int = 4  # semanas con menos días útiles que esto se descartan
    threshold_z: float = 2.0


def _build_weekly(day_vectors: pd.DataFrame, tracked_classes: tuple[str, ...], config: TrendConfig) -> pd.DataFrame:
    """Agrega `day_vectors` (índice hogar/fecha) a nivel `(hogar, semana ISO)`.

    Una semana con menos de `min_dias_validos_por_semana` días sin `cobertura_baja` se descarta
    por completo (no aparece en el resultado) — mismo criterio que `cobertura_baja` a nivel día:
    no se inventa una tendencia sobre una semana con datos insuficientes.
    """
    df = day_vectors.reset_index()
    df = df[~df["cobertura_baja"]].copy()
    fecha = pd.to_datetime(df["fecha"])
    iso = fecha.dt.isocalendar()
    df["semana"] = iso["year"].astype(str) + "-W" + iso["week"].astype(str).str.zfill(2)
    # lunes de esa semana ISO — para poder ordenar/comparar cronológicamente sin ambigüedad
    df["semana_inicio"] = (fecha - pd.to_timedelta(iso["day"] - 1, unit="D")).dt.date

    cols = [f"{c}_min" for c in tracked_classes] + [f"{c}_n" for c in tracked_classes]
    agg = df.groupby(["hogar", "semana", "semana_inicio"], observed=True)[cols].sum()
    n_dias = df.groupby(["hogar", "semana", "semana_inicio"], observed=True).size()
    agg["n_dias_validos"] = n_dias
    agg = agg[agg["n_dias_validos"] >= config.min_dias_validos_por_semana]
    return agg.reset_index().sort_values(["hogar", "semana_inicio"])


def _dias_de_semana(semana_inicio) -> list:
    return list(pd.date_range(semana_inicio, periods=7, freq="D").date)


def score_trend_deviations(
    day_vectors: pd.DataFrame,
    tracked_classes: tuple[str, ...] = DEFAULT_TRACKED_CLASSES,
    config: TrendConfig | None = None,
    exclude_from_reference: set[tuple[str, object]] | None = None,
) -> pd.DataFrame:
    """Evalúa, semana a semana, si el bloque reciente de cada `(hogar, clase)` se corrió respecto
    al bloque base inmediatamente anterior.

    `exclude_from_reference`: pares `(hogar, fecha)` de días que NO deben usarse para construir el
    bloque **base** de ninguna evaluación (mismo espíritu que los otros dos detectores: una
    semana "sospechosa" — ej. inyectada en una prueba — no debe contaminar la rutina de
    referencia). A diferencia de los otros dos detectores, acá la exclusión nunca saca días de la
    semana que se está EVALUANDO como "reciente" — eso rompería la propia medición (una semana
    perturbada a propósito para la prueba dejaría de tener datos para poder detectarla). Una
    semana con al menos un día excluido no entra al bloque base; si eso deja menos de la mitad de
    `n_semanas_base` semanas utilizables, esa evaluación se descarta (no hay base confiable).

    Devuelve una fila por `(hogar, semana evaluada, clase)` marcada con tendencia — la mayoría de
    las semanas no van a tener ninguna.
    """
    config = config or TrendConfig()
    exclude_from_reference = exclude_from_reference or set()
    weekly = _build_weekly(day_vectors, tracked_classes, config)

    rows: list[dict] = []
    n_needed = config.n_semanas_recientes + config.n_semanas_base
    min_base_ok = max(1, config.n_semanas_base // 2)
    for hogar, g in weekly.groupby("hogar", sort=False):
        g = g.sort_values("semana_inicio").reset_index(drop=True)
        semana_excluida = np.array(
            [any((hogar, d) in exclude_from_reference for d in _dias_de_semana(s)) for s in g["semana_inicio"]]
        )
        for clase in tracked_classes:
            col = f"{clase}_min"
            if col not in g.columns:
                continue
            valores = g[col].to_numpy(dtype=float)
            semanas = g["semana_inicio"].to_list()

            for i in range(n_needed, len(valores)):
                base_idx = [j for j in range(i - n_needed, i - config.n_semanas_recientes) if not semana_excluida[j]]
                if len(base_idx) < min_base_ok:
                    continue
                base = valores[base_idx]
                reciente = valores[i - config.n_semanas_recientes : i]

                mediana_base = float(np.median(base))
                mad_base = float(np.median(np.abs(base - mediana_base)))
                sigma_base = mad_base * _MAD_TO_STD
                mediana_reciente = float(np.median(reciente))

                if sigma_base <= 1e-9:
                    continue
                z = (mediana_reciente - mediana_base) / sigma_base
                if abs(z) <= config.threshold_z:
                    continue

                rows.append(
                    {
                        "hogar": hogar,
                        "semana_fin": semanas[i - 1],
                        "clase": clase,
                        "direccion": "sube" if z > 0 else "baja",
                        "mediana_base": mediana_base,
                        "mediana_reciente": mediana_reciente,
                        "z_score": z,
                    }
                )

    return pd.DataFrame(rows)
