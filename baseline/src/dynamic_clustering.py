"""Clustering dinámico (ventana móvil) + score de desvío de rutina.

Interpretación propia del enfoque de Snoun et al. 2024 ("A Deep Learning Based System For a
Long-term Elderly Behavioral Drift Detection", SN Computer Science) citado en el informe
(Sec. 2.1.7/2.2.5): clustering iterativo sobre una ventana móvil de días previos, comparando
cada día nuevo contra los centroides de esa ventana vía un score de similitud/distancia.

**Nota de honestidad académica**: el texto completo del paper está detrás de un paywall — solo
se pudo confirmar el enfoque a nivel de abstract. Esto NO es una reproducción de sus fórmulas
exactas (no publicadas en acceso abierto), es una implementación propia fiel a la idea general
documentada (ventana móvil + reclustering + similitud contra la rutina reciente). Aclarar esto
en el informe si se cita este módulo como implementación de ese paper.

Diseño (ver también `baseline/README.md`):
  1. Para el día D a evaluar, tomar los `window_days` días previos de ESE residente como
     referencia de rutina (nunca incluye D — predicción causal, sin leakage).
  2. Reclusterizar esa ventana con KMeans(k) — es la parte "dinámica": si la rutina cambia de
     forma sostenida, los clusters de referencia se corren solos con la ventana.
  3. Score de distancia de D al centroide más cercano de esos clusters.
  4. Normalización por residente: z-score contra el historial de scores de ESE MISMO residente
     (cada persona tiene un volumen/rutina de base distinto, no son comparables entre sí crudos).
  5. `z_score > threshold_z` → día marcado como posible desvío.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler


@dataclass
class DriftConfig:
    # Defaults calibrados con `evaluate_deviations.py` sobre desvíos sintéticos (ver "Estado" en
    # README): mejor punto recall/FP combinado (con `class_deviations.py`) del barrido de
    # `(window_days, k, threshold_z)`.
    window_days: int = 30
    # período de silencio: decisión de producto (7-10 días) para que el sistema no opine sobre
    # una persona nueva antes de tener una noción mínima de su rutina real.
    min_window_days: int = 7
    k: int = 2
    threshold_z: float = 2.5
    random_state: int = 0


def _feature_columns(day_vectors: pd.DataFrame) -> list[str]:
    return [c for c in day_vectors.columns if c.endswith("_min") or c.endswith("_n")]


def score_days(
    day_vectors: pd.DataFrame,
    config: DriftConfig | None = None,
    exclude_from_reference: set[tuple[str, object]] | None = None,
) -> pd.DataFrame:
    """Corre el clustering dinámico por residente, devuelve un score de desvío por día.

    `day_vectors`: salida de `features.build_day_vectors` (MultiIndex hogar/fecha). Los días
    marcados `cobertura_baja` se excluyen tanto de evaluarse como de servir de referencia (no
    hay señal confiable de rutina en un día casi sin datos).

    `exclude_from_reference`: pares `(hogar, fecha)` que SÍ se evalúan pero que nunca entran en
    la ventana de referencia de otro día ni en el historial de z-score de ese residente — pensado
    para desvíos sintéticos inyectados (`inject_synthetic_deviations.py`): el día perturbado debe
    poder evaluarse contra la rutina real, sin contaminar la rutina de los días posteriores.

    Los primeros `min_window_days` días (útiles) de cada residente no tienen ventana suficiente
    y no aparecen en el resultado.
    """
    config = config or DriftConfig()
    exclude_from_reference = exclude_from_reference or set()
    feat_cols = _feature_columns(day_vectors)

    rows: list[dict] = []
    for hogar, g in day_vectors.groupby(level="hogar", sort=False):
        g = g.droplevel("hogar").sort_index()
        g = g[~g["cobertura_baja"]]
        fechas = g.index.to_list()
        X = g[feat_cols].to_numpy(dtype=float)
        ref_ok = np.array([(hogar, f) not in exclude_from_reference for f in fechas])

        hist_scores: list[float] = []
        for i, fecha in enumerate(fechas):
            lo = max(0, i - config.window_days)
            window_idx = [j for j in range(lo, i) if ref_ok[j]]
            if len(window_idx) < config.min_window_days:
                continue

            window_X = X[window_idx]
            scaler = StandardScaler()
            window_scaled = scaler.fit_transform(window_X)
            k_eff = min(config.k, len(window_idx))
            km = KMeans(n_clusters=k_eff, n_init=10, random_state=config.random_state)
            km.fit(window_scaled)

            day_scaled = scaler.transform(X[i : i + 1])
            dists = np.linalg.norm(km.cluster_centers_ - day_scaled, axis=1)
            nearest = int(np.argmin(dists))
            drift_raw = float(dists[nearest])

            is_ref_day = (hogar, fecha) not in exclude_from_reference
            if len(hist_scores) < config.min_window_days:
                z = float("nan")
            else:
                mu, sigma = float(np.mean(hist_scores)), float(np.std(hist_scores))
                z = (drift_raw - mu) / sigma if sigma > 1e-9 else 0.0
            if is_ref_day:
                hist_scores.append(drift_raw)

            contrib = np.abs(day_scaled[0] - km.cluster_centers_[nearest])
            top_idx = np.argsort(contrib)[::-1][:3]
            top_classes = [feat_cols[j] for j in top_idx]

            rows.append(
                {
                    "hogar": hogar,
                    "fecha": fecha,
                    "drift_score": drift_raw,
                    "z_score": z,
                    "es_desvio": bool(not np.isnan(z) and z > config.threshold_z),
                    "top_contribuyentes": top_classes,
                }
            )

    return pd.DataFrame(rows)
