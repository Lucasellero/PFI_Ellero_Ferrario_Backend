"""Desvíos de **horario habitual** por actividad — ej. "esta persona siempre come entre las 12:00
y las 12:45; hoy comió a las 15:00".

Complementa a `class_deviations.py` (que mira cuánto/cuántas veces ocurre una clase por día, no a
qué hora) y a `dynamic_clustering.py` (que mira la "forma" completa del día). Ninguno de los dos
detecta un corrimiento de horario si la duración/conteo total del día no cambia — este módulo sí,
porque trabaja sobre el timestamp real de cada evento, no sobre el agregado diario.

**No modifica ni perturba nada del dataset real.** Aprende el horario habitual de cada
`(hogar, actividad)` puramente de sus propios eventos pasados (ventana móvil de los últimos
`window_events` eventos de esa clase, causal — nunca mira eventos futuros) y compara cada evento
nuevo contra eso. La única perturbación de datos en este módulo (correr este archivo como script)
es, igual que en `class_deviations.py`, un test de humo para validar el mecanismo — no algo que
corra en producción.

**Por qué clustering circular y no un solo "horario promedio"**: muchas actividades ocurren más
de una vez por día con horarios bien distintos entre sí (`comer` = desayuno+almuerzo+cena;
`medicacion` = toma de mañana + de noche). Comparar un evento de la cena contra el promedio de
desayuno+almuerzo+cena metería ruido. Se agrupan primero los horarios históricos en hasta
`k_time_slots` franjas (ej. "almuerzo" ~12:20, "cena" ~20:00) y el evento nuevo se compara contra
la franja más cercana, no contra el promedio de todas.

**Por qué "circular"**: la hora del día da la vuelta a la medianoche (23:50 y 00:10 están a 20
minutos de distancia real, no a 1420) — relevante para actividades como `dormir`. Se embebe cada
horario como un punto en el círculo unitario (`cos`, `sin` del ángulo) antes de clusterizar, y la
distancia final se reporta en minutos reales sobre el círculo, no en minutos crudos.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans

_MINUTES_PER_DAY = 1440.0
_MAD_TO_STD = 1.4826  # ver class_deviations.py — misma constante, mismo motivo
_SIGMA_MIN_MINUTES = 15.0  # piso de dispersión: evita marcar como "raro" una franja casi sin variación histórica

DEFAULT_TRACKED_CLASSES: tuple[str, ...] = ("comer", "beber", "medicacion", "dormir", "salir_casa")


@dataclass
class TimingConfig:
    window_events: int = 60  # ventana en CANTIDAD de eventos previos de esa clase, no en días
    min_eventos_historial: int = 10
    k_time_slots: int = 2
    excess_k: float = 3.0


def _to_circular(minuto_del_dia: np.ndarray) -> np.ndarray:
    angle = 2 * np.pi * minuto_del_dia / _MINUTES_PER_DAY
    return np.stack([np.cos(angle), np.sin(angle)], axis=1)


def _circular_minute(point_xy: np.ndarray) -> float:
    angle = np.arctan2(point_xy[1], point_xy[0])
    return float((angle / (2 * np.pi)) * _MINUTES_PER_DAY % _MINUTES_PER_DAY)


def _circ_delta_minutes(a: float, b: float) -> float:
    d = abs(a - b) % _MINUTES_PER_DAY
    return min(d, _MINUTES_PER_DAY - d)


def score_timing_deviations(
    events_df: pd.DataFrame,
    tracked_classes: tuple[str, ...] = DEFAULT_TRACKED_CLASSES,
    config: TimingConfig | None = None,
    exclude_from_reference: set[tuple[str, object]] | None = None,
) -> pd.DataFrame:
    """Marca eventos individuales cuyo horario se sale del habitual de esa `(hogar, actividad)`.

    `events_df`: eventos crudos (ej. `casas_events.parquet`, columnas `hogar`, `actividad`,
    `inicio`, `minuto_del_dia`, `fecha`) — a diferencia de los otros dos detectores, trabaja a
    nivel de evento individual, no de vector-día.

    `exclude_from_reference`: pares `(hogar, fecha)` de eventos que se evalúan pero nunca entran
    en el historial de horario habitual de otro evento — mismo uso que en los otros detectores,
    pensado para eventos de prueba inyectados.

    Devuelve una fila por evento marcado (`hogar`, `fecha`, `actividad`, `minuto_del_dia`,
    `horario_habitual_min`, `desvio_min`, `umbral_min`).
    """
    config = config or TimingConfig()
    exclude_from_reference = exclude_from_reference or set()

    df = events_df[events_df["actividad"].isin(tracked_classes)].copy()
    df = df.sort_values("inicio")

    rows: list[dict] = []
    for (hogar, actividad), g in df.groupby(["hogar", "actividad"], observed=True):
        minutos = g["minuto_del_dia"].to_numpy(dtype=float)
        fechas = g["fecha"].to_numpy()
        ref_ok = np.array([(hogar, f) not in exclude_from_reference for f in fechas])

        for i in range(len(g)):
            lo = max(0, i - config.window_events)
            hist_idx = [j for j in range(lo, i) if ref_ok[j]]
            if len(hist_idx) < config.min_eventos_historial:
                continue

            hist_min = minutos[hist_idx]
            hist_xy = _to_circular(hist_min)
            k_eff = min(config.k_time_slots, len(hist_idx))
            km = KMeans(n_clusters=k_eff, n_init=10, random_state=0)
            labels_hist = km.fit_predict(hist_xy)

            hoy_xy = _to_circular(minutos[i : i + 1])[0]
            nearest = int(np.argmin(np.linalg.norm(km.cluster_centers_ - hoy_xy, axis=1)))
            centroid_min = _circular_minute(km.cluster_centers_[nearest])

            miembros_min = hist_min[labels_hist == nearest]
            deltas_hist = np.array([_circ_delta_minutes(m, centroid_min) for m in miembros_min])
            mediana_delta = float(np.median(deltas_hist)) if len(deltas_hist) else 0.0
            mad_delta = float(np.median(np.abs(deltas_hist - mediana_delta))) if len(deltas_hist) else 0.0
            sigma = max(mad_delta * _MAD_TO_STD, _SIGMA_MIN_MINUTES)

            delta_hoy = _circ_delta_minutes(minutos[i], centroid_min)
            umbral = mediana_delta + config.excess_k * sigma

            if delta_hoy > umbral:
                rows.append(
                    {
                        "hogar": hogar,
                        "fecha": fechas[i],
                        "actividad": actividad,
                        "minuto_del_dia": minutos[i],
                        "horario_habitual_min": round(centroid_min, 1),
                        "desvio_min": round(delta_hoy, 1),
                        "umbral_min": round(umbral, 1),
                    }
                )

    return pd.DataFrame(rows)
