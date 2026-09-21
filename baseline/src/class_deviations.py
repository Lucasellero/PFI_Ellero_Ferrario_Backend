"""Desvíos por clase individual — complementario al clustering multivariado de
`dynamic_clustering.py`.

**Por qué existe**: la evaluación con desvíos sintéticos (`evaluate_deviations.py`) encontró que
el escenario `sin_medicacion` nunca cruzaba el umbral del score multivariado, en ningún punto del
barrido de `(window_days, k, threshold_z)`, aunque `medicacion_n`/`medicacion_min` sí aparecían
como el contribuyente principal de la distancia (`top_contribuyentes`). Causa diagnosticada: con
38 dimensiones, un cambio real pero concentrado en una sola clase de bajo volumen diario
(medicación: minutos/día muy chicos frente a `trabajo`/`ver_tv`/`dormir`) queda diluido por el
ruido cotidiano normal de las otras 37 clases — el "día completo" no se ve tan distinto en
agregado, aunque esa clase puntual sí cambió de forma anómala.

Este módulo evalúa cada clase clínicamente relevante **por separado**, contra su propia
distribución histórica (mediana + MAD, robustas a pocos días de historial y a outliers) dentro de
la misma ventana móvil que usa `dynamic_clustering.py` — pensado específicamente para señales
sparse/raras que importan por sí mismas (ausencia total, exceso, o ausencia sostenida varios días
seguidos), no como parte de la forma general del día.

Se usa en conjunto con `dynamic_clustering.score_days`, no en su reemplazo: el multivariado sigue
siendo mejor para detectar que "la rutina en general se ve distinta" cuando el cambio está
repartido en varias clases a la vez; este detector es para cuando el cambio está concentrado en
una sola clase importante.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

# constante estándar para que el MAD (median absolute deviation) aproxime el desvío estándar
# bajo una distribución normal — permite reusar el mismo criterio de "K desvíos" que un z-score,
# pero con una medida robusta a outliers (útil acá: pocos días de historial, valores en 0 frecuentes).
_MAD_TO_STD = 1.4826

TRACKED_CLASSES: tuple[str, ...] = ("medicacion", "salir_casa", "bano_nocturno", "comer")

# Versión "cámara única, sin sensor dedicado" (2026-09-05, ver dataset_events/activity_map.py::COARSE
# y "Vocabulario de cámara" en README): usa las columnas `cam_*` que agrega
# `features.add_camera_columns()`. `medicacion` no existe acá — queda fusionada dentro de
# `cam_comida` (con `beber`), así que una "ausencia" de `cam_comida` significa "no bebió ni tomó
# medicación en el período habitual", sin poder distinguir cuál — el mensaje al usuario final debe
# reflejar esa ambigüedad, no afirmar "se saltó la medicación" como si la cámara lo supiera.
# `cam_comida_mesa` (comer, separado de beber/medicación) **depende de la Fase 1 de zonas del
# hogar, todavía no implementada/validada** — se incluye igual acá para poder medir el mecanismo
# con desvíos sintéticos, pero no debe tratarse como alcanzable en producción hasta que esa fase
# se valide (ver nota en `activity_map.py::COARSE`).
CAMERA_TRACKED_CLASSES: tuple[str, ...] = ("cam_comida", "cam_comida_mesa", "cam_salir_casa", "cam_aseo")


@dataclass
class ClassDeviationConfig:
    window_days: int = 30
    # mismo período de silencio de producto que dynamic_clustering.py (7-10 días)
    min_window_days: int = 7
    excess_k: float = 3.0
    # la clase debe ocurrir en al menos esta fracción de los días de la ventana para que su
    # ausencia hoy cuente como desvío (si ya casi nunca ocurre, no ausentarse no es noticia).
    # Calibrado con cross-validation por hogar sobre los 21 hogares CASAS con historial
    # suficiente (`evaluate_deviations.py --cv 5`, ver "Estado" en README) — parámetros elegidos
    # por voto mayoritario entre folds (0.8/3.0 salió en 3 de 5), no por un único barrido sobre
    # todo el dataset (eso sobreestimaba el resultado: 78.1%/7.1% "en la misma data" vs. 71.4%/
    # 14.2% out-of-fold, la cifra real a citar).
    frac_historica_min: float = 0.8
    dias_consecutivos_ausencia: int = 2
    # 2026-09-20: separar la referencia histórica de un día por si es semana o fin de semana —
    # hoy la rutina de sábado se compara contra un historial que mezcla martes y sábados, así que
    # una diferencia semana/finde perfectamente normal (menos actividad un domingo) se lee como
    # "desvío" cuando no lo es. Idea de Snoun et al. 2024 (contexto: día de semana/feriados), acá
    # implementada con lo mínimo (solo semana/finde, sin feriados todavía). Si la ventana no tiene
    # suficientes días del mismo tipo (fin de semana es 2/7 días, ventanas cortas se quedan sin
    # muestra), cae de vuelta a usar todos los días de la ventana — señal ruidosa es mejor que
    # ninguna señal.
    separar_fin_de_semana: bool = True
    # 2026-09-20: cuántos días seguidos tiene que sostenerse un `exceso`/`deficit` (condición de
    # un solo día) antes de contar como desvío — default 1 = comportamiento original (dispara con
    # un solo día). Pensado para filtrar ruido puntual sin tocar `ausencia`, que ya tiene su propia
    # variante sostenida (`ausencia_sostenida`) y ya mide bien en solitario (ver "Estado" en
    # README). Se agregó al barrido de `evaluate_deviations.py` en vez de fijarlo a mano — mismo
    # criterio que `separar_fin_de_semana`: dejar que el cross-validation decida si ayuda.
    dias_consecutivos_exceso_deficit: int = 1
    # 2026-09-20: diagnóstico manual de FP concretos (ver "Estado" en README) encontró que ~44%
    # de los FP de clase eran `ausencia` de UN SOLO día en clases casi-diarias (ej. `bano_nocturno`
    # con mediana=1 casi todos los días) — si algo pasa el 95% de los días, no pasar 1 de cada 20
    # es esperable por azar, no un desvío. Con esto en `True`, `ausencia` exige la misma racha que
    # `ausencia_sostenida` (no dispara con un solo día) — cuesta recall en escenarios de un solo
    # día real (ej. `sin_medicacion`, hoy 88.9%). Trade-off deliberado, no una mejora gratis: se
    # sumó al barrido para que el cross-validation lo confirme o lo descarte con datos, en vez de
    # decidirlo a mano.
    ausencia_requiere_racha: bool = False


def score_class_deviations(
    day_vectors: pd.DataFrame,
    tracked_classes: tuple[str, ...] = TRACKED_CLASSES,
    config: ClassDeviationConfig | None = None,
    count_col_suffix: str = "_n",
    exclude_from_reference: set[tuple[str, object]] | None = None,
) -> pd.DataFrame:
    """Evalúa cada clase de `tracked_classes` por separado, día a día, por residente.

    Misma semántica de ventana/exclusión que `dynamic_clustering.score_days`: `window_days`
    previos, nunca incluye el día evaluado, y `exclude_from_reference` saca días (ej. inyectados)
    de la referencia sin dejar de evaluarlos.

    Devuelve una fila por `(hogar, fecha, clase)` con desvío detectado — no una fila por día
    evaluado (la mayoría de los días no van a tener ningún desvío en ninguna clase).
    """
    config = config or ClassDeviationConfig()
    exclude_from_reference = exclude_from_reference or set()

    rows: list[dict] = []
    for hogar, g in day_vectors.groupby(level="hogar", sort=False):
        g = g.droplevel("hogar").sort_index()
        g = g[~g["cobertura_baja"]]
        fechas = g.index.to_list()
        ref_ok = np.array([(hogar, f) not in exclude_from_reference for f in fechas])
        es_finde = np.array([pd.Timestamp(f).weekday() >= 5 for f in fechas])

        for cls in tracked_classes:
            col = f"{cls}{count_col_suffix}"
            if col not in g.columns:
                continue
            vals = g[col].to_numpy(dtype=float)
            racha_ausencia = 0
            racha_exceso = 0
            racha_deficit = 0

            for i, fecha in enumerate(fechas):
                hoy = vals[i]
                era_ausente = hoy == 0
                racha_ausencia = racha_ausencia + 1 if era_ausente else 0

                lo = max(0, i - config.window_days)
                idx = [j for j in range(lo, i) if ref_ok[j]]
                if config.separar_fin_de_semana:
                    idx_mismo_tipo = [j for j in idx if es_finde[j] == es_finde[i]]
                    if len(idx_mismo_tipo) >= config.min_window_days:
                        idx = idx_mismo_tipo
                if len(idx) < config.min_window_days:
                    continue

                hist = vals[idx]
                frac_presente = float((hist > 0).mean())
                mediana = float(np.median(hist))
                mad = float(np.median(np.abs(hist - mediana)))
                sigma_robusto = mad * _MAD_TO_STD

                ausencia = era_ausente and frac_presente >= config.frac_historica_min
                if config.ausencia_requiere_racha:
                    ausencia = ausencia and racha_ausencia >= config.dias_consecutivos_ausencia
                ausencia_sostenida = (
                    racha_ausencia >= config.dias_consecutivos_ausencia
                    and frac_presente >= config.frac_historica_min
                )
                exceso_hoy = (
                    not era_ausente
                    and sigma_robusto > 1e-9
                    and hoy > mediana + config.excess_k * sigma_robusto
                )
                # "omisión parcial": la clase sí ocurrió hoy, pero bastante menos de lo habitual
                # (ej. `comer_n=1` cuando lo normal son ~3 comidas) — distinto de `ausencia`
                # (=0) y del techo de dilución que tiene `dynamic_clustering.py` con clases de
                # bajo volumen (ver "Frecuencia/omisión de eventos" en README).
                deficit_hoy = (
                    not era_ausente
                    and sigma_robusto > 1e-9
                    and hoy < mediana - config.excess_k * sigma_robusto
                )
                racha_exceso = racha_exceso + 1 if exceso_hoy else 0
                racha_deficit = racha_deficit + 1 if deficit_hoy else 0
                exceso = exceso_hoy and racha_exceso >= config.dias_consecutivos_exceso_deficit
                deficit = deficit_hoy and racha_deficit >= config.dias_consecutivos_exceso_deficit

                if not (ausencia or ausencia_sostenida or exceso or deficit):
                    continue

                if ausencia_sostenida:
                    tipo = "ausencia_sostenida"
                elif ausencia:
                    tipo = "ausencia"
                elif exceso:
                    tipo = "exceso"
                else:
                    tipo = "deficit"
                rows.append(
                    {
                        "hogar": hogar,
                        "fecha": fecha,
                        "clase": cls,
                        "tipo": tipo,
                        "valor_hoy": hoy,
                        "mediana_historica": mediana,
                        "frac_presente_historica": frac_presente,
                        "racha_ausencia_dias": racha_ausencia,
                    }
                )

    return pd.DataFrame(rows)
