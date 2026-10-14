"""Validación cruzada por hogar del detector por clase con el vocabulario que producción puebla,
no con `CAMERA_TRACKED_CLASSES`, y las pruebas de robustez con ese mismo vocabulario.

`evaluate_deviations.py --cv 5 --camera` (69,5 % / 13,8 %) sigue `cam_comida`, `cam_comida_mesa`,
`cam_salir_casa` y `cam_aseo`, que no es lo que emite la controladora. Este script mide con las
clases que el proceso diario sigue de verdad (`camera_events.LIVE_TRACKED_CLASSES`) y que tienen
equivalente en CASAS: `cam_cocina`, `cam_comida`, `cam_descanso_sofa` y `cam_ocio_sentado`.
`cam_moverse` no tiene equivalente en CASAS y no se puede medir acá.

Con `--con-puerta` se suman `cam_salir_casa` y `cam_entrar_casa` y el desvío `semana_sin_salir`:
**es el vocabulario que quedó en producción** (entradas y salidas por la zona `puerta`,
encendidas por defecto, ver `camera_events.py`). Sin la opción se mide la variante sin puerta,
evaluada y no adoptada. La opción se llama así por el orden en que se hicieron las mediciones.

No modifica ni reemplaza `--cv 5`: usa su propio set de desvíos (en memoria, no pisa
`synthetic_eval/day_vectors_perturbed.parquet`) y el mismo protocolo — mismas particiones por
hogar (5, semilla 0), mismo barrido de `ClassDeviationConfig` (`evaluate_deviations._grid_configs()`),
mismo tope de falsos positivos al elegir (15 %) y, como en modo cámara, sin sacar los días
inyectados de la referencia. Con `--con-puerta` el barrido suma, para las dos clases de puerta,
cuántos días seguidos sin eventos hacen falta (`DIAS_PUERTA_GRID`) y si un solo día ya dispara
(`PUERTA_UN_DIA_GRID`); de esas clases solo sale la falta de eventos, como en el proceso diario.

Columnas `cam_*`: cada clase canónica de CASAS va a la clase que emitiría la controladora
(`evaluate_classifier_noise.CANONICAL_TO_CATEGORY` y después `CAMERA_ACTIVITY_MAP`, el mismo
camino que `evaluate_timing_cv.py`). No es `features.add_camera_columns()`: ahí `cam_comida` es
solo beber + medicación y `ver_tv` cuenta como ocio sentado; en producción comer, beber y
medicación son una sola clase y ver televisión es `Resting`. `cam_salir_casa`/`cam_entrar_casa`
llevan la cantidad de eventos de CASAS y 0 minutos (en producción son eventos puntuales).

Desvíos inyectados (uno por hogar y por tipo, en días distintos):
  - `dia_sin_comida`: comer, beber y medicación en 0.
  - `dia_sin_cocinar`: cocinar y lavar platos en 0.
  - `exceso_descanso`: los minutos de descansar + ver televisión suben a
    max(1,5 x percentil 95 del hogar, mediana + 180 min), con tope de 16 h. La cantidad de
    episodios no cambia: descansar más tiempo no implica sentarse más veces.
  - `exceso_ocio_sentado`: lo mismo con leer + trabajo + teléfono.
  - `semana_sin_salir` (solo con `--con-puerta`): 7 días seguidos sin salidas ni entradas (como
    en `inject_synthetic_deviations.py`), que no pisen ninguno de los días anteriores.
  Los dos excesos solo se inyectan en hogares donde la clase aparece al menos la mitad de los días.

Qué mide (sin `--paso1`):
  1. Validación cruzada por hogar del detector por clase, con 3 semillas de inyección.
  2. El juego de parámetros de producción: la combinación más votada entre las 15 particiones de
     entrenamiento (5 por semilla). Tiene que coincidir con `class_deviations.production_config()`.
  3. Con ese juego fijo, lo que corre `cloud/app/baseline_runner.py` (detector por clase,
     multivariado sobre las columnas `cam_*`, y los dos juntos), la regla de los indicadores de
     `cloud/app/rutina.py` (sedentarismo y cocina) y la "detección del sistema": un desvío
     cuenta si lo marca una alerta o el indicador que le corresponde (`INDICADOR_DEL_DESVIO`).
  4. Con ese juego fijo: error del clasificador (matriz de `evaluate_classifier_noise.py` y,
     con `--matriz`, también la de un archivo), período de aprendizaje (`min_window_days`) y,
     con `--con-puerta`, pérdida de eventos de puerta.

`--paso1` repite la medición del 2026-10-13 (5): las mismas cuatro clases y los mismos cuatro
desvíos, validación cruzada con episodios y con minutos, y producción con los defaults de
`ClassDeviationConfig()` que corrían hasta ese día.

Uso:
  py -3 evaluate_production_vocab.py                    # vocabulario de producción (barridos en synthetic_eval/prod_vocab_grid_sin_puerta_seed*.pkl)
  py -3 evaluate_production_vocab.py --con-puerta       # variante con entradas y salidas (prod_vocab_grid_seed*.pkl)
  py -3 evaluate_production_vocab.py --solo-seleccion   # solo la validación cruzada y el voto
  py -3 evaluate_production_vocab.py --con-puerta --matriz ../fusion/confusion_holdout_produccion.json
  py -3 evaluate_production_vocab.py --paso1
"""
from __future__ import annotations

import os

os.environ.setdefault("OMP_NUM_THREADS", "1")  # KMeans de a un hilo: mucho más rápido en ventanas chicas

import argparse
import json
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np
import pandas as pd

from cv_grid import cached_grid, cross_validate_from_grid
from evaluate_classifier_noise import CANONICAL_TO_CATEGORY, CATEGORIES, CONFUSION, load_events, surviving_events
from evaluate_deviations import DEFAULT_DIR, _grid_configs
from inject_synthetic_deviations import EVENTS_PARQUET, MIN_DIAS_POR_HOGAR, N_NEGATIVOS_POR_HOGAR, _all_homes
from src.camera_events import (
    CAMERA_ACTIVITY_MAP, DOOR_ACTIVITY_MAP, DOOR_TRACKED_CLASSES, LIVE_TRACKED_CLASSES, live_tracked_classes,
)
from src.class_deviations import _MAD_TO_STD, ClassDeviationConfig, production_config, score_class_deviations
from src.dynamic_clustering import DriftConfig, score_days
from src.features import build_day_vectors

# clase canónica de CASAS -> clase que emitiría la controladora
PRODUCTION_CLASS: dict[str, str] = {c: CAMERA_ACTIVITY_MAP[cat] for c, cat in CANONICAL_TO_CATEGORY.items()}
CLASSIFIER_TRACKED: tuple[str, ...] = tuple(sorted({f"cam_{v}" for v in PRODUCTION_CLASS.values()}))
DOOR_CLASSES: tuple[str, ...] = tuple(sorted(DOOR_ACTIVITY_MAP.values()))  # mismas clases canónicas en CASAS
DOOR_TRACKED: tuple[str, ...] = tuple(f"cam_{c}" for c in DOOR_CLASSES)
assert DOOR_TRACKED == DOOR_TRACKED_CLASSES
DOOR_VARIANT_TRACKED: tuple[str, ...] = tuple(sorted(CLASSIFIER_TRACKED + DOOR_TRACKED))
assert set(CLASSIFIER_TRACKED) <= set(live_tracked_classes(con_puerta=False))
assert set(DOOR_VARIANT_TRACKED) <= set(LIVE_TRACKED_CLASSES)
SIN_EQUIVALENTE_EN_CASAS = tuple(c for c in LIVE_TRACKED_CLASSES if c not in DOOR_VARIANT_TRACKED)

# escenario -> (tipo de desvío, clases de cámara que lo muestran)
SCENARIOS: dict[str, tuple[str, tuple[str, ...]]] = {
    "dia_sin_comida": ("ausencia", ("cam_comida",)),
    "dia_sin_cocinar": ("ausencia", ("cam_cocina",)),
    "exceso_descanso": ("exceso", ("cam_descanso_sofa",)),
    "exceso_ocio_sentado": ("exceso", ("cam_ocio_sentado",)),
    "semana_sin_salir": ("semana", DOOR_TRACKED),
}
EXCESO_FACTOR_P95 = 1.5
EXCESO_MIN_EXTRA = 180.0
EXCESO_TOPE_MIN = 960.0
DIAS_SEMANA_SIN_SALIR = 7

SIGNALS = {"conteo": "_n", "minutos": "_min"}
GRID_FIELDS = (
    "frac_historica_min", "excess_k", "dias_consecutivos_ausencia", "separar_fin_de_semana",
    "dias_consecutivos_exceso_deficit", "ausencia_requiere_racha",
)
# Con `--con-puerta` se barren además los dos parámetros de las clases de puerta. Cinco de las
# ocho combinaciones son el mismo detector: si un solo día dispara, la cantidad de días no
# cambia nada, y eso es lo mismo que pedir 1 día. Al elegir, entre candidatos idénticos gana el
# primero del barrido, así que ese comportamiento aparece siempre como (1 día, un solo día: no).
DOOR_GRID_FIELDS = ("dias_consecutivos_ausencia_puerta", "puerta_dispara_un_dia")
DIAS_PUERTA_GRID = (1, 2, 3, 4)
PUERTA_UN_DIA_GRID = (False, True)

# Desvío -> indicador de duración de `cloud/app/rutina.py` que lo mostraría. El día sin comida
# y la semana sin salir no tienen: el indicador `comida` es el horario de la primera comida (un
# día sin comidas no tiene valor que mostrar) y no hay indicador de salidas.
INDICADOR_DEL_DESVIO: dict[str, str] = {
    "dia_sin_cocinar": "cocina",
    "exceso_descanso": "sedentarismo",
    "exceso_ocio_sentado": "sedentarismo",
}
SUBTOTALES: dict[str, tuple[str, ...]] = {
    "ausencias": ("dia_sin_comida", "dia_sin_cocinar", "semana_sin_salir"),
    "excesos": ("exceso_descanso", "exceso_ocio_sentado"),
}

# `cloud/app/rutina.py`: `_EXCESO_K`, `_SIGMA_MIN_MINUTES` y `_INDICADORES_DURACION`.
RUTINA_K = 3.0
RUTINA_SIGMA_MIN = 15.0
RUTINA_INDICADORES = {
    "sedentarismo": (("cam_descanso_sofa_min", "cam_ocio_sentado_min"), "exceso"),
    "cocina": (("cam_cocina_min",), "deficit"),
}


def tracked_classes(con_puerta: bool = False) -> tuple[str, ...]:
    return DOOR_VARIANT_TRACKED if con_puerta else CLASSIFIER_TRACKED


def grid_fields(con_puerta: bool = False) -> tuple[str, ...]:
    return GRID_FIELDS + DOOR_GRID_FIELDS if con_puerta else GRID_FIELDS


def load_confusion(path: Path) -> np.ndarray:
    """Matriz de confusión (real \\ predicho) de un archivo como el que guarda
    `fusion/evaluate_hybrid.py --guardar-matriz`: `{"categorias": [...], "matriz": [[cuentas]]}`.
    Devuelve las filas normalizadas, en el orden de `CATEGORIES`."""
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    orden = [data["categorias"].index(c) for c in CATEGORIES]
    assert len(data["categorias"]) == len(CATEGORIES), f"categorías distintas: {data['categorias']}"
    matriz = np.array(data["matriz"], dtype=float)[np.ix_(orden, orden)]
    return matriz / matriz.sum(axis=1, keepdims=True)


def escenarios(con_puerta: bool = False) -> list[str]:
    return [e for e, (tipo, _) in SCENARIOS.items() if con_puerta or tipo != "semana"]


def _canonicas(clase_cam: str) -> list[str]:
    return [c for c, g in PRODUCTION_CLASS.items() if f"cam_{g}" == clase_cam]


def add_production_columns(day_vectors: pd.DataFrame, con_puerta: bool = False) -> pd.DataFrame:
    """Columnas `cam_*` como las vería producción (ver docstring del módulo)."""
    out = day_vectors.copy()
    for clase in CLASSIFIER_TRACKED:
        canonicas = _canonicas(clase)
        out[f"{clase}_min"] = out[[f"{c}_min" for c in canonicas]].sum(axis=1)
        out[f"{clase}_n"] = out[[f"{c}_n" for c in canonicas]].sum(axis=1)
    if con_puerta:
        for clase in DOOR_CLASSES:
            out[f"cam_{clase}_min"] = 0.0
            out[f"cam_{clase}_n"] = out[f"{clase}_n"]
    return out


def _inject_one_home(day_vectors: pd.DataFrame, hogar: str, rng: np.random.Generator, con_puerta: bool) -> list[dict]:
    """Perturba `day_vectors` (in-place, columnas canónicas) para `hogar`, devuelve las etiquetas."""
    good = day_vectors.xs(hogar, level="hogar")
    good = good[~good["cobertura_baja"]]
    fechas = good.index.to_list()
    if len(fechas) < MIN_DIAS_POR_HOGAR:
        return []

    labels: list[dict] = []
    tocados: set = set()
    for escenario, (tipo, clases) in SCENARIOS.items():
        if tipo == "semana":
            continue
        canonicas = _canonicas(clases[0])
        if tipo == "ausencia":
            total = good[[f"{c}_n" for c in canonicas]].sum(axis=1)
            piso = max(total.median(), 1)
            candidatos = [f for f in fechas if f not in tocados and total.loc[f] >= piso]
            if not candidatos:
                continue
            f = candidatos[rng.integers(len(candidatos))]
            for c in canonicas:
                day_vectors.loc[(hogar, f), f"{c}_n"] = 0
                day_vectors.loc[(hogar, f), f"{c}_min"] = 0.0
        else:
            total = good[[f"{c}_min" for c in canonicas]].sum(axis=1)
            mediana = float(total.median())
            if mediana <= 0:
                continue  # la clase no es habitual en este hogar
            objetivo = min(max(EXCESO_FACTOR_P95 * float(total.quantile(0.95)), mediana + EXCESO_MIN_EXTRA), EXCESO_TOPE_MIN)
            candidatos = [f for f in fechas if f not in tocados and 0 < total.loc[f] <= mediana and total.loc[f] < objetivo]
            if not candidatos:
                continue
            f = candidatos[rng.integers(len(candidatos))]
            factor = objetivo / float(total.loc[f])
            for c in canonicas:
                day_vectors.loc[(hogar, f), f"{c}_min"] = float(good.loc[f, f"{c}_min"]) * factor
        tocados.add(f)
        labels.append({"hogar": hogar, "fecha": f, "escenario": escenario, "es_positivo": True})

    if con_puerta:
        inicios = [
            i for i in range(20, len(fechas) - 10)
            if not tocados.intersection(fechas[i:i + DIAS_SEMANA_SIN_SALIR])
        ]
        if inicios:
            i = inicios[rng.integers(len(inicios))]
            for f in fechas[i:i + DIAS_SEMANA_SIN_SALIR]:
                for c in DOOR_CLASSES:
                    day_vectors.loc[(hogar, f), f"{c}_n"] = 0
                    day_vectors.loc[(hogar, f), f"{c}_min"] = 0.0
                tocados.add(f)
                labels.append({"hogar": hogar, "fecha": f, "escenario": "semana_sin_salir", "es_positivo": True})

    negativos = [f for f in fechas if f not in tocados]
    n_neg = min(N_NEGATIVOS_POR_HOGAR, len(negativos))
    if n_neg:
        for i in rng.choice(len(negativos), size=n_neg, replace=False):
            labels.append({"hogar": hogar, "fecha": negativos[i], "escenario": "real_sin_tocar", "es_positivo": False})
    return labels


def inject(day_vectors: pd.DataFrame, seed: int, con_puerta: bool = False) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Devuelve el vector de día perturbado (solo columnas canónicas, solo los hogares con
    etiquetas) y las etiquetas. Las columnas `cam_*` se agregan después, con
    `add_production_columns`, porque las pruebas de ruido y de puerta las arman distinto."""
    rng = np.random.default_rng(seed)
    perturbed = day_vectors.copy()
    labels: list[dict] = []
    for hogar in sorted(set(day_vectors.index.get_level_values("hogar"))):
        labels.extend(_inject_one_home(perturbed, hogar, rng, con_puerta))
    labels_df = pd.DataFrame(labels)
    homes = sorted(labels_df["hogar"].unique())
    return perturbed[perturbed.index.get_level_values("hogar").isin(homes)], labels_df


def contaminate(
    perturbed: pd.DataFrame, events: pd.DataFrame, seed: int, con_puerta: bool = False, confusion: np.ndarray | None = None,
) -> pd.DataFrame:
    """Columnas `cam_*` de producción bajo el error del clasificador: cada evento con equivalente
    se re-etiqueta al azar con la matriz de confusión de `evaluate_classifier_noise.py` (o con
    `confusion`, ver `load_confusion`) y pasa a la clase de cámara de la categoría predicha.
    Aparece `cam_moverse` (lo que el clasificador confunde con `Moving`). Las entradas y salidas
    no pasan por el clasificador: quedan igual."""
    confusion = CONFUSION if confusion is None else confusion
    rng = np.random.default_rng(seed)
    ev = surviving_events(events, perturbed)
    real = ev["actividad"].map(CANONICAL_TO_CATEGORY).to_numpy()
    predicha = np.empty(len(ev), dtype=object)
    for i, categoria in enumerate(CATEGORIES):
        mascara = real == categoria
        predicha[mascara] = rng.choice(CATEGORIES, size=int(mascara.sum()), p=confusion[i])
    ev["clase_limpia"] = ev["actividad"].map(PRODUCTION_CLASS)
    ev["clase_ruido"] = pd.Series(predicha, index=ev.index).map(CAMERA_ACTIVITY_MAP)

    out = add_production_columns(perturbed, con_puerta)
    out["cam_moverse_n"] = 0
    out["cam_moverse_min"] = 0.0
    movidos = ev[ev["clase_limpia"] != ev["clase_ruido"]]
    por_dia = movidos.groupby(["hogar", "fecha", "actividad", "clase_limpia", "clase_ruido"]).size().rename("n").reset_index()
    for fila in por_dia.itertuples():
        clave = (fila.hogar, fila.fecha)
        minutos = fila.n * perturbed.at[clave, f"{fila.actividad}_min"] / perturbed.at[clave, f"{fila.actividad}_n"]
        out.at[clave, f"cam_{fila.clase_limpia}_n"] -= fila.n
        out.at[clave, f"cam_{fila.clase_limpia}_min"] -= minutos
        out.at[clave, f"cam_{fila.clase_ruido}_n"] += fila.n
        out.at[clave, f"cam_{fila.clase_ruido}_min"] += minutos
    assert (out[[c for c in out.columns if c.startswith("cam_") and c.endswith("_n")]] >= 0).all().all()
    return out


def drop_door_events(perturbed: pd.DataFrame, loss: float, seed: int) -> pd.DataFrame:
    """Copia de `perturbed` donde cada salida y cada entrada se perdió con probabilidad `loss`
    (como `evaluate_door_event_loss.drop_door_events`). Solo eventos perdidos, no inventados."""
    rng = np.random.default_rng(seed)
    out = perturbed.copy()
    for clase in DOOR_CLASSES:
        antes = out[f"{clase}_n"].to_numpy(dtype=int)
        out[f"{clase}_n"] = rng.binomial(antes, 1.0 - loss).astype(out[f"{clase}_n"].dtype)
    return out


def _dias(hits: pd.DataFrame) -> set:
    return set(zip(hits["hogar"], hits["fecha"])) if len(hits) else set()


def evaluate_grid(perturbed: pd.DataFrame, labels: pd.DataFrame, suffix: str, tracked: tuple[str, ...]) -> dict:
    """Como `cv_grid.evaluate_grid(camera=True)`, con las clases `tracked` y la señal `suffix`.

    Si `tracked` incluye las clases de puerta, cada candidato de `_grid_configs()` se combina
    con `DIAS_PUERTA_GRID` x `PUERTA_UN_DIA_GRID` (`grid["campos"]` dice qué se barrió). Las
    clases de puerta se evalúan aparte y se unen: lo que marcan solo depende de sus dos
    parámetros, de `frac_historica_min` y de `separar_fin_de_semana`, y lo que marcan las demás
    clases no depende de los de puerta. Da lo mismo que evaluar cada combinación entera."""
    filas = labels.reset_index(drop=True).copy()
    claves = list(zip(filas["hogar"], filas["fecha"]))

    def _marcados(clases: tuple[str, ...], cfg: ClassDeviationConfig) -> np.ndarray:
        dias = _dias(score_class_deviations(
            perturbed, tracked_classes=clases, config=cfg, count_col_suffix=suffix, clases_de_puerta=DOOR_TRACKED,
        ))
        return np.array([clave in dias for clave in claves], dtype=bool)

    de_puerta = tuple(c for c in tracked if c in DOOR_TRACKED)
    resto = tuple(c for c in tracked if c not in DOOR_TRACKED)
    base = list(_grid_configs())
    if not de_puerta:
        configs = base
        hits = np.array([_marcados(resto, cfg) for cfg in base])
    else:
        opciones = [(d, u) for u in PUERTA_UN_DIA_GRID for d in DIAS_PUERTA_GRID]
        puerta: dict[tuple, np.ndarray] = {}
        for frac, sep in sorted({(c.frac_historica_min, c.separar_fin_de_semana) for c in base}):
            for dias_puerta, un_dia in opciones:
                cfg = ClassDeviationConfig(
                    frac_historica_min=frac, separar_fin_de_semana=sep,
                    dias_consecutivos_ausencia_puerta=dias_puerta, puerta_dispara_un_dia=un_dia,
                )
                puerta[(frac, sep, dias_puerta, un_dia)] = _marcados(de_puerta, cfg)
        configs, filas_hits = [], []
        for cfg in base:
            sin_puerta = _marcados(resto, cfg)
            for dias_puerta, un_dia in opciones:
                configs.append(replace(cfg, dias_consecutivos_ausencia_puerta=dias_puerta, puerta_dispara_un_dia=un_dia))
                filas_hits.append(sin_puerta | puerta[(cfg.frac_historica_min, cfg.separar_fin_de_semana, dias_puerta, un_dia)])
        hits = np.array(filas_hits)
    return {
        "filas": filas, "configs": [asdict(c) for c in configs], "hits": hits,
        "homes": sorted(labels["hogar"].unique()), "etiquetas": labels[["hogar", "es_positivo"]].reset_index(drop=True),
        "campos": grid_fields(bool(de_puerta)),
    }


def _folds(homes: list[str], n_folds: int, seed: int = 0) -> list[np.ndarray]:
    """Las mismas particiones que `evaluate_deviations.cross_validate`."""
    rng = np.random.default_rng(seed)
    mezclados = list(homes)
    rng.shuffle(mezclados)
    return np.array_split(mezclados, n_folds)


def _breakdown(hits: pd.DataFrame, labels: pd.DataFrame, tracked: tuple[str, ...]) -> dict:
    """Qué clase disparó cada alerta: detección por la clase del desvío (no por otra que
    coincidió ese día) y falsos positivos por clase y por tipo."""
    pos, neg = labels[labels["es_positivo"]], labels[~labels["es_positivo"]]
    por_clase = {clase: _dias(hits[hits["clase"] == clase]) for clase in tracked}
    propia = {}
    for escenario, (_, clases) in SCENARIOS.items():
        dias = pos[pos["escenario"] == escenario]
        if len(dias):
            marcados = set().union(*(por_clase.get(c, set()) for c in clases))
            propia[escenario] = float(np.mean([(h, f) in marcados for h, f in zip(dias["hogar"], dias["fecha"])]))
    claves_neg = list(zip(neg["hogar"], neg["fecha"]))
    return {
        "recall_por_clase_del_desvio": propia,
        "fp_por_clase": {clase: float(np.mean([k in dias for k in claves_neg])) for clase, dias in por_clase.items()},
        "fp_por_tipo": {
            tipo: float(np.mean([k in _dias(hits[hits["tipo"] == tipo]) for k in claves_neg]))
            for tipo in ("ausencia", "ausencia_sostenida", "exceso", "deficit")
        },
    }


def _breakdown_cv(perturbed: pd.DataFrame, labels: pd.DataFrame, fold_df: pd.DataFrame, suffix: str, tracked: tuple[str, ...], n_folds: int) -> dict:
    """`_breakdown` con los parámetros elegidos en cada partición, sobre sus hogares de prueba."""
    campos = set(ClassDeviationConfig.__dataclass_fields__)
    partes = []
    for fila, test_homes in zip(fold_df.to_dict(orient="records"), _folds(sorted(labels["hogar"].unique()), n_folds)):
        cfg = ClassDeviationConfig(**{k: v for k, v in fila.items() if k in campos})
        sub = perturbed[perturbed.index.get_level_values("hogar").isin(test_homes.tolist())]
        partes.append(score_class_deviations(sub, tracked_classes=tracked, config=cfg, count_col_suffix=suffix))
    partes = [p for p in partes if len(p)]
    hits = pd.concat(partes, ignore_index=True) if partes else pd.DataFrame(columns=["hogar", "fecha", "clase", "tipo"])
    return _breakdown(hits, labels, tracked)


def _tasas(labels: pd.DataFrame, dias: set) -> dict:
    hit = np.array([(h, f) in dias for h, f in zip(labels["hogar"], labels["fecha"])])
    pos = labels["es_positivo"].to_numpy(dtype=bool)
    por_escenario = pd.Series(hit[pos], index=labels.loc[pos, "escenario"].to_numpy()).groupby(level=0)
    return {
        "n_pos": int(pos.sum()), "n_neg": int((~pos).sum()),
        "detectados": int(hit[pos].sum()), "falsos_positivos": int(hit[~pos].sum()),
        "detectados_por_escenario": por_escenario.sum().astype(int).to_dict(),
        "n_por_escenario": por_escenario.size().to_dict(),
    }


def production_days(
    perturbed: pd.DataFrame, tracked: tuple[str, ...],
    class_config: ClassDeviationConfig, drift_config: DriftConfig | None = None,
) -> tuple[pd.DataFrame, set, set]:
    """Lo que corre `baseline_runner.process_hogar`, con parámetros fijos: las filas del detector
    por clase sobre `tracked` (de las clases de puerta, solo la falta de eventos), los días que
    marca y los que marca el multivariado sobre todas las columnas `cam_*`."""
    hits = score_class_deviations(perturbed, tracked_classes=tracked, config=class_config, clases_de_puerta=DOOR_TRACKED)
    cam_cols = [c for c in perturbed.columns if c.startswith("cam_")]
    scored = score_days(perturbed[cam_cols + ["cobertura_baja"]], drift_config or DriftConfig())
    if not len(hits):
        hits = pd.DataFrame(columns=["hogar", "fecha", "clase", "tipo"])
    return hits, _dias(hits), _dias(scored[scored["es_desvio"]])


def evaluate_production(
    perturbed: pd.DataFrame, labels: pd.DataFrame, tracked: tuple[str, ...],
    class_config: ClassDeviationConfig, drift_config: DriftConfig | None = None,
) -> dict:
    """Detección y falsos positivos de `production_days`: el detector por clase, el
    multivariado y los dos juntos (lo que emite el proceso diario)."""
    hits, clase, multivariado = production_days(perturbed, tracked, class_config, drift_config)
    return {
        "por_clase": {**_tasas(labels, clase), **_breakdown(hits, labels, tracked)},
        "multivariado": _tasas(labels, multivariado),
        "combinado": _tasas(labels, clase | multivariado),
    }


def evaluate_sistema(labels: pd.DataFrame, alertas: set, rutina: dict[str, set]) -> dict:
    """"Detección del sistema": un día con desvío cuenta como detectado si lo marca una alerta
    (`alertas`: los días del detector por clase o del multivariado) o el indicador de
    `rutina.py` que le corresponde (`INDICADOR_DEL_DESVIO`); el día sin comida y la semana sin
    salir no tienen indicador, así que solo cuentan por alerta. Un día normal es un falso
    positivo si lo marca una alerta o cualquiera de los indicadores."""
    cualquiera = set().union(*rutina.values())
    dias = set()
    for fila in labels.itertuples():
        clave = (fila.hogar, fila.fecha)
        del_indicador = rutina.get(INDICADOR_DEL_DESVIO.get(fila.escenario), ()) if fila.es_positivo else cualquiera
        if clave in alertas or clave in del_indicador:
            dias.add(clave)
    return _tasas(labels, dias)


def rutina_days(perturbed: pd.DataFrame, labels: pd.DataFrame) -> dict[str, set]:
    """Regla de los indicadores de duración de `cloud/app/rutina.py`: el día contra la mediana
    del resto de los días del hogar, con sigma robusto y piso de 15 minutos. Devuelve, por
    indicador, los días de `labels` que marca."""
    marcados: dict[str, set] = {clave: set() for clave in RUTINA_INDICADORES}
    for hogar, dias in labels.groupby("hogar"):
        g = perturbed.xs(hogar, level="hogar")
        g = g[~g["cobertura_baja"]]
        for clave, (columnas, direccion) in RUTINA_INDICADORES.items():
            totales = g[list(columnas)].sum(axis=1)
            for fecha in dias["fecha"]:
                historicos = totales.drop(fecha).to_numpy()
                mediana = float(np.median(historicos))
                sigma = max(float(np.median(np.abs(historicos - mediana))) * _MAD_TO_STD, RUTINA_SIGMA_MIN)
                valor = float(totales.loc[fecha])
                desvio = valor > mediana + RUTINA_K * sigma if direccion == "exceso" else valor < mediana - RUTINA_K * sigma
                if desvio:
                    marcados[clave].add((hogar, fecha))
    return marcados


def evaluate_rutina(perturbed: pd.DataFrame, labels: pd.DataFrame) -> dict:
    """Detección y falsos positivos de `rutina_days`, por indicador y con cualquiera de los dos."""
    marcados = rutina_days(perturbed, labels)
    return {
        **{clave: _tasas(labels, dias) for clave, dias in marcados.items()},
        "cualquiera": _tasas(labels, set().union(*marcados.values())),
    }


def _base(events_parquet: str, seed: int, con_puerta: bool) -> tuple[pd.DataFrame, pd.DataFrame]:
    parquet = Path(events_parquet)
    return inject(build_day_vectors(parquet, homes=_all_homes(parquet)), seed, con_puerta)


def _run_grid(job: tuple[int, str, str, bool]) -> tuple[int, dict]:
    seed, events_parquet, out_dir, con_puerta = job

    def _compute() -> dict:
        perturbed, labels = _base(events_parquet, seed, con_puerta)
        return evaluate_grid(add_production_columns(perturbed, con_puerta), labels, "_n", tracked_classes(con_puerta))

    # `_puerta`: el barrido con los parámetros de puerta (2026-10-13 (8)); los
    # `prod_vocab_grid_seed*.pkl` de antes no los tienen y no se reusan.
    sufijo = "_puerta" if con_puerta else "_sin_puerta"
    return seed, cached_grid(Path(out_dir) / f"prod_vocab_grid{sufijo}_seed{seed}.pkl", _compute)


def _run_fixed(job: tuple[str, int, float | None, int | None, str, bool, str | None]) -> dict:
    """Una medición con parámetros fijos. `prueba`: "base", "ruido" (`azar` = semilla del ruido;
    con la matriz de `evaluate_classifier_noise.py`), "ruido_matriz" (lo mismo con la matriz del
    archivo `matriz`), "aprendizaje" (`valor` = `min_window_days`) o "puerta" (`valor` =
    fracción perdida, `azar` = semilla de la pérdida)."""
    prueba, seed, valor, azar, events_parquet, con_puerta, matriz = job
    perturbed, labels = _base(events_parquet, seed, con_puerta)
    class_config, drift_config = production_config(con_puerta), DriftConfig()
    if prueba in ("ruido", "ruido_matriz"):
        confusion = load_confusion(Path(matriz)) if prueba == "ruido_matriz" else None
        eventos = load_events(Path(events_parquet), sorted(labels["hogar"].unique()))
        columnas = contaminate(perturbed, eventos, azar, con_puerta, confusion)
    elif prueba == "puerta":
        columnas = add_production_columns(drop_door_events(perturbed, valor, azar), con_puerta)
    else:
        columnas = add_production_columns(perturbed, con_puerta)
    if prueba == "aprendizaje":
        class_config = replace(class_config, min_window_days=int(valor))
        drift_config = DriftConfig(min_window_days=int(valor))
    out = {"prueba": prueba, "seed": seed, "valor": valor, "azar": azar}
    tracked = tracked_classes(con_puerta)
    hits, clase, multivariado = production_days(columnas, tracked, class_config, drift_config)
    out.update({
        "por_clase": {**_tasas(labels, clase), **_breakdown(hits, labels, tracked)},
        "multivariado": _tasas(labels, multivariado),
        "combinado": _tasas(labels, clase | multivariado),
    })
    if prueba in ("base", "ruido", "ruido_matriz"):
        marcados = rutina_days(columnas, labels)
        out["rutina"] = {
            **{clave: _tasas(labels, dias) for clave, dias in marcados.items()},
            "cualquiera": _tasas(labels, set().union(*marcados.values())),
        }
        out["sistema"] = evaluate_sistema(labels, clase | multivariado, marcados)
    return out


def select_production_config(grids: dict[int, dict], n_folds: int) -> dict:
    """El juego de parámetros de producción: la combinación que más particiones de entrenamiento
    eligieron (5 por semilla de inyección). Solo usa lo que cada partición vio al elegir, nunca
    sus hogares de prueba. Empate: la de menos falsos positivos promedio en entrenamiento."""
    votos: Counter = Counter()
    fp_train: dict[tuple, list[float]] = {}
    cv = {}
    for seed, grid in sorted(grids.items()):
        fold_df, pooled = cross_validate_from_grid(grid, n_folds=n_folds)
        cv[seed] = {"pooled": pooled, "folds": fold_df.to_dict(orient="records")}
        hogar = grid["filas"]["hogar"].to_numpy()
        es_neg = ~grid["filas"]["es_positivo"].to_numpy(dtype=bool)
        campos = tuple(grid.get("campos", GRID_FIELDS))
        for fila, test_homes in zip(cv[seed]["folds"], _folds(grid["homes"], n_folds)):
            elegido_en_la_particion = grid["configs"][fila["candidato"]]
            fila.update({k: elegido_en_la_particion[k] for k in campos})
            clave = tuple(fila[k] for k in campos)
            votos[clave] += 1
            en_train = ~np.isin(hogar, test_homes.tolist())
            fp_train.setdefault(clave, []).append(float(grid["hits"][fila["candidato"]][en_train & es_neg].mean()))
    elegido = min(votos, key=lambda k: (-votos[k], float(np.mean(fp_train[k]))))
    return {
        "elegido": dict(zip(campos, elegido)), "votos": votos[elegido], "particiones": sum(votos.values()),
        "votacion": [{**dict(zip(campos, k)), "votos": v, "fp_entrenamiento": float(np.mean(fp_train[k]))} for k, v in votos.most_common()],
        "cv": cv,
    }


def cv_por_dias_de_puerta(grids: dict[int, dict], n_folds: int) -> dict[tuple, list[dict]]:
    """La misma validación cruzada con los parámetros de puerta fijados de antemano: para cada
    combinación de `DIAS_PUERTA_GRID` x `PUERTA_UN_DIA_GRID`, las particiones eligen solo entre
    los candidatos que la tienen. Es lo que da el procedimiento, fuera de muestra, con cada
    opción de puerta; no elige nada. Devuelve, por opción, una medición por semilla."""
    out: dict[tuple, list[dict]] = {}
    for dias_puerta in DIAS_PUERTA_GRID:
        for un_dia in PUERTA_UN_DIA_GRID:
            for seed, grid in sorted(grids.items()):
                cuales = [
                    i for i, c in enumerate(grid["configs"])
                    if (c["dias_consecutivos_ausencia_puerta"], c["puerta_dispara_un_dia"]) == (dias_puerta, un_dia)
                ]
                sub = {**grid, "configs": [grid["configs"][i] for i in cuales], "hits": grid["hits"][cuales]}
                _, pooled = cross_validate_from_grid(sub, n_folds=n_folds)
                out.setdefault((dias_puerta, un_dia), []).append(_cv_como_tasas(sub, pooled))
    return out


def _pct(a: int, b: int) -> str:
    return f"{100 * a / b:.1f} % ({a}/{b})" if b else "—"


def _fila(nombre: str, tasas: list[dict], con_puerta: bool = False) -> dict:
    """Una fila de tabla sumando las mediciones `tasas` (una por semilla)."""
    det, pos = sum(t["detectados"] for t in tasas), sum(t["n_pos"] for t in tasas)
    fp, neg = sum(t["falsos_positivos"] for t in tasas), sum(t["n_neg"] for t in tasas)
    fila = {"medición": nombre, "detección": _pct(det, pos), "falsos positivos": _pct(fp, neg)}
    for escenario in escenarios(con_puerta):
        n = sum(t["n_por_escenario"].get(escenario, 0) for t in tasas)
        fila[escenario] = _pct(sum(t["detectados_por_escenario"].get(escenario, 0) for t in tasas), n)
    return fila


def _fila_subtotales(nombre: str, tasas: list[dict]) -> dict:
    """Detección global, la de cada grupo de `SUBTOTALES` y falsos positivos."""
    fila = {"medición": nombre}
    for grupo, del_grupo in SUBTOTALES.items():
        n = sum(t["n_por_escenario"].get(e, 0) for t in tasas for e in del_grupo)
        fila[grupo] = _pct(sum(t["detectados_por_escenario"].get(e, 0) for t in tasas for e in del_grupo), n)
    fila["global"] = _pct(sum(t["detectados"] for t in tasas), sum(t["n_pos"] for t in tasas))
    fila["falsos positivos"] = _pct(sum(t["falsos_positivos"] for t in tasas), sum(t["n_neg"] for t in tasas))
    return fila


def _cv_como_tasas(grid: dict, pooled: dict) -> dict:
    n = pooled["n_por_escenario"]
    return {
        "n_pos": pooled["n_pos"], "n_neg": pooled["n_neg"],
        "detectados": pooled["detectados"], "falsos_positivos": pooled["falsos_positivos"],
        "n_por_escenario": n,
        "detectados_por_escenario": {e: round(v * n[e]) for e, v in pooled["recall_por_escenario"].items()},
    }


def _run_paso1(job: tuple[int, str, int]) -> dict:
    seed, events_parquet, n_folds = job
    perturbed, labels = _base(events_parquet, seed, con_puerta=False)
    columnas = add_production_columns(perturbed, con_puerta=False)
    out = {
        "seed": seed, "n_por_escenario": labels["escenario"].value_counts().to_dict(),
        "produccion": evaluate_production(columnas, labels, CLASSIFIER_TRACKED, ClassDeviationConfig()),
        "rutina": evaluate_rutina(columnas, labels),
    }
    for nombre, suffix in SIGNALS.items():
        grid = evaluate_grid(columnas, labels, suffix, CLASSIFIER_TRACKED)
        fold_df, pooled = cross_validate_from_grid(grid, n_folds=n_folds)
        out[nombre] = {
            "pooled": pooled, "folds": fold_df.to_dict(orient="records"), "tasas": _cv_como_tasas(grid, pooled),
            **_breakdown_cv(columnas, labels, fold_df, suffix, CLASSIFIER_TRACKED, n_folds),
        }
    return out


def main_paso1(args) -> None:
    print(f"Medición del 2026-10-13 (5): clases {CLASSIFIER_TRACKED}, sin entradas ni salidas", flush=True)
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        results = list(pool.map(_run_paso1, [(s, str(args.events_parquet), args.cv) for s in args.seeds]))
    for r in results:
        print(f"\n=== semilla de inyección {r['seed']}: {r['n_por_escenario']} ===")
        filas = [_fila(f"CV {nombre}", [r[nombre]["tasas"]]) for nombre in SIGNALS]
        filas += [_fila(f"producción {nombre}", [t]) for nombre, t in r["produccion"].items()]
        filas += [_fila(f"rutina.py {nombre}", [t]) for nombre, t in r["rutina"].items()]
        print(pd.DataFrame(filas).to_string(index=False))
        for nombre in SIGNALS:
            print(f"  CV {nombre} — por la clase del desvío: {r[nombre]['recall_por_clase_del_desvio']}")
            print(f"  CV {nombre} — FP por clase: {r[nombre]['fp_por_clase']}  por tipo: {r[nombre]['fp_por_tipo']}")
    out_path = args.out_dir / "production_vocab_paso1_results.json"
    out_path.write_text(json.dumps(results, indent=2, default=str), encoding="utf-8")
    print(f"\nGuardado: {out_path}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--events-parquet", type=Path, default=EVENTS_PARQUET)
    ap.add_argument("--out-dir", type=Path, default=DEFAULT_DIR)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2], help="semillas de inyección")
    ap.add_argument("--cv", type=int, default=5, metavar="N_FOLDS")
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--paso1", action="store_true", help="la medición del 2026-10-13 (5), con los defaults de entonces")
    ap.add_argument("--con-puerta", action="store_true", help="el vocabulario de producción: con entradas y salidas de casa")
    ap.add_argument("--matriz", type=Path, default=None, help="repetir la prueba del error del clasificador con la matriz de confusión de este archivo (ver load_confusion)")
    ap.add_argument("--solo-seleccion", action="store_true", help="solo la validación cruzada y el voto de parámetros")
    ap.add_argument("--noise-seeds", type=int, nargs="+", default=[0, 1, 2], help="semillas del ruido y de la pérdida de puerta")
    ap.add_argument("--days", type=int, nargs="+", default=[7, 14, 21, 30], help="valores de min_window_days")
    ap.add_argument("--loss", type=float, nargs="+", default=[0.1, 0.2, 0.3], help="fracción de eventos de puerta perdidos")
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    pd.set_option("display.width", 260)
    if args.paso1:
        return main_paso1(args)

    parquet = str(args.events_parquet)
    con_puerta = args.con_puerta
    _f = lambda nombre, tasas: _fila(nombre, tasas, con_puerta)  # noqa: E731
    print(f"Clases seguidas: {tracked_classes(con_puerta)}" + ("  (variante con entradas y salidas)" if con_puerta else ""))
    print(f"Sin equivalente en CASAS, no se miden: {SIN_EQUIVALENTE_EN_CASAS}", flush=True)
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        grids = dict(pool.map(_run_grid, [(s, parquet, str(args.out_dir), con_puerta) for s in args.seeds]))

    seleccion = select_production_config(grids, args.cv)
    print(f"\nValidación cruzada por hogar ({args.cv} particiones), detector por clase, fuera de muestra:")
    cv_tasas = [_cv_como_tasas(grids[s], seleccion["cv"][s]["pooled"]) for s in args.seeds]
    filas = [_f(f"semilla {s}", [t]) for s, t in zip(args.seeds, cv_tasas)] + [_f("las 3 semillas", cv_tasas)]
    print(pd.DataFrame(filas).to_string(index=False))
    print(f"\nVotación de parámetros ({seleccion['particiones']} particiones de entrenamiento):")
    print(pd.DataFrame(seleccion["votacion"]).to_string(index=False))
    actual = {k: getattr(production_config(con_puerta), k) for k in grid_fields(con_puerta)}
    print(f"\nElegido: {seleccion['elegido']}  ({seleccion['votos']} de {seleccion['particiones']} votos)")
    print(f"class_deviations.production_config({con_puerta}): {actual}  -> {'coincide' if actual == seleccion['elegido'] else 'NO COINCIDE'}")
    results = {"seleccion": seleccion}
    if con_puerta:
        print("\nValidación cruzada con los parámetros de puerta fijados (fuera de muestra, 3 semillas; no elige nada):")
        por_puerta = cv_por_dias_de_puerta(grids, args.cv)
        filas = [
            _f(f"{d} día(s) seguidos, un solo día dispara: {'sí' if u else 'no'}", tasas)
            for (d, u), tasas in por_puerta.items()
        ]
        print(pd.DataFrame(filas).to_string(index=False))
        results["cv_por_dias_de_puerta"] = [{"dias": d, "un_dia": u, "tasas": t} for (d, u), t in por_puerta.items()]
    if args.solo_seleccion:
        return

    matriz = str(args.matriz) if args.matriz else None
    jobs = [("base", s, None, None, parquet, con_puerta, None) for s in args.seeds]
    jobs += [("ruido", s, None, a, parquet, con_puerta, None) for s in args.seeds for a in args.noise_seeds]
    if matriz:
        jobs += [("ruido_matriz", s, None, a, parquet, con_puerta, matriz) for s in args.seeds for a in args.noise_seeds]
    jobs += [("aprendizaje", s, d, None, parquet, con_puerta, None) for d in args.days for s in args.seeds]
    if con_puerta:
        jobs += [("puerta", s, p, a, parquet, con_puerta, None) for p in args.loss for s in args.seeds for a in args.noise_seeds]
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        fijos = list(pool.map(_run_fixed, jobs))
    results["fijos"] = fijos

    def _de(prueba: str, **filtro) -> list[dict]:
        return [r for r in fijos if r["prueba"] == prueba and all(r[k] == v for k, v in filtro.items())]

    base = _de("base")
    print("\n=== Parámetros de producción fijos, 3 semillas de inyección (alertas) ===")
    filas = [_f(f"por clase, semilla {r['seed']}", [r["por_clase"]]) for r in base]
    filas += [_f(f"{nombre}, las 3 semillas", [r[nombre] for r in base]) for nombre in ("por_clase", "multivariado", "combinado")]
    print(pd.DataFrame(filas).to_string(index=False))
    for r in base:
        d = r["por_clase"]
        print(f"  semilla {r['seed']} — por la clase del desvío: {d['recall_por_clase_del_desvio']}")
        print(f"  semilla {r['seed']} — FP por clase: {d['fp_por_clase']}  por tipo: {d['fp_por_tipo']}")
    print("\n=== Indicadores de rutina.py, 3 semillas de inyección ===")
    print(pd.DataFrame([_f(nombre, [r["rutina"][nombre] for r in base]) for nombre in base[0]["rutina"]]).to_string(index=False))

    print("\n=== Detección del sistema: alerta (por clase o multivariado) o el indicador de rutina.py que corresponde ===")
    filas = [_f("solo alertas (los dos detectores)", [r["combinado"] for r in base]), _f("sistema (alertas o indicador)", [r["sistema"] for r in base])]
    filas += [_f(f"sistema, semilla {r['seed']}", [r["sistema"]]) for r in base]
    print(pd.DataFrame(filas).to_string(index=False))
    filas = [_fila_subtotales("solo alertas (los dos detectores)", [r["combinado"] for r in base]), _fila_subtotales("sistema (alertas o indicador)", [r["sistema"] for r in base])]
    print(pd.DataFrame(filas).to_string(index=False))

    print("\n=== Error del clasificador (parámetros fijos; 3 semillas de inyección x semillas de ruido) ===")
    ruido = _de("ruido")
    filas = []
    for nombre in ("por_clase", "multivariado", "combinado"):
        filas.append(_f(f"{nombre}, sin ruido", [r[nombre] for r in base]))
        filas.append(_f(f"{nombre}, con ruido", [r[nombre] for r in ruido]))
        filas += [_f(f"{nombre}, ruido semilla {a}", [r[nombre] for r in ruido if r["azar"] == a]) for a in args.noise_seeds]
    filas.append(_f("rutina.py cualquiera, sin ruido", [r["rutina"]["cualquiera"] for r in base]))
    filas.append(_f("rutina.py cualquiera, con ruido", [r["rutina"]["cualquiera"] for r in ruido]))
    filas.append(_f("sistema, sin ruido", [r["sistema"] for r in base]))
    filas.append(_f("sistema, con ruido", [r["sistema"] for r in ruido]))
    print(pd.DataFrame(filas).to_string(index=False))

    if matriz:
        print(f"\n=== Error del clasificador con la matriz de {matriz} (3 semillas de inyección x semillas de ruido) ===")
        print(pd.DataFrame(load_confusion(args.matriz) * 100, index=CATEGORIES, columns=CATEGORIES).round(1).to_string())
        nuevo = _de("ruido_matriz")
        filas = []
        for nombre, titulo in (("por_clase", "por clase"), ("combinado", "los dos juntos"), ("sistema", "sistema")):
            filas.append(_f(f"{titulo}, sin ruido", [r[nombre] for r in base]))
            filas.append(_f(f"{titulo}, matriz anterior", [r[nombre] for r in ruido]))
            filas.append(_f(f"{titulo}, matriz del archivo", [r[nombre] for r in nuevo]))
            filas += [_f(f"{titulo}, matriz del archivo, ruido semilla {a}", [r[nombre] for r in nuevo if r["azar"] == a]) for a in args.noise_seeds]
        filas.append(_f("rutina.py cualquiera, matriz del archivo", [r["rutina"]["cualquiera"] for r in nuevo]))
        print(pd.DataFrame(filas).to_string(index=False))

    print("\n=== Período de aprendizaje (parámetros fijos, 3 semillas de inyección) ===")
    filas = [_f(f"{nombre}, {d} días", [r[nombre] for r in _de("aprendizaje", valor=d)]) for nombre in ("por_clase", "multivariado", "combinado") for d in args.days]
    print(pd.DataFrame(filas).to_string(index=False))

    if con_puerta:
        print("\n=== Pérdida de eventos de puerta (parámetros fijos; 3 semillas de inyección x semillas de pérdida) ===")
        filas = []
        for nombre in ("por_clase", "combinado"):
            filas.append(_f(f"{nombre}, 0 %", [r[nombre] for r in base]))
            filas += [_f(f"{nombre}, {round(100 * p)} %", [r[nombre] for r in _de("puerta", valor=p)]) for p in args.loss]
        print(pd.DataFrame(filas).to_string(index=False))

    out_path = args.out_dir / ("production_vocab_con_puerta_results.json" if con_puerta else "production_vocab_results.json")
    out_path.write_text(json.dumps(results, indent=2, default=str), encoding="utf-8")
    print(f"\nGuardado: {out_path}")


if __name__ == "__main__":
    main()
