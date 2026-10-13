"""Adaptador: eventos reales de `events/events.db` (lo que emite `runner/` en vivo) -> los
mismos formatos de entrada que ya consumen `dynamic_clustering.py` / `class_deviations.py` /
`timing_deviations.py` / `trend_deviations.py` (hasta ahora alimentados solo con
`features.build_day_vectors()` sobre CASAS o con desvíos sintéticos).

Dos responsabilidades, ambas a partir de la misma lista de `MonitoringEvent` (event_type=ACTIVITY):

- `build_day_vectors_from_events`: vector-día (duración/conteo por clase), mismo contrato que
  `features.build_day_vectors` — para `dynamic_clustering.score_days`, `class_deviations.
  score_class_deviations` y `trend_deviations.score_trend_deviations`.
- `build_events_df_from_events`: un registro por ocurrencia (no por heartbeat), con su horario —
  para `timing_deviations.score_timing_deviations`, que trabaja a nivel de evento, no de día.

Vocabulario de cámara (ver "Vocabulario de cámara" en `baseline/README.md`, 2026-09-05): de las 9
clases gruesas que espera `baseline/` (`activity_map.py::COARSE`), solo 5 son alcanzables con lo
que `vlm/`/`fusion/` emiten hoy (6 categorías finas, sin zona de mesa ni detector de puerta) —
`comida_mesa`, `dormir` (requeriría distinguir de `descanso_sofa` por postura/horario), `aseo`,
`salir_casa`, `entrar_casa` quedan sin poblar en esta primera versión, igual que documenta el
roadmap. `Moving` no tiene análogo en `COARSE` (es camera-only, ver nota en `activity_map.py`) y
se agrega acá como clase extra `moverse`.

Entradas y salidas (2026-10-13): **encendidas por defecto**. La controladora emite
`LeaveHome`/`EnterHome` a partir de la zona `puerta` (`runner/door_events.py`, opción
`puerta_activada` de `cameras.yaml`) y este adaptador las lleva a `salir_casa` / `entrar_casa`
(`DOOR_ACTIVITY_MAP`). Son eventos puntuales: suman a la cantidad del día (`_n`), nunca a los
minutos, y no cortan ni alargan la duración de la actividad que los rodea. Una cámara sin zona
`puerta` no los emite y esas dos clases quedan siempre en cero. **Solo generan un tipo de
desvío: la ausencia sostenida** (varios días sin salir o sin entrar). Nunca por horario, ni por
un solo día, ni por exceso, déficit o tendencia: ver `solo_ausencia_sostenida_en_puerta` y
`sin_clases_de_puerta`, que usa el proceso diario. Se apagan con `ENTRADAS_Y_SALIDAS_ACTIVADAS=0`
(o `con_puerta=False`): ahí esas etiquetas se descartan como cualquier etiqueta desconocida y
las dos clases no existen. El detector de puerta no está validado con video (ver "Actualización
2026-10-13 (6)" en `baseline/README.md`).

Zona horaria (`tz`, 2026-10-09): la fecha y el minuto del día de cada evento salen de
`wall_clock_ts` (epoch UTC) convertido a hora local. Sin `tz`, se usa la zona de la máquina
(`datetime.fromtimestamp`), que es lo correcto en la Pi (está en la casa del residente). En la
nube el contenedor corre en UTC, así que `cloud/` pasa `tz` explícito (Argentina): si no, una
toma de las 22:00 caía a la 01:00 del día siguiente y los días cortaban a las 21:00.

Hogar: v1 usa el `video_id` (id de cámara) directamente como clave de hogar — el alcance actual
del proyecto es una sola cámara por hogar (ver README raíz), así que no hace falta un concepto de
"hogar" separado todavía.
"""
from __future__ import annotations

import sys
import os
from datetime import date, datetime, tzinfo
from pathlib import Path

import pandas as pd

_BASELINE_DIR = Path(__file__).resolve().parents[1]
_EVENTS_DIR = _BASELINE_DIR.parent / "events"
if str(_EVENTS_DIR) not in sys.path:
    sys.path.insert(0, str(_EVENTS_DIR))

from schema import EventType, MonitoringEvent  # noqa: E402  (de events/)

# activity_label (CLIP, ver vlm/src/clip_classifier.py::CATEGORY_DESCRIPTIONS) -> clase gruesa
# cam_* que entiende baseline/. None = sin análogo en el vocabulario de baseline/, se descarta.
CAMERA_ACTIVITY_MAP: dict[str, str | None] = {
    "Cooking": "cocina",
    "EatingDrinkingMedication": "comida",
    "Resting": "descanso_sofa",
    "ReadingScreen": "ocio_sentado",
    "Phone": "ocio_sentado",
    "Moving": "moverse",
    "Background": None,
}

# Etiquetas que no salen del clasificador sino de la zona `puerta` (`runner/door_events.py`).
# Van aparte de `CAMERA_ACTIVITY_MAP` porque ese es el mapa de las categorías del clasificador
# (lo usa `evaluate_classifier_noise.py` para aplicarles su error) y estas no lo son.
DOOR_ACTIVITY_MAP: dict[str, str] = {
    "LeaveHome": "salir_casa",
    "EnterHome": "entrar_casa",
}
PUNCTUAL_CLASSES: frozenset[str] = frozenset(DOOR_ACTIVITY_MAP.values())

_CLASE_POR_ETIQUETA_CON_PUERTA: dict[str, str | None] = {**CAMERA_ACTIVITY_MAP, **DOOR_ACTIVITY_MAP}

VARIABLE_ENTRADAS_Y_SALIDAS = "ENTRADAS_Y_SALIDAS_ACTIVADAS"


def entradas_y_salidas_activadas() -> bool:
    """Encendidas salvo que la variable de entorno las apague (ver docstring del módulo).
    Es el interruptor del proceso diario; el de la controladora es `puerta_activada`."""
    return os.environ.get(VARIABLE_ENTRADAS_Y_SALIDAS, "").strip().lower() not in {"0", "false", "no", "off"}


def _mapa(con_puerta: bool) -> dict[str, str | None]:
    return _CLASE_POR_ETIQUETA_CON_PUERTA if con_puerta else CAMERA_ACTIVITY_MAP


def _clases(con_puerta: bool) -> tuple[str, ...]:
    return tuple(sorted(set(v for v in _mapa(con_puerta).values() if v is not None)))


# Clases que esta versión del adaptador puede poblar de verdad (columnas `cam_<clase>`)
# — para pasarle a los detectores en vez del tracked_classes pensado para CASAS/sintético.
LIVE_TRACKED_CLASSES: tuple[str, ...] = tuple(f"cam_{c}" for c in _clases(con_puerta=True))
# Las que salen del clasificador, y las dos de la zona `puerta`.
CLASSIFIER_TRACKED_CLASSES: tuple[str, ...] = tuple(f"cam_{c}" for c in _clases(con_puerta=False))
DOOR_TRACKED_CLASSES: tuple[str, ...] = tuple(f"cam_{c}" for c in sorted(PUNCTUAL_CLASSES))


def live_tracked_classes(con_puerta: bool = True) -> tuple[str, ...]:
    """`LIVE_TRACKED_CLASSES`, o solo las del clasificador si `con_puerta=False`."""
    return tuple(f"cam_{c}" for c in _clases(con_puerta))


def sin_clases_de_puerta(tracked: tuple[str, ...]) -> tuple[str, ...]:
    """Las clases que siguen los detectores de horario y de tendencia: todas menos las de
    puerta, que no generan desvíos de horario ni de tendencia."""
    return tuple(c for c in tracked if c not in DOOR_TRACKED_CLASSES)


def solo_ausencia_sostenida_en_puerta(por_clase: pd.DataFrame) -> pd.DataFrame:
    """Filtra la salida de `class_deviations.score_class_deviations`: de `cam_salir_casa` y
    `cam_entrar_casa` solo deja las ausencias sostenidas. Las demás clases quedan como están."""
    if por_clase.empty:
        return por_clase
    de_puerta = por_clase["clase"].isin(DOOR_TRACKED_CLASSES)
    return por_clase[~de_puerta | (por_clase["tipo"] == "ausencia_sostenida")]

# Gap entre dos eventos de actividad consecutivos por encima del cual se asume que la persona
# se fue de cuadro (motion gate en pausa, cámara sin nadie) y no se le atribuye esa duración a
# la actividad anterior. Generoso respecto al heartbeat por defecto (60s) para no cortar de más
# si simplemente no hubo heartbeat en una ventana. Si se configura `clip_heartbeat_s` más alto
# en producción, este valor debe subir en proporción (ver `--max-gap-s` en `baseline/run_live.py`)
# o se subestima sistemáticamente la duración de cada actividad.
DEFAULT_MAX_GAP_S = 120.0

# `features.COBERTURA_MIN_HORAS` (4.0h) es el umbral de CASAS, donde los sensores cubren casi
# todo el día (incluye dormir, baño, etc.). La cámara en vivo solo puede poblar 5 de las 9 clases
# (ver `CAMERA_ACTIVITY_MAP` — no ve dormir/aseo/salir de casa), así que un día real normal nunca
# acumula 4 horas de actividad *atribuible* aunque la persona esté perfectamente monitoreada todo
# el día — reusar ese umbral dejaría `cobertura_baja=True` siempre y silenciaría `baseline/` para
# siempre, no solo durante el warm-up. Umbral propio, deliberadamente más bajo: "hubo señal
# mínima de que la cámara vio actividad real ese día", no "se cubrió el día entero". **No
# calibrado todavía con datos reales de cámara** (no existen aún) — punto de partida razonable a
# falta de medición, mismo criterio de transparencia que el resto del módulo.
CAMERA_COBERTURA_MIN_HORAS = 1.0


def _hora_local(ts: float, tz: tzinfo | None) -> datetime:
    """Hora local *naive* del epoch `ts`: en `tz` si se pasa, si no en la zona de la máquina
    (ver "Zona horaria" en el docstring del módulo)."""
    if tz is None:
        return datetime.fromtimestamp(ts)
    return datetime.fromtimestamp(ts, tz).replace(tzinfo=None)


def _is_valid(event: MonitoringEvent, con_puerta: bool = True) -> tuple[str, str] | None:
    """Devuelve (clase_cam, activity_label) si el evento es una actividad utilizable, si no None."""
    if event.event_type != EventType.ACTIVITY:
        return None
    if event.wall_clock_ts is None:
        return None
    if event.metadata.get("confirmed") is False:
        return None
    label = event.metadata.get("activity_label")
    cls = _mapa(con_puerta).get(label)
    if cls is None:
        return None
    return cls, label


def build_day_vectors_from_events(
    events: list[MonitoringEvent],
    hogar: str,
    max_gap_s: float = DEFAULT_MAX_GAP_S,
    cobertura_min_horas: float = CAMERA_COBERTURA_MIN_HORAS,
    tz: tzinfo | None = None,
    con_puerta: bool = True,
) -> pd.DataFrame:
    """Arma el vector-día (mismo contrato que `features.build_day_vectors`) a partir de eventos
    `ACTIVITY` reales, reconstruyendo duración por los gaps entre eventos consecutivos (ver
    docstring del módulo). Devuelve un DataFrame vacío (sin filas) si no hay eventos utilizables.
    """
    usable = [(e, _is_valid(e, con_puerta)) for e in events]
    usable = [(e, v) for e, v in usable if v is not None]
    usable.sort(key=lambda ev: ev[0].wall_clock_ts)
    # Entradas y salidas: una ocurrencia cada una, sin duración, y fuera de la cadena de gaps.
    puntuales = [(e, v) for e, v in usable if v[0] in PUNCTUAL_CLASSES]
    usable = [(e, v) for e, v in usable if v[0] not in PUNCTUAL_CLASSES]

    # (fecha, clase) -> minutos acumulados / cantidad de ocurrencias (no-heartbeat)
    minutos: dict[tuple[date, str], float] = {}
    conteos: dict[tuple[date, str], int] = {}

    def _bump(fecha: date, clase: str, min_delta: float, es_ocurrencia: bool) -> None:
        key = (fecha, clase)
        minutos[key] = minutos.get(key, 0.0) + min_delta
        if es_ocurrencia:
            conteos[key] = conteos.get(key, 0) + 1

    for event, (clase, _) in puntuales:
        _bump(_hora_local(event.wall_clock_ts, tz).date(), clase, 0.0, es_ocurrencia=True)

    for i, (event, (clase, _)) in enumerate(usable):
        ts = event.wall_clock_ts
        fecha = _hora_local(ts, tz).date()
        if not event.metadata.get("ongoing", False):
            _bump(fecha, clase, 0.0, es_ocurrencia=True)

        if i + 1 < len(usable):
            next_ts = usable[i + 1][0].wall_clock_ts
            gap_s = next_ts - ts
            if 0 < gap_s <= max_gap_s:
                _bump(fecha, clase, gap_s / 60.0, es_ocurrencia=False)
        # último evento de la lista: no hay forma de saber cuánto duró después, se ignora
        # (mismo criterio conservador que el resto del proyecto — más FN que FP).

    fechas = sorted({f for f, _ in minutos})
    if not fechas:
        return pd.DataFrame()

    rows = []
    for fecha in fechas:
        row = {"hogar": hogar, "fecha": fecha}
        total_min = 0.0
        for clase in _clases(con_puerta):
            m = minutos.get((fecha, clase), 0.0)
            n = conteos.get((fecha, clase), 0)
            row[f"cam_{clase}_min"] = m
            row[f"cam_{clase}_n"] = n
            total_min += m
        row["cobertura_horas"] = total_min / 60.0
        row["cobertura_baja"] = row["cobertura_horas"] < cobertura_min_horas
        rows.append(row)

    out = pd.DataFrame(rows).set_index(["hogar", "fecha"]).sort_index()
    return out


def build_events_df_from_events(
    events: list[MonitoringEvent], hogar: str, tz: tzinfo | None = None, con_puerta: bool = True,
) -> pd.DataFrame:
    """Un registro por ocurrencia real (no heartbeat) de actividad, con columnas
    `hogar`/`actividad`/`inicio`/`fecha`/`minuto_del_dia` — formato que espera
    `timing_deviations.score_timing_deviations`. `actividad` lleva el prefijo `cam_` (ej.
    `cam_cocina`) para que coincida con `LIVE_TRACKED_CLASSES`, que es lo que se le pasa como
    `tracked_classes`.
    """
    rows = []
    for event in events:
        valid = _is_valid(event, con_puerta)
        if valid is None or event.metadata.get("ongoing", False):
            continue
        clase, _ = valid
        inicio = _hora_local(event.wall_clock_ts, tz)
        rows.append(
            {
                "hogar": hogar,
                "actividad": f"cam_{clase}",
                "inicio": inicio,
                "fecha": inicio.date(),
                "minuto_del_dia": inicio.hour * 60 + inicio.minute,
            }
        )
    return pd.DataFrame(rows)
