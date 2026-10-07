"""`GET /hogares/{id}/rutina`: resumen de rutina del residente vs. su propio historial, para la
pantalla Reportes de la app (`docs`/mockup 04, `ResumenRutina`/`IndicadorRutina` en
`src/api/types.ts` del Frontend).

Dos indicadores, los únicos que hoy se pueden calcular de verdad con lo que emite `vlm/` por
cámara única (ver "Vocabulario de cámara" en `baseline/README.md`): no hay clase de "despertar"
(no hay detección de cama/sueño, ver `camera_events.py::CAMERA_ACTIVITY_MAP`) — el mock de la
app lo mostraba, pero inventar ese dato acá violaría la regla de "no inventar campos, verificar
en el backend" (`CLAUDE.md` del repo Frontend).

- **`sedentarismo`**: minutos/día sentado sin actividad (`cam_descanso_sofa` + `cam_ocio_sentado`,
  ver `camera_events.CAMERA_ACTIVITY_MAP`) vs. la mediana histórica diaria del propio residente.
- **`comida`**: horario de la primera comida/toma del día (`cam_comida`, fusiona comer+beber+
  medicación — ver nota de ambigüedad en `class_deviations.py`) vs. su horario habitual.

No reusa `timing_deviations.score_timing_deviations` (ese módulo es un *detector*: solo devuelve
filas para eventos que ya se salen de lo habitual, no "el valor de hoy + el rango habitual" para
mostrar siempre, incluso en un día sin desvío) ni el clustering circular completo — para una sola
clase sin ocurrencias cerca de medianoche, un cálculo directo con mediana+MAD (misma convención
estadística que `class_deviations.py`: MAD robusto a pocos días de historial, `_MAD_TO_STD` para
que se comporte como un desvío estándar) es más simple y más fácil de auditar.

**Ventanas de período = rolling, no calendario** (`hoy`=hoy, `semana`=últimos 7 días,
`mes`=últimos 30 días, hoy inclusive) — igual al criterio que ya usa `ReportesScreen.tsx` del
Frontend (`desde()`) y a los `window_days` de `baseline/`, no una semana/mes de calendario.

**Sin calibrar con datos reales de cámara** (no existen aún, mismo disclaimer que
`camera_events.py`): los umbrales (`_EXCESO_K`, piso de sigma) son puntos de partida razonables,
no un resultado medido.

**Probado (2026-10-07) con un proxy de variabilidad real**: `calcular_resumen_rutina` corrido
contra `dataset_events/events.zip` (CASAS, hogar `hh113`, 490 días de historial real), filtrando
solo las clases `actividad_gruesa` con análogo de cámara (`comida`, `descanso_sofa`,
`ocio_sentado`, `cocina`) y usando la `duracion_s` real de CASAS en vez de reconstruir por gaps.
Con meses de historial real, `comida` dio un rango habitual con varianza genuina (`07:36–08:18`,
no el `05:30–05:30` sin varianza del smoke test sintético) y detectó correctamente un desvío al
simular un "hoy" con la primera comida a las 11:41. **Esto NO es una calibración con cámara
real** (CASAS es un sensor de hogar completo, no una cámara; la cobertura/ambigüedad temporal es
distinta) — sirve para confirmar que la lógica de mediana+MAD se comporta bien con variabilidad
real, no para fijar los umbrales finales.
"""
from __future__ import annotations

import sys
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parents[2]
_BASELINE_SRC = _ROOT / "baseline" / "src"
if str(_BASELINE_SRC) not in sys.path:
    sys.path.insert(0, str(_BASELINE_SRC))

from camera_events import build_day_vectors_from_events, build_events_df_from_events  # noqa: E402

_EVENTS_DIR = _ROOT / "events"
if str(_EVENTS_DIR) not in sys.path:
    sys.path.insert(0, str(_EVENTS_DIR))

from schema import EventType, MonitoringEvent  # noqa: E402  (de events/)

from .models import Evento  # noqa: E402

# mismas constantes/convención que baseline/src/class_deviations.py y timing_deviations.py
_MAD_TO_STD = 1.4826
_EXCESO_K = 3.0
_SIGMA_MIN_MINUTES = 15.0  # piso de dispersión, evita marcar "desvío" por casi-sin-variación

_CLASES_SEDENTARISMO = ("cam_descanso_sofa_min", "cam_ocio_sentado_min")
_CLASE_COMIDA = "cam_comida"

MIN_DIAS_HISTORIAL = 7  # mismo "período de silencio" que dynamic_clustering.DriftConfig


def _dias_periodo(periodo: str, hoy: date) -> tuple[date, date]:
    if periodo == "hoy":
        return hoy, hoy
    dias = 6 if periodo == "semana" else 29  # "semana"=7 días, "mes"=30 días, hoy inclusive
    return hoy - timedelta(days=dias), hoy


def _hhmm(minuto_del_dia: float) -> str:
    m = round(minuto_del_dia) % 1440
    return f"{m // 60:02d}:{m % 60:02d}"


def _duracion(minutos: float) -> str:
    total = round(minutos)
    h, m = total // 60, total % 60
    return f"{h} h {m} min" if h else f"{m} min"


def _evento_a_monitoring_event(evento: Evento) -> MonitoringEvent:
    return MonitoringEvent(
        video_id=str(evento.camara_id) if evento.camara_id is not None else "cloud",
        event_type=EventType(evento.event_type),
        video_timestamp_s=0.0,
        wall_clock_ts=evento.occurred_at.replace(tzinfo=timezone.utc).timestamp(),
        metadata=evento.metadata_json,
    )


def _indicador_sedentarismo(day_vectors, inicio: date, fin: date, periodo: str) -> dict | None:
    cols = [c for c in _CLASES_SEDENTARISMO if c in day_vectors.columns]
    if not cols:
        return None
    totales_por_dia = day_vectors[cols].sum(axis=1)  # index: (hogar, fecha)
    fechas = totales_por_dia.index.get_level_values("fecha")

    actuales = totales_por_dia[(fechas >= inicio) & (fechas <= fin)]
    if actuales.empty:
        return None  # sin datos del período pedido todavía — nada que mostrar

    historicos = totales_por_dia[(fechas < inicio) | (fechas > fin)]
    if len(historicos) < MIN_DIAS_HISTORIAL:
        return None

    mediana_hist = float(np.median(historicos))
    sigma = max(float(np.median(np.abs(historicos - mediana_hist))) * _MAD_TO_STD, _SIGMA_MIN_MINUTES)
    valor = float(actuales.iloc[0]) if periodo == "hoy" else float(actuales.mean())

    estado = "desvio" if valor > mediana_hist + _EXCESO_K * sigma else "ok"
    texto_valor = f"{_duracion(valor)} sin moverse" if periodo == "hoy" else f"{_duracion(valor)} por día"
    return {
        "clave": "sedentarismo", "titulo": "Sedentarismo",
        "valor": texto_valor, "habitual": f"habitual {_duracion(mediana_hist)} por día",
        "estado": estado,
    }


def _indicador_comida(events_df, inicio: date, fin: date, periodo: str) -> dict | None:
    if events_df.empty:
        return None
    comidas = events_df[events_df["actividad"] == _CLASE_COMIDA]
    if comidas.empty:
        return None
    primeras = comidas.groupby("fecha")["minuto_del_dia"].min()  # primera comida de cada día

    actuales = primeras[(primeras.index >= inicio) & (primeras.index <= fin)]
    if actuales.empty:
        return None

    historicos = primeras[(primeras.index < inicio) | (primeras.index > fin)]
    if len(historicos) < MIN_DIAS_HISTORIAL:
        return None

    mediana_hist = float(np.median(historicos))
    sigma = max(float(np.median(np.abs(historicos - mediana_hist))) * _MAD_TO_STD, _SIGMA_MIN_MINUTES)
    valor = float(actuales.iloc[0]) if periodo == "hoy" else float(actuales.mean())

    estado = "desvio" if abs(valor - mediana_hist) > _EXCESO_K * sigma else "ok"
    texto_valor = f"Hoy {_hhmm(valor)}" if periodo == "hoy" else f"Promedio {_hhmm(valor)}"
    p25, p75 = np.percentile(historicos, [25, 75])
    return {
        "clave": "comida", "titulo": "Primera comida",
        "valor": texto_valor, "habitual": f"habitual {_hhmm(p25)}–{_hhmm(p75)}",
        "estado": estado,
    }


def calcular_resumen_rutina(eventos_orm: list[Evento], periodo: str) -> dict:
    """`periodo`: "hoy" | "semana" | "mes". Devuelve `{"periodo", "indicadores"}` — lista vacía
    si el residente todavía no tiene suficiente historial (hogar recién vinculado) o ninguno de
    los dos indicadores tiene datos del período pedido."""
    hoy = datetime.now(timezone.utc).date()
    inicio, fin = _dias_periodo(periodo, hoy)

    eventos = [_evento_a_monitoring_event(e) for e in eventos_orm if e.event_type == EventType.ACTIVITY.value]
    day_vectors = build_day_vectors_from_events(eventos, hogar="hogar")
    events_df = build_events_df_from_events(eventos, hogar="hogar")

    indicadores = [
        ind for ind in (
            _indicador_comida(events_df, inicio, fin, periodo),
            _indicador_sedentarismo(day_vectors, inicio, fin, periodo),
        ) if ind is not None
    ]
    return {"periodo": periodo, "indicadores": indicadores}
