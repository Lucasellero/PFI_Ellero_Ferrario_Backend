"""`GET /hogares/{id}/rutina`: resumen de rutina del residente vs. su propio historial, para la
pantalla Reportes de la app (`docs`/mockup 04, `ResumenRutina`/`IndicadorRutina` en
`src/api/types.ts` del Frontend).

Cuatro indicadores — los que hoy se pueden calcular de verdad con lo que emite `vlm/` por cámara
única, de las 5 clases alcanzables hoy de las 9 del vocabulario acordado (ver "Vocabulario de
cámara" en `baseline/README.md`). No hay clase de "despertar" (no hay detección de cama/sueño, ver
`camera_events.py::CAMERA_ACTIVITY_MAP`) — el mock de la app lo mostraba, pero inventar ese dato
acá violaría la regla de "no inventar campos, verificar en el backend" (`CLAUDE.md` del repo
Frontend).

- **`sedentarismo`**: minutos/día sentado sin actividad (`cam_descanso_sofa` + `cam_ocio_sentado`,
  ver `camera_events.CAMERA_ACTIVITY_MAP`) vs. la mediana histórica diaria del propio residente.
  Desvío = **más** de lo habitual (estar sentado de más es la señal clínica, no al revés).
- **`comida`**: horario de la primera comida/toma del día (`cam_comida`, fusiona comer+beber+
  medicación — ver nota de ambigüedad en `class_deviations.py`) vs. su horario habitual.
- **`cocina`** (2026-10-08): minutos/día cocinando (`cam_cocina`) vs. la mediana histórica. Desvío
  = **menos** de lo habitual (dejar de cocinar es la señal clínica — ej. dificultad para cuidarse
  solo —, cocinar de más no es una alerta).
- **`movilidad`** (2026-10-08): minutos/día moviéndose por la casa (`cam_moverse`) vs. la mediana
  histórica. Desvío = **menos** de lo habitual (mismo criterio que `cocina`: reducción de
  movilidad es la señal clínica relevante, Sección 2.1.2 del informe).

`sedentarismo`/`cocina`/`movilidad` comparten la misma lógica (`_indicador_duracion`/
`_serie_duracion`: suma de minutos/día vs. mediana+MAD, con la dirección del desvío — exceso o
déficit — como parámetro); `comida` es horario de ocurrencia, no duración, y usa su propia función
(`_indicador_comida`/`_serie_comida`, desvío en las dos direcciones).

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

_CLASE_COMIDA = "cam_comida"

# clave -> (columnas de duración a sumar, dirección del desvío, título, texto del valor "hoy")
_INDICADORES_DURACION: dict[str, dict] = {
    "sedentarismo": {
        "columnas": ("cam_descanso_sofa_min", "cam_ocio_sentado_min"),
        "direccion": "exceso", "titulo": "Sedentarismo", "texto_hoy": "sin moverse",
    },
    "cocina": {
        "columnas": ("cam_cocina_min",),
        "direccion": "deficit", "titulo": "Cocina", "texto_hoy": "cocinando",
    },
    "movilidad": {
        "columnas": ("cam_moverse_min",),
        "direccion": "deficit", "titulo": "Movilidad", "texto_hoy": "en movimiento",
    },
}

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


def _es_desvio_duracion(valor: float, mediana_hist: float, sigma: float, direccion: str) -> bool:
    if direccion == "exceso":
        return valor > mediana_hist + _EXCESO_K * sigma
    return valor < mediana_hist - _EXCESO_K * sigma  # "deficit"


def _indicador_duracion(day_vectors, inicio: date, fin: date, periodo: str, clave: str) -> dict | None:
    spec = _INDICADORES_DURACION[clave]
    cols = [c for c in spec["columnas"] if c in day_vectors.columns]
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

    estado = "desvio" if _es_desvio_duracion(valor, mediana_hist, sigma, spec["direccion"]) else "ok"
    texto_valor = f"{_duracion(valor)} {spec['texto_hoy']}" if periodo == "hoy" else f"{_duracion(valor)} por día"
    return {
        "clave": clave, "titulo": spec["titulo"],
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


def _serie_duracion(day_vectors, inicio: date, fin: date, clave: str) -> dict | None:
    """Un punto por día del período (minutos de esa clase ese día) + la mediana histórica ("línea
    de lo habitual") y el límite a partir del cual un día cuenta como desvío — en la dirección que
    corresponda (`exceso` para `sedentarismo`, `deficit` para `cocina`/`movilidad`, ver
    `_INDICADORES_DURACION`)."""
    spec = _INDICADORES_DURACION[clave]
    cols = [c for c in spec["columnas"] if c in day_vectors.columns]
    if not cols:
        return None
    totales_por_dia = day_vectors[cols].sum(axis=1)
    fechas = totales_por_dia.index.get_level_values("fecha")

    historicos = totales_por_dia[(fechas < inicio) | (fechas > fin)]
    if len(historicos) < MIN_DIAS_HISTORIAL:
        return None
    mediana_hist = float(np.median(historicos))
    sigma = max(float(np.median(np.abs(historicos - mediana_hist))) * _MAD_TO_STD, _SIGMA_MIN_MINUTES)
    es_exceso = spec["direccion"] == "exceso"
    limite_superior = mediana_hist + _EXCESO_K * sigma if es_exceso else None
    limite_inferior = None if es_exceso else max(mediana_hist - _EXCESO_K * sigma, 0.0)

    valores_por_fecha = dict(zip(fechas, totales_por_dia.values))
    puntos = []
    dia = inicio
    while dia <= fin:
        valor = valores_por_fecha.get(dia)
        if valor is None:
            estado = "sin_dato"
        else:
            estado = "desvio" if _es_desvio_duracion(valor, mediana_hist, sigma, spec["direccion"]) else "ok"
        puntos.append({"fecha": dia, "valor": valor, "estado": estado})
        dia += timedelta(days=1)

    return {
        "unidad": "minutos", "habitual": mediana_hist,
        "limite_inferior": limite_inferior, "limite_superior": limite_superior,
        "puntos": puntos,
    }


def _serie_comida(events_df, inicio: date, fin: date) -> dict | None:
    """Idem `_serie_sedentarismo` para el horario de la primera comida: acá el desvío es en las
    dos direcciones (comer mucho más temprano o más tarde de lo habitual, ver `_indicador_comida`),
    así que hay límite inferior y superior."""
    if events_df.empty:
        return None
    comidas = events_df[events_df["actividad"] == _CLASE_COMIDA]
    if comidas.empty:
        return None
    primeras = comidas.groupby("fecha")["minuto_del_dia"].min()

    historicos = primeras[(primeras.index < inicio) | (primeras.index > fin)]
    if len(historicos) < MIN_DIAS_HISTORIAL:
        return None
    mediana_hist = float(np.median(historicos))
    sigma = max(float(np.median(np.abs(historicos - mediana_hist))) * _MAD_TO_STD, _SIGMA_MIN_MINUTES)

    valores_por_fecha = primeras.to_dict()
    puntos = []
    dia = inicio
    while dia <= fin:
        valor = valores_por_fecha.get(dia)
        if valor is None:
            estado = "sin_dato"
        else:
            estado = "desvio" if abs(valor - mediana_hist) > _EXCESO_K * sigma else "ok"
        puntos.append({"fecha": dia, "valor": valor, "estado": estado})
        dia += timedelta(days=1)

    return {
        "unidad": "hora", "habitual": mediana_hist,
        "limite_inferior": mediana_hist - _EXCESO_K * sigma, "limite_superior": mediana_hist + _EXCESO_K * sigma,
        "puntos": puntos,
    }


CLAVES_INDICADOR = ("comida", *_INDICADORES_DURACION.keys())


def calcular_serie_rutina(eventos_orm: list[Evento], clave: str, periodo: str) -> dict:
    """Mismo cálculo que `calcular_resumen_rutina`, pero devuelve un punto por día del período
    en vez de un solo valor agregado — para graficar (eje X = día, eje Y = valor) con la línea de
    lo habitual y los límites de desvío. `clave`: una de `CLAVES_INDICADOR`."""
    hoy = datetime.now(timezone.utc).date()
    inicio, fin = _dias_periodo(periodo, hoy)

    eventos = [_evento_a_monitoring_event(e) for e in eventos_orm if e.event_type == EventType.ACTIVITY.value]
    day_vectors = build_day_vectors_from_events(eventos, hogar="hogar")
    events_df = build_events_df_from_events(eventos, hogar="hogar")

    if clave == "comida":
        resultado = _serie_comida(events_df, inicio, fin)
    elif clave in _INDICADORES_DURACION:
        resultado = _serie_duracion(day_vectors, inicio, fin, clave)
    else:
        raise ValueError(f"indicador desconocido: {clave!r} (esperado uno de {CLAVES_INDICADOR})")

    if resultado is None:
        return {
            "clave": clave, "periodo": periodo, "unidad": "hora" if clave == "comida" else "minutos",
            "habitual": None, "limite_inferior": None, "limite_superior": None, "puntos": [],
        }
    return {"clave": clave, "periodo": periodo, **resultado}


def calcular_resumen_rutina(eventos_orm: list[Evento], periodo: str) -> dict:
    """`periodo`: "hoy" | "semana" | "mes". Devuelve `{"periodo", "indicadores"}` — lista vacía
    si el residente todavía no tiene suficiente historial (hogar recién vinculado) o ninguno de
    los indicadores tiene datos del período pedido."""
    hoy = datetime.now(timezone.utc).date()
    inicio, fin = _dias_periodo(periodo, hoy)

    eventos = [_evento_a_monitoring_event(e) for e in eventos_orm if e.event_type == EventType.ACTIVITY.value]
    day_vectors = build_day_vectors_from_events(eventos, hogar="hogar")
    events_df = build_events_df_from_events(eventos, hogar="hogar")

    indicadores = [
        ind for ind in (
            _indicador_comida(events_df, inicio, fin, periodo),
            *(_indicador_duracion(day_vectors, inicio, fin, periodo, clave) for clave in _INDICADORES_DURACION),
        ) if ind is not None
    ]
    return {"periodo": periodo, "indicadores": indicadores}
