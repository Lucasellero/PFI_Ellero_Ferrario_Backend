"""Equivalente cloud de `baseline/run_live.py` (ver su docstring para el diseño general: mismos
4 detectores, misma idempotencia, mismo criterio de evaluar el día anterior completo). La única
diferencia real es la fuente de datos: acá los eventos ya están en la tabla `Evento` (Postgres en
producción, SQLite en desarrollo) en vez de `events/events.db` local.

No se duplica la lógica de `baseline/src/` — se la importa tal cual (mismo patrón
`sys.path.insert` que ya usa `runner/` para importar `vision/`/`vlm/`) y la única pieza nueva acá
es la conversión `Evento` (fila ORM) -> `MonitoringEvent` (el mismo objeto que ya consume
`baseline/src/camera_events.py`), para no reescribir esa lógica contra un segundo formato.
"""
from __future__ import annotations

import sys
from datetime import date, datetime, timezone
from pathlib import Path

import pandas as pd
from sqlalchemy.orm import Session

_ROOT = Path(__file__).resolve().parents[2]
_BASELINE_SRC = _ROOT / "baseline" / "src"
if str(_BASELINE_SRC) not in sys.path:
    sys.path.insert(0, str(_BASELINE_SRC))

from camera_events import (  # noqa: E402  (de baseline/src/)
    LIVE_TRACKED_CLASSES,
    build_day_vectors_from_events,
    build_events_df_from_events,
)
from class_deviations import ClassDeviationConfig, score_class_deviations  # noqa: E402
from dynamic_clustering import DriftConfig, score_days  # noqa: E402
from timing_deviations import TimingConfig, score_timing_deviations  # noqa: E402
from trend_deviations import TrendConfig, score_trend_deviations  # noqa: E402

_EVENTS_DIR = _ROOT / "events"
if str(_EVENTS_DIR) not in sys.path:
    sys.path.insert(0, str(_EVENTS_DIR))

from schema import EventType, MonitoringEvent  # noqa: E402  (de events/)

from .alertas import es_alerta as _es_alerta
from .models import Evento, Hogar
from .notifications import enviar_push_alerta


def _evento_to_monitoring_event(evento: Evento) -> MonitoringEvent:
    """Única pieza de conversión que hace falta: una fila `Evento` de la nube es, en esencia, el
    mismo `MonitoringEvent` que ya produce `runner/` — solo cambia dónde vive guardado."""
    return MonitoringEvent(
        video_id=str(evento.camara_id) if evento.camara_id is not None else "cloud",
        event_type=EventType(evento.event_type),
        video_timestamp_s=0.0,
        wall_clock_ts=evento.occurred_at.replace(tzinfo=timezone.utc).timestamp(),
        metadata=evento.metadata_json,
    )


def _already_emitted_keys(eventos_previos: list[Evento]) -> set[tuple[str, str, str]]:
    return {
        (e.metadata_json.get("fecha"), e.metadata_json.get("detector"), e.metadata_json.get("clave"))
        for e in eventos_previos
        if e.event_type == EventType.ROUTINE_DEVIATION.value
    }


def _registrar_desvio(
    db: Session, hogar: Hogar, target_date: date, detector: str, clave: str, detalle: dict,
) -> None:
    metadata = {"detector": detector, "clave": clave, "fecha": target_date.isoformat(), **detalle}
    evento = Evento(
        camara_id=None,
        residente_id=hogar.residente.id,
        event_type=EventType.ROUTINE_DEVIATION.value,
        occurred_at=datetime.now(timezone.utc),
        synced_at=datetime.now(timezone.utc),
        es_alerta=_es_alerta(EventType.ROUTINE_DEVIATION.value, metadata),
        metadata_json=metadata,
    )
    db.add(evento)
    db.commit()
    print(f"[hogar {hogar.id}] {target_date} desvío ({detector}): {clave} -> {detalle}")
    # RF8: avisar al/a los familiar/es a cargo. Sin un campo de push-token todavía en `Familiar`
    # (fuera de alcance de esta vuelta, ver cloud/README.md "Pendiente"), así que por ahora esto
    # queda en log vía notifications.py hasta sumar ese campo.
    enviar_push_alerta([], titulo="Posible desvío de rutina", cuerpo=clave, data=detalle)


def process_hogar(db: Session, hogar: Hogar, target_date: date) -> None:
    if hogar.residente is None:
        return  # hogar sin residente asociado todavía (alta incompleta) — nada que evaluar

    eventos_orm = (
        db.query(Evento)
        .filter(Evento.residente_id == hogar.residente.id)
        .order_by(Evento.occurred_at)
        .all()
    )
    eventos = [_evento_to_monitoring_event(e) for e in eventos_orm]

    day_vectors = build_day_vectors_from_events(eventos, hogar=str(hogar.id))
    if day_vectors.empty or target_date not in day_vectors.index.get_level_values("fecha"):
        return  # sin cobertura ese día (o sin historial todavía) — nada para evaluar

    tracked = tuple(LIVE_TRACKED_CLASSES)
    ya_emitidos = _already_emitted_keys(eventos_orm)
    fecha_iso = target_date.isoformat()

    def _emit_if_new(detector: str, clave: str, detalle: dict) -> None:
        if (fecha_iso, detector, clave) in ya_emitidos:
            return
        _registrar_desvio(db, hogar, target_date, detector, clave, detalle)

    multi = score_days(day_vectors, DriftConfig())
    multi = multi[(multi["fecha"] == target_date) & multi["es_desvio"]]
    for _, row in multi.iterrows():
        _emit_if_new(
            "multivariado", "forma_del_dia",
            {"z_score": row["z_score"], "top_contribuyentes": row["top_contribuyentes"]},
        )

    por_clase = score_class_deviations(day_vectors, tracked_classes=tracked, config=ClassDeviationConfig())
    if not por_clase.empty:
        por_clase = por_clase[por_clase["fecha"] == target_date]
        for _, row in por_clase.iterrows():
            _emit_if_new(
                "clase", f"{row['clase']}:{row['tipo']}",
                {
                    "clase": row["clase"], "tipo": row["tipo"], "valor_hoy": row["valor_hoy"],
                    "mediana_historica": row["mediana_historica"],
                },
            )

    events_df = build_events_df_from_events(eventos, hogar=str(hogar.id))
    if not events_df.empty:
        timing = score_timing_deviations(events_df, tracked_classes=tracked, config=TimingConfig())
        if not timing.empty:
            timing = timing[timing["fecha"] == target_date]
            for _, row in timing.iterrows():
                _emit_if_new(
                    "timing", f"{row['actividad']}:horario",
                    {
                        "clase": row["actividad"], "minuto_del_dia": row["minuto_del_dia"],
                        "horario_habitual_min": row["horario_habitual_min"], "desvio_min": row["desvio_min"],
                    },
                )

    tendencia = score_trend_deviations(day_vectors, tracked_classes=tracked, config=TrendConfig())
    if not tendencia.empty:
        tendencia = tendencia[pd.to_datetime(tendencia["semana_fin"]).dt.date == target_date]
        for _, row in tendencia.iterrows():
            _emit_if_new(
                "tendencia", f"{row['clase']}:{row['direccion']}",
                {
                    "clase": row["clase"], "direccion": row["direccion"],
                    "mediana_base": row["mediana_base"], "mediana_reciente": row["mediana_reciente"],
                },
            )
