"""Corre los 4 detectores de `baseline/` contra los eventos reales de `events/events.db` (lo que
produce `runner/` en vivo) y escribe los desvíos encontrados de vuelta al store como eventos
`EventType.ROUTINE_DEVIATION` — el tramo que faltaba para que la detección de desvíos de rutina
sea una feature real, no solo un resultado de investigación sobre CASAS (ver `baseline/README.md`,
"Cómo seguir").

Pensado para correr **una vez al día**, temprano a la mañana siguiente, evaluando el día anterior
completo (nunca el día en curso, que está parcial e induciría ruido). El *scheduling* (cron /
Task Scheduler en la Pi) es un paso de infraestructura, no algo que resuelva este script — ver
"Correr en producción" en `baseline/README.md`.

Uso:
    py run_live.py                  # evalúa el día de ayer (hora local), todas las cámaras
    py run_live.py --date 2026-09-20 --video-id webcam   # backfill/debug de un día puntual
"""
from __future__ import annotations

import argparse
import sys
from datetime import date, datetime, timedelta
from pathlib import Path

import pandas as pd

_BASELINE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(_BASELINE_DIR / "src"))

from camera_events import (  # noqa: E402
    DEFAULT_MAX_GAP_S,
    LIVE_TRACKED_CLASSES,
    build_day_vectors_from_events,
    build_events_df_from_events,
)
from class_deviations import ClassDeviationConfig, score_class_deviations  # noqa: E402
from dynamic_clustering import DriftConfig, score_days  # noqa: E402
from timing_deviations import TimingConfig, score_timing_deviations  # noqa: E402
from trend_deviations import TrendConfig, score_trend_deviations  # noqa: E402

_EVENTS_DIR = _BASELINE_DIR.parent / "events"
sys.path.insert(0, str(_EVENTS_DIR))

from schema import EventType, MonitoringEvent  # noqa: E402
from store import EventStore  # noqa: E402


def _already_emitted_keys(events: list[MonitoringEvent]) -> set[tuple[str, str, str]]:
    """(fecha, detector, clave) de los ROUTINE_DEVIATION ya escritos — para no duplicar avisos
    si el script corre más de una vez para el mismo día."""
    return {
        (e.metadata.get("fecha"), e.metadata.get("detector"), e.metadata.get("clave"))
        for e in events
        if e.event_type == EventType.ROUTINE_DEVIATION
    }


def _emit(store: EventStore, video_id: str, fecha: date, detector: str, clave: str, detalle: dict) -> None:
    event = MonitoringEvent(
        video_id=video_id,
        event_type=EventType.ROUTINE_DEVIATION,
        video_timestamp_s=0.0,  # sin sentido en modo vivo por día, ver events/schema.py
        wall_clock_ts=datetime.now().timestamp(),
        metadata={"detector": detector, "clave": clave, "fecha": fecha.isoformat(), **detalle},
    )
    store.add(event)
    print(f"[{video_id}] {fecha} desvío ({detector}): {clave} -> {detalle}")


def process_camera(store: EventStore, video_id: str, target_date: date, max_gap_s: float = DEFAULT_MAX_GAP_S) -> None:
    events = store.all(video_id=video_id)
    day_vectors = build_day_vectors_from_events(events, hogar=video_id, max_gap_s=max_gap_s)
    if day_vectors.empty or target_date not in day_vectors.index.get_level_values("fecha"):
        return  # sin cobertura ese día (o sin historial todavía) — nada para evaluar

    tracked = tuple(LIVE_TRACKED_CLASSES)
    ya_emitidos = _already_emitted_keys(events)
    fecha_iso = target_date.isoformat()

    def _emit_if_new(detector: str, clave: str, detalle: dict) -> None:
        if (fecha_iso, detector, clave) in ya_emitidos:
            return
        _emit(store, video_id, target_date, detector, clave, detalle)

    # 1) Multivariado: forma general del día.
    multi = score_days(day_vectors, DriftConfig())
    multi = multi[(multi["fecha"] == target_date) & multi["es_desvio"]]
    for _, row in multi.iterrows():
        _emit_if_new(
            "multivariado", "forma_del_dia",
            {"z_score": row["z_score"], "top_contribuyentes": row["top_contribuyentes"]},
        )

    # 2) Por clase: ausencia/exceso/déficit de una clase puntual.
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

    # 3) Horario habitual (a nivel de evento, no de día — usa events_df aparte).
    events_df = build_events_df_from_events(events, hogar=video_id)
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

    # 4) Tendencia gradual (nivel semana — solo dispara si target_date es el último día de la
    # semana ISO evaluada, score_trend_deviations ya filtra por semana_fin internamente).
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


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--date", type=str, default=None, help="Fecha a evaluar (YYYY-MM-DD). Default: ayer.")
    parser.add_argument("--video-id", type=str, default=None, help="Solo esta cámara. Default: todas.")
    parser.add_argument(
        "--max-gap-s", type=float, default=DEFAULT_MAX_GAP_S,
        help="Gap máximo (s) entre eventos de actividad para atribuir duración entre ellos. "
             "Debe ser mayor a `clip_heartbeat_s` configurado en cameras.yaml para esa cámara, "
             "o se subestima la duración real de cada actividad (ver camera_events.py).",
    )
    args = parser.parse_args()

    target_date = date.fromisoformat(args.date) if args.date else date.today() - timedelta(days=1)

    store = EventStore()
    try:
        video_ids = [args.video_id] if args.video_id else sorted({e.video_id for e in store.all()})
        for video_id in video_ids:
            process_camera(store, video_id, target_date, max_gap_s=args.max_gap_s)
    finally:
        store.close()


if __name__ == "__main__":
    main()
