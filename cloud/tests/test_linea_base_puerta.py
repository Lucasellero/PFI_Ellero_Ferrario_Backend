"""Entradas y salidas de casa de punta a punta en la nube: los eventos `LeaveHome` / `EnterHome`
que emite la controladora (`runner/door_events.py`) entran por `POST /devices/events` y el
proceso diario los sigue como `cam_salir_casa` / `cam_entrar_casa`. Encendidas por defecto; se
apagan con `ENTRADAS_Y_SALIDAS_ACTIVADAS=0`. Solo generan un desvío: que falten, desde el primer
día sin eventos (parámetros votados el 2026-10-13). Nunca por horario, aunque se enciendan las
alertas de horario.

Mismo historial que `test_linea_base_produccion.py` (60 días reales de un hogar de CASAS). Las
entradas y salidas se agregan a mano: una salida y una entrada por día.
"""
from __future__ import annotations

from datetime import date, datetime, time, timedelta

import pytest

from app.baseline_runner import entradas_y_salidas_activadas
from app.horario import ZONA_ARGENTINA
from app.linea_base_diaria import ejecutar_linea_base
from app.models import Evento

from conftest import como
from test_linea_base_produccion import (
    DIA_DEL_DESVIO, PRIMER_DIA, _claves, _desvios, _enviar_como_la_controladora, _eventos_de_camara,
)

DIAS_SIN_SALIR = 4
DESVIOS_DE_PUERTA = {
    (DIA_DEL_DESVIO.isoformat(), "clase", "cam_salir_casa:ausencia_sostenida"),
    (DIA_DEL_DESVIO.isoformat(), "clase", "cam_entrar_casa:ausencia_sostenida"),
}


def _evento_de_puerta(etiqueta: str, dia: date, hora: time) -> dict:
    return {
        "video_id": "cam1", "event_type": "activity", "video_timestamp_s": 0.0,
        "wall_clock_ts": datetime.combine(dia, hora, tzinfo=ZONA_ARGENTINA).timestamp(),
        "metadata": {"activity_label": etiqueta, "confirmed": True, "source": "door_zone"},
    }


def _historial_con_puerta(sin_salir_los_ultimos: int = 0) -> list[dict]:
    """El historial de siempre más una salida (15:00) y una entrada (17:00) por día, salvo en
    los últimos `sin_salir_los_ultimos` días."""
    eventos = _eventos_de_camara()
    dias = (DIA_DEL_DESVIO - PRIMER_DIA).days + 1
    for n in range(dias - sin_salir_los_ultimos):
        dia = PRIMER_DIA + timedelta(days=n)
        eventos.append(_evento_de_puerta("LeaveHome", dia, time(15, 0)))
        eventos.append(_evento_de_puerta("EnterHome", dia, time(17, 0)))
    eventos.sort(key=lambda e: e["wall_clock_ts"])
    return eventos


def _de_puerta(claves: list[tuple[str, str, str]]) -> set[tuple[str, str, str]]:
    return {c for c in claves if "salir_casa" in c[2] or "entrar_casa" in c[2]}


def test_por_defecto_estan_encendidas():
    assert entradas_y_salidas_activadas() is True


def test_apagadas_varios_dias_sin_salir_no_generan_nada(cliente, db, crear_hogar, monkeypatch):
    """Con `ENTRADAS_Y_SALIDAS_ACTIVADAS=0` los eventos de puerta se guardan (son `activity`
    comunes) pero el proceso diario no los mira. El día evaluado da lo mismo que sin ellos."""
    monkeypatch.setenv("ENTRADAS_Y_SALIDAS_ACTIVADAS", "0")
    hogar = crear_hogar("hh113")
    _enviar_como_la_controladora(cliente, hogar, _historial_con_puerta(sin_salir_los_ultimos=DIAS_SIN_SALIR))

    assert ejecutar_linea_base(DIA_DEL_DESVIO) is True

    assert _desvios(db, hogar) == []  # igual que el mismo día sin eventos de puerta


def _con_la_puerta_corrida(eventos: list[dict], dia: date, horas: int) -> list[dict]:
    """El mismo historial con la salida y la entrada de `dia` corridas `horas` horas."""
    corridos = []
    for e in eventos:
        es_de_puerta = e["metadata"]["activity_label"] in ("LeaveHome", "EnterHome")
        if es_de_puerta and datetime.fromtimestamp(e["wall_clock_ts"], ZONA_ARGENTINA).date() == dia:
            e = {**e, "wall_clock_ts": e["wall_clock_ts"] + horas * 3600}
        corridos.append(e)
    return sorted(corridos, key=lambda e: e["wall_clock_ts"])


def test_una_salida_a_deshora_nunca_genera_un_desvio_de_horario(cliente, db, crear_hogar, monkeypatch):
    """Con las alertas de horario ENCENDIDAS (no es el default), el día evaluado sale a las
    05:00 en vez de las 15:00, como todos los demás días. El detector de horario marca otras
    clases, pero nunca `cam_salir_casa` ni `cam_entrar_casa`: no las sigue."""
    monkeypatch.setenv("ALERTAS_DE_HORARIO_ACTIVADAS", "1")
    hogar = crear_hogar("hh113")
    eventos = _con_la_puerta_corrida(_historial_con_puerta(), DIA_DEL_DESVIO, horas=-10)
    _enviar_como_la_controladora(cliente, hogar, eventos)

    assert ejecutar_linea_base(DIA_DEL_DESVIO) is True

    claves = _claves(_desvios(db, hogar))
    assert [c for c in claves if c[1] == "timing"]  # el detector de horario corrió y marcó algo
    assert _de_puerta(claves) == set()


def test_el_detector_de_horario_marcaria_esa_salida_si_siguiera_la_puerta():
    """Control de la prueba anterior: la salida de las 05:00 sí es un desvío de horario para el
    detector si se le pide seguir `cam_salir_casa`. Que no aparezca es por decisión, no porque
    el caso sea invisible."""
    from camera_events import LIVE_TRACKED_CLASSES, build_events_df_from_events, sin_clases_de_puerta  # baseline/src
    from timing_deviations import TimingConfig, score_timing_deviations
    from schema import EventType, MonitoringEvent  # events/

    eventos = [
        MonitoringEvent(video_id="cam1", event_type=EventType.ACTIVITY, video_timestamp_s=0.0, wall_clock_ts=e["wall_clock_ts"], metadata=e["metadata"])
        for e in _con_la_puerta_corrida(_historial_con_puerta(), DIA_DEL_DESVIO, horas=-10)
        if e["metadata"]["activity_label"] in ("LeaveHome", "EnterHome")
    ]
    df = build_events_df_from_events(eventos, hogar="1", tz=ZONA_ARGENTINA)
    solo_ese_dia = {("1", DIA_DEL_DESVIO)}

    siguiendo_la_puerta = score_timing_deviations(df, tracked_classes=LIVE_TRACKED_CLASSES, config=TimingConfig(), evaluate_only=solo_ese_dia)
    assert "cam_salir_casa" in set(siguiendo_la_puerta["actividad"])
    como_produccion = score_timing_deviations(
        df, tracked_classes=sin_clases_de_puerta(LIVE_TRACKED_CLASSES), config=TimingConfig(), evaluate_only=solo_ese_dia,
    )
    assert como_produccion.empty


def test_un_solo_dia_sin_salir_ya_genera_un_desvio(cliente, db, crear_hogar):
    """Con los parámetros votados la puerta dispara desde el primer día sin eventos. Ese primer
    día sale como `ausencia`, no como `ausencia_sostenida`."""
    hogar = crear_hogar("hh113")
    _enviar_como_la_controladora(cliente, hogar, _historial_con_puerta(sin_salir_los_ultimos=1))

    assert ejecutar_linea_base(DIA_DEL_DESVIO) is True

    assert _de_puerta(_claves(_desvios(db, hogar))) == {
        (DIA_DEL_DESVIO.isoformat(), "clase", "cam_salir_casa:ausencia"),
        (DIA_DEL_DESVIO.isoformat(), "clase", "cam_entrar_casa:ausencia"),
    }


def test_un_solo_dia_sin_salir_no_alcanza_si_se_piden_dos(cliente, db, crear_hogar, monkeypatch):
    """La cantidad de días es un parámetro del detector por clase: con 2 vuelve a hacer falta
    una racha."""
    import class_deviations  # baseline/src

    monkeypatch.setitem(class_deviations.PRODUCTION_PARAMS, "dias_consecutivos_ausencia_puerta", 2)
    hogar = crear_hogar("hh113")
    _enviar_como_la_controladora(cliente, hogar, _historial_con_puerta(sin_salir_los_ultimos=1))

    assert ejecutar_linea_base(DIA_DEL_DESVIO) is True

    assert _de_puerta(_claves(_desvios(db, hogar))) == set()


def test_varios_dias_sin_salir_generan_una_alerta_de_desvio(cliente, db, crear_hogar):
    hogar = crear_hogar("hh113")
    eventos = _historial_con_puerta(sin_salir_los_ultimos=DIAS_SIN_SALIR)
    _enviar_como_la_controladora(cliente, hogar, eventos)

    # Entran como eventos de actividad comunes: no son alertas por sí mismos.
    de_puerta = [
        e for e in db.query(Evento).filter(Evento.residente_id == hogar["residente_id"]).all()
        if e.metadata_json.get("activity_label") in ("LeaveHome", "EnterHome")
    ]
    assert len(de_puerta) == 2 * (60 - DIAS_SIN_SALIR)
    assert all(e.event_type == "activity" and e.es_alerta is False for e in de_puerta)

    assert ejecutar_linea_base(DIA_DEL_DESVIO) is True

    desvios = _desvios(db, hogar)
    assert _de_puerta(_claves(desvios)) == DESVIOS_DE_PUERTA
    salida = next(d for d in desvios if d.metadata_json["clave"] == "cam_salir_casa:ausencia_sostenida")
    assert salida.es_alerta is True
    assert salida.metadata_json["clase"] == "cam_salir_casa"
    assert salida.metadata_json["valor_hoy"] == 0 and salida.metadata_json["mediana_historica"] == 1

    # La app lo recibe con la clase que traduce `presentacion.ts` del frontend.
    alertas = cliente.get(f"/hogares/{hogar['hogar_id']}/alertas", headers=como(hogar["uid_comun"]))
    assert alertas.status_code == 200
    en_la_api = next(a for a in alertas.json() if a["id"] == salida.id)
    assert en_la_api["metadata"]["clase"] == "cam_salir_casa"
    assert en_la_api["metadata"]["tipo"] == "ausencia_sostenida"


def test_saliendo_todos_los_dias_no_hay_alerta_de_puerta(cliente, db, crear_hogar):
    hogar = crear_hogar("hh113")
    _enviar_como_la_controladora(cliente, hogar, _historial_con_puerta())

    assert ejecutar_linea_base(DIA_DEL_DESVIO) is True

    assert _de_puerta(_claves(_desvios(db, hogar))) == set()


def test_una_camara_sin_zona_de_puerta_nunca_da_alerta_de_puerta(cliente, db, crear_hogar):
    """Sin zona `puerta` la controladora no emite entradas ni salidas: la clase queda siempre
    en cero y que falte no es un desvío. Se evalúan los últimos 5 días del historial."""
    hogar = crear_hogar("hh113")
    _enviar_como_la_controladora(cliente, hogar, _eventos_de_camara())

    for n in range(5):
        assert ejecutar_linea_base(DIA_DEL_DESVIO - timedelta(days=n)) is True

    assert _de_puerta(_claves(_desvios(db, hogar))) == set()
