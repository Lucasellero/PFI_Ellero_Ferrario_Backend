"""La línea base por el camino de producción: el historial entra por `POST /devices/events`
(como lo manda la controladora) y lo evalúa el proceso diario de la API
(`linea_base_diaria.ejecutar_linea_base`), no los scripts de evaluación de `baseline/`.

El historial son 60 días reales de un hogar de CASAS (`hh113`, `datos/casas_hh113_60dias.csv`,
ver `datos/generar_casas_hh113.py`), convertidos a los eventos que emitiría la cámara. El desvío
se inyecta en el último día: ese día no se envía ninguna actividad de comida.

El detector de horario está apagado por defecto (2026-10-13): no emite alertas salvo con
`ALERTAS_DE_HORARIO_ACTIVADAS=1`.
"""
from __future__ import annotations

import csv
from datetime import date, datetime, timedelta
from pathlib import Path

import pytest

from app.baseline_runner import VARIABLE_ALERTAS_DE_HORARIO, alertas_de_horario_activadas
from app.horario import ZONA_ARGENTINA
from app.linea_base_diaria import ejecutar_linea_base
from app.models import Evento

from conftest import como

_HISTORIAL = Path(__file__).resolve().parent / "datos" / "casas_hh113_60dias.csv"
PRIMER_DIA = date(2011, 9, 5)
DIA_DEL_DESVIO = date(2011, 11, 3)  # el último de los 60
COMIDA = "EatingDrinkingMedication"
DESVIO_ESPERADO = ("clase", "cam_comida:ausencia")
# Todo lo que el proceso diario emite ese día: los dos desvíos que provoca la falta de comidas.
DESVIOS_DEL_DIA_SIN_COMIDAS = [
    (DIA_DEL_DESVIO.isoformat(), "multivariado", "forma_del_dia"),
    (DIA_DEL_DESVIO.isoformat(), "clase", "cam_comida:ausencia"),
]
# Lo que agrega el detector de horario si se lo enciende, también sobre el día real sin tocar.
DESVIOS_DE_HORARIO = [
    (DIA_DEL_DESVIO.isoformat(), "timing", "cam_cocina:horario"),
    (DIA_DEL_DESVIO.isoformat(), "timing", "cam_ocio_sentado:horario"),
]
LATIDO_S = 60   # `clip_heartbeat_s` por default en cameras.yaml
LOTE = 200      # `_BATCH_SIZE` de runner/cloud_sync.py


def _eventos_de_camara(sin_comida_el: date | None = None) -> list[dict]:
    """Cada actividad de CASAS como la reportaría el runner: un evento cuando empieza y uno cada
    60 s mientras sigue (`ongoing`). Las horas de CASAS son la hora local del hogar; se cargan
    como hora de Argentina, que es la zona en la que la nube corta los días."""
    eventos = []
    with _HISTORIAL.open(encoding="utf-8", newline="") as archivo:
        for fila in csv.DictReader(archivo):
            inicio = datetime.fromisoformat(fila["inicio"]).replace(tzinfo=ZONA_ARGENTINA)
            if fila["activity_label"] == COMIDA and inicio.date() == sin_comida_el:
                continue
            for latido in range(int(fila["duracion_s"]) // LATIDO_S + 1):
                metadata = {"activity_label": fila["activity_label"], "confirmed": True}
                if latido:
                    metadata["ongoing"] = True
                eventos.append({
                    "video_id": "cam1", "event_type": "activity", "video_timestamp_s": 0.0,
                    "wall_clock_ts": (inicio + timedelta(seconds=latido * LATIDO_S)).timestamp(),
                    "metadata": metadata,
                })
    eventos.sort(key=lambda e: e["wall_clock_ts"])
    return eventos


def _enviar_como_la_controladora(cliente, hogar: dict, eventos: list[dict]) -> None:
    for i in range(0, len(eventos), LOTE):
        respuesta = cliente.post("/devices/events", headers={"X-API-Key": hogar["api_key"]}, json=eventos[i:i + LOTE])
        assert respuesta.status_code == 201, respuesta.text
        assert respuesta.json()["insertados"] == len(eventos[i:i + LOTE])


def _desvios(db, hogar: dict) -> list[Evento]:
    db.expire_all()
    return (
        db.query(Evento)
        .filter(Evento.residente_id == hogar["residente_id"], Evento.event_type == "routine_deviation")
        .order_by(Evento.id).all()
    )


def _claves(desvios: list[Evento]) -> list[tuple[str, str, str]]:
    return [(d.metadata_json["fecha"], d.metadata_json["detector"], d.metadata_json["clave"]) for d in desvios]


def test_el_historial_cubre_60_dias_y_el_ultimo_tiene_comidas():
    """El desvío es inyectado: en los datos reales ese día sí hay actividad de comida."""
    eventos = _eventos_de_camara()
    dias = {datetime.fromtimestamp(e["wall_clock_ts"], ZONA_ARGENTINA).date() for e in eventos}
    assert min(dias) == PRIMER_DIA and max(dias) == DIA_DEL_DESVIO and len(dias) == 60
    comidas_ese_dia = [
        e for e in eventos
        if e["metadata"]["activity_label"] == COMIDA and "ongoing" not in e["metadata"]
        and datetime.fromtimestamp(e["wall_clock_ts"], ZONA_ARGENTINA).date() == DIA_DEL_DESVIO
    ]
    assert len(comidas_ese_dia) >= 3
    sin_comida = _eventos_de_camara(sin_comida_el=DIA_DEL_DESVIO)
    assert len(sin_comida) < len(eventos)


def test_un_dia_sin_comidas_genera_una_alerta_de_desvio_de_rutina(cliente, db, crear_hogar):
    hogar = crear_hogar("hh113")
    eventos = _eventos_de_camara(sin_comida_el=DIA_DEL_DESVIO)
    _enviar_como_la_controladora(cliente, hogar, eventos)
    assert db.query(Evento).filter(Evento.residente_id == hogar["residente_id"]).count() == len(eventos)
    assert _desvios(db, hogar) == []

    assert ejecutar_linea_base(DIA_DEL_DESVIO) is True

    desvios = _desvios(db, hogar)
    assert _claves(desvios) == DESVIOS_DEL_DIA_SIN_COMIDAS  # ninguno del detector de horario
    desvio = next(d for d in desvios if (d.metadata_json["detector"], d.metadata_json["clave"]) == DESVIO_ESPERADO)
    assert desvio.es_alerta is True
    assert desvio.camara_id is None
    assert desvio.metadata_json["valor_hoy"] == 0
    assert desvio.metadata_json["mediana_historica"] > 0
    assert all(d.es_alerta for d in desvios)

    # La app lo ve entre las alertas del hogar.
    alertas = cliente.get(f"/hogares/{hogar['hogar_id']}/alertas", headers=como(hogar["uid_comun"]))
    assert alertas.status_code == 200
    por_id = {a["id"]: a for a in alertas.json()}
    assert set(por_id) == {d.id for d in desvios}
    en_la_api = por_id[desvio.id]
    assert en_la_api["event_type"] == "routine_deviation"
    assert en_la_api["es_alerta"] is True and en_la_api["resuelta"] is False
    assert en_la_api["metadata"]["clave"] == "cam_comida:ausencia"
    assert en_la_api["metadata"]["fecha"] == DIA_DEL_DESVIO.isoformat()

    # Correr el proceso otra vez sobre el mismo día no duplica nada.
    antes = _claves(desvios)
    assert ejecutar_linea_base(DIA_DEL_DESVIO) is True
    assert _claves(_desvios(db, hogar)) == antes
    assert len(set(antes)) == len(antes)
    alertas = cliente.get(f"/hogares/{hogar['hogar_id']}/alertas", headers=como(hogar["uid_comun"]))
    assert len(alertas.json()) == len(antes)


def test_el_mismo_dia_con_sus_comidas_reales_no_genera_ninguna_alerta(cliente, db, crear_hogar):
    """Control: con el historial tal cual, sin inyectar nada, el proceso diario no emite nada.
    (Hasta el 2026-10-13 emitía dos alertas del detector de horario sobre este día real.)"""
    hogar = crear_hogar("hh113")
    _enviar_como_la_controladora(cliente, hogar, _eventos_de_camara())

    assert ejecutar_linea_base(DIA_DEL_DESVIO) is True

    assert _desvios(db, hogar) == []
    alertas = cliente.get(f"/hogares/{hogar['hogar_id']}/alertas", headers=como(hogar["uid_comun"]))
    assert alertas.json() == []


def test_con_la_opcion_encendida_el_detector_de_horario_vuelve_a_emitir(cliente, db, crear_hogar, monkeypatch):
    """La opción existe y funciona: encendida, el mismo día real recibe las dos alertas de
    horario de antes. Es el comportamiento que se apagó, no uno recomendado."""
    monkeypatch.setenv(VARIABLE_ALERTAS_DE_HORARIO, "1")
    hogar = crear_hogar("hh113")
    _enviar_como_la_controladora(cliente, hogar, _eventos_de_camara())

    assert ejecutar_linea_base(DIA_DEL_DESVIO) is True

    assert _claves(_desvios(db, hogar)) == DESVIOS_DE_HORARIO


@pytest.mark.parametrize("valor, encendida", [
    (None, False), ("", False), ("0", False), ("false", False), ("no", False), ("cualquier cosa", False),
    ("1", True), ("true", True), ("TRUE", True), (" si ", True), ("sí", True), ("yes", True),
])
def test_las_alertas_de_horario_estan_apagadas_salvo_que_se_pidan(monkeypatch, valor, encendida):
    if valor is None:
        monkeypatch.delenv(VARIABLE_ALERTAS_DE_HORARIO, raising=False)
    else:
        monkeypatch.setenv(VARIABLE_ALERTAS_DE_HORARIO, valor)
    assert alertas_de_horario_activadas() is encendida


def test_evaluar_solo_el_dia_pedido_da_lo_mismo_que_evaluar_todo_el_historial():
    """El proceso diario le pide al detector de horario solo el día evaluado (`evaluate_only`).
    Tiene que marcar exactamente lo mismo que marcaría ese día recorriendo todo el historial."""
    import pandas as pd
    from timing_deviations import TimingConfig, score_timing_deviations  # baseline/src, vía baseline_runner

    filas = []
    with _HISTORIAL.open(encoding="utf-8", newline="") as archivo:
        for fila in csv.DictReader(archivo):
            inicio = datetime.fromisoformat(fila["inicio"])
            if inicio.date() >= DIA_DEL_DESVIO - timedelta(days=20):
                filas.append({
                    "hogar": "1", "actividad": fila["activity_label"], "inicio": inicio, "fecha": inicio.date(),
                    "minuto_del_dia": inicio.hour * 60 + inicio.minute,
                })
    eventos = pd.DataFrame(filas)
    clases = tuple(sorted(eventos["actividad"].unique()))
    config = TimingConfig(por_clase={clases[0]: {"excess_k": 2.0, "k_time_slots": 3}})

    todo = score_timing_deviations(eventos, tracked_classes=clases, config=config)
    assert len(todo) > 5  # con un solo resultado la comparación no probaría nada
    for dia in sorted(todo["fecha"].unique())[-4:] + [DIA_DEL_DESVIO]:
        solo_ese_dia = score_timing_deviations(eventos, tracked_classes=clases, config=config, evaluate_only={("1", dia)})
        esperado = todo[todo["fecha"] == dia].reset_index(drop=True)
        assert solo_ese_dia.to_dict(orient="records") == esperado.to_dict(orient="records")


def test_un_dia_sin_ningun_evento_no_se_evalua(cliente, db, crear_hogar):
    """Si del día a evaluar no llegó nada (cámara apagada o sin conexión), no se emite un desvío."""
    hogar = crear_hogar("hh113")
    _enviar_como_la_controladora(cliente, hogar, _eventos_de_camara())

    assert ejecutar_linea_base(DIA_DEL_DESVIO + timedelta(days=1)) is True

    assert _desvios(db, hogar) == []
