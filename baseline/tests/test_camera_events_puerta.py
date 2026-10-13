"""`camera_events.py` con las entradas y salidas que emite la controladora (`LeaveHome` /
`EnterHome`, ver `runner/door_events.py`). Están encendidas por defecto: son eventos puntuales
de `cam_salir_casa` / `cam_entrar_casa`, cuentan como ocurrencias y no tocan la duración de las
demás actividades. De esas dos clases solo sale un desvío, que falten, con sus propios
parámetros en el detector por clase (`clases_de_puerta`).
Apagadas (`con_puerta=False`), el adaptador las descarta."""
from __future__ import annotations

from datetime import date, datetime, timedelta, timezone

import pandas as pd
import pytest

from schema import EventType, MonitoringEvent  # de events/

UTC = timezone.utc
DIA = datetime(2026, 10, 12, 10, 0, tzinfo=UTC)


def _evento(etiqueta: str, cuando: datetime, **metadata) -> MonitoringEvent:
    return MonitoringEvent(
        video_id="cam1", event_type=EventType.ACTIVITY, video_timestamp_s=0.0,
        wall_clock_ts=cuando.timestamp(), metadata={"activity_label": etiqueta, "confirmed": True, **metadata},
    )


def _cocinando_tres_minutos() -> list[MonitoringEvent]:
    """Cocina de 10:00 a 10:03 (un evento al empezar y un latido por minuto) y después descansa."""
    return [
        _evento("Cooking", DIA),
        _evento("Cooking", DIA + timedelta(minutes=1), ongoing=True),
        _evento("Cooking", DIA + timedelta(minutes=2), ongoing=True),
        _evento("Resting", DIA + timedelta(minutes=3)),
        _evento("Resting", DIA + timedelta(minutes=4), ongoing=True),
    ]


@pytest.fixture
def camera_events(baseline_modulo):
    return baseline_modulo("src.camera_events")


def test_por_defecto_produccion_sigue_entradas_y_salidas(camera_events):
    siete = (
        "cam_cocina", "cam_comida", "cam_descanso_sofa", "cam_entrar_casa", "cam_moverse",
        "cam_ocio_sentado", "cam_salir_casa",
    )
    cinco = ("cam_cocina", "cam_comida", "cam_descanso_sofa", "cam_moverse", "cam_ocio_sentado")
    assert camera_events.LIVE_TRACKED_CLASSES == siete
    assert camera_events.live_tracked_classes() == siete
    assert camera_events.live_tracked_classes(con_puerta=False) == camera_events.CLASSIFIER_TRACKED_CLASSES == cinco
    assert camera_events.DOOR_TRACKED_CLASSES == ("cam_entrar_casa", "cam_salir_casa")


def _hogar_con_salidas(salidas: list[int]) -> pd.DataFrame:
    """Un hogar con `salidas[i]` salidas el día i y siempre 3 comidas."""
    fechas = pd.date_range("2026-03-02", periods=len(salidas)).date
    filas = [
        {"hogar": "h", "fecha": f, "cam_salir_casa_n": n, "cam_comida_n": 3, "cobertura_baja": False}
        for f, n in zip(fechas, salidas)
    ]
    return pd.DataFrame(filas).set_index(["hogar", "fecha"])


def _desvios_de_salida(baseline_modulo, salidas: list[int], **parametros) -> list[tuple[int, str]]:
    """(día, tipo) de cada desvío de `cam_salir_casa`, tratada como clase de puerta."""
    cd = baseline_modulo("src.class_deviations")
    dias = _hogar_con_salidas(salidas)
    hits = cd.score_class_deviations(
        dias, tracked_classes=("cam_salir_casa", "cam_comida"), config=cd.ClassDeviationConfig(**parametros),
        clases_de_puerta=("cam_salir_casa",),
    )
    if hits.empty:
        return []
    orden = {f: i for i, f in enumerate(dias.index.get_level_values("fecha"))}
    return [(orden[r.fecha], r.tipo) for r in hits[hits["clase"] == "cam_salir_casa"].itertuples()]


@pytest.mark.parametrize("dias_puerta", [1, 2, 3, 4])
def test_la_puerta_dispara_recien_con_los_dias_seguidos_pedidos(baseline_modulo, dias_puerta):
    """20 días saliendo y después 5 sin salir: el desvío aparece el día `dias_puerta` de la
    racha. El primer día sale como `ausencia`; del segundo en adelante, `ausencia_sostenida`."""
    desvios = _desvios_de_salida(baseline_modulo, [1] * 20 + [0] * 5, dias_consecutivos_ausencia_puerta=dias_puerta)

    assert [dia for dia, _ in desvios] == list(range(20 + dias_puerta - 1, 25))
    assert [tipo for dia, tipo in desvios if dia == 20] == (["ausencia"] if dias_puerta == 1 else [])
    assert all(tipo == "ausencia_sostenida" for dia, tipo in desvios if dia > 20)


@pytest.mark.parametrize("dias_puerta", [1, 2, 3, 4])
def test_si_un_solo_dia_dispara_la_cantidad_de_dias_no_cambia_nada(baseline_modulo, dias_puerta):
    salidas = [1] * 20 + [0] * 5
    desde_el_primer_dia = _desvios_de_salida(baseline_modulo, salidas, dias_consecutivos_ausencia_puerta=1)
    con_un_solo_dia = _desvios_de_salida(
        baseline_modulo, salidas, dias_consecutivos_ausencia_puerta=dias_puerta, puerta_dispara_un_dia=True,
    )
    assert con_un_solo_dia == desde_el_primer_dia


def test_de_la_puerta_nunca_sale_un_exceso_ni_un_deficit(baseline_modulo):
    """Un día con muchas más salidas que lo habitual, o con menos, no es un desvío de puerta."""
    salidas = [4, 5, 6] * 7 + [40, 1, 5]
    assert _desvios_de_salida(baseline_modulo, salidas) == []
    # La misma serie en una clase común sí da el exceso: es la regla de puerta la que lo saca.
    cd = baseline_modulo("src.class_deviations")
    hits = cd.score_class_deviations(_hogar_con_salidas(salidas), tracked_classes=("cam_salir_casa",), config=cd.ClassDeviationConfig())
    assert "exceso" in set(hits["tipo"])


def test_los_parametros_de_puerta_no_tocan_a_las_otras_clases(baseline_modulo):
    """Un día sin comidas se marca igual con cualquier valor de los parámetros de puerta."""
    cd = baseline_modulo("src.class_deviations")
    dias = _hogar_con_salidas([1] * 25)
    dias.iloc[22, dias.columns.get_loc("cam_comida_n")] = 0
    resultados = [
        cd.score_class_deviations(
            dias, tracked_classes=("cam_salir_casa", "cam_comida"), clases_de_puerta=("cam_salir_casa",),
            config=cd.ClassDeviationConfig(dias_consecutivos_ausencia_puerta=d, puerta_dispara_un_dia=u),
        )
        for d in (1, 4) for u in (False, True)
    ]
    for hits in resultados:
        assert hits[["clase", "tipo"]].values.tolist() == [["cam_comida", "ausencia"]]


def test_los_detectores_de_horario_y_de_tendencia_no_siguen_las_clases_de_puerta(camera_events):
    assert camera_events.sin_clases_de_puerta(camera_events.LIVE_TRACKED_CLASSES) == camera_events.CLASSIFIER_TRACKED_CLASSES
    assert camera_events.sin_clases_de_puerta(camera_events.CLASSIFIER_TRACKED_CLASSES) == camera_events.CLASSIFIER_TRACKED_CLASSES
    assert camera_events.DOOR_ACTIVITY_MAP == {"LeaveHome": "salir_casa", "EnterHome": "entrar_casa"}
    # El mapa de categorías del clasificador no cambia: las de puerta no salen del clasificador.
    assert set(camera_events.CAMERA_ACTIVITY_MAP) == {
        "Cooking", "EatingDrinkingMedication", "Resting", "ReadingScreen", "Phone", "Moving", "Background",
    }


def test_las_etiquetas_son_las_mismas_que_emite_la_controladora(camera_events):
    from runner.door_events import DOOR_LABELS

    assert set(camera_events.DOOR_ACTIVITY_MAP) == set(DOOR_LABELS)


@pytest.mark.parametrize("valor, encendidas", [
    (None, True), ("", True), ("1", True), ("true", True), (" si ", True), ("cualquier cosa", True),
    ("0", False), ("false", False), ("FALSE", False), (" no ", False), ("off", False),
])
def test_la_variable_de_entorno_apaga_las_entradas_y_salidas(camera_events, monkeypatch, valor, encendidas):
    if valor is None:
        monkeypatch.delenv(camera_events.VARIABLE_ENTRADAS_Y_SALIDAS, raising=False)
    else:
        monkeypatch.setenv(camera_events.VARIABLE_ENTRADAS_Y_SALIDAS, valor)
    assert camera_events.VARIABLE_ENTRADAS_Y_SALIDAS == "ENTRADAS_Y_SALIDAS_ACTIVADAS"
    assert camera_events.entradas_y_salidas_activadas() is encendidas


def test_apagadas_los_eventos_de_puerta_se_descartan(camera_events):
    """Con la opción apagada, que lleguen eventos de puerta no cambia nada: ni columnas nuevas,
    ni ocurrencias, ni la duración de las otras actividades."""
    sin_puerta = _cocinando_tres_minutos()
    con_eventos = sin_puerta + [
        _evento("LeaveHome", DIA + timedelta(minutes=1, seconds=30)),
        _evento("EnterHome", DIA + timedelta(hours=2)),
    ]
    a = camera_events.build_day_vectors_from_events(sin_puerta, hogar="h", tz=UTC, con_puerta=False)
    b = camera_events.build_day_vectors_from_events(con_eventos, hogar="h", tz=UTC, con_puerta=False)

    assert not any("salir_casa" in c or "entrar_casa" in c for c in b.columns)
    assert sorted(c for c in b.columns if c.endswith("_n")) == [f"{c}_n" for c in camera_events.CLASSIFIER_TRACKED_CLASSES]
    assert b.equals(a)
    df = camera_events.build_events_df_from_events(con_eventos, hogar="h", tz=UTC, con_puerta=False)
    assert sorted(df["actividad"]) == ["cam_cocina", "cam_descanso_sofa"]


def test_una_salida_y_una_entrada_cuentan_como_ocurrencias_sin_minutos(camera_events):
    eventos = _cocinando_tres_minutos() + [
        _evento("LeaveHome", DIA + timedelta(minutes=30)),
        _evento("EnterHome", DIA + timedelta(hours=2)),
        _evento("LeaveHome", DIA + timedelta(hours=5)),
    ]
    dia = camera_events.build_day_vectors_from_events(eventos, hogar="h", tz=UTC, con_puerta=True).loc[("h", DIA.date())]

    assert (dia["cam_salir_casa_n"], dia["cam_entrar_casa_n"]) == (2, 1)
    assert (dia["cam_salir_casa_min"], dia["cam_entrar_casa_min"]) == (0.0, 0.0)


def test_un_evento_de_puerta_no_cambia_la_duracion_de_las_actividades(camera_events):
    """La salida cae entre dos latidos de cocina: no se queda con ese minuto ni corta la cadena."""
    sin_puerta = _cocinando_tres_minutos()
    con_puerta = sin_puerta + [
        _evento("LeaveHome", DIA + timedelta(minutes=1, seconds=30)),
        _evento("EnterHome", DIA + timedelta(minutes=3, seconds=10)),
    ]
    a = camera_events.build_day_vectors_from_events(sin_puerta, hogar="h", tz=UTC, con_puerta=True).loc[("h", DIA.date())]
    b = camera_events.build_day_vectors_from_events(con_puerta, hogar="h", tz=UTC, con_puerta=True).loc[("h", DIA.date())]

    assert b["cam_cocina_min"] == pytest.approx(3.0)
    assert b["cam_descanso_sofa_min"] == pytest.approx(1.0)
    for columna in a.index:
        if columna not in ("cam_salir_casa_n", "cam_entrar_casa_n"):
            assert b[columna] == a[columna], columna
    assert b["cobertura_horas"] == pytest.approx(4.0 / 60.0)


def test_encendidas_y_sin_eventos_de_puerta_las_columnas_existen_y_valen_cero(camera_events):
    """Cámara sin zona `puerta`: la clase nunca aparece, así que su ausencia no puede ser un desvío."""
    dia = camera_events.build_day_vectors_from_events(_cocinando_tres_minutos(), hogar="h", tz=UTC, con_puerta=True).loc[("h", DIA.date())]
    for columna in ("cam_salir_casa_n", "cam_salir_casa_min", "cam_entrar_casa_n", "cam_entrar_casa_min"):
        assert dia[columna] == 0


def test_la_puerta_se_cuenta_en_el_dia_local_del_hogar(camera_events):
    argentina = timezone(timedelta(hours=-3))
    eventos = _cocinando_tres_minutos() + [_evento("EnterHome", datetime(2026, 10, 13, 1, 30, tzinfo=UTC))]  # 22:30 del 12 en Argentina
    vectores = camera_events.build_day_vectors_from_events(eventos, hogar="h", tz=argentina, con_puerta=True)
    assert vectores.loc[("h", date(2026, 10, 12)), "cam_entrar_casa_n"] == 1
    assert list(vectores.index.get_level_values("fecha")) == [date(2026, 10, 12)]


def test_un_dia_con_solo_eventos_de_puerta_queda_como_de_cobertura_baja(camera_events):
    """No hubo actividad atribuible: el día existe pero los detectores no lo evalúan."""
    vectores = camera_events.build_day_vectors_from_events([_evento("LeaveHome", DIA)], hogar="h", tz=UTC, con_puerta=True)
    assert camera_events.build_day_vectors_from_events([_evento("LeaveHome", DIA)], hogar="h", tz=UTC, con_puerta=False).empty  # apagadas: no hay día
    dia = vectores.loc[("h", DIA.date())]
    assert dia["cam_salir_casa_n"] == 1 and dia["cobertura_horas"] == 0.0 and bool(dia["cobertura_baja"]) is True


def test_las_entradas_y_salidas_aparecen_como_ocurrencias_con_su_horario(camera_events):
    eventos = _cocinando_tres_minutos() + [_evento("LeaveHome", DIA + timedelta(minutes=30))]
    df = camera_events.build_events_df_from_events(eventos, hogar="h", tz=UTC, con_puerta=True)

    assert sorted(df["actividad"]) == ["cam_cocina", "cam_descanso_sofa", "cam_salir_casa"]
    salida = df[df["actividad"] == "cam_salir_casa"].iloc[0]
    assert salida["minuto_del_dia"] == 10 * 60 + 30 and salida["fecha"] == DIA.date()


def test_una_etiqueta_desconocida_se_sigue_descartando(camera_events):
    eventos = _cocinando_tres_minutos() + [_evento("OtraCosa", DIA + timedelta(minutes=30))]
    for con_puerta in (False, True):
        df = camera_events.build_events_df_from_events(eventos, hogar="h", tz=UTC, con_puerta=con_puerta)
        assert sorted(df["actividad"]) == ["cam_cocina", "cam_descanso_sofa"]
