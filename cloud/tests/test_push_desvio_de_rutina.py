"""El aviso al celular de un desvío de rutina (`alertas.texto_desvio`): dice lo mismo que la
pantalla de la app (`presentacion.ts` del Frontend) y nunca muestra la clave interna del desvío
(`cam_comida:ausencia`), el nombre de una clase ni un porcentaje.
"""
from __future__ import annotations

import re

import pytest

from app import baseline_runner
from app.alertas import texto_desvio
from app.linea_base_diaria import ejecutar_linea_base

from test_linea_base_produccion import DIA_DEL_DESVIO, _enviar_como_la_controladora, _eventos_de_camara
from test_linea_base_puerta import _historial_con_puerta

DEL_CLASIFICADOR = ("cam_cocina", "cam_comida", "cam_descanso_sofa", "cam_ocio_sentado", "cam_moverse")
DE_PUERTA = ("cam_salir_casa", "cam_entrar_casa")
CLAVE_INTERNA = re.compile(r"[a-z_]+:[a-z_]+")
# No se nombra a la persona ni se la trata en masculino o femenino.
CON_GENERO = re.compile(r"\b(él|ella|el residente|la residente|abuel[oa]|señor|señora|solo|sola)\b", re.IGNORECASE)


def _desvios_que_emite_el_proceso_diario() -> list[tuple[str, dict]]:
    """Todo lo que `baseline_runner.process_hogar` puede emitir con las alertas de horario
    apagadas (el default): por clase, tendencia y multivariado. De las clases de puerta solo
    sale que falten."""
    casos = [("multivariado", {"z_score": 4.2, "top_contribuyentes": ["cam_comida_n"]})]
    for clase in DEL_CLASIFICADOR:
        casos += [("clase", {"clase": clase, "tipo": t, "valor_hoy": 0, "mediana_historica": 3}) for t in ("ausencia", "ausencia_sostenida", "exceso", "deficit")]
        casos += [("tendencia", {"clase": clase, "direccion": d, "mediana_base": 40.0, "mediana_reciente": 10.0}) for d in ("sube", "baja")]
    for clase in DE_PUERTA:
        casos += [("clase", {"clase": clase, "tipo": t, "valor_hoy": 0, "mediana_historica": 1}) for t in ("ausencia", "ausencia_sostenida")]
    return casos


@pytest.mark.parametrize("detector, detalle", _desvios_que_emite_el_proceso_diario())
def test_ningun_aviso_muestra_claves_nombres_tecnicos_ni_porcentajes(detector, detalle):
    titulo, cuerpo = texto_desvio(detector, detalle)

    for texto in (titulo, cuerpo):
        assert texto.strip()
        assert "cam_" not in texto and ":" not in texto and "_" not in texto
        assert "%" not in texto and not re.search(r"\d", texto)
        assert detector not in texto and str(detalle.get("tipo")) not in texto
        assert not CON_GENERO.search(texto)


@pytest.mark.parametrize("detector, detalle, titulo, cuerpo", [
    ("clase", {"clase": "cam_comida", "tipo": "ausencia"}, "Cambio en la rutina", "Hoy no se registró comer o beber"),
    ("clase", {"clase": "cam_cocina", "tipo": "ausencia_sostenida"}, "Cambio en la rutina", "Varios días sin cocinar"),
    ("clase", {"clase": "cam_descanso_sofa", "tipo": "exceso"}, "Cambio en la rutina", "Más tiempo de lo habitual en descansar"),
    ("clase", {"clase": "cam_moverse", "tipo": "deficit"}, "Cambio en la rutina", "Menos tiempo de lo habitual en moverse por la casa"),
    ("clase", {"clase": "cam_salir_casa", "tipo": "ausencia"}, "Cambio en la rutina", "Hoy no se registró ninguna salida de casa"),
    ("clase", {"clase": "cam_entrar_casa", "tipo": "ausencia"}, "Cambio en la rutina", "Hoy no se registró ninguna entrada a casa"),
    ("clase", {"clase": "cam_salir_casa", "tipo": "ausencia_sostenida"}, "Cambio en la rutina", "Varios días sin salir de casa"),
    ("tendencia", {"clase": "cam_ocio_sentado", "direccion": "sube"}, "Tendencia de varias semanas", "Cada vez más tiempo en estar sin moverse (leer, TV, teléfono)"),
    ("tendencia", {"clase": "cam_cocina", "direccion": "baja"}, "Tendencia de varias semanas", "Cada vez menos tiempo en cocinar"),
    ("multivariado", {"z_score": 3.1}, "Día distinto a lo habitual", "La forma general del día se apartó de su rutina"),
])
def test_el_aviso_dice_lo_mismo_que_la_pantalla_de_la_app(detector, detalle, titulo, cuerpo):
    """Los textos de `desvioDeRutina` en `presentacion.ts` del Frontend, uno por uno."""
    assert texto_desvio(detector, detalle) == (titulo, cuerpo)


@pytest.mark.parametrize("detalle", [
    {"clase": "cam_clase_que_no_existe", "tipo": "ausencia"},
    {"clase": "cam_cocina", "tipo": "tipo_nuevo"},
    {"tipo": "exceso"},
    {},
])
def test_una_clase_o_un_tipo_desconocidos_no_se_filtran_al_aviso(detalle):
    """Si el detector por clase suma una clase o un tipo que este módulo no conoce, el aviso
    habla de "su rutina": nunca copia el nombre interno."""
    for detector in ("clase", "tendencia", "timing", "detector_nuevo"):
        titulo, cuerpo = texto_desvio(detector, detalle)
        assert "cam_" not in titulo + cuerpo and "_" not in titulo + cuerpo
        assert not CLAVE_INTERNA.search(cuerpo)
        assert "rutina" in cuerpo or "cocinar" in cuerpo


def test_el_aviso_de_horario_muestra_las_horas_y_ninguna_clave():
    """Las alertas de horario están apagadas por defecto. Encendidas, el aviso lleva dos horas
    (por eso tiene dos puntos), como en la app, y nada más técnico que eso."""
    titulo, cuerpo = texto_desvio("timing", {"clase": "cam_cocina", "minuto_del_dia": 485, "horario_habitual_min": 750.4, "desvio_min": 265})
    assert (titulo, cuerpo) == ("Horario fuera de lo habitual", "Momento de cocinar: 8:05 · habitual 12:30")
    assert "cam_" not in cuerpo and "%" not in cuerpo and not CLAVE_INTERNA.search(cuerpo)
    assert texto_desvio("timing", {"clase": "cam_comida"}) == ("Horario fuera de lo habitual", "Cambió el horario de comer o beber")


@pytest.fixture
def avisos(monkeypatch):
    """Los avisos que el proceso diario manda al celular, en vez de mandarlos."""
    enviados: list[dict] = []
    monkeypatch.setattr(baseline_runner, "enviar_push_alerta", lambda tokens, **datos: enviados.append(datos))
    return enviados


def test_el_dia_sin_comidas_avisa_con_texto_legible(cliente, crear_hogar, avisos):
    hogar = crear_hogar("hh113")
    _enviar_como_la_controladora(cliente, hogar, _eventos_de_camara(sin_comida_el=DIA_DEL_DESVIO))

    assert ejecutar_linea_base(DIA_DEL_DESVIO) is True

    assert [(a["titulo"], a["cuerpo"]) for a in avisos] == [
        ("Día distinto a lo habitual", "La forma general del día se apartó de su rutina"),
        ("Cambio en la rutina", "Hoy no se registró comer o beber"),
    ]
    for aviso in avisos:
        assert "cam_" not in aviso["cuerpo"] and ":" not in aviso["cuerpo"]


@pytest.mark.parametrize("dias_sin_salir, cuerpos", [
    (1, {"Hoy no se registró ninguna salida de casa", "Hoy no se registró ninguna entrada a casa"}),
    (4, {"Varios días sin salir de casa", "Varios días sin entrar a casa"}),
])
def test_los_dias_sin_salir_avisan_con_texto_legible(cliente, crear_hogar, avisos, dias_sin_salir, cuerpos):
    hogar = crear_hogar("hh113")
    _enviar_como_la_controladora(cliente, hogar, _historial_con_puerta(sin_salir_los_ultimos=dias_sin_salir))

    assert ejecutar_linea_base(DIA_DEL_DESVIO) is True

    assert {a["cuerpo"] for a in avisos} == cuerpos
    assert {a["titulo"] for a in avisos} == {"Cambio en la rutina"}
    for aviso in avisos:
        assert "cam_" not in aviso["cuerpo"] and ":" not in aviso["cuerpo"]
