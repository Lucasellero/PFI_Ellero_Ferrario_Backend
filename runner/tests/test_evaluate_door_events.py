"""`runner/evaluate_door_events.py`: las piezas de la validación con video del detector de
puerta, con trayectorias sintéticas. No corre YOLO ni necesita los videos: prueba cómo se
ajusta la zona y cómo se agrega la ausencia que le falta a un clip recortado.
"""
from __future__ import annotations

import pytest

from runner import evaluate_door_events as ede
from runner.door_events import LABEL_ENTRAR, LABEL_SALIR

PUERTA = [[0.8, 0.0], [1.0, 0.0], [1.0, 1.0], [0.8, 1.0]]
EN_PUERTA = (0.9, 0.5)
EN_SALA = (0.4, 0.5)


def _clip(*posiciones) -> list:
    """Una muestra cada 0,1 s; cada posición dura 1 s."""
    return [(i * 0.1, p) for i, p in enumerate(pos for p in posiciones for pos in [p] * 10)]


def test_los_participantes_de_ajuste_y_de_prueba_no_se_pisan():
    grupos = (ede.FIT_PARTICIPANTS, ede.TEST_PARTICIPANTS, ede.UNTRIMMED_PARTICIPANTS)
    todos = [p for g in grupos for p in g]
    assert len(todos) == len(set(todos)) == 18


def test_el_nombre_del_clip_dice_actividad_participante_y_camara():
    assert ede._partes("Leave_p02_r00_v01_c07") == ("Leave", 2, "07")
    assert ede._partes("Cook.Cleandishes_p15_r03_v18_c03") == ("Cook.Cleandishes", 15, "03")
    assert ede._partes("P07T05C05") is None


def test_la_zona_es_el_rectangulo_de_los_percentiles_con_margen():
    puntos = [(0.80 + 0.01 * i, 0.40 + 0.02 * i) for i in range(11)]  # x 0,80–0,90; y 0,40–0,60
    (x1, y1), (x2, _), (_, y2), _ = ede.ajustar_zona(puntos)
    assert (x1, x2) == pytest.approx((0.81 - ede.MARGEN, 0.89 + ede.MARGEN))
    assert (y1, y2) == pytest.approx((0.42 - ede.MARGEN, 0.58 + ede.MARGEN))
    # Nunca se sale del cuadro.
    zona = ede.ajustar_zona([(0.99, 0.01), (1.0, 0.0), (0.98, 0.02)])
    assert all(0.0 <= v <= 1.0 for punto in zona for v in punto)


def test_un_clip_recortado_tal_cual_no_da_eventos():
    assert ede.correr(_clip(EN_SALA, EN_PUERTA), PUERTA) == []


def test_una_salida_necesita_la_ausencia_agregada_al_final():
    eventos = ede.correr(_clip(EN_SALA, EN_PUERTA), PUERTA, despues=True)
    assert [label for label, _ in eventos] == [LABEL_SALIR]
    assert eventos[0][1] == pytest.approx(1.9)  # la última vez que se la vio, en segundos del clip
    # Si lo último que se ve es lejos de la puerta, no es una salida.
    assert ede.correr(_clip(EN_PUERTA, EN_SALA, EN_SALA, EN_SALA), PUERTA, despues=True) == []


def test_una_entrada_necesita_la_ausencia_agregada_al_principio():
    eventos = ede.correr(_clip(EN_PUERTA, EN_SALA), PUERTA, antes=True)
    assert [label for label, _ in eventos] == [LABEL_ENTRAR]
    assert eventos[0][1] == pytest.approx(0.0)
    assert ede.correr(_clip(EN_SALA, EN_PUERTA), PUERTA, antes=True) == []  # aparece lejos de la puerta


def test_el_peor_caso_cuenta_una_entrada_y_una_salida_si_el_clip_empieza_y_termina_en_la_zona():
    peor_caso = ede.correr(_clip(EN_PUERTA, EN_SALA, EN_PUERTA), PUERTA, antes=True, despues=True)
    assert [label for label, _ in peor_caso] == [LABEL_ENTRAR, LABEL_SALIR]
    assert ede.correr(_clip(EN_SALA, EN_SALA), PUERTA, antes=True, despues=True) == []


def test_un_clip_sin_ninguna_deteccion_no_da_eventos():
    sin_nadie = [(i * 0.1, None) for i in range(30)]
    assert ede.correr(sin_nadie, PUERTA, antes=True, despues=True) == []
    assert ede._primera(sin_nadie) is None and ede._ultima(sin_nadie) is None
