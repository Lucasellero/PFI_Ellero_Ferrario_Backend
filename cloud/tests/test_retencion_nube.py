"""Retención de eventos en la nube (`app/retencion.py`), como paso de la tarea diaria de las
03:00 (`app/linea_base_diaria.py`)."""
from __future__ import annotations

from datetime import date, datetime, timedelta, timezone

import pytest

from app import linea_base_diaria
from app.models import Evento
from app.retencion import VARIABLE_ACTIVIDAD, VARIABLE_ALERTAS, aplicar_retencion, restar_meses

AHORA = datetime(2026, 10, 13, 6, 0, tzinfo=timezone.utc)  # las 03:00 de Argentina
TIPOS_DE_ALERTA = ("fall", "immobility", "routine_deviation")


def _evento(db, hogar: dict, tipo: str, meses: int, *, dias_extra: int = 0) -> int:
    """Crea un evento ocurrido hace `meses` meses de calendario (más `dias_extra` días)."""
    evento = Evento(
        camara_id=None if tipo == "routine_deviation" else hogar["camara_ids"][0],
        residente_id=hogar["residente_id"], event_type=tipo,
        occurred_at=restar_meses(AHORA, meses) - timedelta(days=dias_extra),
        synced_at=AHORA, es_alerta=tipo != "activity", metadata_json={},
    )
    db.add(evento)
    db.commit()
    return evento.id


def _ids(db) -> set[int]:
    db.expire_all()
    return {e.id for e in db.query(Evento).all()}


@pytest.fixture
def hogar(crear_hogar):
    return crear_hogar("A")


def test_sin_las_variables_de_entorno_no_se_borra_nada(db, hogar):
    viejos = {_evento(db, hogar, tipo, 120) for tipo in ("activity", *TIPOS_DE_ALERTA)}

    assert aplicar_retencion(db, AHORA) == {}
    assert _ids(db) == viejos


def test_la_tarea_diaria_no_borra_nada_sin_las_variables(db, hogar):
    viejos = {_evento(db, hogar, tipo, 120) for tipo in ("activity", *TIPOS_DE_ALERTA)}

    assert linea_base_diaria.ejecutar_linea_base(date(2026, 10, 12)) is True
    assert _ids(db) == viejos


def test_actividad_se_borra_pasado_su_plazo(db, hogar, monkeypatch):
    monkeypatch.setenv(VARIABLE_ACTIVIDAD, "12")
    vencido = _evento(db, hogar, "activity", 12, dias_extra=1)
    vigente = _evento(db, hogar, "activity", 12, dias_extra=-1)
    caida_de_13_meses = _evento(db, hogar, "fall", 13)

    assert aplicar_retencion(db, AHORA) == {VARIABLE_ACTIVIDAD: 1}
    assert _ids(db) == {vigente, caida_de_13_meses}
    assert vencido not in _ids(db)


@pytest.mark.parametrize("tipo", TIPOS_DE_ALERTA)
def test_caidas_inmovilidad_y_desvios_se_borran_pasado_su_plazo(db, hogar, monkeypatch, tipo):
    monkeypatch.setenv(VARIABLE_ALERTAS, "24")
    _evento(db, hogar, tipo, 24, dias_extra=1)
    vigente = _evento(db, hogar, tipo, 24, dias_extra=-1)
    actividad_de_30_meses = _evento(db, hogar, "activity", 30)

    assert aplicar_retencion(db, AHORA) == {VARIABLE_ALERTAS: 1}
    # La actividad vieja sigue: su plazo depende de la otra variable, que no está definida.
    assert _ids(db) == {vigente, actividad_de_30_meses}


def test_cada_grupo_usa_su_propio_plazo(db, hogar, monkeypatch):
    monkeypatch.setenv(VARIABLE_ACTIVIDAD, "12")
    monkeypatch.setenv(VARIABLE_ALERTAS, "24")
    _evento(db, hogar, "activity", 13)
    _evento(db, hogar, "immobility", 25)
    quedan = {
        _evento(db, hogar, "activity", 11),
        _evento(db, hogar, "fall", 13),
        _evento(db, hogar, "immobility", 23),
        _evento(db, hogar, "routine_deviation", 23),
    }

    assert aplicar_retencion(db, AHORA) == {VARIABLE_ACTIVIDAD: 1, VARIABLE_ALERTAS: 1}
    assert _ids(db) == quedan


def test_los_plazos_son_configurables(db, hogar, monkeypatch):
    monkeypatch.setenv(VARIABLE_ACTIVIDAD, "3")
    _evento(db, hogar, "activity", 4)
    vigente = _evento(db, hogar, "activity", 2)

    aplicar_retencion(db, AHORA)
    assert _ids(db) == {vigente}


@pytest.mark.parametrize("valor", ["", "  ", "0", "-12", "doce", "1.5"])
def test_un_valor_invalido_no_borra_nada(db, hogar, monkeypatch, valor):
    monkeypatch.setenv(VARIABLE_ACTIVIDAD, valor)
    monkeypatch.setenv(VARIABLE_ALERTAS, valor)
    viejos = {_evento(db, hogar, tipo, 120) for tipo in ("activity", *TIPOS_DE_ALERTA)}

    assert aplicar_retencion(db, AHORA) == {}
    assert _ids(db) == viejos


def test_un_tipo_de_evento_desconocido_no_se_toca(db, hogar, monkeypatch):
    monkeypatch.setenv(VARIABLE_ACTIVIDAD, "12")
    monkeypatch.setenv(VARIABLE_ALERTAS, "24")
    otro = _evento(db, hogar, "tipo_futuro", 120)

    aplicar_retencion(db, AHORA)
    assert _ids(db) == {otro}


def test_la_retencion_corre_despues_de_la_linea_base(db, crear_hogar, monkeypatch):
    crear_hogar("A")
    crear_hogar("B")
    pasos = []
    monkeypatch.setattr(linea_base_diaria, "process_hogar", lambda db, hogar, fecha: pasos.append(f"linea_base:{hogar.id}"))
    monkeypatch.setattr(linea_base_diaria, "aplicar_retencion", lambda db, ahora: pasos.append("retencion"))

    assert linea_base_diaria.ejecutar_linea_base(date(2026, 10, 12)) is True
    assert pasos == ["linea_base:1", "linea_base:2", "retencion"]


def test_la_linea_base_ve_los_eventos_antes_de_que_se_borren(db, hogar, monkeypatch):
    """De punta a punta con la variable definida: cuando se evalúa el hogar el evento vencido
    todavía está, y cuando termina la tarea ya no."""
    monkeypatch.setenv(VARIABLE_ACTIVIDAD, "12")
    vencido = _evento(db, hogar, "activity", 40)
    vigente = _evento(db, hogar, "activity", 1)
    vistos = []
    monkeypatch.setattr(
        linea_base_diaria, "process_hogar",
        lambda db, hogar, fecha: vistos.append({e.id for e in db.query(Evento).all()}),
    )

    linea_base_diaria.ejecutar_linea_base(date(2026, 10, 12))

    assert vistos == [{vencido, vigente}]
    assert _ids(db) == {vigente}


def test_la_retencion_corre_aunque_falle_la_linea_base_de_un_hogar(db, hogar, monkeypatch):
    monkeypatch.setenv(VARIABLE_ACTIVIDAD, "12")
    _evento(db, hogar, "activity", 40)

    def falla(db, hogar, fecha):
        raise RuntimeError("hogar con datos raros")

    monkeypatch.setattr(linea_base_diaria, "process_hogar", falla)

    assert linea_base_diaria.ejecutar_linea_base(date(2026, 10, 12)) is True
    assert _ids(db) == set()


def test_un_error_en_la_retencion_no_tira_la_tarea_diaria(db, hogar, monkeypatch):
    def falla(db, ahora):
        raise RuntimeError("base caída")

    monkeypatch.setattr(linea_base_diaria, "aplicar_retencion", falla)

    assert linea_base_diaria.ejecutar_linea_base(date(2026, 10, 12)) is True


@pytest.mark.parametrize("momento, meses, esperado", [
    (datetime(2026, 10, 13, tzinfo=timezone.utc), 12, datetime(2025, 10, 13, tzinfo=timezone.utc)),
    (datetime(2026, 10, 13, tzinfo=timezone.utc), 24, datetime(2024, 10, 13, tzinfo=timezone.utc)),
    (datetime(2026, 1, 15, tzinfo=timezone.utc), 3, datetime(2025, 10, 15, tzinfo=timezone.utc)),
    (datetime(2026, 3, 31, tzinfo=timezone.utc), 1, datetime(2026, 2, 28, tzinfo=timezone.utc)),
    (datetime(2024, 2, 29, tzinfo=timezone.utc), 12, datetime(2023, 2, 28, tzinfo=timezone.utc)),
    (datetime(2026, 12, 31, tzinfo=timezone.utc), 12, datetime(2025, 12, 31, tzinfo=timezone.utc)),
])
def test_restar_meses_de_calendario(momento, meses, esperado):
    assert restar_meses(momento, meses) == esperado
