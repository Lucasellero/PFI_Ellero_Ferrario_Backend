"""Push (RF8) del lado de la cuenta y de lo que viaja en el aviso:

- `DELETE /me/push-token` borra el token solo si es el de ese celular (cerrar sesión en un
  celular viejo no le corta las push al nuevo).
- La `data` de la push trae `evento_id` y `hogar_id`, para que la app abra esa alerta al
  tocarla — tanto en caídas/inmovilidad (`POST /devices/events`) como en desvíos de rutina
  (`baseline_runner`).
"""
from __future__ import annotations

from datetime import date

import pytest

from app import baseline_runner
from app.models import Evento, Familiar, Hogar
from app.routers import devices

from conftest import como
from test_ingestion_idempotente import _enviar, _evento


@pytest.fixture
def hogar(crear_hogar):
    return crear_hogar("push")


def _token_guardado(db, uid: str) -> str | None:
    db.expire_all()
    return db.query(Familiar).filter(Familiar.firebase_uid == uid).one().push_token


def _registrar(cliente, uid: str, token: str) -> None:
    respuesta = cliente.post("/me/push-token", headers=como(uid), json={"token": token})
    assert respuesta.status_code == 204, respuesta.text


def test_cerrar_sesion_borra_el_token_de_ese_celular(cliente, db, hogar):
    uid = hogar["uid_responsable"]
    _registrar(cliente, uid, "token-celular-a")

    respuesta = cliente.delete("/me/push-token", headers=como(uid), params={"token": "token-celular-a"})

    assert respuesta.status_code == 204
    assert _token_guardado(db, uid) is None


def test_cerrar_sesion_en_el_celular_viejo_no_borra_el_del_nuevo(cliente, db, hogar):
    uid = hogar["uid_responsable"]
    _registrar(cliente, uid, "token-celular-viejo")
    _registrar(cliente, uid, "token-celular-nuevo")  # inició sesión en otro celular

    respuesta = cliente.delete("/me/push-token", headers=como(uid), params={"token": "token-celular-viejo"})

    assert respuesta.status_code == 204
    assert _token_guardado(db, uid) == "token-celular-nuevo"


def test_borrar_sin_token_guardado_es_idempotente(cliente, db, hogar):
    uid = hogar["uid_comun"]

    for _ in range(2):
        respuesta = cliente.delete("/me/push-token", headers=como(uid), params={"token": "cualquiera"})
        assert respuesta.status_code == 204
    assert _token_guardado(db, uid) is None


def test_borrar_sin_perfil_da_404(cliente):
    respuesta = cliente.delete("/me/push-token", headers=como("uid-sin-perfil"), params={"token": "t"})
    assert respuesta.status_code == 404


def test_borrar_pide_el_token(cliente, hogar):
    respuesta = cliente.delete("/me/push-token", headers=como(hogar["uid_responsable"]))
    assert respuesta.status_code == 422


@pytest.fixture
def avisos(monkeypatch):
    """`(tokens, data)` de cada push enviada, desde la ingestión y desde la corrida diaria."""
    enviados = []

    def capturar(tokens, **datos):
        enviados.append((tokens, datos["data"]))

    monkeypatch.setattr(devices, "enviar_push_alerta", capturar)
    monkeypatch.setattr(baseline_runner, "enviar_push_alerta", capturar)
    return enviados


def test_la_push_de_una_caida_lleva_el_evento_y_el_hogar(cliente, db, hogar, avisos):
    _registrar(cliente, hogar["uid_responsable"], "token-a")

    _enviar(cliente, hogar, [_evento("caida-1", tipo="fall", confirmed=True)])

    db.expire_all()
    caida = db.query(Evento).filter(Evento.event_type == "fall").one()
    assert avisos == [(["token-a"], {"event_type": "fall", "evento_id": caida.id, "hogar_id": hogar["hogar_id"]})]


def test_cada_alerta_de_un_lote_lleva_su_propio_evento(cliente, db, hogar, avisos):
    _registrar(cliente, hogar["uid_responsable"], "token-a")

    _enviar(cliente, hogar, [
        _evento("caida-1", tipo="fall", confirmed=True),
        _evento("actividad-1"),  # no es alerta: no avisa
        _evento("inmovil-1", tipo="immobility", camara="cam2", severity="alta"),
    ])

    db.expire_all()
    ids = {e.event_type: e.id for e in db.query(Evento).filter(Evento.es_alerta.is_(True))}
    assert [data["evento_id"] for _, data in avisos] == [ids["fall"], ids["immobility"]]


def test_la_push_de_un_desvio_de_rutina_lleva_el_evento_y_el_hogar(db, hogar, avisos):
    hogar_orm = db.get(Hogar, hogar["hogar_id"])
    detalle = {"clase": "cam_comida", "tipo": "ausencia", "valor_hoy": 0, "mediana_historica": 3}

    baseline_runner._registrar_desvio(db, hogar_orm, date(2026, 10, 15), "clase", "cam_comida:ausencia", detalle)

    desvio = db.query(Evento).filter(Evento.event_type == "routine_deviation").one()
    (_, data), = avisos
    assert data == {**detalle, "event_type": "routine_deviation", "evento_id": desvio.id, "hogar_id": hogar["hogar_id"]}
