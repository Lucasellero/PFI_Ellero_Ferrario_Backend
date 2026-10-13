"""Baja del hogar: `DELETE /hogares/{hogar_id}` (`app/routers/mobile.py`)."""
from __future__ import annotations

import time
from datetime import datetime, timezone

import pytest

import main
from app.models import (
    Camara, Evento, Familiar, Hogar, HogarFamiliar, Kit, LineaBaseComportamiento, Residente,
)

from conftest import como


def _evento(camara: str = "cam1", tipo: str = "activity") -> dict:
    return {"video_id": camara, "event_type": tipo, "video_timestamp_s": 0.0, "wall_clock_ts": time.time(), "metadata": {}}


def _cargar_datos(cliente, db, hogar: dict) -> None:
    """Eventos enviados por la controladora, un desvío de rutina, una alerta resuelta por el
    familiar común y la línea base: todo lo que cuelga del residente."""
    respuesta = cliente.post(
        "/devices/events", headers={"X-API-Key": hogar["api_key"]},
        json=[_evento("cam1"), _evento("cam2"), _evento("cam1", "immobility")],
    )
    assert respuesta.status_code == 201, respuesta.text
    ahora = datetime.now(timezone.utc)
    db.add(Evento(
        camara_id=None, residente_id=hogar["residente_id"], event_type="routine_deviation",
        occurred_at=ahora, synced_at=ahora, es_alerta=True, metadata_json={},
    ))
    db.add(LineaBaseComportamiento(residente_id=hogar["residente_id"], parametros_cluster={"k": 3}))
    db.commit()
    un_evento = db.query(Evento).filter(Evento.residente_id == hogar["residente_id"]).first()
    resuelto = cliente.post(
        f"/hogares/{hogar['hogar_id']}/eventos/{un_evento.id}/resolver", headers=como(hogar["uid_comun"]),
    )
    assert resuelto.status_code == 204


def _estado(db, hogar: dict) -> dict:
    db.expire_all()
    return {
        "hogares": db.query(Hogar).filter(Hogar.id == hogar["hogar_id"]).count(),
        "residentes": db.query(Residente).filter(Residente.id == hogar["residente_id"]).count(),
        "eventos": db.query(Evento).filter(Evento.residente_id == hogar["residente_id"]).count(),
        "lineas_base": db.query(LineaBaseComportamiento).filter(
            LineaBaseComportamiento.residente_id == hogar["residente_id"]).count(),
        "vinculos": db.query(HogarFamiliar).filter(HogarFamiliar.hogar_id == hogar["hogar_id"]).count(),
    }


COMPLETO = {"hogares": 1, "residentes": 1, "eventos": 4, "lineas_base": 1, "vinculos": 2}
VACIO = {"hogares": 0, "residentes": 0, "eventos": 0, "lineas_base": 0, "vinculos": 0}


@pytest.fixture
def hogar(cliente, db, crear_hogar):
    datos = crear_hogar("A")
    _cargar_datos(cliente, db, datos)
    return datos


def test_un_familiar_comun_no_puede_dar_de_baja_el_hogar(cliente, db, hogar):
    respuesta = cliente.delete(f"/hogares/{hogar['hogar_id']}", headers=como(hogar["uid_comun"]))

    assert respuesta.status_code == 403
    assert _estado(db, hogar) == COMPLETO
    assert db.get(Kit, hogar["kit_id"]).hogar_id == hogar["hogar_id"]


def test_alguien_que_no_es_del_hogar_no_puede_darlo_de_baja(cliente, db, hogar, crear_hogar):
    otro = crear_hogar("B")

    # Ni siendo responsable principal de otro hogar.
    respuesta = cliente.delete(f"/hogares/{hogar['hogar_id']}", headers=como(otro["uid_responsable"]))
    assert respuesta.status_code == 403

    sin_cuenta = cliente.delete(f"/hogares/{hogar['hogar_id']}", headers=como("uid-desconocido"))
    assert sin_cuenta.status_code == 403

    assert _estado(db, hogar) == COMPLETO


def test_sin_token_no_se_puede_dar_de_baja(cliente, db, hogar):
    assert cliente.delete(f"/hogares/{hogar['hogar_id']}").status_code == 422
    assert _estado(db, hogar) == COMPLETO


def test_el_responsable_principal_da_de_baja_el_hogar(cliente, db, hogar):
    respuesta = cliente.delete(f"/hogares/{hogar['hogar_id']}", headers=como(hogar["uid_responsable"]))

    assert respuesta.status_code == 204
    assert _estado(db, hogar) == VACIO


def test_el_kit_y_las_camaras_quedan_desvinculados(cliente, db, hogar):
    cliente.delete(f"/hogares/{hogar['hogar_id']}", headers=como(hogar["uid_responsable"]))
    db.expire_all()

    kit = db.get(Kit, hogar["kit_id"])
    assert (kit.hogar_id, kit.vinculado_por) == (None, None)
    camaras = db.query(Camara).filter(Camara.kit_id == hogar["kit_id"]).order_by(Camara.id).all()
    assert [c.identificador_local for c in camaras] == ["cam1", "cam2"]  # las cámaras no se borran
    assert all(c.hogar_id is None and c.ubicacion == "" for c in camaras)


def test_las_cuentas_de_los_familiares_no_se_borran(cliente, db, hogar):
    cliente.delete(f"/hogares/{hogar['hogar_id']}", headers=como(hogar["uid_responsable"]))

    assert db.query(Familiar).count() == 2
    for uid in (hogar["uid_responsable"], hogar["uid_comun"]):
        perfil = cliente.get("/me", headers=como(uid))
        assert perfil.status_code == 200
        assert perfil.json()["hogares"] == []
    # El hogar ya no existe para nadie: mismo 403 que un hogar ajeno.
    assert cliente.get(f"/hogares/{hogar['hogar_id']}/eventos", headers=como(hogar["uid_responsable"])).status_code == 403


def test_no_toca_los_datos_de_otro_hogar(cliente, db, hogar, crear_hogar):
    otro = crear_hogar("B")
    _cargar_datos(cliente, db, otro)

    cliente.delete(f"/hogares/{hogar['hogar_id']}", headers=como(hogar["uid_responsable"]))

    assert _estado(db, hogar) == VACIO
    assert _estado(db, otro) == COMPLETO
    assert db.get(Kit, otro["kit_id"]).hogar_id == otro["hogar_id"]
    camaras = db.query(Camara).filter(Camara.kit_id == otro["kit_id"]).order_by(Camara.id).all()
    assert [(c.hogar_id, c.ubicacion) for c in camaras] == [(otro["hogar_id"], "Cocina"), (otro["hogar_id"], "Dormitorio")]


def test_despues_de_la_baja_la_controladora_recibe_410(cliente, db, hogar):
    """410 y no 409: le dice a la controladora que descarte esos eventos en vez de guardarlos."""
    cliente.delete(f"/hogares/{hogar['hogar_id']}", headers=como(hogar["uid_responsable"]))

    respuesta = cliente.post("/devices/events", headers={"X-API-Key": hogar["api_key"]}, json=[_evento()])

    assert respuesta.status_code == 410
    assert db.query(Evento).count() == 0


def test_un_kit_que_nunca_se_vinculo_sigue_recibiendo_409(cliente, db):
    """Kit recién salido de fábrica: sus eventos se guardan en la controladora hasta vincularlo."""
    _, codigo_hash = Kit.generar_codigo_vinculacion()
    api_key, api_key_hash = Kit.generar_api_key()
    db.add(Kit(numero_serie="LR-NUEVO", cantidad_camaras=1, codigo_vinculacion_hash=codigo_hash, device_api_key_hash=api_key_hash))
    db.commit()

    respuesta = cliente.post("/devices/events", headers={"X-API-Key": api_key}, json=[_evento()])

    assert respuesta.status_code == 409


def test_revinculado_pero_sin_residente_vuelve_a_409(cliente, hogar):
    """Mientras el familiar nuevo termina la vinculación, los eventos se guardan (no se descartan)."""
    cliente.delete(f"/hogares/{hogar['hogar_id']}", headers=como(hogar["uid_responsable"]))
    cliente.post("/kits/vincular", headers=como(hogar["uid_comun"]), json={"codigo": hogar["codigo"]})

    respuesta = cliente.post("/devices/events", headers={"X-API-Key": hogar["api_key"]}, json=[_evento()])

    assert respuesta.status_code == 409


def test_el_kit_se_puede_volver_a_vincular(cliente, db, hogar):
    cliente.delete(f"/hogares/{hogar['hogar_id']}", headers=como(hogar["uid_responsable"]))

    # Lo vincula otra persona, con el mismo código del QR.
    vinculado = cliente.post("/kits/vincular", headers=como(hogar["uid_comun"]), json={"codigo": hogar["codigo"]})

    assert vinculado.status_code == 200, vinculado.text
    cuerpo = vinculado.json()
    assert cuerpo["hogar_id"] is not None
    assert [(c["identificador_local"], c["ubicacion"]) for c in cuerpo["camaras"]] == [("cam1", None), ("cam2", None)]
    nuevo_hogar = cuerpo["hogar_id"]
    assert cliente.put(
        f"/hogares/{nuevo_hogar}/residente", headers=como(hogar["uid_comun"]), json={"nombre": "Otra Persona"},
    ).status_code == 204
    # La controladora vuelve a sincronizar, y el hogar nuevo no hereda nada del anterior.
    assert cliente.post(
        "/devices/events", headers={"X-API-Key": hogar["api_key"]}, json=[_evento()],
    ).status_code == 201
    eventos = cliente.get(f"/hogares/{nuevo_hogar}/eventos", headers=como(hogar["uid_comun"])).json()
    assert len(eventos) == 1


def test_dar_de_baja_un_hogar_sin_residente(cliente, db, crear_hogar):
    """Vinculación a medio hacer: kit vinculado pero todavía sin residente cargado."""
    datos = crear_hogar("A")
    db.query(Residente).filter(Residente.id == datos["residente_id"]).delete()
    db.commit()

    respuesta = cliente.delete(f"/hogares/{datos['hogar_id']}", headers=como(datos["uid_responsable"]))

    assert respuesta.status_code == 204
    assert db.query(Hogar).count() == 0
    assert db.get(Kit, datos["kit_id"]).hogar_id is None


def test_dar_de_baja_dos_veces_da_403(cliente, hogar):
    assert cliente.delete(f"/hogares/{hogar['hogar_id']}", headers=como(hogar["uid_responsable"])).status_code == 204
    assert cliente.delete(f"/hogares/{hogar['hogar_id']}", headers=como(hogar["uid_responsable"])).status_code == 403


def test_aparece_en_el_grupo_hogar_de_swagger():
    operacion = main.app.openapi()["paths"]["/hogares/{hogar_id}"]["delete"]

    assert operacion["tags"] == ["Hogar"]
    assert operacion["summary"] == "Dar de baja el hogar y borrar sus datos"
    assert operacion["security"] == [{"TokenFirebase": []}]
