"""Las pruebas de `cloud/` corren siempre contra un SQLite temporal, nunca contra la base que
tenga configurada la terminal. `app/db.py` crea el engine al importarse a partir de
`DATABASE_URL`, así que la variable se pisa acá, antes de importar nada de `app`.
"""
from __future__ import annotations

import atexit
import os
import shutil
import tempfile
from pathlib import Path

_DIR = Path(tempfile.mkdtemp(prefix="lares-cloud-tests-"))
os.environ["DATABASE_URL"] = f"sqlite:///{(_DIR / 'cloud_test.db').as_posix()}"
os.environ["LINEA_BASE_DIARIA_DESACTIVADA"] = "1"
for _variable in (
    "RETENCION_ACTIVIDAD_MESES", "RETENCION_ALERTAS_MESES",
    "GOOGLE_APPLICATION_CREDENTIALS", "GOOGLE_APPLICATION_CREDENTIALS_JSON",
    "ALERTAS_DE_HORARIO_ACTIVADAS", "ENTRADAS_Y_SALIDAS_ACTIVADAS",
):
    os.environ.pop(_variable, None)

import pytest  # noqa: E402
from fastapi import Header  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402
from sqlalchemy import event  # noqa: E402

import main  # noqa: E402  (cloud/main.py)
from app import db as app_db  # noqa: E402
from app.auth import require_firebase_token  # noqa: E402
from app.models import (  # noqa: E402
    ROL_FAMILIAR, ROL_RESPONSABLE_PRINCIPAL, Camara, Familiar, Hogar, HogarFamiliar, Kit, Residente,
)

# Si `app.db` ya estaba importado con otra URL, mejor frenar acá que escribir en esa base.
assert app_db.engine.dialect.name == "sqlite", app_db.engine.url
assert _DIR.name in str(app_db.engine.url.database), app_db.engine.url


@event.listens_for(app_db.engine, "connect")
def _claves_foraneas(conexion, _registro) -> None:
    """SQLite no valida claves foráneas salvo que se le pida. Postgres (producción) sí: con esto
    un borrado en el orden equivocado falla acá igual que fallaría allá."""
    conexion.execute("PRAGMA foreign_keys=ON")


@atexit.register
def _limpiar() -> None:
    app_db.engine.dispose()
    shutil.rmtree(_DIR, ignore_errors=True)


def _token_de_prueba(authorization: str = Header(...)) -> dict:
    """Reemplaza la verificación de Firebase: el "token" es directamente el uid."""
    return {"uid": authorization.removeprefix("Bearer "), "email": "prueba@example.com"}


@pytest.fixture(autouse=True)
def base_limpia():
    app_db.Base.metadata.drop_all(app_db.engine)
    app_db.Base.metadata.create_all(app_db.engine)
    yield


@pytest.fixture
def db():
    sesion = app_db.SessionLocal()
    yield sesion
    sesion.close()


@pytest.fixture
def cliente():
    main.app.dependency_overrides[require_firebase_token] = _token_de_prueba
    yield TestClient(main.app)
    main.app.dependency_overrides.clear()


def como(uid: str) -> dict:
    """Encabezados de un familiar logueado."""
    return {"Authorization": f"Bearer {uid}"}


@pytest.fixture
def crear_hogar(db):
    """Arma un hogar completo, como queda después de vincular el kit desde la app: kit con dos
    cámaras ubicadas, residente, un `responsable_principal` y un `familiar` común."""

    def _crear(sufijo: str) -> dict:
        codigo, codigo_hash = Kit.generar_codigo_vinculacion()
        api_key, api_key_hash = Kit.generar_api_key()
        hogar = Hogar(nombre=f"Casa de {sufijo}")
        responsable = Familiar(firebase_uid=f"uid-resp-{sufijo}", nombre="Responsable", email=f"resp-{sufijo}@example.com")
        comun = Familiar(firebase_uid=f"uid-comun-{sufijo}", nombre="Familiar", email=f"comun-{sufijo}@example.com")
        db.add_all([hogar, responsable, comun])
        db.flush()
        kit = Kit(
            numero_serie=f"LR-{sufijo}", cantidad_camaras=2, codigo_vinculacion_hash=codigo_hash,
            device_api_key_hash=api_key_hash, hogar_id=hogar.id, vinculado_por=responsable.id,
            vinculado_at=hogar.fecha_alta,
        )
        db.add(kit)
        db.flush()
        camaras = [
            Camara(kit_id=kit.id, hogar_id=hogar.id, identificador_local="cam1", ubicacion="Cocina"),
            Camara(kit_id=kit.id, hogar_id=hogar.id, identificador_local="cam2", ubicacion="Dormitorio"),
        ]
        residente = Residente(hogar_id=hogar.id, nombre=f"Residente {sufijo}", telefono="+5491123456789")
        db.add_all([
            *camaras, residente,
            HogarFamiliar(hogar_id=hogar.id, familiar_id=responsable.id, rol=ROL_RESPONSABLE_PRINCIPAL),
            HogarFamiliar(hogar_id=hogar.id, familiar_id=comun.id, rol=ROL_FAMILIAR),
        ])
        db.commit()
        return {
            "hogar_id": hogar.id, "kit_id": kit.id, "residente_id": residente.id,
            "camara_ids": [c.id for c in camaras], "codigo": codigo, "api_key": api_key,
            "uid_responsable": responsable.firebase_uid, "uid_comun": comun.firebase_uid,
            "responsable_id": responsable.id, "comun_id": comun.id,
        }

    return _crear
