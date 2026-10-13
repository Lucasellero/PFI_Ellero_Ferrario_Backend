"""Levanta el backend `cloud/` real (FastAPI + uvicorn) en esta máquina, contra un SQLite
temporal, para las pruebas de corte y reconexión (`test_cloud_sync_corte_reconexion.py`).

Se ejecuta como proceso aparte, con `DATABASE_URL` ya definida por la prueba:

    python backend_nube_local.py --crear-esquema     # crea las tablas y termina
    python backend_nube_local.py --puerto 8123       # sirve la API hasta que lo maten

Lo único que se reemplaza es la verificación del token de Firebase (el "token" es directamente el
uid, igual que en `cloud/tests/conftest.py`): la ingestión, la clave del kit, la vinculación y la
base son las reales. Se niega a arrancar si `DATABASE_URL` no es SQLite.
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

_CLOUD = Path(__file__).resolve().parents[2] / "cloud"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--puerto", type=int)
    parser.add_argument("--crear-esquema", action="store_true")
    args = parser.parse_args()

    if not os.environ.get("DATABASE_URL", "").startswith("sqlite:///"):
        sys.exit("backend_nube_local: DATABASE_URL tiene que ser un SQLite local")
    os.environ["LINEA_BASE_DIARIA_DESACTIVADA"] = "1"
    for variable in ("GOOGLE_APPLICATION_CREDENTIALS", "GOOGLE_APPLICATION_CREDENTIALS_JSON"):
        os.environ.pop(variable, None)
    sys.path.insert(0, str(_CLOUD))

    from fastapi import Header

    import main as cloud_main  # cloud/main.py
    from app import db as app_db
    from app import models  # noqa: F401  (registra las tablas en Base.metadata)
    from app.auth import require_firebase_token

    if args.crear_esquema:
        app_db.Base.metadata.create_all(app_db.engine)
        return

    def _token_de_prueba(authorization: str = Header(...)) -> dict:
        uid = authorization.removeprefix("Bearer ")
        return {"uid": uid, "email": f"{uid}@example.com"}

    cloud_main.app.dependency_overrides[require_firebase_token] = _token_de_prueba

    import uvicorn

    uvicorn.run(cloud_main.app, host="127.0.0.1", port=args.puerto, log_level="warning")


if __name__ == "__main__":
    main()
