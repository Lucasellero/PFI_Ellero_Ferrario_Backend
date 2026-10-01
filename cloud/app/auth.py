"""Dos esquemas de autenticación separados (ver `cloud/README.md`, "Autenticación"):

- Pi -> nube: API key simple por hogar, header `X-API-Key` (`require_hogar_by_api_key`).
- App móvil -> nube: `Authorization: Bearer <firebase_id_token>` (`require_familiar`). Si no hay
  credenciales de Firebase configuradas en este proceso, devuelve 501 en vez de romper el arranque
  de la app o fallar de forma confusa — así se puede probar/desplegar la ingestión y
  `baseline_runner` sin tener un proyecto de Firebase todavía.
"""
from __future__ import annotations

import os

from fastapi import Depends, Header, HTTPException
from sqlalchemy.orm import Session

from .db import get_db
from .models import Hogar, hash_api_key

FIREBASE_CREDENTIALS_ENV = "GOOGLE_APPLICATION_CREDENTIALS"

_firebase_app = None
_firebase_init_attempted = False


def _get_firebase_app():
    """Inicializa `firebase_admin` de forma perezosa (solo si alguien pega contra un endpoint
    que lo necesita) y una sola vez por proceso. Devuelve `None` si no hay credenciales
    configuradas — el caller decide qué hacer (acá: 501)."""
    global _firebase_app, _firebase_init_attempted
    if _firebase_init_attempted:
        return _firebase_app
    _firebase_init_attempted = True
    if not os.environ.get(FIREBASE_CREDENTIALS_ENV):
        return None
    import firebase_admin
    from firebase_admin import credentials

    _firebase_app = firebase_admin.initialize_app(credentials.ApplicationDefault())
    return _firebase_app


def require_hogar_by_api_key(
    x_api_key: str = Header(..., alias="X-API-Key"),
    db: Session = Depends(get_db),
) -> Hogar:
    hogar = db.query(Hogar).filter(Hogar.device_api_key_hash == hash_api_key(x_api_key)).first()
    if hogar is None:
        raise HTTPException(status_code=401, detail="API key inválida")
    return hogar


def require_familiar_uid(authorization: str = Header(...)) -> str:
    """Verifica el ID token de Firebase del header `Authorization: Bearer <token>` y devuelve el
    `firebase_uid`. 501 si este proceso no tiene Firebase configurado (ver docstring del módulo),
    401 si el token es inválido/expiró."""
    app = _get_firebase_app()
    if app is None:
        raise HTTPException(
            status_code=501,
            detail=(
                f"Firebase no está configurado en este backend (falta {FIREBASE_CREDENTIALS_ENV}) "
                "— este endpoint necesita un proyecto de Firebase real para autenticar la app móvil."
            ),
        )
    if not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Header Authorization mal formado")
    token = authorization.removeprefix("Bearer ").strip()

    from firebase_admin import auth as firebase_auth

    try:
        decoded = firebase_auth.verify_id_token(token, app=app)
    except Exception as exc:  # firebase_admin lanza varias excepciones propias, todas = token inválido
        raise HTTPException(status_code=401, detail="Token de Firebase inválido") from exc
    return decoded["uid"]
