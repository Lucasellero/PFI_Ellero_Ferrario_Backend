"""Perfil del familiar logueado — lo primero que pide la app móvil después del login
(`obtenerPerfil` / `crearPerfil` en `src/api/client.ts` del repo Frontend).

- `GET /me`: quién soy y a qué hogares accedo. 404 si la cuenta de Firebase todavía no tiene
  `Familiar` (la app entonces llama a `POST /me`).
- `POST /me`: crea el `Familiar` al registrarse. Idempotente: si ya existe, lo devuelve tal cual
  (la app puede reintentar si se cortó la red a mitad del registro).
"""
from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, Response
from sqlalchemy.orm import Session

from ..auth import require_familiar_uid, require_firebase_token
from ..db import get_db
from ..models import Familiar
from ..schemas import (
    FamiliarOut, HogarPerfilOut, KitResumenOut, PerfilIn, PerfilOut, PushTokenIn, ResidenteOut,
)

router = APIRouter(prefix="/me")


def _perfil(familiar: Familiar) -> PerfilOut:
    hogares = []
    for vinculo in familiar.hogares:  # HogarFamiliar
        h = vinculo.hogar
        r, k = h.residente, h.kit_activo
        hogares.append(HogarPerfilOut(
            id=h.id, nombre=h.nombre, rol=vinculo.rol,
            residente=ResidenteOut(id=r.id, nombre=r.nombre, telefono=r.telefono) if r else None,
            kit=KitResumenOut(numero_serie=k.numero_serie, cantidad_camaras=k.cantidad_camaras) if k else None,
        ))
    return PerfilOut(
        familiar=FamiliarOut(id=familiar.id, nombre=familiar.nombre, email=familiar.email),
        hogares=hogares,
    )


@router.get("", tags=["Cuenta"], summary="Ver el perfil y los hogares del familiar", response_model=PerfilOut)
def obtener_perfil(uid: str = Depends(require_familiar_uid), db: Session = Depends(get_db)) -> PerfilOut:
    familiar = db.query(Familiar).filter(Familiar.firebase_uid == uid).first()
    if familiar is None:
        raise HTTPException(status_code=404, detail="Tu cuenta todavía no tiene perfil.")
    return _perfil(familiar)


@router.post("", tags=["Cuenta"], summary="Crear el perfil del familiar", response_model=PerfilOut)
def crear_perfil(
    datos: PerfilIn, claims: dict = Depends(require_firebase_token), db: Session = Depends(get_db),
) -> PerfilOut:
    uid = claims["uid"]
    familiar = db.query(Familiar).filter(Familiar.firebase_uid == uid).first()
    if familiar is not None:
        return _perfil(familiar)
    email = claims.get("email")
    if not email:
        # solo pasa con proveedores sin email (teléfono, anónimo) — hoy solo está Email/Password
        raise HTTPException(status_code=400, detail="La cuenta de Firebase no tiene email.")
    familiar = Familiar(firebase_uid=uid, nombre=datos.nombre, email=email)
    db.add(familiar)
    db.commit()
    db.refresh(familiar)
    return _perfil(familiar)


@router.post("/push-token", tags=["Cuenta"], summary="Guardar el token de notificaciones del celular", status_code=204)
def registrar_push_token(
    datos: PushTokenIn, uid: str = Depends(require_familiar_uid), db: Session = Depends(get_db),
) -> Response:
    """La app lo llama al loguearse y cada vez que el SDK de FCM avisa que el token rotó
    (`onTokenRefresh`). Un token por familiar: se pisa al iniciar sesión en otro equipo."""
    familiar = db.query(Familiar).filter(Familiar.firebase_uid == uid).first()
    if familiar is None:
        raise HTTPException(status_code=404, detail="Tu cuenta todavía no tiene perfil.")
    familiar.push_token = datos.token
    db.commit()
    return Response(status_code=204)
