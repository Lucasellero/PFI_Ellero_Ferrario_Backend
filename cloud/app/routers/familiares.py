"""CRUD de familiares/roles (RF12): invitar/quitar familiares de un hogar y cambiar su rol.
Diseño y decisiones en `cloud/README.md` → "Cómo seguir" → punto 4.

Invitar es "a quien ya tiene cuenta" (sin invitación pendiente/por mail): se busca el
`firebase_uid` por email con `firebase_admin.auth.get_user_by_email`, 404 si esa persona nunca se
registró en la app — consistente con "no sobre-ingenierizar antes de medir si hace falta".
"""
from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, Response
from sqlalchemy.orm import Session

from ..auth import FIREBASE_CREDENTIALS_ENV, _get_firebase_app, require_familiar_uid
from ..db import get_db
from ..models import ROL_RESPONSABLE_PRINCIPAL, ROLES, Familiar, HogarFamiliar
from ..permisos import hogar_autorizado
from ..schemas import CambiarRolIn, FamiliarVinculadoOut, InvitarFamiliarIn

router = APIRouter(prefix="/hogares")


def _out(vinculo: HogarFamiliar) -> FamiliarVinculadoOut:
    f = vinculo.familiar
    return FamiliarVinculadoOut(id=f.id, nombre=f.nombre, email=f.email, rol=vinculo.rol)


def _validar_rol(rol: str) -> None:
    if rol not in ROLES:
        raise HTTPException(status_code=422, detail=f"rol tiene que ser uno de {ROLES}.")


def _es_unico_responsable(db: Session, hogar_id: int, familiar_id: int) -> bool:
    """True si `familiar_id` es el único `responsable_principal` del hogar (sacarlo, o bajarle
    el rol, dejaría el hogar sin nadie que pueda administrarlo)."""
    otros = (
        db.query(HogarFamiliar)
        .filter(
            HogarFamiliar.hogar_id == hogar_id,
            HogarFamiliar.rol == ROL_RESPONSABLE_PRINCIPAL,
            HogarFamiliar.familiar_id != familiar_id,
        )
        .count()
    )
    return otros == 0


@router.get("/{hogar_id}/familiares", tags=["Familiares"], summary="Listar los familiares del hogar", response_model=list[FamiliarVinculadoOut])
def listar_familiares(
    hogar_id: int, uid: str = Depends(require_familiar_uid), db: Session = Depends(get_db),
) -> list[FamiliarVinculadoOut]:
    """Cualquier familiar vinculado al hogar puede ver con quién lo comparte."""
    hogar_autorizado(hogar_id, uid, db)
    vinculos = db.query(HogarFamiliar).filter(HogarFamiliar.hogar_id == hogar_id).all()
    return [_out(v) for v in vinculos]


@router.post("/{hogar_id}/familiares", tags=["Familiares"], summary="Invitar a un familiar", response_model=FamiliarVinculadoOut, status_code=201)
def invitar_familiar(
    hogar_id: int, datos: InvitarFamiliarIn,
    uid: str = Depends(require_familiar_uid), db: Session = Depends(get_db),
) -> FamiliarVinculadoOut:
    """Solo `responsable_principal`. 404 si esa persona nunca se registró en la app (`POST /me`)
    — por ahora no hay invitación pendiente, tiene que crearse una cuenta primero. Idempotente:
    invitar a alguien ya vinculado devuelve su vínculo tal cual, no duplica ni pisa el rol."""
    _validar_rol(datos.rol)
    hogar_autorizado(hogar_id, uid, db, solo_responsable=True)

    app = _get_firebase_app()
    if app is None:
        raise HTTPException(
            status_code=501,
            detail=f"Firebase no está configurado en este backend (falta {FIREBASE_CREDENTIALS_ENV}).",
        )
    from firebase_admin import auth as firebase_auth

    try:
        usuario = firebase_auth.get_user_by_email(datos.email, app=app)
    except firebase_auth.UserNotFoundError:
        raise HTTPException(
            status_code=404,
            detail="Esa persona todavía no tiene una cuenta en la app. Pedile que se registre primero.",
        )

    familiar = db.query(Familiar).filter(Familiar.firebase_uid == usuario.uid).first()
    if familiar is None:
        # Cuenta de Firebase real pero sin `POST /me` todavía (nombre del token, no hay otro dato).
        nombre = usuario.display_name or datos.email.split("@")[0]
        familiar = Familiar(firebase_uid=usuario.uid, nombre=nombre, email=datos.email)
        db.add(familiar)
        db.flush()

    existente = (
        db.query(HogarFamiliar)
        .filter(HogarFamiliar.hogar_id == hogar_id, HogarFamiliar.familiar_id == familiar.id)
        .first()
    )
    if existente is not None:
        return _out(existente)

    vinculo = HogarFamiliar(hogar_id=hogar_id, familiar_id=familiar.id, rol=datos.rol)
    db.add(vinculo)
    db.commit()
    db.refresh(vinculo)
    return _out(vinculo)


@router.patch("/{hogar_id}/familiares/{familiar_id}", tags=["Familiares"], summary="Cambiar el rol de un familiar", response_model=FamiliarVinculadoOut)
def cambiar_rol(
    hogar_id: int, familiar_id: int, datos: CambiarRolIn,
    uid: str = Depends(require_familiar_uid), db: Session = Depends(get_db),
) -> FamiliarVinculadoOut:
    """Solo `responsable_principal`. 409 si es el único `responsable_principal` del hogar y se
    intenta bajarle el rol (nadie podría administrar el hogar después)."""
    _validar_rol(datos.rol)
    hogar_autorizado(hogar_id, uid, db, solo_responsable=True)
    vinculo = (
        db.query(HogarFamiliar)
        .filter(HogarFamiliar.hogar_id == hogar_id, HogarFamiliar.familiar_id == familiar_id)
        .first()
    )
    if vinculo is None:
        raise HTTPException(status_code=404, detail="Ese familiar no está vinculado a este hogar.")
    if (
        vinculo.rol == ROL_RESPONSABLE_PRINCIPAL
        and datos.rol != ROL_RESPONSABLE_PRINCIPAL
        and _es_unico_responsable(db, hogar_id, familiar_id)
    ):
        raise HTTPException(
            status_code=409,
            detail="Es el único responsable principal del hogar — asigná otro responsable antes de cambiarle el rol.",
        )
    vinculo.rol = datos.rol
    db.commit()
    db.refresh(vinculo)
    return _out(vinculo)


@router.delete("/{hogar_id}/familiares/{familiar_id}", tags=["Familiares"], summary="Desvincular a un familiar", status_code=204)
def quitar_familiar(
    hogar_id: int, familiar_id: int,
    uid: str = Depends(require_familiar_uid), db: Session = Depends(get_db),
) -> Response:
    """Un familiar puede desvincularse a sí mismo siempre; desvincular a otro requiere
    `responsable_principal`. 409 si es el único `responsable_principal` del hogar."""
    vinculo_propio = hogar_autorizado(hogar_id, uid, db)
    es_self = vinculo_propio.familiar_id == familiar_id
    if not es_self and vinculo_propio.rol != ROL_RESPONSABLE_PRINCIPAL:
        raise HTTPException(
            status_code=403, detail="Solo el responsable principal puede desvincular a otro familiar.",
        )
    objetivo = (
        db.query(HogarFamiliar)
        .filter(HogarFamiliar.hogar_id == hogar_id, HogarFamiliar.familiar_id == familiar_id)
        .first()
    )
    if objetivo is None:
        raise HTTPException(status_code=404, detail="Ese familiar no está vinculado a este hogar.")
    if objetivo.rol == ROL_RESPONSABLE_PRINCIPAL and _es_unico_responsable(db, hogar_id, familiar_id):
        raise HTTPException(
            status_code=409,
            detail="Es el único responsable principal del hogar — asigná otro responsable antes de desvincularlo.",
        )
    db.delete(objetivo)
    db.commit()
    return Response(status_code=204)
