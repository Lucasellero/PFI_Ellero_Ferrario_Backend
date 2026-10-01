"""Endpoints mínimos para la futura app móvil — autenticados con Firebase (`auth.require_familiar_uid`).
Alcance recortado a propósito para esta vuelta (ver `cloud/README.md`, "Pendiente"): alcanza para
probar el circuito completo evento -> alerta -> consulta, no el CRUD completo de familiares/roles
(RF12), que no tiene todavía un consumidor real (la app no existe).
"""
from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from ..auth import require_familiar_uid
from ..db import get_db
from ..models import Evento, Familiar, HogarFamiliar
from ..schemas import EventoOut

router = APIRouter(prefix="/hogares", tags=["mobile"])


def _hogar_autorizado(hogar_id: int, uid: str, db: Session) -> None:
    familiar = db.query(Familiar).filter(Familiar.firebase_uid == uid).first()
    if familiar is None:
        raise HTTPException(status_code=403, detail="Familiar no registrado")
    vinculo = db.query(HogarFamiliar).filter(
        HogarFamiliar.hogar_id == hogar_id, HogarFamiliar.familiar_id == familiar.id,
    ).first()
    if vinculo is None:
        raise HTTPException(status_code=403, detail="No autorizado para este hogar")


@router.get("/{hogar_id}/eventos", response_model=list[EventoOut])
def listar_eventos(
    hogar_id: int, limit: int = 50,
    uid: str = Depends(require_familiar_uid), db: Session = Depends(get_db),
) -> list[EventoOut]:
    _hogar_autorizado(hogar_id, uid, db)
    eventos = (
        db.query(Evento).join(Evento.residente)
        .filter(Evento.residente.has(hogar_id=hogar_id))
        .order_by(Evento.occurred_at.desc()).limit(limit).all()
    )
    return [EventoOut.from_orm_evento(e) for e in eventos]


@router.get("/{hogar_id}/alertas", response_model=list[EventoOut])
def listar_alertas(
    hogar_id: int, limit: int = 50,
    uid: str = Depends(require_familiar_uid), db: Session = Depends(get_db),
) -> list[EventoOut]:
    _hogar_autorizado(hogar_id, uid, db)
    eventos = (
        db.query(Evento).join(Evento.residente)
        .filter(Evento.residente.has(hogar_id=hogar_id), Evento.es_alerta.is_(True))
        .order_by(Evento.occurred_at.desc()).limit(limit).all()
    )
    return [EventoOut.from_orm_evento(e) for e in eventos]
