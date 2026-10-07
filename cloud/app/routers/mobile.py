"""Endpoints de la app móvil sobre un hogar — autenticados con Firebase (`auth.require_familiar_uid`),
permisos en `permisos.py`. Eventos/alertas (lectura) y la configuración inicial que sigue a la
vinculación del kit: ubicación de cada cámara (RF10) y datos del residente. Falta el CRUD de
familiares/roles (RF12, ver `cloud/README.md` → "Cómo seguir" punto 4).
"""
from __future__ import annotations

from datetime import datetime, timezone

from fastapi import APIRouter, Depends, HTTPException, Response
from sqlalchemy.orm import Session

from ..auth import require_familiar_uid
from ..db import get_db
from ..models import Camara, Evento, Residente
from ..permisos import hogar_autorizado
from ..schemas import CamaraOut, EventoOut, ResidenteIn, UbicacionIn

router = APIRouter(prefix="/hogares", tags=["mobile"])

# nombre provisorio que pone `POST /kits/vincular`; el residente lo reemplaza por "Casa de Rosa"
NOMBRE_HOGAR_PROVISORIO = "Mi hogar"


@router.get("/{hogar_id}/eventos", response_model=list[EventoOut])
def listar_eventos(
    hogar_id: int, limit: int = 50,
    tipo: str | None = None, desde: datetime | None = None, hasta: datetime | None = None,
    uid: str = Depends(require_familiar_uid), db: Session = Depends(get_db),
) -> list[EventoOut]:
    """RF13: filtros opcionales por tipo (`EventType`) y rango de fechas (`occurred_at`, ISO
    8601). Sin filtros, mismo comportamiento que antes (los últimos `limit`)."""
    hogar_autorizado(hogar_id, uid, db)
    query = db.query(Evento).join(Evento.residente).filter(Evento.residente.has(hogar_id=hogar_id))
    if tipo is not None:
        query = query.filter(Evento.event_type == tipo)
    if desde is not None:
        query = query.filter(Evento.occurred_at >= desde)
    if hasta is not None:
        query = query.filter(Evento.occurred_at <= hasta)
    eventos = query.order_by(Evento.occurred_at.desc()).limit(limit).all()
    return [EventoOut.from_orm_evento(e) for e in eventos]


@router.get("/{hogar_id}/alertas", response_model=list[EventoOut])
def listar_alertas(
    hogar_id: int, limit: int = 50,
    uid: str = Depends(require_familiar_uid), db: Session = Depends(get_db),
) -> list[EventoOut]:
    hogar_autorizado(hogar_id, uid, db)
    eventos = (
        db.query(Evento).join(Evento.residente)
        .filter(Evento.residente.has(hogar_id=hogar_id), Evento.es_alerta.is_(True))
        .order_by(Evento.occurred_at.desc()).limit(limit).all()
    )
    return [EventoOut.from_orm_evento(e) for e in eventos]


@router.post("/{hogar_id}/eventos/{evento_id}/resolver", status_code=204)
def resolver_evento(
    hogar_id: int, evento_id: int,
    uid: str = Depends(require_familiar_uid), db: Session = Depends(get_db),
) -> Response:
    """"Marcar como resuelta" (mockup 03, RF8): cualquier familiar vinculado puede atenderla, no
    solo el `responsable_principal`. Idempotente: si ya estaba resuelta, no pisa quién la marcó
    primero (reintento tras un corte de conexión)."""
    vinculo = hogar_autorizado(hogar_id, uid, db)
    evento = (
        db.query(Evento).join(Evento.residente)
        .filter(Evento.id == evento_id, Evento.residente.has(hogar_id=hogar_id))
        .first()
    )
    if evento is None:
        raise HTTPException(status_code=404, detail="Ese evento no es de este hogar.")
    if evento.resuelta_at is None:
        evento.resuelta_at = datetime.now(timezone.utc)
        evento.resuelta_por = vinculo.familiar_id
        db.commit()
    return Response(status_code=204)


@router.get("/{hogar_id}/camaras", response_model=list[CamaraOut])
def listar_camaras(
    hogar_id: int, uid: str = Depends(require_familiar_uid), db: Session = Depends(get_db),
) -> list[Camara]:
    hogar_autorizado(hogar_id, uid, db)
    return db.query(Camara).filter(Camara.hogar_id == hogar_id).order_by(Camara.id).all()


@router.patch("/{hogar_id}/camaras/{camara_id}", response_model=CamaraOut)
def actualizar_camara(
    hogar_id: int, camara_id: int, cambios: UbicacionIn,
    uid: str = Depends(require_familiar_uid), db: Session = Depends(get_db),
) -> Camara:
    """RF10: el familiar asigna desde la app la ubicación de cada cámara. Sin repetir entre
    cámaras del mismo hogar (la app valida lo mismo, `validarUbicaciones`)."""
    hogar_autorizado(hogar_id, uid, db, solo_responsable=True)
    camara = db.query(Camara).filter(Camara.id == camara_id, Camara.hogar_id == hogar_id).first()
    if camara is None:
        raise HTTPException(status_code=404, detail="Esa cámara no es de este hogar.")
    repetida = any(
        c.id != camara.id and c.ubicacion.lower() == cambios.ubicacion.lower()
        for c in db.query(Camara).filter(Camara.hogar_id == hogar_id)
    )
    if repetida:
        raise HTTPException(
            status_code=409,
            detail=f'"{cambios.ubicacion}" ya es otra cámara. Sumale un detalle (ej. "{cambios.ubicacion} 2").',
        )
    camara.ubicacion = cambios.ubicacion
    db.commit()
    db.refresh(camara)
    return camara


@router.put("/{hogar_id}/residente", status_code=204)
def guardar_residente(
    hogar_id: int, datos: ResidenteIn,
    uid: str = Depends(require_familiar_uid), db: Session = Depends(get_db),
) -> Response:
    """Crea o actualiza el residente (1:1 con el hogar, RNF11). Es el último paso de la
    configuración inicial: con residente, `GET /me` hace que la app pase a las pestañas."""
    vinculo = hogar_autorizado(hogar_id, uid, db, solo_responsable=True)
    hogar = vinculo.hogar
    if hogar.residente is None:
        db.add(Residente(hogar_id=hogar.id, nombre=datos.nombre, telefono=datos.telefono))
    else:
        hogar.residente.nombre, hogar.residente.telefono = datos.nombre, datos.telefono
    if hogar.nombre == NOMBRE_HOGAR_PROVISORIO:
        hogar.nombre = f"Casa de {datos.nombre.split()[0]}"
    db.commit()
    return Response(status_code=204)
