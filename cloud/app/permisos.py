"""Quién puede hacer qué sobre un hogar (RF12). Compartido por `routers/mobile.py` y
`routers/kits.py`; los futuros endpoints de familiares/roles también lo usan.

- Cualquier vínculo (`familiar` o `responsable_principal`): leer eventos, alertas, cámaras.
- Solo `responsable_principal`: configurar el hogar (residente, ubicación de cámaras) y, a
  futuro, invitar o quitar familiares.
"""
from __future__ import annotations

from fastapi import HTTPException
from sqlalchemy.orm import Session

from .models import ROL_RESPONSABLE_PRINCIPAL, Familiar, HogarFamiliar


def familiar_de(uid: str, db: Session) -> Familiar:
    """El `Familiar` del token. 403 si la cuenta de Firebase todavía no hizo `POST /me`."""
    familiar = db.query(Familiar).filter(Familiar.firebase_uid == uid).first()
    if familiar is None:
        raise HTTPException(status_code=403, detail="Familiar no registrado")
    return familiar


def hogar_autorizado(hogar_id: int, uid: str, db: Session, *, solo_responsable: bool = False) -> HogarFamiliar:
    """Devuelve el vínculo familiar↔hogar. 403 si no existe, o si `solo_responsable` y el rol no
    es `responsable_principal`."""
    familiar = familiar_de(uid, db)
    vinculo = db.query(HogarFamiliar).filter(
        HogarFamiliar.hogar_id == hogar_id, HogarFamiliar.familiar_id == familiar.id,
    ).first()
    if vinculo is None:
        raise HTTPException(status_code=403, detail="No autorizado para este hogar")
    if solo_responsable and vinculo.rol != ROL_RESPONSABLE_PRINCIPAL:
        raise HTTPException(status_code=403, detail="Solo el responsable principal del hogar puede cambiar esto.")
    return vinculo
