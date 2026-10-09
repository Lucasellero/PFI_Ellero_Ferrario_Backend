"""`POST /kits/vincular`: el familiar escanea el QR de la controladora y el kit pasa a ser de un
hogar nuevo, con él como `responsable_principal` (RF14). Contrato y decisiones en
`docs/vinculacion-kit.md` del repo Frontend, sección 5.

Límite de intentos fallidos por familiar (5 en 15 min → 429): con 8 caracteres Crockford el
código no se adivina, pero evita barridos. Se guarda en memoria del proceso — alcanza para una
instancia; con varias instancias detrás del balanceador (EB) cada una cuenta por separado.
"""
from __future__ import annotations

import time
from collections import defaultdict, deque
from datetime import datetime, timezone

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from ..auth import require_familiar_uid
from ..db import get_db
from ..models import ROL_RESPONSABLE_PRINCIPAL, Hogar, HogarFamiliar, Kit, hash_codigo
from ..permisos import familiar_de
from ..schemas import CamaraOut, KitVinculadoOut, VincularKitIn
from .mobile import NOMBRE_HOGAR_PROVISORIO

router = APIRouter(prefix="/kits")

MAX_INTENTOS_FALLIDOS = 5
VENTANA_INTENTOS_S = 15 * 60
_fallidos: dict[int, deque[float]] = defaultdict(deque)  # familiar_id -> timestamps


def _intentos_recientes(familiar_id: int) -> deque[float]:
    intentos = _fallidos[familiar_id]
    limite = time.monotonic() - VENTANA_INTENTOS_S
    while intentos and intentos[0] < limite:
        intentos.popleft()
    return intentos


def _respuesta(kit: Kit) -> KitVinculadoOut:
    return KitVinculadoOut(
        hogar_id=kit.hogar_id, numero_serie=kit.numero_serie, cantidad_camaras=kit.cantidad_camaras,
        camaras=[CamaraOut.model_validate(c) for c in sorted(kit.camaras, key=lambda c: c.id)],
    )


@router.post("/vincular", tags=["Vinculación"], summary="Vincular un kit con el código QR", response_model=KitVinculadoOut)
def vincular_kit(
    datos: VincularKitIn, uid: str = Depends(require_familiar_uid), db: Session = Depends(get_db),
) -> KitVinculadoOut:
    familiar = familiar_de(uid, db)
    intentos = _intentos_recientes(familiar.id)
    if len(intentos) >= MAX_INTENTOS_FALLIDOS:
        raise HTTPException(status_code=429, detail="Demasiados intentos. Esperá unos minutos y probá de nuevo.")

    # FOR UPDATE: si dos personas escanean el mismo QR a la vez, la segunda espera y ve el kit
    # ya vinculado (Postgres; SQLite lo ignora, pero serializa las escrituras igual).
    kit = (db.query(Kit).filter(Kit.codigo_vinculacion_hash == hash_codigo(datos.codigo))
           .with_for_update().first())
    if kit is None:
        intentos.append(time.monotonic())
        # mismo mensaje para "no existe" y "mal tipeado"
        raise HTTPException(status_code=404, detail="Ese código no corresponde a ningún kit. Revisalo en la etiqueta.")

    if kit.hogar_id is not None:
        ya_es_suyo = any(v.hogar_id == kit.hogar_id for v in familiar.hogares)
        if ya_es_suyo:  # reintento tras un corte de conexión: idempotente
            return _respuesta(kit)
        raise HTTPException(
            status_code=409,
            detail="Este kit ya está vinculado a otro hogar. Pedile a quien lo configuró que te invite desde su app.",
        )

    intentos.clear()
    ahora = datetime.now(timezone.utc)
    hogar = Hogar(nombre=NOMBRE_HOGAR_PROVISORIO, fecha_alta=ahora)
    db.add(hogar)
    db.flush()
    kit.hogar_id, kit.vinculado_por, kit.vinculado_at = hogar.id, familiar.id, ahora
    for camara in kit.camaras:
        camara.hogar_id = hogar.id
    db.add(HogarFamiliar(hogar_id=hogar.id, familiar_id=familiar.id, rol=ROL_RESPONSABLE_PRINCIPAL))
    db.commit()
    db.refresh(kit)
    return _respuesta(kit)
