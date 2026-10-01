"""Endpoints que consume la Pi (`runner/cloud_sync.py`), autenticados por API key de hogar —
ver `auth.require_hogar_by_api_key`. Nunca requieren que la Pi acepte conexiones entrantes: todo
es la Pi iniciando un POST saliente (ver informe, "arquitectura de red").
"""
from __future__ import annotations

from datetime import datetime, timezone

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from ..alertas import es_alerta
from ..auth import require_hogar_by_api_key
from ..db import get_db
from ..models import Camara, Evento, Hogar
from ..schemas import CamaraIn, CamaraOut, MonitoringEvent

router = APIRouter(prefix="/devices", tags=["devices"])


@router.post("/events", status_code=201)
def ingest_events(
    eventos: list[MonitoringEvent],
    hogar: Hogar = Depends(require_hogar_by_api_key),
    db: Session = Depends(get_db),
) -> dict:
    """Ingestión en lote de eventos sincronizados desde `runner/cloud_sync.py`. `video_id` de
    cada `MonitoringEvent` debe coincidir con una `Camara.local_id` ya registrada para este hogar
    (ver `POST /devices/camaras` — RF10 exige alta explícita, no auto-registro silencioso)."""
    if not eventos:
        return {"insertados": 0}

    camaras_por_local_id = {c.local_id: c for c in hogar.camaras}
    insertados = 0
    for ev in eventos:
        camara = camaras_por_local_id.get(ev.video_id)
        if camara is None:
            raise HTTPException(
                status_code=404,
                detail=f"Cámara '{ev.video_id}' no está registrada para el hogar {hogar.id}. "
                       "Registrarla primero con POST /devices/camaras.",
            )
        if ev.wall_clock_ts is None:
            raise HTTPException(
                status_code=422,
                detail="Evento sin wall_clock_ts — la nube necesita el reloj real del evento.",
            )
        occurred_at = datetime.fromtimestamp(ev.wall_clock_ts, tz=timezone.utc)
        db.add(Evento(
            camara_id=camara.id,
            residente_id=hogar.residente.id,
            event_type=ev.event_type.value,
            occurred_at=occurred_at,
            synced_at=datetime.now(timezone.utc),
            es_alerta=es_alerta(ev.event_type.value, ev.metadata),
            metadata_json=ev.metadata,
        ))
        insertados += 1
    db.commit()
    return {"insertados": insertados}


@router.post("/camaras", response_model=CamaraOut, status_code=201)
def registrar_camara(
    camara_in: CamaraIn,
    hogar: Hogar = Depends(require_hogar_by_api_key),
    db: Session = Depends(get_db),
) -> Camara:
    existente = db.query(Camara).filter(
        Camara.hogar_id == hogar.id, Camara.local_id == camara_in.local_id,
    ).first()
    if existente is not None:
        return existente
    camara = Camara(hogar_id=hogar.id, local_id=camara_in.local_id, ubicacion=camara_in.ubicacion)
    db.add(camara)
    db.commit()
    db.refresh(camara)
    return camara
