"""Endpoints que consume la Pi (`runner/cloud_sync.py`), autenticados por API key del kit —
ver `auth.require_kit_by_api_key`. Nunca requieren que la Pi acepte conexiones entrantes: todo
es la Pi iniciando un POST saliente (ver informe, "arquitectura de red").
"""
from __future__ import annotations

from datetime import datetime, timezone

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from ..alertas import es_alerta, texto_alerta
from ..auth import require_kit_by_api_key
from ..db import get_db
from ..models import Camara, Evento, Kit
from ..notifications import enviar_push_alerta
from ..schemas import CamaraIn, CamaraOut, MonitoringEvent

router = APIRouter(prefix="/devices", tags=["devices"])


@router.post("/events", status_code=201)
def ingest_events(
    eventos: list[MonitoringEvent],
    kit: Kit = Depends(require_kit_by_api_key),
    db: Session = Depends(get_db),
) -> dict:
    """Ingestión en lote de eventos sincronizados desde `runner/cloud_sync.py`. `video_id` de
    cada `MonitoringEvent` debe coincidir con el `identificador_local` de una cámara del kit
    (creadas en fábrica, ver `alta_kit.py`). 409 si el kit todavía no está vinculado a un hogar
    con residente: la Pi no marca nada como sincronizado y reintenta en el próximo ciclo, así
    que no se pierde ningún evento mientras el familiar termina la vinculación."""
    if not eventos:
        return {"insertados": 0}
    hogar = kit.hogar
    if hogar is None or hogar.residente is None:
        raise HTTPException(
            status_code=409,
            detail=f"El kit {kit.numero_serie} todavía no está vinculado a un hogar con residente.",
        )

    camaras_por_id_local = {c.identificador_local: c for c in kit.camaras}
    insertados = 0
    alertas_nuevas: list[tuple[str, dict, str | None]] = []  # (event_type, metadata, ubicacion)
    for ev in eventos:
        camara = camaras_por_id_local.get(ev.video_id)
        if camara is None:
            raise HTTPException(
                status_code=404,
                detail=f"Cámara '{ev.video_id}' no está registrada para el kit {kit.numero_serie}. "
                       "Registrarla primero con POST /devices/camaras.",
            )
        if ev.wall_clock_ts is None:
            raise HTTPException(
                status_code=422,
                detail="Evento sin wall_clock_ts — la nube necesita el reloj real del evento.",
            )
        occurred_at = datetime.fromtimestamp(ev.wall_clock_ts, tz=timezone.utc)
        es_alerta_dura = es_alerta(ev.event_type.value, ev.metadata)
        db.add(Evento(
            camara_id=camara.id,
            residente_id=hogar.residente.id,
            event_type=ev.event_type.value,
            occurred_at=occurred_at,
            synced_at=datetime.now(timezone.utc),
            es_alerta=es_alerta_dura,
            metadata_json=ev.metadata,
        ))
        insertados += 1
        if es_alerta_dura:
            alertas_nuevas.append((ev.event_type.value, ev.metadata, camara.ubicacion or None))
    db.commit()

    # RF8: caídas/inmovilidad avisan al momento, no recién al otro día con el resumen de rutina
    # (eso es solo para `routine_deviation`, ver `baseline_runner.py::_registrar_desvio`).
    if alertas_nuevas:
        tokens = [f.familiar.push_token for f in hogar.familiares if f.familiar.push_token]
        for event_type, metadata, ubicacion in alertas_nuevas:
            titulo, cuerpo = texto_alerta(event_type, metadata, ubicacion)
            enviar_push_alerta(tokens, titulo=titulo, cuerpo=cuerpo, data={"event_type": event_type})

    return {"insertados": insertados}


@router.post("/camaras", response_model=CamaraOut, status_code=201)
def registrar_camara(
    camara_in: CamaraIn,
    kit: Kit = Depends(require_kit_by_api_key),
    db: Session = Depends(get_db),
) -> Camara:
    """Las cámaras del plan se crean en fábrica junto con el kit; esto queda para cuando la
    controladora informa una cámara que el backend no tiene. Idempotente, y nunca deja pasar más
    cámaras que `Kit.cantidad_camaras` (el plan comprado)."""
    existente = next(
        (c for c in kit.camaras if c.identificador_local == camara_in.identificador_local), None,
    )
    if existente is not None:
        return existente
    if len(kit.camaras) >= kit.cantidad_camaras:
        raise HTTPException(
            status_code=409,
            detail=f"El plan de este kit es de {kit.cantidad_camaras} cámara(s).",
        )
    camara = Camara(
        kit_id=kit.id, hogar_id=kit.hogar_id,
        identificador_local=camara_in.identificador_local, ubicacion=camara_in.ubicacion or "",
    )
    db.add(camara)
    db.commit()
    db.refresh(camara)
    return camara
