"""Retención de eventos en la nube (informe, capítulo 3, Viabilidad Legal): los eventos de
actividad se conservan 12 meses y los de caída, inmovilidad y desvío de rutina, 24. Lo aplica la
tarea diaria de las 03:00 (`linea_base_diaria.py`), siempre después de calcular la línea base.

Los plazos no están escritos acá: salen de dos variables de entorno, en meses.

- `RETENCION_ACTIVIDAD_MESES`: eventos `activity`.
- `RETENCION_ALERTAS_MESES`: eventos `fall`, `immobility` y `routine_deviation`.

**Sin la variable, ese grupo no se borra.** Es a propósito: desplegar este código no borra nada
en producción hasta que alguien define el plazo. Un valor que no sea un entero mayor que cero
tampoco borra nada (queda un error en el log); en particular `0` no significa "borrar todo".

La antigüedad se mide con `Evento.occurred_at` (cuándo ocurrió, según el reloj de la
controladora), no con `synced_at`.
"""
from __future__ import annotations

import logging
import os
from datetime import datetime

from sqlalchemy.orm import Session

from .models import Evento

logger = logging.getLogger(__name__)

VARIABLE_ACTIVIDAD = "RETENCION_ACTIVIDAD_MESES"
VARIABLE_ALERTAS = "RETENCION_ALERTAS_MESES"

# variable de entorno -> tipos de evento (`events.schema.EventType`) a los que aplica ese plazo
GRUPOS = {
    VARIABLE_ACTIVIDAD: ("activity",),
    VARIABLE_ALERTAS: ("fall", "immobility", "routine_deviation"),
}


def meses_configurados(variable: str) -> int | None:
    """Plazo en meses de esa variable, o None si no está definida o no es un entero mayor que 0."""
    crudo = os.environ.get(variable, "").strip()
    if not crudo:
        return None
    try:
        meses = int(crudo)
    except ValueError:
        meses = 0
    if meses <= 0:
        logger.error("retencion: %s=%r no es un entero mayor que 0, no se borra nada de ese grupo", variable, crudo)
        return None
    return meses


def restar_meses(momento: datetime, meses: int) -> datetime:
    """`momento` menos `meses` meses de calendario. Si el día no existe en el mes de destino
    (31 de marzo menos un mes) se usa el último día de ese mes."""
    indice = momento.year * 12 + (momento.month - 1) - meses
    anio, mes = divmod(indice, 12)
    mes += 1
    for dia in range(momento.day, 27, -1):
        try:
            return momento.replace(year=anio, month=mes, day=dia)
        except ValueError:
            continue
    return momento.replace(year=anio, month=mes)


def aplicar_retencion(db: Session, ahora: datetime) -> dict[str, int]:
    """Borra los eventos vencidos de cada grupo que tenga su plazo configurado. Devuelve cuántos
    borró por variable (solo las configuradas; vacío = el paso no corrió). `ahora` con zona horaria."""
    borrados: dict[str, int] = {}
    for variable, tipos in GRUPOS.items():
        meses = meses_configurados(variable)
        if meses is None:
            continue
        corte = restar_meses(ahora, meses)
        borrados[variable] = (
            db.query(Evento)
            .filter(Evento.event_type.in_(tipos), Evento.occurred_at < corte)
            .delete(synchronize_session=False)
        )
    if borrados:
        db.commit()
        logger.info("retencion: eventos borrados por antigüedad %s", borrados)
    return borrados
