"""Push vía Firebase Cloud Messaging (RF8). No-op con log si Firebase no está configurado en
este proceso (ver `auth.py` — mismo criterio: no bloquear el resto del sistema por falta de
credenciales que todavía no existen)."""
from __future__ import annotations

import logging

from .auth import _get_firebase_app

logger = logging.getLogger(__name__)


def enviar_push_alerta(tokens_dispositivo: list[str], titulo: str, cuerpo: str, data: dict | None = None) -> None:
    app = _get_firebase_app()
    if app is None:
        logger.warning(
            "Firebase no configurado — push NO enviado (título=%r, destinatarios=%d). "
            "Configurar GOOGLE_APPLICATION_CREDENTIALS para habilitarlo.",
            titulo, len(tokens_dispositivo),
        )
        return
    if not tokens_dispositivo:
        return

    from firebase_admin import messaging

    message = messaging.MulticastMessage(
        notification=messaging.Notification(title=titulo, body=cuerpo),
        data={k: str(v) for k, v in (data or {}).items()},
        tokens=tokens_dispositivo,
    )
    response = messaging.send_each_for_multicast(message, app=app)
    if response.failure_count:
        logger.warning("%d/%d push fallaron", response.failure_count, len(tokens_dispositivo))
