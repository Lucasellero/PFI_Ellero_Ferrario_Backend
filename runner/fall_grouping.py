"""Agrupado de caídas: como mucho una posible caída publicada por minuto y por cámara.

La heurística de `vision/` puede disparar varias veces seguidas por la misma situación (alguien
que se mueve al borde del cuadro, o en el sillón): en una prueba con un video de Toyota
Smarthome (2026-10-14) salieron 7 sospechas en 26 segundos. Cada una llegaba a la app como un
evento aparte.

Regla, después de la segunda etapa (`CameraWorker._confirm_fall_event`), es decir, solo sobre
eventos que se iban a publicar:

- Dentro de `ventana_s` desde la última caída publicada, otra del **mismo nivel o menor** no se
  publica (queda en el log).
- Una caída **confirmada** después de una **sospecha** sí se publica: lo más grave nunca se tapa.
- No toca la vigilancia de `vision/` después de una sospecha: si la persona queda inmóvil, el
  evento `immobility` con `trigger: post_fall_sospecha` sale igual (no es un evento `fall`).
"""
from __future__ import annotations

DEFAULT_VENTANA_S = 60.0
SOSPECHA = "sospecha"
CONFIRMADA = "confirmada"
_GRAVEDAD = {SOSPECHA: 0, CONFIRMADA: 1}


class AgrupadorDeCaidas:
    """Una instancia por cámara. `publicar(ts, nivel)` dice si el evento se publica y, si sí,
    lo registra como la última caída publicada."""

    def __init__(self, ventana_s: float = DEFAULT_VENTANA_S):
        if ventana_s < 0:
            raise ValueError("ventana_s no puede ser negativa")
        self._ventana_s = ventana_s
        self._ultima: tuple[float, str] | None = None

    def publicar(self, ts: float, nivel: str) -> bool:
        if nivel not in _GRAVEDAD:
            raise ValueError(f"nivel desconocido: {nivel!r}")
        if self._ultima is not None:
            ts_ultima, nivel_ultima = self._ultima
            if ts < ts_ultima:
                self._ultima = None  # video en loop: el reloj volvió atrás
            elif ts - ts_ultima < self._ventana_s and _GRAVEDAD[nivel] <= _GRAVEDAD[nivel_ultima]:
                return False
        self._ultima = (ts, nivel)
        return True
