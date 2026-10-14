"""Deriva `Evento.es_alerta` a partir de `event_type`/`metadata` — el informe lo describe como
"un booleano derivado de la severidad del evento, en lugar de modelar una entidad Notificación
aparte" (ver capítulo 3, "Diagrama de Base de Datos"). Un solo lugar para esta regla: la usan
tanto la ingestión (`routers/devices.py`, eventos que vienen de `vision/`/`vlm/`) como
`baseline_runner.py` (eventos `routine_deviation` que genera la nube).
"""
from __future__ import annotations


def es_alerta(event_type: str, metadata: dict) -> bool:
    if event_type == "fall":
        # ver runner/camera_worker.py: una caída de evidencia parcial se marca confirmed=False
        # y escala a inmovilidad después — no es la alerta dura en sí misma todavía.
        return bool(metadata.get("confirmed", True))
    if event_type == "immobility":
        return metadata.get("severity") == "alta"
    if event_type == "routine_deviation":
        return True
    return False  # "activity": nunca es una alerta


def texto_alerta(event_type: str, metadata: dict, ubicacion: str | None) -> tuple[str, str]:
    """Título/cuerpo del push para un evento agudo (`fall`/`immobility` con `es_alerta=True`).
    Texto simple (RNF8): la app tiene su propia presentación más rica
    (`src/features/eventos/presentacion.ts` del Frontend) para cuando el familiar abre el
    detalle; esto es solo lo que entra en la notificación del celular."""
    lugar = f" en {ubicacion.lower()}" if ubicacion else ""
    if event_type == "fall":
        return "Posible caída detectada", f"Revisá cómo está{lugar}."
    if event_type == "immobility":
        if metadata.get("trigger") == "post_fall_sospecha":
            return "Sin movimiento después de una posible caída", f"Revisá cómo está{lugar}."
        return "Sin movimiento en el piso", f"Revisá cómo está{lugar}."
    return "Alerta", "Revisá la app para más detalles."


# Los mismos textos que muestra la app para un desvío de rutina
# (`src/features/eventos/presentacion.ts` del Frontend, `desvioDeRutina`): el aviso del celular y
# la pantalla tienen que decir lo mismo. Si se cambia uno, cambiar el otro.
_CLASES_RUTINA: dict[str, str] = {
    "cam_cocina": "cocinar",
    "cam_comida": "comer o beber",
    "cam_descanso_sofa": "descansar",
    "cam_ocio_sentado": "estar sin moverse (leer, TV, teléfono)",
    "cam_moverse": "moverse por la casa",
    "cam_salir_casa": "salir de casa",
    "cam_entrar_casa": "entrar a casa",
}
# Entradas y salidas son eventos puntuales: se cuentan veces, no tiempo.
_CLASES_PUNTUALES = frozenset({"cam_salir_casa", "cam_entrar_casa"})
_SIN_EVENTOS_HOY: dict[str, str] = {
    "cam_salir_casa": "Hoy no se registró ninguna salida de casa",
    "cam_entrar_casa": "Hoy no se registró ninguna entrada a casa",
}


def _hora(minuto_del_dia: float) -> str:
    """`horaDesdeMinutos` del Frontend: 8:05, sin cero adelante."""
    return f"{int(minuto_del_dia // 60)}:{round(minuto_del_dia % 60):02d}"


def texto_desvio(detector: str, detalle: dict) -> tuple[str, str]:
    """Título y cuerpo del push de un desvío de rutina (`routine_deviation`), a partir del
    detector que lo generó y de lo que `baseline_runner` guarda en la metadata. Nunca muestra la
    clave interna del desvío ni el nombre de una clase: una clase desconocida es "su rutina"."""
    actividad = _CLASES_RUTINA.get(detalle.get("clase"), "su rutina")
    if detector == "timing":
        real, habitual = detalle.get("minuto_del_dia"), detalle.get("horario_habitual_min")
        if real is not None and habitual is not None:
            return "Horario fuera de lo habitual", f"Momento de {actividad}: {_hora(real)} · habitual {_hora(habitual)}"
        return "Horario fuera de lo habitual", f"Cambió el horario de {actividad}"
    if detector == "clase":
        puntual = detalle.get("clase") in _CLASES_PUNTUALES
        textos = {
            "ausencia": _SIN_EVENTOS_HOY.get(detalle.get("clase"), f"Hoy no se registró {actividad}"),
            "ausencia_sostenida": f"Varios días sin {actividad}",
            "exceso": f"Más veces de lo habitual: {actividad}" if puntual else f"Más tiempo de lo habitual en {actividad}",
            "deficit": f"Menos veces de lo habitual: {actividad}" if puntual else f"Menos tiempo de lo habitual en {actividad}",
        }
        return "Cambio en la rutina", textos.get(detalle.get("tipo"), f"Cambio en {actividad}")
    if detector == "tendencia":
        sube = detalle.get("direccion") == "sube"
        return "Tendencia de varias semanas", f"Cada vez {'más' if sube else 'menos'} tiempo en {actividad}"
    return "Día distinto a lo habitual", "La forma general del día se apartó de su rutina"
