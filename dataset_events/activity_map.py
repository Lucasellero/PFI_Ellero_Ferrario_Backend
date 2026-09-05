"""Mapeo de etiquetas de actividad CASAS -> vocabulario del proyecto.

Dos niveles:
  - `canonical`: 19 clases, la granularidad que el dataset de sensores soporta
    con volumen. Es lo que consume `baseline/` para aprender la rutina.
  - `coarse`: 9 clases (+ `otro`), el subconjunto que una cámara única puede llegar a
    producir (ver `fusion/PLAN_sustitucion_clip.md` y "Vocabulario de cámara" en
    `baseline/README.md`, 2026-09-05). Permite comparar "desvíos detectables con
    sensores" vs "solo con cámara". `salir_casa`/`entrar_casa` son eventos
    puntuales acá, no un estado "fuera_casa" con duración (la cámara nunca ve
    "afuera"). `comida_mesa` (comer) vs `comida` (beber/medicación, residual
    ambiguo) depende de la Fase 1 de zonas del hogar, todavía no implementada —
    ver nota en `COARSE` abajo. No incluye `moverse` (sin análogo en los
    sensores de CASAS).

`descartar` = la etiqueta existe pero no representa actividad del residente
(ruido de sensor médico, entradas/salidas de personal de cuidado). El ETL las
cuenta y las omite.
"""
from __future__ import annotations

# etiqueta CASAS (case-insensitive) -> clase canónica
CANONICAL: dict[str, str] = {
    # --- dormir ---
    "sleep": "dormir",
    "go_to_sleep": "dormir",
    "wake_up": "dormir",
    "wakeup": "dormir",
    "nap": "dormir",
    "sleep_out_of_bed": "dormir",
    # --- baño ---
    "toilet": "bano",
    "bed_toilet_transition": "bano_nocturno",
    # --- aseo personal ---
    "personal_hygiene": "aseo",
    "bathe": "aseo",
    "shower": "aseo",
    "groom": "aseo",
    "dress": "aseo",
    # --- cocinar / lavar platos ---
    "cook": "cocinar",
    "cook_breakfast": "cocinar",
    "cook_lunch": "cocinar",
    "cook_dinner": "cocinar",
    "wash_dishes": "lavar_platos",
    "wash_breakfast_dishes": "lavar_platos",
    "wash_lunch_dishes": "lavar_platos",
    "wash_dinner_dishes": "lavar_platos",
    # --- comer / beber / medicación ---
    "eat": "comer",
    "eat_breakfast": "comer",
    "eat_lunch": "comer",
    "eat_dinner": "comer",
    "drink": "beber",
    "morning_meds": "medicacion",
    "evening_meds": "medicacion",
    "take_medicine": "medicacion",
    # --- ocio ---
    "relax": "descansar",
    "watch_tv": "ver_tv",
    "read": "leer",
    "phone": "telefono",
    "usetelephone": "telefono",
    # --- trabajo ---
    "work": "trabajo",
    "work_at_desk": "trabajo",
    "work_at_table": "trabajo",
    "work_on_computer": "trabajo",
    # --- transiciones casa ---
    "leave_home": "salir_casa",
    "step_out": "salir_casa",
    "single_leave": "salir_casa",
    "single_step_out": "salir_casa",
    "single_leave_home": "salir_casa",
    "enter_home": "entrar_casa",
    "single_enter": "entrar_casa",
    "single_enter_home": "entrar_casa",
    # --- tareas del hogar ---
    "housekeeping": "tareas_hogar",
    "laundry": "tareas_hogar",
    "put_groceries_away": "tareas_hogar",
    # --- social ---
    "entertain_guests": "visitas",
    # --- otro (actividad real pero fuera de las clases de interés) ---
    "pet_activity": "otro",
    "smoking": "otro",
    "sew": "otro",
    "exercise": "otro",
    # --- descartar (no es actividad del residente) ---
    "resperate": "descartar",   # ruido de un sensor médico (documentado por CASAS)
    "staff_enter": "descartar",
    "staff_leave": "descartar",
}

# clase canónica -> clase gruesa (nivel que una cámara puede alcanzar)
#
# Vocabulario alineado con el usuario 2026-09-05 (ver "Vocabulario de cámara" en baseline/README.md):
# 9 clases de actividad + los eventos `caida`/`inmovilidad` que ya produce `vision/` aparte (no
# son actividades, no pasan por este mapeo). Reemplaza una versión anterior que fusionaba
# `salir_casa`/`entrar_casa` en un estado "fuera_casa" con duración — inconsistente con lo que una
# cámara puede observar: nunca ve "afuera", solo el instante de cruzar la puerta. Se corrigió para
# tratarlos como EVENTOS puntuales, cada uno su propia clase (¿salió hoy? ¿entró hoy?), no un
# estado a acumular. Si en el futuro hace falta saber "¿está afuera ahora?", se deriva con lógica
# simple sobre la secuencia de estos dos eventos, no prediciendo un tercer estado.
#
# También se sacaron de la lista objetivo (quedan mapeadas a `otro`, no a una clase propia):
# - `visitas` → detectar una segunda persona en cuadro está fuera de alcance (el informe excluye
#   el caso multi-residente, ver `vision/README.md`).
# - `tareas_hogar` → no estaba en el vocabulario de 9 clases acordado con el usuario.
# `bano_nocturno` se fusiona con `aseo` (misma actividad visible por cámara; la distinción de
# "de noche" es una propiedad derivable del horario del evento, no algo que el clasificador de
# escena necesite predecir como clase aparte).
#
# `comer` → `comida_mesa` (2026-09-05, ajuste con el usuario): se separa de `beber`/`medicacion`
# porque comer una comida es fisiológicamente distinto (dura más, gesto repetido, ocurre en una
# zona fija — la mesa) de un trago de agua o una pastilla (gesto breve, puede pasar en cualquier
# lado). **Depende de la detección de zona del hogar (Fase 1 de
# `fusion/PLAN_sustitucion_clip.md`), todavía no implementada/validada** — sin esa señal, la
# cámara no puede confirmar "está en la mesa", solo ve el gesto mano-boca igual que para
# beber/medicación. Se documenta el mapeo aspiracional ahora para que el vocabulario quede listo
# cuando se retome esa Fase 1; hasta entonces, tratar `comida_mesa` como no alcanzable todavía en
# producción (ver "Vocabulario de cámara" en `baseline/README.md`).
COARSE: dict[str, str] = {
    "dormir": "dormir",
    "descansar": "descanso_sofa",
    "ver_tv": "ocio_sentado",
    "leer": "ocio_sentado",
    "telefono": "ocio_sentado",
    "trabajo": "ocio_sentado",
    "cocinar": "cocina",
    "lavar_platos": "cocina",
    "comer": "comida_mesa",       # requiere zona (Fase 1, no implementada) — ver nota arriba
    "beber": "comida",            # residual ambiguo: gesto breve, puede pasar en cualquier zona
    "medicacion": "comida",       # ídem — indistinguible de beber sin zona ni objeto sostenido
    "aseo": "aseo",
    "bano": "aseo",
    "bano_nocturno": "aseo",      # "de noche" se deriva del horario, no es una clase de escena aparte
    "tareas_hogar": "otro",       # fuera del vocabulario de 9 clases acordado
    "salir_casa": "salir_casa",   # evento puntual, no un estado con duración
    "entrar_casa": "entrar_casa", # evento puntual, no un estado con duración
    "visitas": "otro",            # multi-residente fuera de alcance (ver vision/README.md)
    "otro": "otro",
}

# Nota: `moverse` (caminar/tránsito general) es parte del vocabulario objetivo de 9 clases pero no
# tiene análogo en CASAS (los sensores PIR no anotan "caminando" como actividad discreta) — solo
# existe del lado de la cámara (`vlm/`, categoría `Moving`). No aparece acá ni en `COARSE_CLASSES`.

CANONICAL_CLASSES = sorted(set(v for v in CANONICAL.values() if v != "descartar"))
COARSE_CLASSES = sorted(set(COARSE.values()))


def normalize_label(raw: str) -> str:
    """Normaliza una etiqueta CASAS cruda a la clave del diccionario.

    Quita comillas, espacios, y el sufijo de estado si viniera pegado.
    """
    return raw.strip().strip('"').strip("'").strip().lower()


def to_canonical(raw: str) -> str | None:
    """Devuelve la clase canónica, 'descartar', o None si la etiqueta es desconocida."""
    return CANONICAL.get(normalize_label(raw))


def to_coarse(canonical: str) -> str:
    return COARSE.get(canonical, "otro")
