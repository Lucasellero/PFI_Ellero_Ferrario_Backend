"""Mapeo de etiquetas de actividad CASAS -> vocabulario del proyecto.

Dos niveles:
  - `canonical`: ~16 clases, la granularidad que el dataset de sensores soporta
    con volumen. Es lo que consume `baseline/` para aprender la rutina.
  - `coarse`: ~10 clases, el subconjunto que una cámara única puede llegar a
    producir (ver `fusion/PLAN_sustitucion_clip.md`). Permite comparar
    "desvíos detectables con sensores" vs "solo con cámara".

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
COARSE: dict[str, str] = {
    "dormir": "dormir",
    "descansar": "descanso_sofa",
    "ver_tv": "ocio_sentado",
    "leer": "ocio_sentado",
    "telefono": "ocio_sentado",
    "trabajo": "ocio_sentado",
    "cocinar": "cocina",
    "lavar_platos": "cocina",
    "comer": "comida",
    "beber": "comida",
    "medicacion": "comida",       # indistinguible de comer/beber desde cámara única
    "aseo": "aseo",
    "bano": "aseo",
    "bano_nocturno": "bano_nocturno",
    "tareas_hogar": "tareas_hogar",
    "salir_casa": "fuera_casa",
    "entrar_casa": "fuera_casa",
    "visitas": "social",
    "otro": "otro",
}

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
