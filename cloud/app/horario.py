"""Zona horaria de los hogares. El contenedor de Elastic Beanstalk corre en UTC, pero los días y
las horas que importan (qué es "hoy", a qué hora fue la primera comida, dónde corta cada día)
son los del residente. Todo lo que agrupa eventos por día u hora en `cloud/` tiene que usar esto,
nunca la zona de la máquina ni UTC.

UTC-3 fijo: Argentina no tiene horario de verano desde 2009. Si algún día hay hogares en otras
zonas, esto pasa a ser un campo del `Hogar`.
"""
from __future__ import annotations

from datetime import date, datetime, timedelta, timezone

ZONA_ARGENTINA = timezone(timedelta(hours=-3), "ART")


def hoy_argentina() -> date:
    return datetime.now(ZONA_ARGENTINA).date()
