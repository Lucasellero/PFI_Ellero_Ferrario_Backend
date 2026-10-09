"""CLI para correr `app.baseline_runner` a mano (backfill/debug). En producción la corre la propia
API una vez por día (`app/linea_base_diaria.py`). Mismo criterio que `baseline/run_live.py` del lado local (ver su docstring): evalúa **un día
completo por vez**, por default el de ayer, nunca el día en curso.

Uso:
    py run_baseline_daily.py                              # ayer, todos los hogares
    py run_baseline_daily.py --date 2026-09-20 --hogar-id 1   # backfill/debug de un hogar puntual
"""
from __future__ import annotations

import argparse
from datetime import date, timedelta

from app.db import SessionLocal
from app.baseline_runner import process_hogar
from app.horario import hoy_argentina
from app.models import Hogar


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--date", type=str, default=None, help="Fecha a evaluar (YYYY-MM-DD). Default: ayer.")
    parser.add_argument("--hogar-id", type=int, default=None, help="Solo este hogar. Default: todos.")
    args = parser.parse_args()

    target_date = date.fromisoformat(args.date) if args.date else hoy_argentina() - timedelta(days=1)

    db = SessionLocal()
    try:
        hogares = [db.get(Hogar, args.hogar_id)] if args.hogar_id else db.query(Hogar).all()
        for hogar in hogares:
            if hogar is None:
                continue
            process_hogar(db, hogar, target_date)
    finally:
        db.close()


if __name__ == "__main__":
    main()
