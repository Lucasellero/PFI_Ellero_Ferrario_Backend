"""Simula el historial de un hogar para probar los desvíos de rutina de punta a punta (RF8): carga
60 días de rutina (CASAS hh113, el mismo CSV de `tests/`) como si los mandara el kit, corridos
para que el último día sea `--hoy`, y ese día sin comidas. A las 03:00 siguientes, el proceso
diario (`app/linea_base_diaria.py`) evalúa ese día, emite `cam_comida:ausencia` (y el
multivariado "Día distinto a lo habitual") y manda las push.

Mismos eventos que `_eventos_de_camara` de `tests/test_linea_base_produccion.py`: uno al empezar
cada actividad y uno por minuto mientras sigue. Sin `--enviar` solo muestra qué cargaría.

    URL_BACKEND=http://... API_KEY_KIT=... python simular_historial.py --hoy 2026-10-17 --enviar

Usado el 2026-10-17 con el kit de prueba `LR-000003` (ver README, "Cómo seguir" → 3). Los
eventos quedan en el hogar: para limpiarlos, dar de baja ese hogar (`DELETE /hogares/{id}`).
Solo con un hogar de prueba: a un hogar real le mete 60 días de actividad que no pasó.
"""
import argparse
import csv
import json
import os
import urllib.request
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

CSV = Path(__file__).resolve().parent / "tests" / "datos" / "casas_hh113_60dias.csv"
ZONA_ARGENTINA = timezone(timedelta(hours=-3))
ULTIMO_DIA_CASAS = date(2011, 11, 3)
COMIDA = "EatingDrinkingMedication"
LATIDO_S = 60
LOTE = 200

parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument("--hoy", type=date.fromisoformat, required=True, help="día que queda sin comidas (AAAA-MM-DD)")
parser.add_argument("--enviar", action="store_true", help="mandarlos de verdad (sin esto, solo muestra)")
args = parser.parse_args()
HOY = args.hoy

desplazamiento = HOY - ULTIMO_DIA_CASAS
eventos = []
with open(CSV, encoding="utf-8", newline="") as archivo:
    for fila in csv.DictReader(archivo):
        inicio = datetime.fromisoformat(fila["inicio"]).replace(tzinfo=ZONA_ARGENTINA) + desplazamiento
        if fila["activity_label"] == COMIDA and inicio.date() == HOY:
            continue
        for latido in range(int(fila["duracion_s"]) // LATIDO_S + 1):
            metadata = {"activity_label": fila["activity_label"], "confirmed": True}
            if latido:
                metadata["ongoing"] = True
            ts = (inicio + timedelta(seconds=latido * LATIDO_S)).timestamp()
            eventos.append({
                "video_id": "cam1", "event_type": "activity", "video_timestamp_s": 0.0,
                "wall_clock_ts": ts, "metadata": metadata,
                # Reintentar el script no duplica: el backend descarta event_id repetidos.
                "event_id": f"sim-casas-{int(ts)}-{fila['activity_label']}",
            })
eventos.sort(key=lambda e: e["wall_clock_ts"])

dias = sorted({datetime.fromtimestamp(e["wall_clock_ts"], ZONA_ARGENTINA).date() for e in eventos})
de_hoy = [e for e in eventos if datetime.fromtimestamp(e["wall_clock_ts"], ZONA_ARGENTINA).date() == HOY]
comidas_hoy = [e for e in de_hoy if e["metadata"]["activity_label"] == COMIDA]
print(f"{len(eventos)} eventos, {len(dias)} días ({dias[0]} a {dias[-1]}); hoy: {len(de_hoy)} eventos, {len(comidas_hoy)} de comida")

if not args.enviar:
    raise SystemExit(0)

url = os.environ["URL_BACKEND"].rstrip("/") + "/devices/events"
clave = os.environ["API_KEY_KIT"]
insertados = duplicados = 0
for i in range(0, len(eventos), LOTE):
    pedido = urllib.request.Request(
        url, data=json.dumps(eventos[i:i + LOTE]).encode(), method="POST",
        headers={"Content-Type": "application/json", "X-API-Key": clave},
    )
    with urllib.request.urlopen(pedido, timeout=60) as respuesta:
        r = json.load(respuesta)
    insertados += r["insertados"]
    duplicados += r["duplicados"]
    if (i // LOTE) % 10 == 0:
        print(f"  lote {i // LOTE + 1}/{-(-len(eventos) // LOTE)}: {insertados} insertados, {duplicados} duplicados")
print(f"Listo: {insertados} insertados, {duplicados} duplicados")
