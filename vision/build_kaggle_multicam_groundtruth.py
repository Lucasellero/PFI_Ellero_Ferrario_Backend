"""Genera `videos/Kaggle/Fall.csv` a partir de `videos/Kaggle/kaggle_data.csv`
(Multiple Cameras Fall Dataset, Auvinet et al. — 24 escenas grabadas con 8
cámaras simultáneas cada una).

`kaggle_data.csv` trae, por (chute, cam), segmentos [start_frame, end_frame]
con label 1=caída / 0=no-caída. El chute24 no tiene ninguna fila en ese CSV
(no está documentado como caída ni como confusor en esta fuente), así que se
excluye en vez de asumir su contenido.

El FPS para convertir frame -> segundo se lee del .avi de cada (chute, cam),
igual que en `build_caucafall_groundtruth.py`. El CSV resultante respeta el
formato que ya consume `evaluate_falls.py`.

Los clips de este dataset son mucho más largos que los de GMDCSA-24/CAUCAFall
(hasta ~4200 frames vs. ~100-300), y correr YOLO-pose frame a frame en CPU
sobre los 24 chutes x 8 cámaras (184 clips) tarda horas. Por eso se limita a
`INCLUDED_CHUTES`: un solo chute con sus 8 cámaras alcanza para lo que
importa acá (ver si la heurística generaliza a distintos ángulos del mismo
evento), sin el costo de las 23 escenas restantes. Ampliar `INCLUDED_CHUTES`
si se necesita más volumen más adelante.

Uso:
    python build_kaggle_multicam_groundtruth.py
"""
from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path

import cv2

INPUT_CSV = Path("../videos/Kaggle/kaggle_data.csv")
KAGGLE_ROOT = Path("../videos/Kaggle")
OUTPUT_CSV = Path("../videos/Kaggle/Fall.csv")
FIELDNAMES = ["File Name", "Length (seconds)", "Time of Recording", "Attire", "Description", " Classes"]
INCLUDED_CHUTES = {1}  # prueba rápida: 1 escena x 8 cámaras, ampliar de a tandas después


def load_segments() -> dict[tuple[int, int], list[tuple[int, int, int]]]:
    """Devuelve {(chute, cam): [(start_frame, end_frame, label), ...]}."""
    segments = defaultdict(list)
    with open(INPUT_CSV, encoding="utf-8-sig") as f:
        for row in csv.DictReader(f):
            chute, cam = int(float(row["chute"])), int(float(row["cam"]))
            start, end, label = int(float(row["start"])), int(float(row["end"])), int(float(row["label"]))
            segments[(chute, cam)].append((start, end, label))
    return segments


def build_row(chute: int, cam: int, segs: list[tuple[int, int, int]]) -> dict | None:
    avi_path = KAGGLE_ROOT / f"chute{chute:02d}" / f"cam{cam}.avi"
    if not avi_path.exists():
        print(f"  [aviso] {avi_path} no existe, se ignora")
        return None

    cap = cv2.VideoCapture(str(avi_path))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    cap.release()

    segs = sorted(segs, key=lambda s: s[0])
    parts = []
    for start, end, label in segs:
        tag = "Fall" if label == 1 else "NoFall"
        parts.append(f"{tag}[{start / fps:.2f} to {end / fps:.2f}]")
    classes_field = "; ".join(parts)
    total_s = segs[-1][1] / fps

    return {
        "File Name": f"chute{chute:02d}/cam{cam}.avi",
        "Length (seconds)": round(total_s, 1),
        "Time of Recording": "",
        "Attire": "",
        "Description": f"chute{chute:02d} - cam{cam}",
        " Classes": classes_field,
    }


def main() -> None:
    segments = load_segments()
    rows = []
    for (chute, cam), segs in sorted(segments.items()):
        if chute not in INCLUDED_CHUTES:
            continue
        row = build_row(chute, cam, segs)
        if row is not None:
            rows.append(row)

    with open(OUTPUT_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)

    n_fall = sum(1 for r in rows if any(p.strip().lower().startswith("fall") for p in r[" Classes"].split(";")))
    print(f"Guardado {OUTPUT_CSV} ({len(rows)} clips, {n_fall} con caída anotada)")


if __name__ == "__main__":
    main()
