"""Genera `videos/URFD/Fall.csv` a partir de las etiquetas por frame que trae
el UR Fall Detection Dataset (`_meta/urfall-cam0-falls.csv`, columna `label`:
`-1` = de pie/moviéndose antes de la caída, `0`/`1` = cayendo/tirado en el
piso — no se distingue entre ambos acá, ver docstring del dataset).

El inicio de la ventana `Fall` es el primer frame con `label != -1`; el fin
es el final del clip (la persona queda en el piso, no hay "se levanta" en
estas secuencias cortas). Las secuencias `adl-XX` no tienen CSV de labels
(son "casi caída" por diseño: sentarse, agacharse, acostarse) y se marcan
`NoFall` completas.

El CSV resultante respeta el mismo formato que `videos/GMDCSA-24/fall*/Fall.csv`
para que `evaluate_falls.py` lo consuma sin cambios en el parser.

Uso:
    python build_urfd_groundtruth.py
"""
from __future__ import annotations

import csv
from pathlib import Path

import cv2

URFD_ROOT = Path("../videos/URFD")
META_DIR = URFD_ROOT / "_meta"
OUTPUT_CSV = URFD_ROOT / "Fall.csv"
FIELDNAMES = ["File Name", "Length (seconds)", "Time of Recording", "Attire", "Description", " Classes"]
N_FALLS = 30
N_ADLS = 40


def clip_length_s(video_path: Path) -> float:
    cap = cv2.VideoCapture(str(video_path))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    n_frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    cap.release()
    return n_frames / fps, fps


def load_fall_start_frames() -> dict[str, int]:
    """Primer frame con label != -1 por secuencia, leído de urfall-cam0-falls.csv."""
    start_frame: dict[str, int] = {}
    with open(META_DIR / "urfall-cam0-falls.csv", encoding="utf-8") as f:
        for seq, frame, label, *_ in csv.reader(f):
            if int(label) != -1 and seq not in start_frame:
                start_frame[seq] = int(frame)
    return start_frame


def main() -> None:
    fall_start_frame = load_fall_start_frames()
    rows = []

    for i in range(1, N_FALLS + 1):
        seq = f"fall-{i:02d}"
        fname = f"{seq}-cam0.mp4"
        length_s, fps = clip_length_s(URFD_ROOT / fname)
        start_frame = fall_start_frame.get(seq)
        start_s = (start_frame - 1) / fps if start_frame else 0.0
        classes_field = f"Fall[{start_s:.2f} to {length_s:.2f}]"
        if start_s > 0:
            classes_field = f"NoFall[0 to {start_s:.2f}]; " + classes_field
        rows.append({
            "File Name": fname,
            "Length (seconds)": round(length_s, 2),
            "Time of Recording": "",
            "Attire": "",
            "Description": f"URFD (UR Fall Detection Dataset) - {seq}",
            " Classes": classes_field,
        })

    for i in range(1, N_ADLS + 1):
        seq = f"adl-{i:02d}"
        fname = f"{seq}-cam0.mp4"
        length_s, _ = clip_length_s(URFD_ROOT / fname)
        rows.append({
            "File Name": fname,
            "Length (seconds)": round(length_s, 2),
            "Time of Recording": "",
            "Attire": "",
            "Description": f"URFD (UR Fall Detection Dataset) - {seq}, actividad cotidiana sin caida",
            " Classes": f"NoFall[0 to {length_s:.2f}]",
        })

    with open(OUTPUT_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)

    n_fall = sum(
        1 for r in rows
        if any(part.strip().lower().startswith("fall[") for part in r[" Classes"].split(";"))
    )
    print(f"Guardado {OUTPUT_CSV} ({len(rows)} clips, {n_fall} con caída anotada)")


if __name__ == "__main__":
    main()
