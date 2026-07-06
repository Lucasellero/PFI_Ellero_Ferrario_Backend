"""Genera `videos/CAUCAFall/Fall.csv` a partir de las etiquetas por frame que
trae CAUCAFall (`Subject.N/<actividad>/*.txt`, formato YOLO: `<class_id> ...`,
con `classes.txt` mapeando class_id -> "nofall"/"fall").

CAUCAFall no trae un CSV de intervalos como GMDCSA-24 (`Fall.csv`), pero sí
etiqueta cada frame, así que el segundo exacto donde arranca la caída se
puede derivar: es el primer frame cuya clase es "fall". El FPS real se lee
del .avi correspondiente (20 fps en las muestras verificadas), no se asume.

El CSV resultante respeta el mismo formato que `videos/GMDCSA-24/fall*/Fall.csv`
para que `evaluate_falls.py` lo consuma sin cambios en el parser.

Uso:
    python build_caucafall_groundtruth.py
"""
from __future__ import annotations

import csv
from pathlib import Path

import cv2

CAUCAFALL_ROOT = Path("../videos/CAUCAFall/CAUCAFall")
OUTPUT_CSV = Path("../videos/CAUCAFall/Fall.csv")
FIELDNAMES = ["File Name", "Length (seconds)", "Time of Recording", "Attire", "Description", " Classes"]


def build_row(subject_dir: Path, activity_dir: Path) -> dict | None:
    classes_path = activity_dir / "classes.txt"
    avi_candidates = list(activity_dir.glob("*.avi"))
    if not classes_path.exists() or not avi_candidates:
        print(f"  [aviso] {activity_dir}: falta classes.txt o .avi, se ignora")
        return None
    avi_path = avi_candidates[0]

    classes = classes_path.read_text().split()
    frame_files = sorted(f for f in activity_dir.glob("*.txt") if f.name != "classes.txt")
    class_ids = [int(f.read_text().split()[0]) for f in frame_files]

    cap = cv2.VideoCapture(str(avi_path))
    fps = cap.get(cv2.CAP_PROP_FPS) or 20.0
    cap.release()

    total_s = (len(class_ids) - 1) / fps
    fall_idx = classes.index("fall") if "fall" in classes else None

    if fall_idx is not None and fall_idx in class_ids:
        start_frame = class_ids.index(fall_idx)
        start_s = start_frame / fps
        classes_field = f"Fall[{start_s:.2f} to {total_s:.2f}]; NoFall[0 to {start_s:.2f}]"
    else:
        classes_field = f"NoFall[0 to {total_s:.2f}]"

    return {
        "File Name": f"CAUCAFall/{subject_dir.name}/{activity_dir.name}/{avi_path.name}",
        "Length (seconds)": round(total_s, 1),
        "Time of Recording": "",
        "Attire": "",
        "Description": f"{subject_dir.name} - {activity_dir.name}",
        " Classes": classes_field,
    }


def main() -> None:
    rows = []
    for subject_dir in sorted(CAUCAFALL_ROOT.glob("Subject.*")):
        for activity_dir in sorted(p for p in subject_dir.iterdir() if p.is_dir()):
            row = build_row(subject_dir, activity_dir)
            if row is not None:
                rows.append(row)

    with open(OUTPUT_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)

    n_fall = sum(1 for r in rows if r[" Classes"].lower().startswith("fall"))
    print(f"Guardado {OUTPUT_CSV} ({len(rows)} clips, {n_fall} con caída anotada)")


if __name__ == "__main__":
    main()
