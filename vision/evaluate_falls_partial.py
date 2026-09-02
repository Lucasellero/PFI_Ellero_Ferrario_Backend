"""Test de encuadre parcial: ¿el detector sigue cazando una caída cuando
sólo se ve el torso de la persona?

Motivación (ver plan y vision/README.md): la queja concreta es que la
heurística geométrica se rompe cuando la persona no entra entera en el cuadro
— el bbox recortado ensancha el aspect ratio y hace saltar hip_y, generando
falsos positivos, y en el otro sentido puede perder caídas reales. La Fase 0
marca esos eventos como `evidence="partial"` (no los descarta, los escala por
inmovilidad). La Fase 1 entrena el clasificador temporal con augmentación de
truncamiento para que aprenda "torso visible + este movimiento = caída".

Este script mide las dos cosas sobre los mismos clips de caída anotados que
usa `evaluate_falls.py`, pero **enmascarando cada frame a una banda alrededor
del torso** (hombros→cadera, con margen), de modo que YOLO sólo vea esa
región — el peor caso de "sólo se ve el torso". Compara:

  - heurística geométrica sola
  - heurística + clasificador temporal (fusion/, si está exportado)

sobre frames enteros (baseline) y sobre frames enmascarados al torso.

Uso:
    python evaluate_falls_partial.py [--limit 8] [--only GMDCSA-24,CAUCAFall]
        [--margin 0.35] [--no-temporal]
"""
from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np

from evaluate_falls import load_ground_truth, _load_temporal, build_pose_sample
from src.pose_pipeline import FallImmobilityDetector

_ROOT = Path(__file__).resolve().parents[1]
_TEMPORAL_REJECT = 0.15

ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument("--limit", type=int, default=8, help="Clips de caída por dataset")
ap.add_argument("--only", default=None, help="Datasets a incluir (coma-separado)")
ap.add_argument("--margin", type=float, default=0.35, help="Margen alrededor del torso, como fracción del alto del bbox de la persona")
ap.add_argument("--no-temporal", action="store_true")
args = ap.parse_args()
ONLY = set(args.only.split(",")) if args.only else None


def torso_band(img, kpts, box, margin):
    """Devuelve una copia de `img` con todo en negro salvo una banda
    horizontal alrededor del torso (hombros/cadera) con `margin` de holgura."""
    h, w = img.shape[:2]
    ys = []
    for i in (5, 6, 11, 12):
        if kpts[i][1] > 0:
            ys.append(kpts[i][1])
    if not ys:
        return None
    bh = max(box[3] - box[1], 1.0)
    y0 = int(max(0, min(ys) - margin * bh))
    y1 = int(min(h, max(ys) + margin * bh))
    out = np.zeros_like(img)
    out[y0:y1, :] = img[y0:y1, :]
    return out


def run(video, temporal, mask_torso: bool):
    det = FallImmobilityDetector()
    # detector aparte, sólo para localizar el torso en frame entero
    scout = FallImmobilityDetector() if mask_torso else None
    buffer = ff = clf = None
    if temporal is not None:
        clf, BufferCls, ff = temporal
        buffer = BufferCls()
    cap = cv2.VideoCapture(str(video))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    fall_events, i = [], 0
    while True:
        ok, img = cap.read()
        if not ok:
            break
        t = i / fps
        frame = img
        if mask_torso:
            scout.process_frame(img, t)
            if scout.last_detection is not None:
                banded = torso_band(img, *scout.last_detection, args.margin)
                if banded is not None:
                    frame = banded
        raw = det.process_frame(frame, t)
        if temporal is not None:
            if det.last_detection is not None:
                kpts, box = det.last_detection
                hh, ww = frame.shape[:2]
                ps = build_pose_sample(kpts, box, ww, hh, t, kpts_conf=det.last_detection_conf)
                buffer.push(t, ff(kpts, det.last_detection_conf, box,
                                 ps.aspect_ratio, ps.torso_angle_deg, ps.hip_y, ps.bbox_height_frac))
            else:
                buffer.reset()
        for e in raw:
            if e["event_type"] != "fall":
                continue
            if temporal is not None:
                win = buffer.build_window(t)
                if win is not None and clf.classify(win)["fall_score"] <= _TEMPORAL_REJECT:
                    continue
            fall_events.append((t, e["metadata"].get("evidence", "full")))
        i += 1
    cap.release()
    return fall_events


def recall_over(cases, temporal, mask_torso):
    hits = partial_hits = 0
    for c in cases:
        evs = run(c["video"], temporal, mask_torso)
        got = False
        got_partial = False
        for t, ev in evs:
            for s, e in c["fall_segments"]:
                if s <= t <= e:
                    got = True
                    if ev == "partial":
                        got_partial = True
        hits += got
        partial_hits += (got and got_partial and not any(
            ev == "full" for t, ev in evs if any(s <= t <= e for s, e in c["fall_segments"])
        ))
    return hits, partial_hits, len(cases)


def main():
    all_cases = [c for c in load_ground_truth() if c["fall_segments"]]
    by_ds: dict[str, list] = {}
    for c in all_cases:
        ds = Path(c["video"]).relative_to((_ROOT / "videos")).parts[0]
        if ONLY and ds not in ONLY:
            continue
        by_ds.setdefault(ds, []).append(c)
    cases = [c for ds in by_ds for c in by_ds[ds][: args.limit]]
    print(f"{len(cases)} clips de caída ({ {ds: min(len(v), args.limit) for ds, v in by_ds.items()} })\n")

    temporal = None
    if not args.no_temporal:
        try:
            temporal = _load_temporal()
            print("Clasificador temporal: ACTIVO\n")
        except Exception as e:
            print(f"Clasificador temporal no disponible ({e}) — sólo heurística\n")

    for label, mask in [("frame entero", False), ("enmascarado al torso", True)]:
        print(f"--- {label} ---")
        h_geom, _, n = recall_over(cases, None, mask)
        print(f"  heurística sola:      recall {h_geom}/{n} ({h_geom/n*100:.0f}%)")
        if temporal is not None:
            h_t, p_t, _ = recall_over(cases, temporal, mask)
            print(f"  heurística+temporal:  recall {h_t}/{n} ({h_t/n*100:.0f}%)  "
                  f"[{p_t} sólo vía evento 'partial']")
        print()


if __name__ == "__main__":
    main()
