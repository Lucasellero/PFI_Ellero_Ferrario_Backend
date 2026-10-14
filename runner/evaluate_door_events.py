"""Validación con video del detector de entradas y salidas (`door_events.py`), sobre Toyota
Smarthome. No es parte del pipeline productivo.

Qué tiene el dataset y qué no: `Enter` y `Leave` son entrar y salir **del ambiente que ve la
cámara** (el departamento tiene 7 cámaras fijas), no de la casa por la puerta de calle. Para
el detector es lo mismo — solo ve que alguien aparece o desaparece en una zona —, pero la zona
que se valida acá es "por dónde se entra y se sale de cuadro", no una puerta de calle dibujada
por un instalador.

**Zona `puerta` por cámara** (se fija antes de medir, con participantes que después no se
usan para medir): por cada cámara se toma dónde estaba la persona (`posicion_normalizada`, el
mismo punto que usa el runner) en la primera detección de cada clip `Enter` y en la última de
cada clip `Leave` de los participantes de ajuste. La zona es el rectángulo que va del
percentil 10 al 90 de esos puntos en cada eje, más un margen de 0,05 por lado. Los
participantes de ajuste son los de entrenamiento del protocolo cross-subject del dataset
(3, 4, 6, 7, 9, 12, 13, 15, 17, 19, 25) menos el 7 y el 9, que se reservan para los videos
largos. Se mide con los de prueba de ese protocolo (2, 10, 11, 14, 16, 18, 20).

**Tres mediciones**, todas con los valores por defecto del detector (60 s de ausencia, 2 s de
memoria de puerta):

1. *Entradas y salidas reales* (clips `Enter` / `Leave` de los participantes de prueba). Los
   clips son recortes de pocos segundos y el detector necesita 60 s sin nadie, así que esa
   ausencia se agrega: a un `Leave` se le suman 61 s sin detección al final y a un `Enter`,
   61 s al principio. Se cuenta si sale el evento.
2. *Entradas y salidas inventadas* (clips de las demás actividades, participantes de prueba).
   Tal cual, un clip corto nunca puede dar un evento (no hay 60 s de ausencia). Se mide el
   peor caso: los mismos 61 s sin detección antes y después de cada clip, como si la persona
   hubiera aparecido y desaparecido justo ahí. Es una cota superior, no una tasa esperable:
   da un evento cada vez que el clip empieza o termina con la persona dentro de la zona.
   También se cuentan los eventos que salen *dentro* del clip sin agregar nada (clips de más
   de 60 s donde YOLO pierde a la persona).
3. *Video continuo* (`--untrimmed`: los videos largos sin recortar de los participantes 7 y
   9, con las anotaciones originales del dataset): el detector corre como en producción, sin
   agregar nada. Cada par `Leave` -> `Enter` anotado es una ausencia real; solo las de 60 s o
   más deberían dar eventos. Lo que salga fuera de un par anotado es inventado.

Las trayectorias (una pasada de YOLO por clip, a `--fps-cap`) se guardan en `--cache` para no
repetirlas.

Uso (desde la raíz del repo):
    python -m runner.evaluate_door_events <carpeta con los .mp4 recortados>
        [--untrimmed videos/ToyotaSmarthome_FP --annotations <carpeta Annotation>]
        [--max-otras 30] [--zonas runner/toyota_smarthome_puerta.yaml]
"""
from __future__ import annotations

import argparse
import csv
import json
import pickle
import random
import re
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT / "vision"))

import cv2
import numpy as np
import yaml

from runner.door_events import (
    DEFAULT_AUSENCIA_MIN_S, DEFAULT_MEMORIA_PUERTA_S, LABEL_ENTRAR, LABEL_SALIR, DoorEventDetector, posicion_normalizada,
)

FIT_PARTICIPANTS = (3, 4, 6, 12, 13, 15, 17, 19, 25)
TEST_PARTICIPANTS = (2, 10, 11, 14, 16, 18, 20)
UNTRIMMED_PARTICIPANTS = (7, 9)
PERCENTILES = (10, 90)
MARGEN = 0.05
RELLENO_S = DEFAULT_AUSENCIA_MIN_S + 1.0
PASO_RELLENO_S = 0.1
TOLERANCIA_PAR_S = 3.0

Trayectoria = list[tuple[float, tuple[float, float] | None]]


def _partes(stem: str) -> tuple[str, int, str] | None:
    """`Leave_p02_r00_v01_c07` -> ("Leave", 2, "07")."""
    m = re.match(r"(.+?)_p(\d+)_.*_c(\d+)$", stem)
    return (m.group(1), int(m.group(2)), m.group(3)) if m else None


def trayectoria(path: Path, yolo, fps_cap: float) -> Trayectoria:
    """(segundo, posición de la persona o None) de cada frame analizado, igual que
    `CameraWorker._door_tick_unsafe`."""
    from src.pose_pipeline import detect_person

    cap = cv2.VideoCapture(str(path))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    stride = max(1, round(fps / fps_cap))
    out: Trayectoria = []
    i = 0
    while cap.grab():
        if i % stride == 0:
            ok, frame = cap.retrieve()
            if not ok:
                break
            det = detect_person(yolo, frame)
            point = None
            if det is not None:
                kpts, box, kpts_conf = det
                h, w = frame.shape[:2]
                point = posicion_normalizada(kpts, box, w, h, kpts_conf)
            out.append((i / fps, point))
        i += 1
    cap.release()
    return out


def ajustar_zona(puntos: list[tuple[float, float]]) -> list[list[float]]:
    """El rectángulo de la zona `puerta` (ver docstring del módulo), como polígono."""
    xs, ys = np.array([p[0] for p in puntos]), np.array([p[1] for p in puntos])
    x1, x2 = (float(np.percentile(xs, q)) for q in PERCENTILES)
    y1, y2 = (float(np.percentile(ys, q)) for q in PERCENTILES)
    x1, y1 = max(0.0, x1 - MARGEN), max(0.0, y1 - MARGEN)
    x2, y2 = min(1.0, x2 + MARGEN), min(1.0, y2 + MARGEN)
    return [[round(v, 3) for v in p] for p in ([x1, y1], [x2, y1], [x2, y2], [x1, y2])]


def _primera(tr: Trayectoria):
    return next((p for _, p in tr if p is not None), None)


def _ultima(tr: Trayectoria):
    return next((p for _, p in reversed(tr) if p is not None), None)


def correr(tr: Trayectoria, zona: list, antes: bool = False, despues: bool = False) -> list[tuple[str, float]]:
    """Eventos del detector sobre la trayectoria, con `RELLENO_S` sin detección antes y/o
    después. Devuelve (etiqueta, segundo del clip); un evento del relleno previo no existe."""
    det = DoorEventDetector(zona, DEFAULT_AUSENCIA_MIN_S, DEFAULT_MEMORIA_PUERTA_S)
    eventos = []
    desplazamiento = 0.0
    if antes:
        for t in np.arange(0.0, RELLENO_S, PASO_RELLENO_S):
            det.update(float(t), None)
        desplazamiento = RELLENO_S
    fin = desplazamiento
    for t, point in tr:
        fin = t + desplazamiento
        eventos += [(e.label, e.timestamp_s - desplazamiento) for e in det.update(fin, point)]
    if despues:
        for t in np.arange(fin + PASO_RELLENO_S, fin + RELLENO_S + PASO_RELLENO_S, PASO_RELLENO_S):
            eventos += [(e.label, e.timestamp_s - desplazamiento) for e in det.update(float(t), None)]
    return eventos


def _pct(a: int, b: int) -> str:
    return f"{100 * a / b:.1f} % ({a}/{b})" if b else "—"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("dataset_dir", type=Path, help="Carpeta con los clips recortados <Actividad>_pXX_..._cNN.mp4")
    ap.add_argument("--untrimmed", type=Path, default=None, help="Carpeta con videos sin recortar PxxTyyCzz.mp4")
    ap.add_argument("--annotations", type=Path, default=None, help="Carpeta Annotation del dataset (Pxx/PxxTyyCzz.csv)")
    ap.add_argument("--max-otras", type=int, default=30, help="Tope de clips por actividad para la medición 2")
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--fps-cap", type=float, default=10.0)
    ap.add_argument("--cache", type=Path, default=_ROOT / "runner_state" / "door_validation_trajectories.pkl")
    ap.add_argument("--zonas", type=Path, default=Path(__file__).parent / "toyota_smarthome_puerta.yaml")
    ap.add_argument("--resultados", type=Path, default=_ROOT / "runner_state" / "door_validation_results.json")
    args = ap.parse_args()

    clips: dict[str, list[Path]] = {}
    for f in sorted(args.dataset_dir.glob("**/*.mp4")):
        if _partes(f.stem):
            clips.setdefault(_partes(f.stem)[0], []).append(f)

    def _de(actividad: str, participantes: tuple[int, ...]) -> list[Path]:
        return [f for f in clips.get(actividad, []) if _partes(f.stem)[1] in participantes]

    rng = random.Random(args.seed)
    otras: list[Path] = []
    for actividad in sorted(clips):
        if actividad in ("Enter", "Leave"):
            continue
        candidatos = _de(actividad, TEST_PARTICIPANTS)
        otras += rng.sample(candidatos, args.max_otras) if len(candidatos) > args.max_otras else candidatos
    de_ajuste = _de("Enter", FIT_PARTICIPANTS) + _de("Leave", FIT_PARTICIPANTS)
    de_prueba = _de("Enter", TEST_PARTICIPANTS) + _de("Leave", TEST_PARTICIPANTS)
    largos = sorted(args.untrimmed.glob("P*T*C*.mp4")) if args.untrimmed else []

    cache: dict[str, Trayectoria] = {}
    if args.cache.exists():
        with args.cache.open("rb") as fh:
            cache = pickle.load(fh)
    pendientes = [f for f in de_ajuste + de_prueba + otras + largos if f.stem not in cache]
    if pendientes:
        from ultralytics import YOLO
        yolo = YOLO(str(_ROOT / "yolo11n-pose.pt"))
        args.cache.parent.mkdir(parents=True, exist_ok=True)
        for n, f in enumerate(pendientes, 1):
            cache[f.stem] = trayectoria(f, yolo, args.fps_cap)
            if n % 100 == 0 or n == len(pendientes):
                print(f"  trayectorias: {n}/{len(pendientes)}", flush=True)
                with args.cache.open("wb") as fh:
                    pickle.dump(cache, fh)

    # ---- Zona por cámara, solo con los participantes de ajuste ----
    puntos: dict[str, list] = {}
    for f in de_ajuste:
        actividad, _, camara = _partes(f.stem)
        p = _primera(cache[f.stem]) if actividad == "Enter" else _ultima(cache[f.stem])
        if p is not None:
            puntos.setdefault(camara, []).append(p)
    zonas = {camara: ajustar_zona(pts) for camara, pts in sorted(puntos.items())}
    print(f"\nZona `puerta` por cámara (participantes de ajuste {FIT_PARTICIPANTS}):")
    for camara, zona in zonas.items():
        (x1, y1), _, (x2, y2), _ = zona
        print(f"  c{camara}: x {x1:.2f}–{x2:.2f}, y {y1:.2f}–{y2:.2f}  "
              f"({100 * (x2 - x1) * (y2 - y1):.0f} % del cuadro, {len(puntos[camara])} clips de ajuste)")
    encabezado = (
        "# Zona `puerta` de cada cámara de Toyota Smarthome para validar runner/door_events.py.\n"
        "# La genera runner/evaluate_door_events.py con los participantes de ajuste (ver su docstring):\n"
        "# rectángulo del percentil 10 al 90, más 0,05 de margen, de dónde aparece la persona en los\n"
        "# clips Enter y dónde se la ve por última vez en los clips Leave. Coordenadas normalizadas 0-1.\n"
        "# Es \"por dónde se entra y se sale de cuadro\", no una puerta de calle. No es cameras.yaml.\n"
    )
    args.zonas.write_text(encabezado + yaml.safe_dump({c: {"puerta": z} for c, z in zonas.items()}, default_flow_style=None), encoding="utf-8")

    resultados: dict = {"zonas": zonas, "clips_de_ajuste": {c: len(p) for c, p in puntos.items()}}

    # ---- 1. Entradas y salidas reales ----
    print(f"\n1. Entradas y salidas reales (participantes de prueba {TEST_PARTICIPANTS}; 61 s de ausencia agregados)")
    reales: dict[tuple[str, str], list[int]] = {}
    for f in de_prueba:
        actividad, _, camara = _partes(f.stem)
        tr = cache[f.stem]
        esperado = LABEL_ENTRAR if actividad == "Enter" else LABEL_SALIR
        eventos = correr(tr, zonas[camara], antes=actividad == "Enter", despues=actividad == "Leave") if camara in zonas else []
        cuenta = reales.setdefault((actividad, camara), [0, 0, 0])
        cuenta[0] += 1
        cuenta[1] += any(label == esperado for label, _ in eventos)
        cuenta[2] += all(p is None for _, p in tr)
    for actividad in ("Enter", "Leave"):
        por_camara = {c: v for (a, c), v in sorted(reales.items()) if a == actividad}
        total = [sum(v[i] for v in por_camara.values()) for i in range(3)]
        print(f"  {actividad}: detectadas {_pct(total[1], total[0])}   (clips donde YOLO no vio a nadie: {total[2]})")
        print("    por cámara: " + "   ".join(f"c{c} {_pct(v[1], v[0])}" for c, v in por_camara.items()))
    resultados["reales"] = {f"{a}_c{c}": v for (a, c), v in sorted(reales.items())}

    # ---- 2. Inventadas en clips de otras actividades ----
    print(f"\n2. Clips de otras actividades (participantes de prueba, hasta {args.max_otras} por actividad)")
    inventadas: dict[str, dict] = {}
    por_camara: dict[str, list[int]] = {}
    for f in otras:
        actividad, _, camara = _partes(f.stem)
        if camara not in zonas:
            continue
        tr = cache[f.stem]
        tal_cual = correr(tr, zonas[camara])
        peor_caso = correr(tr, zonas[camara], antes=True, despues=True)
        d = inventadas.setdefault(actividad, {"clips": 0, "segundos": 0.0, "tal_cual": 0, "entrada": 0, "salida": 0, "alguna": 0})
        d["clips"] += 1
        d["segundos"] += tr[-1][0] if tr else 0.0
        d["tal_cual"] += len(tal_cual)
        entrada = any(label == LABEL_ENTRAR for label, _ in peor_caso)
        salida = any(label == LABEL_SALIR for label, _ in peor_caso)
        d["entrada"] += entrada
        d["salida"] += salida
        d["alguna"] += entrada or salida
        c = por_camara.setdefault(camara, [0, 0, 0])
        c[0] += 1
        c[1] += entrada
        c[2] += salida
    n = sum(d["clips"] for d in inventadas.values())
    horas = sum(d["segundos"] for d in inventadas.values()) / 3600
    print(f"  {n} clips, {horas:.2f} h de video")
    print(f"  Tal cual (sin agregar ausencia): {sum(d['tal_cual'] for d in inventadas.values())} eventos")
    print(f"  Peor caso (61 s de ausencia antes y después de cada clip): entrada inventada en "
          f"{_pct(sum(d['entrada'] for d in inventadas.values()), n)}, salida inventada en "
          f"{_pct(sum(d['salida'] for d in inventadas.values()), n)}, alguna de las dos en "
          f"{_pct(sum(d['alguna'] for d in inventadas.values()), n)}")
    print("    por cámara (entrada / salida): " + "   ".join(f"c{c} {_pct(v[1], v[0])} / {_pct(v[2], v[0])}" for c, v in sorted(por_camara.items())))
    for actividad, d in sorted(inventadas.items()):
        print(f"    {actividad:22s} entrada {_pct(d['entrada'], d['clips']):>16s}   salida {_pct(d['salida'], d['clips']):>16s}   tal cual {d['tal_cual']}")
    resultados["otras"] = inventadas
    resultados["otras_por_camara"] = por_camara

    # ---- 3. Video continuo ----
    if largos:
        print("\n3. Video continuo, sin agregar nada (detector como en producción)")
        continuo = []
        for f in largos:
            m = re.match(r"P(\d+)T\d+C(\d+)$", f.stem)
            camara = m.group(2)
            tr = cache[f.stem]
            eventos = correr(tr, zonas[camara]) if camara in zonas else []
            pares = []
            anotacion = args.annotations / f"P{m.group(1)}" / f"{f.stem}.csv" if args.annotations else None
            if anotacion is not None and anotacion.exists():
                fps = cv2.VideoCapture(str(f)).get(cv2.CAP_PROP_FPS) or 25.0
                filas = [r for r in csv.DictReader(anotacion.open(encoding="utf-8-sig")) if r["event"] in ("Enter", "Leave")]
                for a, b in zip(filas, filas[1:]):
                    if a["event"] == "Leave" and b["event"] == "Enter":
                        pares.append((int(a["start_frame"]) / fps, int(a["end_frame"]) / fps, int(b["start_frame"]) / fps, int(b["end_frame"]) / fps))
            sin_nadie = sum(1 for _, p in tr if p is None) / max(len(tr), 1)
            print(f"  {f.stem} (c{camara}, {tr[-1][0] / 60:.1f} min, {100 * sin_nadie:.0f} % de los frames sin detección)")
            usados: set[int] = set()
            for s0, s1, e0, e1 in pares:
                ausencia = e0 - s1
                dentro = [i for i, (_, t) in enumerate(eventos) if s0 - TOLERANCIA_PAR_S <= t <= e1 + TOLERANCIA_PAR_S]
                usados.update(dentro)
                print(f"    ausencia anotada {s1:7.1f} s -> {e0:7.1f} s ({ausencia:5.1f} s, "
                      f"{'debería' if ausencia >= DEFAULT_AUSENCIA_MIN_S else 'no debería'} dar eventos): "
                      f"{[f'{eventos[i][0]} @ {eventos[i][1]:.1f}s' for i in dentro] or 'ningún evento'}")
            fuera = [f"{label} @ {t:.1f}s" for i, (label, t) in enumerate(eventos) if i not in usados]
            print(f"    eventos fuera de una ausencia anotada: {fuera or 'ninguno'}")
            continuo.append({
                "video": f.stem, "camara": camara, "minutos": tr[-1][0] / 60, "frac_sin_deteccion": sin_nadie,
                "pares": pares, "eventos": eventos, "fuera_de_par": fuera,
            })
        resultados["continuo"] = continuo

    args.resultados.parent.mkdir(parents=True, exist_ok=True)
    args.resultados.write_text(json.dumps(resultados, indent=1, default=str), encoding="utf-8")
    print(f"\nZonas: {args.zonas}   Resultados: {args.resultados}")


if __name__ == "__main__":
    main()
