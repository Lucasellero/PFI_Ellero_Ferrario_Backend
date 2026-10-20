"""Benchmark del tiempo por frame del pipeline de pose sobre una cámara en vivo, CPU contra GPU.
No es parte del pipeline productivo.

Qué se mide como "tiempo por frame": `FallImmobilityDetector.process_frame()` (YOLO pose + la
heurística de caída/inmovilidad), que es lo que el runner corre sobre cada frame mientras la
cámara está despierta (ver `CameraWorker.run()`). No entra CLIP ni la fusión de esqueleto
(corren por ventanas de muestreo, no por frame) ni el gate de movimiento (acá la cámara está
siempre despierta).

Dos fases:

1. *En vivo*, una corrida de `--segundos` por dispositivo, con la fuente leída igual que en el
   runner (`make_source(..., realtime=True)`: siempre el frame más reciente, los que llegan
   mientras YOLO procesa se descartan). Por dispositivo: media y p95 del tiempo por frame, % de
   frames por debajo de `--umbral-ms`, frames analizados por segundo (frames procesados / tiempo
   de pared; tope: los FPS de la fuente) y % de frames con persona detectada. Los primeros
   `--warmup` frames no cuentan (la primera inferencia en MPS/CUDA compila kernels). El % de
   persona depende de quién esté en cuadro en cada corrida, no solo del dispositivo.
2. *Keypoints CPU contra GPU*: sobre una muestra de los frames de la fase 1 (los mismos frames
   para los dos), corre `detect_person()` con cada dispositivo y compara: en cuántos frames
   coinciden en detectar o no a la persona, y la distancia entre keypoints (píxeles y fracción
   del alto del bbox), contando solo los keypoints con confianza >= 0.5 en los dos.

"GPU" es `mps` en una Mac con Apple Silicon y `cuda` en una máquina con NVIDIA. Ojo: el runner
hoy elige `cuda` si hay y si no `cpu` (ver `FallImmobilityDetector.__init__`), así que en una
Mac corre en CPU aunque haya MPS.

Uso (desde la raíz del repo):
    python -m runner.benchmark_frame_time [--source 0] [--segundos 60] [--devices cpu mps]
        [--json resultados.json]
`--source` por defecto es la de la cámara `webcam` de `cameras.yaml`; "0" es la webcam de la
máquina por índice. Un archivo de video también sirve (se reproduce a velocidad real, en loop).
"""
from __future__ import annotations

import argparse
import json
import sys
import threading
import time
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT / "vision"))

import numpy as np
import torch

from runner.camera_source import make_source
from runner.config import load_config
from src.pose_pipeline import FallImmobilityDetector, detect_person

KEYPOINT_CONF_COMPARACION = 0.5


def _gpu_disponible() -> str | None:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return None


def _source_por_defecto() -> str:
    try:
        config = load_config(_ROOT / "cameras.yaml")
    except Exception:
        return "0"
    for cam in config.cameras:
        if cam.id == "webcam":
            return cam.source
    return "0"


def correr_en_vivo(source: str, device: str, model_path: str, segundos: float, warmup: int,
                   cada_n_guardar: int, max_guardados: int) -> dict:
    """Fase 1: procesa la fuente en vivo durante `segundos` (después del warmup)."""
    detector = FallImmobilityDetector(model_path=model_path, device=device)
    stop = threading.Event()
    fuente = make_source(source, realtime=True, loop=True, stop_event=stop)

    tiempos_ms: list[float] = []
    con_persona = 0
    guardados: list[np.ndarray] = []
    vistos = 0
    inicio = None
    resolucion = None
    try:
        for frame in fuente.frames():
            t0 = time.perf_counter()
            detector.process_frame(frame.image, frame.timestamp_s)
            dt_ms = (time.perf_counter() - t0) * 1000
            vistos += 1
            if vistos <= warmup:
                continue
            if inicio is None:
                inicio = time.perf_counter()
                resolucion = frame.image.shape[1], frame.image.shape[0]
            tiempos_ms.append(dt_ms)
            con_persona += detector.last_detection is not None
            if len(tiempos_ms) % cada_n_guardar == 0 and len(guardados) < max_guardados:
                guardados.append(frame.image)
            if time.perf_counter() - inicio >= segundos:
                break
    finally:
        stop.set()
    duracion = time.perf_counter() - inicio if inicio else 0.0

    if not tiempos_ms:
        raise RuntimeError(f"La fuente {source!r} no entregó frames (¿permiso de cámara? ¿stream caído?)")
    t = np.array(tiempos_ms)
    return {
        "device": device,
        "resolucion": resolucion,
        "frames_analizados": len(t),
        "duracion_s": round(duracion, 2),
        "frames_por_segundo": round(len(t) / duracion, 2),
        "media_ms": round(float(t.mean()), 1),
        "mediana_ms": round(float(np.median(t)), 1),
        "p95_ms": round(float(np.percentile(t, 95)), 1),
        "max_ms": round(float(t.max()), 1),
        "pct_bajo_umbral": None,  # se completa en main() con el umbral
        "pct_con_persona": round(100 * con_persona / len(t), 1),
        "_tiempos_ms": tiempos_ms,
        "_frames": guardados,
    }


def _sig(x: float) -> float:
    return float(f"{float(x):.3g}")


def comparar_keypoints(frames: list[np.ndarray], model_path: str, dev_a: str, dev_b: str) -> dict:
    """Fase 2: mismos frames con los dos dispositivos, compara la persona detectada."""
    from ultralytics import YOLO

    modelos = {}
    for dev in (dev_a, dev_b):
        modelos[dev] = YOLO(model_path)
        modelos[dev].to(dev)

    ambos = solo_a = solo_b = ninguno = 0
    dist_px: list[float] = []
    dist_rel: list[float] = []
    for image in frames:
        det_a = detect_person(modelos[dev_a], image)
        det_b = detect_person(modelos[dev_b], image)
        if det_a is None and det_b is None:
            ninguno += 1
            continue
        if det_a is None:
            solo_b += 1
            continue
        if det_b is None:
            solo_a += 1
            continue
        ambos += 1
        (kp_a, box_a, conf_a), (kp_b, _, conf_b) = det_a, det_b
        confiables = np.ones(len(kp_a), dtype=bool)
        if conf_a is not None and conf_b is not None:
            confiables = (conf_a >= KEYPOINT_CONF_COMPARACION) & (conf_b >= KEYPOINT_CONF_COMPARACION)
        if not confiables.any():
            continue
        d = np.linalg.norm(kp_a[confiables] - kp_b[confiables], axis=1)
        alto_bbox = max(float(box_a[3] - box_a[1]), 1.0)
        dist_px.extend(d.tolist())
        dist_rel.extend((d / alto_bbox).tolist())

    px = np.array(dist_px) if dist_px else np.array([np.nan])
    rel = np.array(dist_rel) if dist_rel else np.array([np.nan])
    total = len(frames)
    return {
        "dispositivos": [dev_a, dev_b],
        "frames_comparados": total,
        "pct_coinciden_deteccion": round(100 * (ambos + ninguno) / total, 1) if total else None,
        "frames_ambos_detectan": ambos,
        f"frames_solo_{dev_a}": solo_a,
        f"frames_solo_{dev_b}": solo_b,
        "frames_ninguno": ninguno,
        "keypoints_comparados": len(dist_px),
        # 3 cifras significativas, no decimales fijos: entre CPU y GPU la diferencia suele ser
        # de milésimas de píxel y redondear a 2 decimales la mostraba como 0.
        "dist_media_px": _sig(np.nanmean(px)),
        "dist_p95_px": _sig(np.nanpercentile(px, 95)),
        "dist_max_px": _sig(np.nanmax(px)),
        "dist_media_pct_alto_bbox": _sig(100 * np.nanmean(rel)),
        "dist_p95_pct_alto_bbox": _sig(100 * np.nanpercentile(rel, 95)),
    }


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--source", default=None, help="URL, índice de webcam o archivo (default: cámara 'webcam' de cameras.yaml)")
    p.add_argument("--devices", nargs="+", default=None, help="default: cpu y la GPU disponible (mps o cuda)")
    p.add_argument("--segundos", type=float, default=60.0, help="duración de la medición en vivo por dispositivo")
    p.add_argument("--warmup", type=int, default=20, help="frames iniciales que no cuentan")
    p.add_argument("--umbral-ms", type=float, default=200.0)
    p.add_argument("--max-frames-comparacion", type=int, default=200)
    p.add_argument("--pose-model", default=str(_ROOT / "yolo11n-pose.pt"))
    p.add_argument("--json", type=Path, default=None, help="guardar los resultados en este archivo")
    args = p.parse_args()

    source = args.source if args.source is not None else _source_por_defecto()
    devices = args.devices
    if devices is None:
        gpu = _gpu_disponible()
        devices = ["cpu"] + ([gpu] if gpu else [])
        if gpu is None:
            print("No hay GPU (ni cuda ni mps): solo se mide CPU, sin comparación de keypoints.")

    print(f"Fuente: {source}  |  dispositivos: {', '.join(devices)}  |  {args.segundos:.0f}s por dispositivo")
    # Guardar 1 de cada N frames para la fase 2, repartidos sobre toda la corrida. N asume
    # ~30 FPS como techo; si se procesa menos, se guardan menos frames, no se pierde nada.
    por_device = max(1, args.max_frames_comparacion // len(devices))
    cada_n = max(1, int(args.segundos * 30 / por_device))

    resultados = []
    for dev in devices:
        print(f"\n[{dev}] midiendo... (ponete en cuadro y movete normalmente)")
        r = correr_en_vivo(source, dev, args.pose_model, args.segundos, args.warmup, cada_n, por_device)
        t = np.array(r["_tiempos_ms"])
        r["pct_bajo_umbral"] = round(100 * float((t < args.umbral_ms).mean()), 1)
        resultados.append(r)

    print(f"\n{'':28}" + "".join(f"{r['device']:>12}" for r in resultados))
    filas = [
        ("resolución", lambda r: f"{r['resolucion'][0]}x{r['resolucion'][1]}"),
        ("frames analizados", lambda r: r["frames_analizados"]),
        ("frames analizados/s", lambda r: r["frames_por_segundo"]),
        ("media ms/frame", lambda r: r["media_ms"]),
        ("mediana ms/frame", lambda r: r["mediana_ms"]),
        ("p95 ms/frame", lambda r: r["p95_ms"]),
        ("máx ms/frame", lambda r: r["max_ms"]),
        (f"% frames < {args.umbral_ms:.0f} ms", lambda r: r["pct_bajo_umbral"]),
        ("% frames con persona", lambda r: r["pct_con_persona"]),
    ]
    for nombre, f in filas:
        print(f"{nombre:28}" + "".join(f"{f(r)!s:>12}" for r in resultados))

    comparacion = None
    if len(devices) >= 2:
        frames = [img for r in resultados for img in r["_frames"]]
        print(f"\nKeypoints {devices[0]} vs {devices[1]} sobre {len(frames)} frames idénticos...")
        comparacion = comparar_keypoints(frames, args.pose_model, devices[0], devices[1])
        for k, v in comparacion.items():
            print(f"  {k:30} {v}")

    if args.json:
        salida = {
            "fuente": source,
            "umbral_ms": args.umbral_ms,
            "por_dispositivo": [{k: v for k, v in r.items() if not k.startswith("_")} for r in resultados],
            "keypoints": comparacion,
        }
        args.json.write_text(json.dumps(salida, indent=1, ensure_ascii=False), encoding="utf-8")
        print(f"\nResultados en {args.json}")


if __name__ == "__main__":
    main()
