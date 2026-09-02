"""Runner multi-cámara: arranca N cámaras en paralelo y corre indefinidamente.

Uso:
    python -m runner                          # usa cameras.yaml en la raíz del proyecto
    python -m runner --config mi_config.yaml  # config alternativa
"""
from __future__ import annotations

import argparse
import logging
import signal
import sys
import threading
from pathlib import Path

from .camera_worker import CameraWorker, _CLIP_AVAILABLE
from .config import load_config

_ROOT = Path(__file__).resolve().parents[1]
_DEFAULT_CONFIG = _ROOT / "cameras.yaml"
_DB_PATH = _ROOT / "events" / "events.db"


def _load_clip(log: logging.Logger):
    """Carga el clasificador CLIP una sola vez para compartirlo entre cámaras."""
    if not _CLIP_AVAILABLE:
        log.warning("vlm/ no importable — CLIP desactivado")
        return None
    try:
        from vlm.src.clip_classifier import ClipActivityClassifier
        log.info("Cargando clasificador CLIP (ONNX)…")
        clf = ClipActivityClassifier(quantized=True)
        log.info("CLIP listo.")
        return clf
    except FileNotFoundError as e:
        log.warning("Modelos CLIP no encontrados (%s) — ejecutá vlm/export_onnx.py primero", e)
        return None
    except Exception as e:
        log.warning("No se pudo cargar CLIP: %s", e)
        return None


def _load_skeleton(log: logging.Logger):
    """Carga el clasificador temporal de esqueleto una sola vez, para
    compartir entre cámaras (ver fusion/README.md, "Integración"). Degrada a
    None (CameraWorker cae a CLIP solo) si fusion/ no está entrenado/
    exportado todavía — mismo patrón que _load_clip."""
    try:
        from fusion.src.temporal_classifier import SkeletonActivityClassifier
        log.info("Cargando clasificador de esqueleto (ONNX)…")
        clf = SkeletonActivityClassifier(quantized=True)
        log.info("Esqueleto listo.")
        return clf
    except FileNotFoundError as e:
        log.warning(
            "Modelo de esqueleto no encontrado (%s) — ejecutá "
            "fusion/train_temporal_classifier.py + fusion/export_temporal_onnx.py primero",
            e,
        )
        return None
    except Exception as e:
        log.warning("No se pudo cargar el clasificador de esqueleto: %s", e)
        return None


def _load_fall_temporal(log: logging.Logger):
    """Carga el clasificador temporal de caída una sola vez (ver
    fusion/src/fall_temporal.py). Es la segunda etapa del evento de caída,
    igual que la confirmación CLIP: degrada a None (CameraWorker sigue usando
    la confirmación CLIP) si fusion/ no lo tiene entrenado/exportado."""
    try:
        from fusion.src.fall_temporal import FallTemporalClassifier
        log.info("Cargando clasificador temporal de caída (ONNX)…")
        clf = FallTemporalClassifier(quantized=True)
        log.info("Clasificador temporal de caída listo.")
        return clf
    except FileNotFoundError as e:
        log.warning(
            "Modelo temporal de caída no encontrado (%s) — ejecutá "
            "fusion/extract_falls.py + fusion/train_fall_temporal.py + "
            "fusion/export_fall_temporal_onnx.py primero. Se sigue con confirmación CLIP.",
            e,
        )
        return None
    except Exception as e:
        log.warning("No se pudo cargar el clasificador temporal de caída: %s", e)
        return None


def main() -> None:
    parser = argparse.ArgumentParser(description="Runner multi-cámara de monitoreo")
    parser.add_argument(
        "--config", type=Path, default=_DEFAULT_CONFIG,
        help=f"Archivo YAML de configuración (default: {_DEFAULT_CONFIG})"
    )
    args = parser.parse_args()

    if not args.config.exists():
        print(f"ERROR: No se encontró el archivo de config: {args.config}", file=sys.stderr)
        print("Creá cameras.yaml en la raíz del proyecto (ver cameras.yaml.example).", file=sys.stderr)
        sys.exit(1)

    config = load_config(args.config)

    logging.basicConfig(
        level=getattr(logging, config.log_level),
        format="%(asctime)s  %(name)-20s  %(levelname)s  %(message)s",
        datefmt="%H:%M:%S",
    )
    logger = logging.getLogger("runner")

    # CLIP y esqueleto se cargan una sola vez y se comparten entre todos los
    # workers. ONNX Runtime es thread-safe para inferencia concurrente.
    clip = _load_clip(logger)
    skeleton = _load_skeleton(logger)
    fall_temporal = _load_fall_temporal(logger)

    stop_event = threading.Event()

    def _shutdown(signum, frame):
        logger.info("Señal %s recibida — deteniendo cámaras...", signum)
        stop_event.set()

    signal.signal(signal.SIGINT, _shutdown)
    signal.signal(signal.SIGTERM, _shutdown)

    workers = [
        CameraWorker(
            config=cam,
            db_path=_DB_PATH,
            pose_model=config.pose_model,
            stop_event=stop_event,
            clip_classifier=clip,
            skeleton_classifier=skeleton,
            fall_temporal_classifier=fall_temporal,
            skeleton_weight=config.skeleton_weight,
            clip_interval_s=config.clip_interval_s,
            clip_min_confidence=config.clip_min_confidence,
            clip_heartbeat_s=config.clip_heartbeat_s,
            motion_gate_enabled=config.motion_gate_enabled,
            motion_min_area_ratio=config.motion_min_area_ratio,
            motion_absence_timeout_s=config.motion_absence_timeout_s,
        )
        for cam in config.cameras
    ]

    logger.info(
        "=== Iniciando %d cámara(s) | CLIP: %s | esqueleto: %s | caída temporal: %s | intervalo: %.0fs | gate de movimiento: %s — Ctrl+C para detener ===",
        len(workers),
        "activo" if clip else "inactivo",
        "activo" if skeleton else "inactivo",
        "activo" if fall_temporal else "inactivo (confirmación CLIP)",
        config.clip_interval_s,
        "activo" if config.motion_gate_enabled else "inactivo",
    )
    for w in workers:
        w.start()

    for w in workers:
        w.join()

    logger.info("=== Runner detenido. ===")


if __name__ == "__main__":
    main()
