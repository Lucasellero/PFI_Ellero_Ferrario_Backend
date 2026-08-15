"""Carga la configuración del runner desde un archivo YAML."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import yaml


@dataclass
class CameraConfig:
    id: str
    source: str          # ruta a archivo, URL RTSP/HTTP, o índice de webcam ("0")
    realtime: bool = True    # si True, descarta frames viejos para no ir más lento que el tiempo real
    loop: bool = True        # si True, reinicia el video cuando termina (solo archivos)


@dataclass
class RunnerConfig:
    cameras: list[CameraConfig]
    pose_model: str = "yolo11n-pose.pt"
    log_level: str = "INFO"
    clip_interval_s: float = 10.0        # segundos entre ventanas de clasificación CLIP por cámara
    clip_min_confidence: float = 0.4     # confianza mínima para emitir un evento de actividad
    clip_heartbeat_s: float = 60.0       # cada cuánto re-emitir una actividad que continúa igual
    motion_gate_enabled: bool = True     # si True, YOLO/CLIP solo corren mientras hay movimiento o presencia confirmada (ver motion_gate.py)
    motion_min_area_ratio: float = 0.005 # fracción del frame (reducido) que debe cambiar para contar como movimiento
    motion_absence_timeout_s: float = 45.0  # segundos sin detección de persona (estando despierto) antes de volver a modo bajo consumo


def load_config(path: Path) -> RunnerConfig:
    data = yaml.safe_load(path.read_text())
    cameras = [CameraConfig(**c) for c in data["cameras"]]
    return RunnerConfig(
        cameras=cameras,
        pose_model=data.get("pose_model", "yolo11n-pose.pt"),
        log_level=data.get("log_level", "INFO"),
        clip_interval_s=float(data.get("clip_interval_s", 10.0)),
        clip_min_confidence=float(data.get("clip_min_confidence", 0.4)),
        clip_heartbeat_s=float(data.get("clip_heartbeat_s", 60.0)),
        motion_gate_enabled=bool(data.get("motion_gate_enabled", True)),
        motion_min_area_ratio=float(data.get("motion_min_area_ratio", 0.005)),
        motion_absence_timeout_s=float(data.get("motion_absence_timeout_s", 45.0)),
    )
