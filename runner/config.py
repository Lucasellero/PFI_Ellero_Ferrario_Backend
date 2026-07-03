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
    clip_interval_s: float = 10.0  # segundos entre clasificaciones CLIP por cámara


def load_config(path: Path) -> RunnerConfig:
    data = yaml.safe_load(path.read_text())
    cameras = [CameraConfig(**c) for c in data["cameras"]]
    return RunnerConfig(
        cameras=cameras,
        pose_model=data.get("pose_model", "yolo11n-pose.pt"),
        log_level=data.get("log_level", "INFO"),
        clip_interval_s=float(data.get("clip_interval_s", 10.0)),
    )
