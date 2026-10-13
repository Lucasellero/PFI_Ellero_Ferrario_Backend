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
    # Zonas del hogar vistas por esta cámara: {nombre_zona: [[x,y], ...]} en
    # coordenadas normalizadas 0-1 del frame — se configuran una sola vez por
    # cámara real, en el setup de instalación (ver ejemplo comentado en
    # cameras.yaml y fusion/src/zone_context.py). None = sin configurar
    # todavía: el clasificador degrada a "sin zona" (vector en cero), no
    # rompe nada. Medido en vlm/README.md ("Fase 1 de PLAN_sustitucion_clip.md...",
    # 2026-09-21): +6pts de accuracy en Cooking cuando está configurado.
    zones: dict[str, list[list[float]]] | None = None


@dataclass
class CloudSyncConfig:
    """Sección opcional `cloud:` de `cameras.yaml` — sin ella, `runner/cloud_sync.py` no se
    arranca y nada cambia respecto a una instalación sin nube. Ver `cloud/README.md`."""
    url: str
    api_key: str
    interval_s: float = 30.0
    # Días que un evento ya confirmado por la nube sigue en events/events.db antes de borrarse
    # (ver `EventStore.purge_synced` y `runner/cloud_sync.py`). Los no sincronizados no se tocan.
    retencion_local_dias: float = 7.0


@dataclass
class RunnerConfig:
    cameras: list[CameraConfig]
    pose_model: str = "yolo11n-pose.pt"
    log_level: str = "INFO"
    clip_interval_s: float = 10.0        # segundos entre ventanas de clasificación CLIP por cámara
    clip_min_confidence: float = 0.4     # confianza mínima para emitir un evento de actividad
    clip_heartbeat_s: float = 60.0       # cada cuánto re-emitir una actividad que continúa igual
    skeleton_weight: float = 0.5         # peso del esqueleto vs. CLIP al fusionar categoría (ver fusion/src/fusion.py); sin efecto si fusion/ no está entrenado/exportado
    motion_gate_enabled: bool = True     # si True, YOLO/CLIP solo corren mientras hay movimiento o presencia confirmada (ver motion_gate.py)
    motion_min_area_ratio: float = 0.005 # fracción del frame (reducido) que debe cambiar para contar como movimiento
    motion_absence_timeout_s: float = 45.0  # segundos sin detección de persona (estando despierto) antes de volver a modo bajo consumo
    cloud_sync: CloudSyncConfig | None = None  # None = sin sincronización a la nube (default)


def load_config(path: Path) -> RunnerConfig:
    data = yaml.safe_load(path.read_text())
    cameras = [CameraConfig(**c) for c in data["cameras"]]
    cloud_data = data.get("cloud")
    cloud_sync = (
        CloudSyncConfig(
            url=cloud_data["url"], api_key=cloud_data["api_key"],
            interval_s=float(cloud_data.get("interval_s", 30.0)),
            retencion_local_dias=float(cloud_data.get("retencion_local_dias", 7.0)),
        )
        if cloud_data else None
    )
    if cloud_sync is not None and cloud_sync.retencion_local_dias <= 0:
        # 0 borraría cada evento apenas se sincroniza: se rechaza en vez de adivinar la intención.
        raise ValueError("cloud.retencion_local_dias tiene que ser mayor que 0")
    return RunnerConfig(
        cameras=cameras,
        pose_model=data.get("pose_model", "yolo11n-pose.pt"),
        log_level=data.get("log_level", "INFO"),
        clip_interval_s=float(data.get("clip_interval_s", 10.0)),
        clip_min_confidence=float(data.get("clip_min_confidence", 0.4)),
        clip_heartbeat_s=float(data.get("clip_heartbeat_s", 60.0)),
        skeleton_weight=float(data.get("skeleton_weight", 0.5)),
        motion_gate_enabled=bool(data.get("motion_gate_enabled", True)),
        motion_min_area_ratio=float(data.get("motion_min_area_ratio", 0.005)),
        motion_absence_timeout_s=float(data.get("motion_absence_timeout_s", 45.0)),
        cloud_sync=cloud_sync,
    )
