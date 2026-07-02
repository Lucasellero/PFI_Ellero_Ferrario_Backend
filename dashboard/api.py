"""Dashboard web para el sistema de monitoreo.

Expone:
  GET  /            → panel HTML
  GET  /api/status  → estado del runner
  POST /api/start   → inicia el runner como subproceso
  POST /api/stop    → detiene el runner
  GET  /api/cameras → lee cameras.yaml
  POST /api/cameras/add      → agrega una cámara
  DELETE /api/cameras/{id}   → elimina una cámara
  GET  /api/videos  → lista videos disponibles
  POST /api/upload-video     → sube un video
  GET  /api/events  → lee eventos de SQLite
  WS   /ws          → stream de logs + estado en tiempo real
"""
from __future__ import annotations

import asyncio
import json
import sqlite3
import sys
from pathlib import Path
from typing import Optional

import re

import aiofiles
import yaml
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, UploadFile, File, HTTPException
from fastapi.responses import FileResponse, HTMLResponse

ROOT = Path(__file__).resolve().parents[1]
CAMERAS_YAML = ROOT / "cameras.yaml"
DB_PATH = ROOT / "events" / "events.db"
VIDEOS_DIR = ROOT / "videos"
UPLOADS_DIR = VIDEOS_DIR / "uploads"
FRAMES_DIR = ROOT / "frames"
STATIC_DIR = Path(__file__).parent / "static"

app = FastAPI(title="Monitor Dashboard")

# ── Estado global ──────────────────────────────────────────────────────────
runner_proc: Optional[asyncio.subprocess.Process] = None
_ws_clients: set[WebSocket] = set()
_last_broadcast_event_id: int = 0


# ── Utilidades ─────────────────────────────────────────────────────────────

async def _broadcast(msg: dict) -> None:
    dead: set[WebSocket] = set()
    for ws in list(_ws_clients):
        try:
            await ws.send_json(msg)
        except Exception:
            dead.add(ws)
    _ws_clients.difference_update(dead)


def _is_running() -> bool:
    return runner_proc is not None and runner_proc.returncode is None


def _read_cameras_yaml() -> dict:
    return yaml.safe_load(CAMERAS_YAML.read_text()) or {}


def _write_cameras_yaml(data: dict) -> None:
    CAMERAS_YAML.write_text(
        yaml.dump(data, allow_unicode=True, default_flow_style=False, sort_keys=False)
    )


def _fetch_events_since(last_id: int, limit: int = 50) -> list[dict]:
    if not DB_PATH.exists():
        return []
    conn = sqlite3.connect(str(DB_PATH))
    rows = conn.execute(
        "SELECT id, video_id, event_type, video_timestamp_s, metadata "
        "FROM events WHERE id > ? ORDER BY id ASC LIMIT ?",
        (last_id, limit),
    ).fetchall()
    conn.close()
    return [
        {
            "id": r[0],
            "video_id": r[1],
            "event_type": r[2],
            "video_timestamp_s": r[3],
            "metadata": json.loads(r[4]),
        }
        for r in rows
    ]


# ── Tareas de fondo ────────────────────────────────────────────────────────

async def _log_streamer(proc: asyncio.subprocess.Process) -> None:
    """Lee stdout del runner y lo transmite a todos los clientes WS."""
    try:
        assert proc.stdout is not None
        async for line in proc.stdout:
            text = line.decode(errors="replace").rstrip()
            await _broadcast({"type": "log", "text": text})
    except Exception:
        pass
    await _broadcast({"type": "log", "text": "■ Runner finalizado."})
    await _broadcast({"type": "status", "running": False})


async def _background_broadcaster() -> None:
    """Cada segundo: envía estado del sistema. Cada 2s: envía eventos nuevos."""
    global _last_broadcast_event_id
    tick = 0
    while True:
        await asyncio.sleep(1)
        if not _ws_clients:
            tick += 1
            continue

        await _broadcast({"type": "status", "running": _is_running()})

        tick += 1
        if tick % 2 == 0:
            rows = await asyncio.get_event_loop().run_in_executor(
                None, _fetch_events_since, _last_broadcast_event_id
            )
            for row in rows:
                _last_broadcast_event_id = max(_last_broadcast_event_id, row["id"])
                await _broadcast({"type": "event", **row})


# ── Startup ────────────────────────────────────────────────────────────────

@app.on_event("startup")
async def _startup() -> None:
    asyncio.create_task(_background_broadcaster())


# ── Rutas HTTP ─────────────────────────────────────────────────────────────

@app.get("/", response_class=HTMLResponse)
async def index() -> str:
    return (STATIC_DIR / "index.html").read_text(encoding="utf-8")


@app.get("/api/status")
async def get_status() -> dict:
    return {"running": _is_running(), "pid": runner_proc.pid if _is_running() else None}


@app.post("/api/start")
async def start_runner() -> dict:
    global runner_proc
    if _is_running():
        raise HTTPException(400, "El runner ya está en ejecución")
    if not CAMERAS_YAML.exists():
        raise HTTPException(400, "No se encontró cameras.yaml — configurá al menos una fuente")

    runner_proc = await asyncio.create_subprocess_exec(
        sys.executable, "-m", "runner", "--config", str(CAMERAS_YAML),
        cwd=str(ROOT),
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.STDOUT,
    )
    asyncio.create_task(_log_streamer(runner_proc))
    await _broadcast({"type": "log", "text": f"▶ Runner iniciado (PID {runner_proc.pid})"})
    return {"status": "started", "pid": runner_proc.pid}


@app.post("/api/stop")
async def stop_runner() -> dict:
    global runner_proc
    if not _is_running():
        raise HTTPException(400, "El runner no está en ejecución")
    runner_proc.terminate()
    await runner_proc.wait()
    return {"status": "stopped"}


@app.get("/api/cameras")
async def get_cameras() -> dict:
    data = _read_cameras_yaml()
    return data


@app.post("/api/cameras/add")
async def add_camera(body: dict) -> dict:
    cam_id = body.get("id", "").strip()
    source = body.get("source", "").strip()
    if not cam_id:
        raise HTTPException(400, "El campo 'id' es requerido")
    if not source:
        raise HTTPException(400, "El campo 'source' es requerido")

    data = _read_cameras_yaml()
    cameras: list[dict] = data.get("cameras", [])
    if any(c["id"] == cam_id for c in cameras):
        raise HTTPException(400, f"Ya existe una cámara con id '{cam_id}'")

    cameras.append({
        "id": cam_id,
        "source": source,
        "realtime": bool(body.get("realtime", False)),
        "loop": bool(body.get("loop", True)),
    })
    data["cameras"] = cameras
    _write_cameras_yaml(data)
    return {"ok": True}


@app.delete("/api/cameras/{cam_id}")
async def remove_camera(cam_id: str) -> dict:
    data = _read_cameras_yaml()
    original = data.get("cameras", [])
    data["cameras"] = [c for c in original if c["id"] != cam_id]
    if len(data["cameras"]) == len(original):
        raise HTTPException(404, f"No existe cámara con id '{cam_id}'")
    _write_cameras_yaml(data)
    return {"ok": True}


@app.get("/api/videos")
async def list_videos() -> dict:
    videos: list[dict] = []
    for ext in ("*.mp4", "*.avi", "*.MP4", "*.AVI"):
        for f in sorted(VIDEOS_DIR.rglob(ext)):
            rel = str(f.relative_to(ROOT))
            folder = f.parent.name
            videos.append({"name": f.name, "path": rel, "folder": folder})
    return {"videos": videos}


@app.post("/api/upload-video")
async def upload_video(file: UploadFile = File(...)) -> dict:
    UPLOADS_DIR.mkdir(parents=True, exist_ok=True)
    safe_name = Path(file.filename).name  # strip any path components
    dest = UPLOADS_DIR / safe_name
    async with aiofiles.open(dest, "wb") as out:
        content = await file.read()
        await out.write(content)
    rel = str(dest.relative_to(ROOT))
    return {"path": rel, "name": safe_name}


@app.get("/api/frame/{cam_id}")
async def get_frame(cam_id: str) -> FileResponse:
    safe = re.sub(r"[^a-zA-Z0-9_-]", "_", cam_id)
    path = FRAMES_DIR / f"{safe}.jpg"
    if not path.exists():
        raise HTTPException(404, "Frame no disponible")
    return FileResponse(
        str(path),
        media_type="image/jpeg",
        headers={"Cache-Control": "no-store, no-cache, must-revalidate", "Pragma": "no-cache"},
    )


@app.get("/api/frames/list")
async def list_frames() -> dict:
    if not FRAMES_DIR.exists():
        return {"available": []}
    return {"available": [f.stem for f in FRAMES_DIR.glob("*.jpg") if not f.stem.endswith(".tmp")]}


@app.get("/api/video-file/{video_path:path}")
async def serve_video_file(video_path: str) -> FileResponse:
    """Sirve un archivo de video del proyecto para reproducción en el browser."""
    resolved = (ROOT / video_path).resolve()
    try:
        resolved.relative_to(ROOT.resolve())
    except ValueError:
        raise HTTPException(403, "Acceso denegado")
    if not resolved.exists() or not resolved.is_file():
        raise HTTPException(404, "Archivo no encontrado")
    ext = resolved.suffix.lower()
    media_type = "video/mp4" if ext == ".mp4" else "video/x-msvideo"
    return FileResponse(str(resolved), media_type=media_type)


@app.get("/api/events")
async def get_events(limit: int = 100, since_id: int = 0) -> dict:
    rows = await asyncio.get_event_loop().run_in_executor(
        None, _fetch_events_since, since_id, limit
    )
    return {"events": rows}


# ── WebSocket ──────────────────────────────────────────────────────────────

@app.websocket("/ws")
async def ws_endpoint(ws: WebSocket) -> None:
    await ws.accept()
    _ws_clients.add(ws)

    # Estado inicial
    await ws.send_json({"type": "status", "running": _is_running()})

    # Últimos 30 eventos al conectar
    rows = await asyncio.get_event_loop().run_in_executor(
        None, lambda: _fetch_events_since(0, 30)
    )
    for row in rows:
        await ws.send_json({"type": "event", **row})

    try:
        while True:
            try:
                await asyncio.wait_for(ws.receive_text(), timeout=25)
            except asyncio.TimeoutError:
                await ws.send_json({"type": "ping"})
    except WebSocketDisconnect:
        pass
    except Exception:
        pass
    finally:
        _ws_clients.discard(ws)
