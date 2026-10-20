"""Dashboard web para el sistema de monitoreo.

Expone:
  GET  /            → panel HTML
  GET  /logs        → panel de logs del sistema
  GET  /api/status  → estado del runner
  POST /api/start   → inicia el runner como subproceso
  POST /api/stop    → detiene el runner
  GET  /api/cameras → lee cameras.yaml
  POST /api/cameras/add      → agrega una cámara
  DELETE /api/cameras/{id}   → elimina una cámara
  GET  /api/videos  → lista videos disponibles
  POST /api/upload-video     → sube un video
  GET  /api/events  → lee eventos de SQLite
  GET  /api/overlay → estado del overlay de pose (bbox + keypoints)
  POST /api/overlay → activa/desactiva el overlay en todas las cámaras
  GET  /api/motion  → estado del gate de movimiento (runner/motion_gate.py) por cámara
  POST /api/focus   → marca qué cámara está en vista única (runner/focus_state.py)
  GET  /api/demo    → si el modo demo está activado (clave `demo` de cameras.yaml, true por defecto)
  GET  /api/playback → estado de reproducción de los videos en modo demo (runner/playback_state.py)
  POST /api/playback/{cam_id} → play/pausa del video de esa cámara
  GET  /api/frame-stream/{cam_id} → preview en vivo como stream MJPEG (fluido si está en foco)
  GET  /api/kit     → kit de esta controladora (kit.yaml): QR de vinculación y estado de la nube
  POST /api/kit/fuente → elige qué fuente se sube a la nube como la cámara del kit
  GET  /kit         → etiqueta imprimible con el QR del kit
  WS   /ws          → stream de logs + estado en tiempo real
"""
from __future__ import annotations

import asyncio
import html
import json
import sqlite3
import sys
import time
from pathlib import Path
from typing import Optional

import re

import aiofiles
import yaml
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, UploadFile, File, HTTPException
from fastapi.responses import FileResponse, HTMLResponse, StreamingResponse

from runner.cloud_state import STATE_PATH as CLOUD_STATE_PATH
from runner.cloud_state import SOURCE_PATH as CLOUD_SOURCE_PATH
from runner.cloud_state import read_cloud_source, read_cloud_state, write_cloud_source
from runner.config import KIT_FILENAME
from runner.focus_state import write_focused_camera
from runner.motion_state import read_all_motion_state
from runner.overlay_state import read_overlay_enabled, write_overlay_enabled
from runner.playback_state import delete_status as delete_playback_status
from runner.playback_state import read_all_status as read_all_playback_status
from runner.playback_state import write_paused
from runner.preview_file import FRAMES_DIR, delete_preview

ROOT = Path(__file__).resolve().parents[1]
CAMERAS_YAML = ROOT / "cameras.yaml"
KIT_YAML = ROOT / KIT_FILENAME
DB_PATH = ROOT / "events" / "events.db"
VIDEOS_DIR = ROOT / "videos"
UPLOADS_DIR = VIDEOS_DIR / "uploads"
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


def _rows_to_events(rows: list[tuple]) -> list[dict]:
    return [
        {
            "id": r[0],
            "video_id": r[1],
            "event_type": r[2],
            "video_timestamp_s": r[3],
            "wall_clock_ts": r[4],
            "metadata": json.loads(r[5]),
        }
        for r in rows
    ]


def _connect() -> sqlite3.Connection:
    """Abre la conexión y agrega `wall_clock_ts` si la DB es de antes de ese campo (ver
    events/store.py — misma migración, acá porque esta API lee con SQL crudo, no con EventStore)."""
    conn = sqlite3.connect(str(DB_PATH))
    cols = {row[1] for row in conn.execute("PRAGMA table_info(events)")}
    if "wall_clock_ts" not in cols:
        conn.execute("ALTER TABLE events ADD COLUMN wall_clock_ts REAL")
        conn.commit()
    return conn


def _fetch_events_since(last_id: int, limit: int = 50) -> list[dict]:
    if not DB_PATH.exists():
        return []
    conn = _connect()
    rows = conn.execute(
        "SELECT id, video_id, event_type, video_timestamp_s, wall_clock_ts, metadata "
        "FROM events WHERE id > ? ORDER BY id ASC LIMIT ?",
        (last_id, limit),
    ).fetchall()
    conn.close()
    return _rows_to_events(rows)


def _fetch_last_events(limit: int = 30) -> list[dict]:
    if not DB_PATH.exists():
        return []
    conn = _connect()
    rows = conn.execute(
        "SELECT id, video_id, event_type, video_timestamp_s, wall_clock_ts, metadata "
        "FROM events ORDER BY id DESC LIMIT ?",
        (limit,),
    ).fetchall()
    conn.close()
    return _rows_to_events(list(reversed(rows)))


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
    global _last_broadcast_event_id
    if DB_PATH.exists():
        conn = sqlite3.connect(str(DB_PATH))
        row = conn.execute("SELECT MAX(id) FROM events").fetchone()
        conn.close()
        _last_broadcast_event_id = row[0] or 0
    asyncio.create_task(_background_broadcaster())


@app.on_event("shutdown")
async def _shutdown() -> None:
    # Si el dashboard se cierra con el runner andando, no dejarlo huérfano.
    if _is_running():
        runner_proc.terminate()
        await runner_proc.wait()


# ── Rutas HTTP ─────────────────────────────────────────────────────────────

@app.get("/", response_class=HTMLResponse)
async def index() -> str:
    return (STATIC_DIR / "index.html").read_text(encoding="utf-8")


@app.get("/logs", response_class=HTMLResponse)
async def logs_page() -> str:
    return (STATIC_DIR / "logs.html").read_text(encoding="utf-8")


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
    # El runner borra la vista previa de cada cámara al detenerse, pero en Windows `terminate()`
    # lo mata sin darle tiempo: se borra también acá para que la imagen no quede en disco.
    for cam in _read_cameras_yaml().get("cameras") or []:
        delete_preview(str(cam.get("id", "")), FRAMES_DIR)
        delete_playback_status(str(cam.get("id", "")))
    return {"status": "stopped"}


@app.get("/api/demo")
async def get_demo() -> dict:
    # Mismo default que runner/config.py: encendido salvo `demo: false` en cameras.yaml.
    return {"demo": bool(_read_cameras_yaml().get("demo", True))}


@app.get("/api/playback")
async def get_playback() -> dict:
    """{"cameras": {"<id>": {"paused", "ended", "position_s", "duration_s", "updated_at"}}} de
    los videos que se están reproduciendo en modo demo. Las demás cámaras no aparecen."""
    return {"cameras": read_all_playback_status()}


@app.post("/api/playback/{cam_id}")
async def set_playback(cam_id: str, body: dict) -> dict:
    """Pausa o reanuda el video de esa cámara. Si el video ya terminó, play lo arranca de nuevo."""
    if "paused" not in body:
        raise HTTPException(400, "El campo 'paused' es requerido")
    paused = bool(body["paused"])
    write_paused(cam_id, paused)
    return {"cam_id": cam_id, "paused": paused}


def _read_kit() -> dict | None:
    """kit.yaml (ver "Vincular con la app móvil" en runner/README.md), o None si no hay."""
    if not KIT_YAML.exists():
        return None
    return yaml.safe_load(KIT_YAML.read_text()) or None


def _qr_kit(kit: dict) -> str:
    """Lo que lee el escáner de la app (src/features/vinculacion/codigoKit.ts del Frontend)."""
    return f"lares://vincular?serie={kit['numero_serie']}&codigo={kit['codigo_vinculacion']}"


def _qr_svg(texto: str) -> Optional[str]:
    try:
        import segno
    except ImportError:  # sin segno el panel muestra el código para tipearlo a mano
        return None
    # omitsize: SVG con viewBox y sin width/height fijos, para que el CSS lo escale sin recortarlo.
    return segno.make(texto, error="m").svg_inline(omitsize=True, border=4, dark="#000", light="#fff")


def _codigo_legible(codigo: str) -> str:
    return f"{codigo[:4]}-{codigo[4:]}"


@app.get("/api/kit")
async def get_kit() -> dict:
    """Kit de esta controladora para vincularlo desde la app: serie, código, QR (SVG) y el último
    estado de la sincronización con la nube que dejó el runner (runner/cloud_state.py). Nunca
    devuelve la API key. `estado_vigente` es False si el runner no escribió nada hace rato
    (detenido, o la nube no configurada en esta corrida)."""
    kit = _read_kit()
    if kit is None:
        return {"configurado": False}
    codigo = str(kit.get("codigo_vinculacion") or "")
    serie = str(kit.get("numero_serie") or "")
    if not codigo or not serie:
        raise HTTPException(500, "kit.yaml no tiene numero_serie y codigo_vinculacion")
    estado = read_cloud_state(CLOUD_STATE_PATH)
    margen = max(3 * float(kit.get("interval_s", 30.0)), 30.0)
    vigente = bool(estado) and time.time() - float(estado.get("actualizado_at", 0)) < margen
    return {
        "configurado": True,
        "numero_serie": serie,
        "codigo": _codigo_legible(codigo),
        "qr_svg": _qr_svg(_qr_kit(kit)),
        "url": kit.get("url"),
        "solo_desde_vinculacion": bool(kit.get("solo_desde_vinculacion", False)),
        "estado": estado,
        "estado_vigente": vigente,
        # Botón "App" de cada fuente: solo con `camara` en kit.yaml.
        "camara_kit": kit.get("camara"),
        "fuente_nube": read_cloud_source(CLOUD_SOURCE_PATH),
    }


@app.post("/api/kit/fuente")
async def set_kit_source(body: dict) -> dict:
    """Elige la fuente de cameras.yaml que se sube a la nube como la cámara del kit
    (`{"cam_id": "..."}`, o `null` para ninguna). El runner lo toma en el próximo ciclo de
    sincronización, sin reiniciar; los eventos de las demás fuentes no se suben."""
    kit = _read_kit()
    if kit is None or not kit.get("camara"):
        raise HTTPException(400, "kit.yaml no tiene la clave `camara` (la cámara del kit en la nube)")
    cam_id = body.get("cam_id") or None
    if cam_id is not None and not any(
        str(c.get("id")) == cam_id for c in _read_cameras_yaml().get("cameras") or []
    ):
        raise HTTPException(404, f"No existe una fuente con id '{cam_id}'")
    write_cloud_source(cam_id, CLOUD_SOURCE_PATH)
    return {"cam_id": cam_id}


@app.get("/kit", response_class=HTMLResponse)
async def kit_label() -> str:
    """Etiqueta del kit para imprimir: QR, código para tipear y número de serie."""
    kit = _read_kit()
    if kit is None:
        raise HTTPException(404, "No hay kit.yaml en esta controladora")
    serie = html.escape(str(kit["numero_serie"]))
    codigo = html.escape(_codigo_legible(str(kit["codigo_vinculacion"])))
    qr = _qr_svg(_qr_kit(kit)) or ""
    return f"""<!DOCTYPE html>
<html lang="es"><head><meta charset="UTF-8"><title>Kit {serie}</title>
<style>
  body {{ font-family: -apple-system, 'Segoe UI', system-ui, sans-serif; background: #fff; color: #000;
         display: flex; justify-content: center; padding: 40px 16px; }}
  .etiqueta {{ border: 2px solid #000; border-radius: 16px; padding: 28px 36px; text-align: center; }}
  h1 {{ font-size: 22px; margin: 0 0 4px; }}
  p {{ margin: 4px 0; color: #333; }}
  .qr svg {{ display: block; width: 300px; height: 300px; margin: 16px auto; }}
  .codigo {{ font: 700 30px 'SF Mono', Consolas, monospace; letter-spacing: .12em; color: #000; }}
  @media print {{ .no-print {{ display: none; }} }}
</style></head>
<body><div class="etiqueta">
  <h1>Kit Lares</h1>
  <p>Escaneá este código desde la app para vincular el kit</p>
  <div class="qr">{qr}</div>
  <p>o ingresá el código a mano</p>
  <p class="codigo">{codigo}</p>
  <p>N.º de serie {serie}</p>
  <p class="no-print"><button onclick="print()">Imprimir</button></p>
</div></body></html>"""


@app.get("/api/overlay")
async def get_overlay() -> dict:
    return {"show_overlay": read_overlay_enabled()}


@app.post("/api/overlay")
async def set_overlay(body: dict) -> dict:
    if "show_overlay" not in body:
        raise HTTPException(400, "El campo 'show_overlay' es requerido")
    enabled = bool(body["show_overlay"])
    write_overlay_enabled(enabled)
    return {"show_overlay": enabled}


@app.get("/api/motion")
async def get_motion() -> dict:
    """Estado del gate de movimiento por cámara (ver runner/motion_gate.py):
    despierta (YOLO/CLIP corriendo) o dormida (pausados, sin movimiento ni
    persona detectada). Una cámara sin el gate activo, o mientras el runner
    no arrancó, simplemente no aparece en la respuesta."""
    return {"cameras": read_all_motion_state()}


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
        "realtime": bool(body.get("realtime", True)),
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


def _resolve_project_file(rel_path: str) -> Path:
    resolved = (ROOT / rel_path).resolve()
    try:
        resolved.relative_to(ROOT.resolve())
    except ValueError:
        raise HTTPException(403, "Acceso denegado")
    if not resolved.exists() or not resolved.is_file():
        raise HTTPException(404, "Archivo no encontrado")
    return resolved


@app.get("/api/video-file/{video_path:path}")
async def serve_video_file(video_path: str) -> FileResponse:
    """Sirve un archivo de video del proyecto para reproducción en el browser."""
    resolved = _resolve_project_file(video_path)
    ext = resolved.suffix.lower()
    media_type = "video/mp4" if ext == ".mp4" else "video/x-msvideo"
    return FileResponse(str(resolved), media_type=media_type)


@app.post("/api/focus")
async def set_focus(body: dict) -> dict:
    """Le avisa a runner/ qué cámara está en vista única (ver
    runner/focus_state.py) — mientras está enfocada, su CameraWorker escribe
    el preview en cada frame en vez de 1 de cada 5, para que
    GET /api/frame-stream pueda mostrarla fluida y con overlay de YOLO."""
    cam_id = body.get("cam_id") or None
    write_focused_camera(cam_id)
    return {"cam_id": cam_id}


@app.get("/api/frame-stream/{cam_id}")
async def stream_frame(cam_id: str) -> StreamingResponse:
    """Stream MJPEG (multipart/x-mixed-replace) del preview que ya escribe
    runner/ en runner_state/frames/{cam_id}.jpg (con overlay de pose si está activado) —
    tailea el archivo por mtime y empuja cada versión nueva. Fluido solo
    mientras esa cámara está enfocada (ver /api/focus); si no, se actualiza a
    la tasa reducida de siempre (1 de cada 5 frames)."""
    safe = re.sub(r"[^a-zA-Z0-9_-]", "_", cam_id)
    path = FRAMES_DIR / f"{safe}.jpg"

    async def gen():
        last_mtime: Optional[float] = None
        while True:
            try:
                mtime = path.stat().st_mtime
            except OSError:
                mtime = None
            if mtime is not None and mtime != last_mtime:
                last_mtime = mtime
                try:
                    data = path.read_bytes()
                except OSError:
                    data = None
                if data:
                    yield (
                        b"--frame\r\n"
                        b"Content-Type: image/jpeg\r\n"
                        b"Content-Length: " + str(len(data)).encode() + b"\r\n\r\n"
                        + data + b"\r\n"
                    )
            await asyncio.sleep(0.02)

    return StreamingResponse(gen(), media_type="multipart/x-mixed-replace; boundary=frame")


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
        None, lambda: _fetch_last_events(30)
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
