# `dashboard/` — Panel web de control y monitoreo

## Qué hace

API FastAPI + panel HTML de una sola página para operar el sistema sin
línea de comandos: arrancar/parar `runner/`, gestionar `cameras.yaml`,
subir videos de prueba, ver el frame en vivo de cada cámara y los eventos
que van llegando al store de `events/` en tiempo real (vía WebSocket).

Es la capa de operación sobre `runner/` — no reemplaza ni duplica lógica de
detección, solo la orquesta y la visualiza.

## Estructura

```
dashboard/
├── __main__.py        # entry point: uvicorn sirviendo api.py
├── api.py              # rutas FastAPI + WebSocket
└── static/index.html   # panel (HTML/CSS/JS, sin build step ni framework)
```

## Cómo arranca `runner/`

`POST /api/start` lanza `python -m runner --config cameras.yaml` como
subproceso (`asyncio.create_subprocess_exec`), con `cwd` en la raíz del
proyecto. El dashboard no importa código de `runner/` ni de `vision/`/`vlm/`
directamente — la única comunicación es a través del subproceso (stdout
para logs) y del store de eventos (SQLite) para los datos, igual que
`baseline/` o `evaluation/` lo harían.

## Rutas

| Ruta | Qué hace |
|---|---|
| `GET /` | sirve el panel (`static/index.html`) |
| `GET /api/status` | si el runner está corriendo, y su PID |
| `POST /api/start` | arranca `runner/` como subproceso |
| `POST /api/stop` | lo detiene (`terminate()` + espera) |
| `GET /api/cameras` | lee `cameras.yaml` |
| `POST /api/cameras/add` | agrega una cámara al YAML |
| `DELETE /api/cameras/{id}` | la saca del YAML |
| `GET /api/videos` | lista videos disponibles en `videos/` |
| `POST /api/upload-video` | sube un archivo a `videos/uploads/` |
| `GET /api/video-file/{path}` | sirve un video para reproducirlo en el browser |
| `GET /api/frame/{cam_id}` | último frame preview de esa cámara (`frames/{id}.jpg`, con esqueleto de YOLO dibujado encima si hay detección) |
| `GET /api/frames/list` | qué cámaras tienen preview disponible |
| `GET /api/events` | eventos del store, paginado por `since_id` |
| `WS /ws` | push de logs del runner + eventos nuevos + estado, cada ~1s |

El WebSocket manda el estado (`running`) cada segundo y consulta eventos
nuevos cada 2 segundos (`_background_broadcaster`), solo si hay clientes
conectados. Al conectar, un cliente recibe el estado actual y los últimos
30 eventos antes de empezar a recibir novedades.

## Vistas del panel

- **Todas las fuentes**: grilla con el preview de cada cámara y una banda
  común con todos los eventos mezclados.
- **Vista única**: una cámara seleccionada en grande, con sus eventos
  separados por detector en dos paneles apilados: **Postura · YOLO**
  (caídas e inmovilidad, del pipeline de pose de `vision/`) y
  **Actividad · CLIP** (clasificación de escena de `vlm/`), cada uno con su
  contador y su scroll — así las actividades frecuentes de CLIP no
  entierran las caídas. La separación se decide por `event_type` (solo CLIP
  emite `activity`).

Las tarjetas de eventos CLIP muestran como título la actividad predicha
traducida al español con ícono propio (mapa `ACTIVITY_ES` en
`static/index.html`: `Walk` → "🚶 Caminando", `Readbook` → "📖 Leyendo un
libro", etc.) y la confianza de la predicción en la línea de metadata. Una
clase que no esté en el mapa se muestra con su etiqueta cruda.

## Seguridad de rutas de archivo

`upload-video` sanitiza el nombre (`Path(file.filename).name`, descarta
cualquier componente de path) antes de guardarlo. `video-file/{path}`
resuelve la ruta pedida y verifica con `relative_to(ROOT)` que quede dentro
de la raíz del proyecto antes de servirla, para no exponer el filesystem.

## Setup

```
cd dashboard
pip install -r requirements.txt
```

## Cómo probarlo

```
python -m dashboard
```

Sirve en `http://localhost:8000` (ver `__main__.py` para el puerto exacto).
Desde ahí: cargar/editar `cameras.yaml` desde el panel, apretar "iniciar" y
ver logs/eventos/preview en vivo.

## Fuera de alcance por ahora

- Autenticación — pensado para uso local, no exposición a internet.
- Multi-usuario / control de acceso.
