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
└── static/
    ├── index.html      # panel principal (HTML/CSS/JS, sin build step ni framework)
    └── logs.html        # panel de logs del sistema, servido en GET /logs
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
| `GET /logs` | sirve el panel de logs del sistema (`static/logs.html`) |
| `GET /api/status` | si el runner está corriendo, y su PID |
| `POST /api/start` | arranca `runner/` como subproceso |
| `POST /api/stop` | lo detiene (`terminate()` + espera) y borra la vista previa de cada cámara |
| `GET /api/cameras` | lee `cameras.yaml` |
| `POST /api/cameras/add` | agrega una cámara al YAML |
| `DELETE /api/cameras/{id}` | la saca del YAML |
| `GET /api/videos` | lista videos disponibles en `videos/` |
| `POST /api/upload-video` | sube un archivo a `videos/uploads/` |
| `GET /api/video-file/{path}` | sirve un video para reproducirlo en el browser |
| `GET /api/frame/{cam_id}` | último frame preview de esa cámara (`runner_state/frames/{id}.jpg`, con esqueleto de YOLO dibujado encima si hay detección y el overlay está activado) |
| `GET /api/frames/list` | qué cámaras tienen preview disponible |
| `GET /api/events` | eventos del store, paginado por `since_id` |
| `GET /api/overlay` | estado actual del toggle de overlay de pose (bbox + keypoints) |
| `POST /api/overlay` | activa/desactiva el overlay en el preview de todas las cámaras |
| `GET /api/motion` | estado del gate de movimiento (despierta/dormida) por cámara |
| `POST /api/focus` | marca qué cámara está en vista única (`{"cam_id": "..."}` o `null`) |
| `GET /api/frame-stream/{cam_id}` | preview de esa cámara como stream MJPEG (`multipart/x-mixed-replace`) |
| `GET /api/demo` | si el modo demo está activado (clave `demo` de `cameras.yaml`, `true` por defecto) |
| `GET /api/playback` | estado de reproducción (pausado, terminado, posición, duración) de los videos en modo demo |
| `POST /api/playback/{cam_id}` | play/pausa del video de esa cámara (`{"paused": bool}`); play sobre un video terminado lo reinicia |
| `GET /api/kit` | kit de esta controladora (`kit.yaml`): serie, código, QR en SVG y estado de la sincronización con la nube (`runner_state/cloud.json`). Nunca devuelve la API key |
| `POST /api/kit/fuente` | elige qué fuente se sube a la nube como la cámara del kit (`{"cam_id": "..."}` o `null`); botón APP de cada fuente |
| `GET /kit` | etiqueta del kit para imprimir (QR + código + serie) |
| `WS /ws` | push de logs del runner + eventos nuevos + estado, cada ~1s |

`GET`/`POST /api/overlay` leen y escriben `runner_state/overlay.json` vía
`runner/overlay_state.py` — el mismo archivo que usa `camera_worker.py` para
decidir si dibuja el esqueleto en el preview (ver "Frame preview" en
`runner/README.md`). Es la forma en que el dashboard y el runner, procesos
separados, comparten ese estado sin acoplar código.

`GET /api/motion` lee `runner_state/motion/{id}.json` vía
`runner/motion_state.py` — un archivo por cámara, escrito por su
`CameraWorker` cada vez que despierta/duerme (ver "Gate de movimiento" en
`runner/README.md`). Devuelve `{"cameras": {"<id>": {"awake": bool,
"updated_at": <epoch>}}}`; una cámara con el gate desactivado, o mientras el
runner no arrancó, no aparece en la respuesta.

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

## Vista única fluida

El preview normal (`GET /api/frame/{cam_id}`, pollado cada 400ms desde la
grilla) solo se actualiza 1 de cada 5 frames procesados
(`_FRAME_PREVIEW_EVERY_N` en `runner/camera_worker.py`) — a propósito, para
no gastar CPU/disco en cámaras que nadie está mirando. Eso lo hace lento
para ver un video fluido.

`selectCamera()` en `static/index.html` resuelve esto avisándole a `runner/`
qué cámara está en foco (`POST /api/focus`, ver `runner/focus_state.py`):
mientras una cámara está enfocada, su `CameraWorker` guarda el preview en
**cada** frame en vez de 1 de cada 5 (la pose ya corre sobre todos los
frames mientras la cámara está despierta — ver "Gate de movimiento" más
abajo — así que esto no agrega costo de inferencia, solo de encode+escritura
del JPEG). `GET /api/frame-stream/{cam_id}` tailea ese archivo por `mtime` y
lo empuja como stream MJPEG (`multipart/x-mixed-replace`) — el `<img>` de la
vista única lo consume directo, sin polling, y de paso trae dibujado el
overlay de pose si está activado.

Este mecanismo depende de que YOLO corra lo bastante rápido como para no
atrasarse frame a frame — en CPU (~150-300ms/frame, ver `vision/README.md`)
el preview fluido se nota entrecortado; con GPU (CUDA) baja a
~20ms/frame y el stream sale fluido de verdad. `vision/src/pose_pipeline.py`
ya elige `cuda` automáticamente si `torch.cuda.is_available()` — solo hace
falta tener instalado un `torch` con build CUDA (no la `+cpu`).

Al salir de la vista única (`setViewMode('multi')`) se manda
`POST /api/focus {"cam_id": null}` para que esa cámara vuelva a la tasa
reducida y no quede escribiendo a disco en cada frame sin que nadie la mire.

## Sin vista previa (2026-10-13)

`runner/` ya no deja `runner_state/frames/{id}.jpg` en disco cuando no está analizando esa
cámara: lo borra al detenerse y al dormirse la cámara por el gate de
movimiento, y dormida no lo escribe (ver "La vista previa no queda en disco"
en `runner/README.md`). Para el panel, que el archivo no exista es un estado
normal:

- `GET /api/frame/{cam_id}` responde 404 y la celda de la grilla muestra su
  placeholder (`onCellError` en `static/index.html`), con el texto "SIN
  MOVIMIENTO" si la cámara está dormida.
- `GET /api/frames/list` devuelve la lista vacía.
- `GET /api/frame-stream/{cam_id}` no manda nada mientras no hay archivo, sin
  cortar la conexión, y retoma solo cuando la cámara despierta. En la vista
  única el navegador se queda mostrando el último frame que recibió, quieto,
  junto al texto "SIN MOVIMIENTO": esa imagen está en la memoria del navegador,
  no en el disco de la controladora.
- `POST /api/stop` borra la vista previa de cada cámara de `cameras.yaml`
  después de detener el runner. Hace falta porque en Windows `terminate()`
  mata el proceso sin que llegue a limpiar; en Linux el runner ya lo hace solo.

Pruebas en `tests/test_vista_previa_ausente.py` (`pytest dashboard/tests`
desde la raíz del repo).

## Indicador de movimiento

Junto al nombre de cada cámara (grilla de "Todas las fuentes" y overlay de
la vista única) hay un indicador del gate de movimiento
(`runner/motion_gate.py`): un punto verde mientras YOLO/CLIP están activos
para esa cámara (hay movimiento reciente o YOLO sigue viendo a la persona,
se mueva o no — ver "Gate de movimiento" en `runner/README.md`), o el texto
"SIN MOVIMIENTO" mientras están en pausa por no detectar a nadie. Si la
cámara no tiene el gate activo (`motion_gate_enabled: false`) o el runner
todavía no arrancó, no se muestra nada — no hay estado que mostrar, y
mostrar un punto apagado por defecto se podría confundir con "sin
movimiento".

`pollMotion()` en `static/index.html` consulta `GET /api/motion` cada
segundo (aparte del poll de frames, que es cada 400ms pero por cámara — el
de movimiento es una sola consulta liviana para todas) y actualiza el badge
de cada celda de la grilla y el de la vista única vía `renderMotionBadge()`.

Antes existía además un `.live-dot` junto al nombre de cada cámara (si el
último frame pedido a `/api/frame/{id}` cargó bien en el browser) — se sacó
al agregar el badge de movimiento: los dos puntos uno al lado del otro no se
distinguían y esa señal ya la muestra el placeholder de la celda
("Conectando…"/ícono de pausa cuando el `<img>` no carga).

## Modo demo (2026-10-19)

Encendido por defecto, sin botón en el panel (se sacó el 2026-10-19): solo se apaga con
`demo: false` en `cameras.yaml`, y aplica al reiniciar el runner. Con el modo activo, el
checkbox de loop del formulario de video se oculta: el runner reproduce los archivos una sola vez igual (ver "Modo demo" en
`runner/README.md`).

Cada cámara de archivo arranca en pausa (la celda dice "En pausa — dale play" mientras no hay
vista previa) y muestra un botón play/pausa y la posición (`0:12 / 0:45`, o `FIN` al
terminar) en el pie de su celda de la grilla y en el recuadro de la vista única.
`pollPlayback()` en `static/index.html` consulta `GET /api/playback` cada 0,5 s; una cámara que
no está en modo demo, o el runner apagado, no aparece en la respuesta y no muestra controles.
`POST /api/stop` borra también el estado de reproducción de cada cámara.

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

## App móvil (2026-10-19)

El botón **App móvil** del encabezado muestra el QR para vincular esta controladora desde la app,
y si la nube ya acepta sus eventos. Necesita `kit.yaml` en la raíz del repo y `segno`
(`requirements.txt`). Sin `segno` muestra solo el código para tipear. Cómo generar `kit.yaml`,
qué significa cada estado y el guion de la demo: "Vincular con la app móvil" en
`runner/README.md`.
