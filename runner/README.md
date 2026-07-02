# `runner/` — Orquestador multi-cámara

## Qué hace

Corre el pipeline completo (pose estimation de `vision/` + clasificación de
actividad de `vlm/`) en paralelo sobre varias fuentes de video a la vez —
archivos locales en loop o streams en vivo (RTSP/HTTP/webcam) — y escribe
los eventos resultantes al mismo store de `events/`. Es el reemplazo, para
la fase con cámaras reales, de correr `vision/main.py` y `vlm/main.py` por
separado sobre un dataset grabado.

Lo arranca `dashboard/` como subproceso (`python -m runner`), pero también
se puede correr standalone.

## Estructura

```
runner/
├── main.py           # entry point: carga config, arranca un CameraWorker por cámara
├── config.py          # lee cameras.yaml → RunnerConfig / CameraConfig
├── camera_source.py    # VideoFileLoop (archivo con loop) / LiveStream (RTSP/HTTP/webcam)
└── camera_worker.py    # thread por cámara: pose pipeline + CLIP periódico + frame preview
```

Un `CameraWorker` (thread) por cámara. Cada worker corre su propia instancia
de `FallImmobilityDetector` (de `vision/`), pero todas comparten una única
instancia de `ClipActivityClassifier` (de `vlm/`) cargada una sola vez en
`main.py` — ONNX Runtime es thread-safe para inferencia concurrente, y
cargar CLIP por cámara sería redundante.

## Config (`cameras.yaml`)

Vive en la raíz del proyecto:

```yaml
cameras:
- id: asd
  source: videos/Uselaptop_p12_r00_v14_c05.mp4   # archivo, URL RTSP/HTTP, o índice de webcam ("0")
  realtime: false   # si true, throttlea a los FPS del video (más lento pero fiel a un stream real)
  loop: true         # si true, reinicia el archivo al terminar (ignorado en streams en vivo)
pose_model: yolo11n-pose.pt
log_level: INFO
clip_interval_s: 10.0   # segundos entre clasificaciones CLIP por cámara
```

Si `source` empieza con `rtsp://`, `http(s)://`, o es un entero (índice de
webcam), `camera_source.make_source()` lo trata como stream en vivo
(`LiveStream`, con reconexión automática ante cortes). Cualquier otra cosa
se resuelve como ruta a archivo, relativa a la raíz del proyecto si no es
absoluta (`VideoFileLoop`, con loop opcional).

Los timestamps de los eventos son wall-clock (segundos desde que arrancó la
fuente), no el tiempo relativo al archivo — necesario para que el detector
de inmovilidad de `vision/` tenga sentido cuando un archivo hace loop o
cuando la fuente es un stream en vivo sin timestamps propios.

## CLIP es opcional

Si `vlm/` no está disponible (no se pudo importar) o no se generaron los
modelos ONNX (`vlm/export_onnx.py`), el runner sigue funcionando solo con
pose estimation — loggea una advertencia y no clasifica actividad. Ver
`../vlm/README.md` para generar los modelos.

## Frame preview

Cada worker guarda 1 de cada 5 frames como `../frames/<cam_id>.jpg`
(escritura atómica vía archivo temporal + `replace`), que expone
`dashboard/` en `GET /api/frame/{cam_id}` para mostrar la cámara en vivo en
el panel.

## Setup

```
pip install -r requirements.txt
pip install -r ../vision/requirements.txt
pip install -r ../events/requirements.txt
```

## Cómo probarlo

```
python -m runner                          # usa cameras.yaml en la raíz
python -m runner --config otra_config.yaml
```

Ctrl+C (o `SIGTERM`) detiene todas las cámaras de forma prolija.

## Fuera de alcance por ahora

- Multi-persona por cámara (mismo límite que `vision/`).
- Balanceo de carga entre cámaras si el CPU no alcanza para todas en
  paralelo — asume que el hardware objetivo (Pi 5) soporta la cantidad de
  cámaras configuradas.
