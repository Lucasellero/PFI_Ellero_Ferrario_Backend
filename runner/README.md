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
├── camera_worker.py    # thread por cámara: pose pipeline + CLIP periódico + frame preview
└── overlay_state.py     # lee/escribe runner_state/overlay.json (toggle de overlay compartido con dashboard/)
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
  realtime: true    # ver "Tiempo real y frame dropping" abajo — recomendado en true
  loop: true         # si true, reinicia el archivo al terminar (ignorado en streams en vivo)
pose_model: yolo11n-pose.pt
log_level: INFO
clip_interval_s: 10.0      # segundos entre ventanas de clasificación CLIP por cámara
clip_min_confidence: 0.4   # confianza mínima para emitir un evento de actividad
clip_heartbeat_s: 60.0     # cada cuánto re-emitir una actividad que sigue igual
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

## Tiempo real y frame dropping

YOLO11n-pose tarda ~150-300ms por frame en CPU — mucho más que la duración
real de un frame de video (~33ms a 30fps). Si se procesara cada frame en
orden sin descartar ninguno, el contenido se "estira" en el tiempo: se ve
en cámara lenta con un archivo (`VideoFileLoop`), y con una cámara real se
traduce en un atraso creciente en vez de cámara lenta, porque los frames
siguen llegando aunque no se procesen a tiempo (`LiveStream`).

Con `realtime: true`, ambas fuentes descartan frames "viejos" para no
quedarse atrás en vez de encolarlos:

- `VideoFileLoop` compara cada frame leído contra su tiempo "programado"
  (`frame_idx / fps`) y el tiempo real transcurrido; si ya está atrasado
  más de un frame de duración, lo saltea sin entregarlo al pipeline.
- `LiveStream` usa un thread lector en background que drena la
  cámara/stream tan rápido como puede y siempre deja el frame más reciente
  disponible en una variable compartida — `frames()` nunca entrega un frame
  viejo, aunque el consumidor tarde más que el intervalo real entre frames.

Se pierde densidad de muestreo bajo carga (no se procesan todos los
frames), pero la heurística de caída de `vision/` ya trabaja con ventanas
de ~0.4-1.5s, no depende de ver cada frame individual — ver
`vision/README.md`. Con `realtime: false` no hay descarte: se procesa cada
frame del archivo en orden, tan rápido como el CPU pueda, útil para pasadas
batch donde importa completitud y no la sensación de tiempo real (por
ejemplo, reprocesar un archivo para popular el store de eventos).

## Clasificación de actividades (CLIP)

Pensada para una cámara fija transmitiendo continuamente la habitación de
una persona. Cada `clip_interval_s` se abre una **ventana de muestreo**: se
clasifican 3 frames espaciados ~1s y se promedian los scores — un frame
único es ruidoso (persona de espaldas, a mitad de un movimiento) y el
promedio estabiliza la predicción. Además:

- **Recorte a la persona**: cada frame se recorta al bounding box que ya
  calculó el detector de pose (con 25% de margen) antes de pasar por CLIP.
  En una cámara fija el fondo constante domina la similitud (una TV visible
  empuja todo hacia `WatchTV`); clasificar solo a la persona elimina ese
  sesgo.
- **Sin persona, sin evento**: si YOLO no detecta a nadie en la ventana, no
  se clasifica nada (una habitación vacía no tiene actividad). Además hay
  una clase negativa `Background` en los prompts: si el recorte no muestra
  una persona reconocible (bbox desfasado, persona saliendo de cuadro), esa
  clase absorbe la probabilidad y la ventana se descarta.
- **Categorías gruesas**: se clasifica con las 14 clases finas del dataset
  (mejor cobertura de prompts) pero el evento se emite a nivel de categoría
  (`COARSE_CATEGORIES` en `vlm/src/clip_classifier.py`: `Cooking`,
  `EatingDrinking`, `Medication`, `Resting`, `ReadingScreen`, `Moving`),
  sumando las probabilidades de las subclases. Distinguir `Drink.Fromcan`
  de `Drink.Fromcup` en un frame estático es imposible y al familiar
  tampoco le aporta; agregarlas elimina ese error. `activity_label` en el
  metadata es la categoría; la clase fina ganadora queda en `fine_label`.
- **Umbral de confianza** (`clip_min_confidence`): se aplica sobre la
  confianza de la *categoría* (suma de sus subclases); las ventanas
  ambiguas se descartan (quedan en el log) en vez de emitir una etiqueta
  dudosa.
- **Deduplicación**: se emite evento cuando la categoría *cambia*; si sigue
  igual se re-emite cada `clip_heartbeat_s` con `ongoing: true` en el
  metadata, para que el análisis posterior pueda reconstruir duraciones sin
  ahogarse en eventos repetidos.

## Confirmación de caídas con CLIP

El detector geométrico de `vision/` (velocidad vertical + aspect ratio) no
distingue una caída real de sentarse/acostarse a propósito en algunos setups
de cámara (bbox ancho y bajo genera la misma firma geométrica que una caída
— ver "Detección de caída como evento" en `vision/README.md` para el
detalle y por qué recalibrar los umbrales geométricos no alcanza). Como
segunda etapa, cuando `FallImmobilityDetector` dispara un evento `fall`, el
worker le pregunta a CLIP zero-shot (`vlm/src/fall_confirmation.py`,
reusando el mismo encoder ONNX que la clasificación de actividad, sin cargar
un segundo modelo) si el recorte de la persona en ese frame se parece más a
una caída accidental o a una postura voluntaria — si no se confirma, el
evento se descarta y solo queda logueado, no se guarda en `events/`.
Calibrado sobre 445 eventos geométricos de 183 clips
(`vision/calibrate_fall_confirmation.py`): retiene 82.0% del recall
geométrico y elimina 67% de los falsos positivos totales (70% en el setup de
cámara más afectado). El margen de confirmación queda en el `metadata` del
evento (`clip_confirmation_margin`) cuando se confirma.

## CLIP es opcional

Si `vlm/` no está disponible (no se pudo importar) o no se generaron los
modelos ONNX (`vlm/export_onnx.py`), el runner sigue funcionando solo con
pose estimation — loggea una advertencia y no clasifica actividad ni
confirma caídas (quedan solo con la heurística geométrica). La confirmación
de caídas además necesita `fall_confirmation_features.npy` específicamente
(mismo `export_onnx.py`); si CLIP está disponible pero ese archivo no, el
runner clasifica actividad normalmente y cae al comportamiento puramente
geométrico solo para caídas. Ver `../vlm/README.md` para generar los
modelos.

## Frame preview

Cada worker guarda 1 de cada 5 frames como `../frames/<cam_id>.jpg`
(escritura atómica vía archivo temporal + `replace`), que expone
`dashboard/` en `GET /api/frame/{cam_id}` para mostrar la cámara en vivo en
el panel. Si YOLO detectó a alguien en ese frame, el preview incluye el
overlay del esqueleto (bounding box + keypoints + huesos, vía
`vision/src/pose_pipeline.draw_pose()`) — así se puede ver en el dashboard
qué está detectando el pipeline, no solo la imagen cruda de la cámara.

Dibujar ese overlay es opcional: cada worker consulta
`overlay_state.read_overlay_enabled()` antes de generar el preview, que lee
`../runner_state/overlay.json`. El dashboard escribe ese mismo archivo desde
`POST /api/overlay` (ver `dashboard/README.md`) — como `dashboard/` y
`runner/` corren en procesos separados, el archivo es la forma más simple de
que el toggle de la UI afecte a todas las cámaras ya en ejecución sin que un
proceso importe código del otro.

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
