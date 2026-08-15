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
├── camera_worker.py    # thread por cámara: gate de movimiento + pose pipeline + CLIP periódico + frame preview
├── motion_gate.py       # detector de movimiento barato (MOG2) que pausa YOLO/CLIP sin nadie en cuadro
├── motion_state.py      # publica awake/dormida por cámara en runner_state/motion/, para que dashboard/ lo muestre
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
  source: videos/ToyotaSmarthome/Uselaptop_p12_r00_v14_c05.mp4   # archivo, URL RTSP/HTTP, o índice de webcam ("0")
  realtime: true    # ver "Tiempo real y frame dropping" abajo — recomendado en true
  loop: true         # si true, reinicia el archivo al terminar (ignorado en streams en vivo)
pose_model: yolo11n-pose.pt
log_level: INFO
clip_interval_s: 10.0      # segundos entre ventanas de clasificación CLIP por cámara
clip_min_confidence: 0.4   # confianza mínima para emitir un evento de actividad
clip_heartbeat_s: 60.0     # cada cuánto re-emitir una actividad que sigue igual
motion_gate_enabled: true          # ver "Gate de movimiento" abajo
motion_min_area_ratio: 0.005       # sensibilidad del detector de movimiento
motion_absence_timeout_s: 45.0     # segundos sin persona detectada antes de volver a modo bajo consumo
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
- **Categorías gruesas**: se clasifica con las 31 clases finas del catálogo
  de Toyota Smarthome (mejor cobertura de prompts, ver `vlm/README.md`) pero
  el evento se emite a nivel de categoría (`COARSE_CATEGORIES` en
  `vlm/src/clip_classifier.py`: `Cooking`, `EatingDrinkingMedication`,
  `Resting`, `ReadingScreen`, `Moving`, `Phone`), sumando las probabilidades
  de las subclases (zero-shot) o vía el linear probe entrenado. Hubo una
  séptima categoría, `Standing` ("parado sin hacer nada", vía OmniFall), que
  se agregó y se terminó sacando — no generalizaba a una cámara distinta a
  la de entrenamiento (0% recall medido contra GMDCSA-24) y su valor de
  producto era bajo frente a lo que ya cubre la inmovilidad geométrica de
  `vision/`; ver "`Standing`: se probó y se descartó" en `vlm/README.md`.
  Distinguir
  `Drink.Fromcan` de `Drink.Fromcup` en un frame estático es imposible y al
  familiar tampoco le aporta; agregarlas elimina ese error. `activity_label`
  en el metadata es la categoría; la clase fina ganadora queda en
  `fine_label`.
- **Umbral de confianza** (`clip_min_confidence`): se aplica sobre la
  confianza de la *categoría* (suma de sus subclases); las ventanas
  ambiguas se descartan (quedan en el log) en vez de emitir una etiqueta
  dudosa.
- **Umbral extra para `EatingDrinkingMedication`**: es la categoría más
  floja del probe (56.1%, ver "Precisión y evaluación" en `vlm/README.md`) y
  la más sensible de reportar mal (medicación) — `_EATING_CONFIRM_THRESHOLD`
  en `camera_worker.py` le exige más confianza que `clip_min_confidence`
  antes de afirmarla. Si no llega: se reporta la segunda categoría más
  probable si esa sí supera `clip_min_confidence` (aprovecha la señal ya
  calculada en vez de descartarla), o se emite igual pero con
  `confirmed: false` en el metadata — el dashboard lo usa para mostrar un
  texto genérico ("posiblemente comiendo, bebiendo o con medicación, sin
  confirmar") en vez de afirmar la categoría con seguridad.
- **Doble check probe/zero-shot**: `ClipActivityClassifier.classify()`
  calcula dos señales independientes del mismo embedding — la categoría vía
  el linear probe (la que se reporta) y, sin costo extra, la agregación
  zero-shot de prompts de texto (la que ya usaba `evaluate_activity.py` antes
  de entrenar el probe, ver `vlm/README.md`). Medido offline sobre el dataset
  de validación: cuando ambas coinciden el probe acierta 74.1%, cuando no
  coinciden cae a 57.8% — sigue siendo mejor que el azar, pero es una señal
  de confianza gratis. `camera_worker.py` promedia la categoría zero-shot por
  ventana igual que la del probe; si no coincide con la reportada, el evento
  se emite con `confirmed: false` (mismo mecanismo de degradación que
  `EatingDrinkingMedication` abajo, generalizado a las 7 categorías) y queda
  logueado con cuál categoría decía cada señal. El `metadata` del evento
  siempre incluye `zeroshot_category` (coincida o no con `activity_label`)
  para poder auditar desacuerdos analizando eventos ya guardados, sin
  reprocesar el video.
- **Umbral extra para `EatingDrinkingMedication`**: es la categoría más
  floja del probe (56.1%, ver "Precisión y evaluación" en `vlm/README.md`) y
  la más sensible de reportar mal (medicación) — `_EATING_CONFIRM_THRESHOLD`
  en `camera_worker.py` le exige más confianza que `clip_min_confidence`
  antes de afirmarla. Si no llega: se reporta la segunda categoría más
  probable si esa sí supera `clip_min_confidence` (aprovecha la señal ya
  calculada en vez de descartarla), o se emite igual pero con
  `confirmed: false` en el metadata — el dashboard lo usa para mostrar un
  texto genérico ("posiblemente comiendo, bebiendo o con medicación, sin
  confirmar") en vez de afirmar la categoría con seguridad. Corre antes del
  doble check general de arriba y no se pisan: si esta rama ya bajó
  `confirmed` a `false`, el doble check no tiene nada que revertir.
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

## Gate de movimiento

YOLO11n-pose tarda ~150-300ms/frame en CPU y CLIP agrega más costo encima
(ver "Tiempo real y frame dropping" arriba) — correr ambos sin parar sobre
una cámara que apunta a una habitación vacía la mayor parte del día es
desperdicio puro, más relevante todavía en el hardware objetivo (Pi 5, ver
`../README.md`). `motion_gate.py` agrega un filtro previo mucho más barato
(sustracción de fondo con `cv2.createBackgroundSubtractorMOG2` sobre una
copia del frame reducida a 160px de ancho, ~1-2ms/frame) que decide si vale
la pena correr el pipeline pesado en absoluto.

**Diseño: el movimiento solo despierta, YOLO decide cuándo dormir.** Cada
`CameraWorker` arranca en modo dormido (si `motion_gate_enabled: true`) y
solo corre `MotionDetector.detect()` sobre cada frame — nada de YOLO ni
CLIP. Apenas detecta movimiento, pasa a modo despierto y el pipeline corre
normal (igual que con el gate desactivado). Volver a dormir **no** lo decide
el detector de movimiento: lo decide `FallImmobilityDetector` al confirmar
que no ve a nadie (`last_detection is None`) durante
`motion_absence_timeout_s` segundos seguidos.

Esta asimetría es intencional y no es un detalle menor: si el gate durmiera
el pipeline apenas cesa el movimiento, una persona inmóvil en el piso
(posible caída) o sentada sin moverse (sedentarismo, ver "Inmovilidad
prolongada" en `vision/README.md`) dejaría de generar movimiento como
cualquier habitación vacía — el sistema apagaría YOLO justo en el momento en
que más importa que siga viendo a esa persona. Con este diseño, mientras
YOLO siga detectando a alguien en cuadro (se mueva o no) el pipeline sigue
despierto; solo se duerme cuando confirma ausencia real.

Costo de este diseño: no ahorra CPU mientras hay alguien presente e
inmóvil, solo cuando la habitación está genuinamente vacía. Es la
contrapartida aceptada frente a un gate puramente por movimiento, que
ahorraría más pero a costa de la detección de inmovilidad/caída.

**Calibración de `motion_min_area_ratio`**: el umbral por defecto (0.005) se
ajustó a mano contra clips de `videos/` (Toyota Smarthome, cámara fija) —
ver el ratio de píxeles en movimiento típico de un video con actividad real
contra el de una escena estática con solo ruido de sensor (~0, con
`varThreshold=16` de MOG2 filtrando ruido normal). Primer valor probado,
0.008: nunca cruzaba el umbral en `Drink.Fromcan_p25_r06_v15_c03.mp4` (3.5s,
persona ya de pie, solo levanta una lata — el clip completo no supera un
ratio de 0.0069), así que esa cámara quedaba dormida todo el tiempo pese a
tener a alguien en cuadro. Bajado a 0.005, con margen de sobra contra ruido
sintético de sensor (probado sin falsos despertares hasta con σ=5, más
agresivo que el ruido típico de una webcam). No está calibrado contra una
cámara real de la Pi ni contra una habitación vacía de verdad (el dataset
son todo clips recortados a actividad, casi siempre cortos) — igual que el
resto de los umbrales del proyecto, conviene reajustarlo empíricamente por
cámara si en la práctica se ven demasiados falsos despertares (iluminación
variable, sombras, cambios de exposición automática) o si tarda en
despertar con movimiento real sutil.

**Estado visible en el dashboard**: cada `CameraWorker` publica si está
despierta o dormida en `runner_state/motion/{id}.json` (`motion_state.py`,
mismo patrón que `overlay_state.py` — un archivo porque `dashboard/` corre
en un proceso separado del runner). Un archivo por cámara, no uno
compartido: varios threads (uno por cámara) escribirían al mismo archivo si
fuera uno solo, y cada worker sanitiza su propio id para el nombre de
archivo igual que ya hace para el frame preview (`frames/{id}.jpg`). Se
escribe en las transiciones de despertar/dormir (para que el dashboard
reaccione al toque) y además cada `_FRAME_PREVIEW_EVERY_N` frames como
heartbeat mientras está despierta. `dashboard/` expone ese estado en
`GET /api/motion` (ver `dashboard/README.md`) y lo pinta como un punto verde
(despierta) o el texto "SIN MOVIMIENTO" (dormida) sobre cada cámara — ver
"Indicador de movimiento" en `dashboard/README.md`. Una cámara con
`motion_gate_enabled: false` no escribe archivo, así que no aparece en
`/api/motion` y el dashboard no muestra nada para ella, en vez de asumir un
estado que no corresponde.

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
