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
instancia de `ClipActivityClassifier` (de `vlm/`) y, si está disponible, una
única instancia de `SkeletonActivityClassifier` (de `fusion/`) cargadas una
sola vez en `main.py` — ONNX Runtime es thread-safe para inferencia
concurrente, y cargar cualquiera de los dos por cámara sería redundante. Ver
"Clasificación de actividades (CLIP)" abajo para cómo se combinan.

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
skeleton_weight: 0.5       # peso del esqueleto vs. CLIP al fusionar categoría (ver fusion/README.md); sin efecto si fusion/ no está entrenado/exportado
motion_gate_enabled: true          # ver "Gate de movimiento" abajo
motion_min_area_ratio: 0.005       # sensibilidad del detector de movimiento
motion_absence_timeout_s: 45.0     # segundos sin persona detectada antes de volver a modo bajo consumo
```

Si `source` empieza con `rtsp://`, `http(s)://`, o es un entero (índice de
webcam), `camera_source.make_source()` lo trata como stream en vivo
(`LiveStream`, con reconexión automática ante cortes). Cualquier otra cosa
se resuelve como ruta a archivo, relativa a la raíz del proyecto si no es
absoluta (`VideoFileLoop`, con loop opcional).

Los timestamps de los eventos son relativos a cuando arrancó la fuente (no
el tiempo del archivo), para que el detector de inmovilidad de `vision/`
tenga sentido cuando un archivo hace loop. En streams en vivo (`LiveStream`)
y en `VideoFileLoop` con `realtime: true` son wall-clock (no hay otra
referencia posible, o el frame ya está acompasado a tiempo real). En
`VideoFileLoop` con `realtime: false` son `frame_idx / fps` acumulado
across loops, no wall-clock — ver "Bug real encontrado y corregido" más
abajo (2026-10-01).

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

**Bug real encontrado y corregido (2026-10-01), validando con
`evaluation/annotate_video.py` sobre `videos/ToyotaSmarthome_FP/`:** con
`realtime: false`, antes del fix, el timestamp de evento era reloj real
("wall-clock"), que corre a la velocidad que el hardware logre procesar, no
a la velocidad del video. Con GPU, el pipeline completo (pose + CLIP +
clasificador temporal) resultó ~2.6x más rápido que la reproducción real
(un clip de 1168.5s se procesó en 449s de reloj real, corrida aislada, sin
otras cámaras compitiendo por la GPU). Como los timestamps de evento eran
ese mismo reloj real, todos los umbrales temporales de `vision/` (ventana
de inmovilidad de 180s, ventana de velocidad de caída de 0.4s, cooldowns)
terminaban midiéndose contra una cantidad de contenido de video distinta a
la calibrada — en este caso, 180s de reloj real equivalían a ~470s reales
de video. Si además corrían varias cámaras en paralelo compitiendo por la
misma GPU, el factor de distorsión dejaba de ser constante (variaba con la
contención instantánea) y podía ir en cualquier dirección — en una corrida
de prueba con 4 clips en paralelo esto generó un evento de inmovilidad
espurio que no aparecía corriendo el mismo clip solo.

**Fix:** `VideoFileLoop` (`camera_source.py`) ahora usa `frame_idx / fps`
acumulado across loops (nunca wall-clock) cuando `realtime: false` — el
mismo criterio que ya usan `vision/main.py` y `evaluation/annotate_video.py`
con `--rerun`. Verificado: timestamps exactos frame a frame (delta
constante de `1/fps`) y monótonos crecientes a través de loops (no
resetean a 0 en cada vuelta). `realtime: true` no cambió — ahí wall-clock
ya era equivalente al tiempo de video por construcción (el frame se
acompasa a tiempo real antes de entregarse), así que nunca tuvo este
problema. `LiveStream` tampoco cambió: para un stream en vivo wall-clock es
la única referencia de tiempo posible.

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
- **Contexto de zona del hogar** (2026-09-21): si la cámara tiene `zones:`
  configurado en `cameras.yaml` (polígonos normalizados por nombre de zona —
  ver ejemplo comentado ahí y `fusion/src/zone_context.py`), se calcula en
  qué zona cae la posición de cadera de la persona (mismos keypoints que ya
  calcula el detector de pose, sin modelo adicional) y se suma como feature
  del linear probe. Es la mejora más fuerte medida hasta ahora en
  `vlm/README.md` ("Fase 1 de PLAN_sustitucion_clip.md...", 2026-09-21):
  `Cooking` 80.2%→86.0%. **Configuración manual, una sola vez por cámara
  real**, en el setup de instalación (como las zonas de interés de una
  cámara de seguridad) — no es algo que el sistema infiera solo. Sin
  `zones:` configurado, el vector de zona es todo cero y el sistema se
  comporta igual que sin esta feature (no rompe nada, solo no se beneficia
  todavía).
- **Umbral de confianza** (`clip_min_confidence`): se aplica sobre la
  confianza de la *categoría* (suma de sus subclases); las ventanas
  ambiguas se descartan (quedan en el log) en vez de emitir una etiqueta
  dudosa.
- **Umbral extra para `EatingDrinkingMedication`**: es la categoría más
  floja del probe (56.1%, ver "Precisión y evaluación" en `vlm/README.md`) y
  la más sensible de reportar mal (medicación) — `_EATING_CONFIRM_THRESHOLD`
  en `camera_worker.py` le exige más confianza que `clip_min_confidence`
  antes de afirmarla. Si no llega y la segunda categoría más probable supera
  `clip_min_confidence`, se reporta esa en su lugar (aprovecha la señal ya
  calculada en vez de descartarla).
- **Zero-shot como señal auxiliar, no como gate**: `ClipActivityClassifier.classify()`
  calcula dos señales independientes del mismo embedding — la categoría vía
  el linear probe (la que se reporta) y, sin costo extra, la agregación
  zero-shot de prompts de texto (la que ya usaba `evaluate_activity.py` antes
  de entrenar el probe, ver `vlm/README.md`). Medido offline sobre el dataset
  de validación: cuando ambas coinciden el probe acierta 74.1%, cuando no
  coinciden cae a 57.8% — sigue siendo mejor que el azar, así que ya no se usa
  para degradar el evento (antes bajaba `confirmed` a `false` cuando no
  coincidían; se sacó porque generaba demasiados eventos marcados "sin
  confirmar" en producción). El `metadata` del evento siempre incluye
  `zeroshot_category` (coincida o no con `activity_label`) para poder auditar
  desacuerdos analizando eventos ya guardados, sin reprocesar el video.
- **Deduplicación**: se emite evento cuando la categoría *cambia*; si sigue
  igual se re-emite cada `clip_heartbeat_s` con `ongoing: true` en el
  metadata, para que el análisis posterior pueda reconstruir duraciones sin
  ahogarse en eventos repetidos.

## Fusión con esqueleto (`fusion/`) (2026-08-18)

Además de CLIP, cada `CameraWorker` puede sumar la señal de movimiento de
`fusion/` (clasificador temporal de esqueleto sobre la secuencia de
keypoints que `vision/` ya calcula, ver `fusion/README.md`) a la
clasificación de actividad, en vez de depender solo de CLIP.

- **Buffer de esqueleto por frame**: en cada frame procesado (mientras la
  cámara está despierta), si YOLO detectó a la persona, se arma un vector de
  features (keypoints normalizados + confianzas + descriptores geométricos
  que `vision/` ya calcula gratis) y se empuja a un `SkeletonSequenceBuffer`
  propio de esa cámara. Sin detección, el buffer se resetea (no interpolar
  sobre un hueco) — mismo criterio que el historial de inmovilidad de
  `vision/`. También se resetea al reiniciar un video en loop y al
  despertar del gate de movimiento (ventana nueva, no una a medio llenar con
  contexto de antes de dormir).
- **Fusión en cada tick de CLIP**: en el mismo muestreo de `clip_interval_s`
  que ya usa CLIP, se arma la ventana de esqueleto (últimos ~2s
  resampleados) y se clasifica junto con CLIP vía `HybridActivityClassifier`
  (`fusion/src/fusion.py`), que combina las dos distribuciones de categoría
  con un promedio pesado (`skeleton_weight`, configurable arriba). El
  resultado fusionado es el que se promedia por ventana y decide
  `activity_label` — todo lo demás (umbral de confianza, umbral extra para
  `EatingDrinkingMedication`, doble check, deduplicación) sigue operando
  igual, ahora sobre la categoría ya fusionada.
- **El doble check probe/zero-shot sigue siendo solo de CLIP**: compara la
  categoría del probe contra la agregación zero-shot, ambas señales de CLIP
  — no compara esqueleto contra CLIP. Es una verificación de consistencia
  interna de CLIP consigo mismo, no del acuerdo entre las dos modalidades.
- **Degradación elegante**: si `fusion/` no está entrenado/exportado
  (`fusion/models/temporal_classifier_int8.onnx` no existe —
  `fusion/train_temporal_classifier.py` + `fusion/export_temporal_onnx.py`),
  el runner sigue funcionando solo con CLIP, exactamente igual que antes de
  este enganche. Se loguea al arrancar cada cámara (`fusión de esqueleto
  (fusion/): activa (modo hybrid)` o `inactiva (solo CLIP)`).
- **Auditoría**: cada evento de actividad guarda `fusion_mode` (`"hybrid"` o
  `"clip_only"`) en su metadata.
- **`skeleton_weight` validado (2026-08-18)**: sobre 273 videos genuinamente
  no vistos en entrenamiento (ver "Estado" en `fusion/README.md`), el
  barrido de `evaluate_hybrid.py` da el mejor accuracy (82.8%, +7pts sobre
  CLIP solo) en la zona `skeleton_weight=0.2-0.5` — el default actual (0.5)
  cae justo ahí, así que no hace falta cambiarlo. Cobertura parcial: 11 de
  las 18 clases del dataset local (las otras 7 no tienen video sobrante para
  medirse sin leakage), desbalanceadas entre sí.
- **Cuidado al probar la mejora con un `cameras.yaml` de clips cortos en
  loop**: con un clip de pocos segundos repitiéndose y `clip_interval_s=10s`
  (default), cada ventana de clasificación promedia varias vueltas del mismo
  clip cortadas de forma abrupta — un patrón de input que no se da en el uso
  real (cámara continua). Esto puede hacer que el resultado en vivo se vea
  peor de lo que realmente mide el modelo offline (ver el hallazgo
  documentado en "Reentrenamiento con dataset local categorizado por
  carpetas" en `vlm/README.md`). Para clips cortos en loop, considerar bajar
  `clip_interval_s` a algo cercano a la duración real del clip.

## Confirmación de caídas (`_confirm_fall_event`)

El detector geométrico de `vision/` (velocidad vertical + aspect ratio) no
distingue una caída real de sentarse/acostarse a propósito en algunos setups
de cámara (bbox ancho y bajo genera la misma firma geométrica que una caída
— ver "Detección de caída como evento" en `vision/README.md` para el
detalle y por qué recalibrar los umbrales geométricos no alcanza). Cuando
`FallImmobilityDetector` dispara un evento `fall`, el worker decide en 2ª
etapa según la **evidencia de encuadre** que trae el evento (Fase 0, ver
`vision/README.md`):

- **`evidence == "partial"`** (persona parcialmente fuera de cuadro): la firma
  geométrica y un recorte truncado para CLIP son poco confiables. El evento
  **no se descarta ni se confirma** — se emite con `severity: "sospecha"` /
  `confirmed: False`, y `vision/` lo escala por inmovilidad si la persona
  queda quieta (evento `immobility` con `trigger: "post_fall_sospecha"`).

- **`evidence == "full"`** → `_confirm_fall_event`, en este orden:
  1. **Clasificador temporal** (`fusion/src/fall_temporal.py`, GRU entrenado
     sobre la ventana de keypoints — ver `fusion/README.md`): mira la *forma
     del movimiento* de los últimos ~2 s. `fall_score ≥ 0.45` confirma,
     `≤ 0.15` descarta, en el medio → paso 2.
  2. **CLIP zero-shot/probe** (`vlm/src/fall_confirmation.py`, reusa el mismo
     encoder ONNX que la clasificación de actividad): compara el recorte de
     la persona contra "caída accidental" vs. "postura voluntaria". Se usa de
     desempate del temporal, o como filtro principal si el temporal no está
     disponible **o no pudo armar ventana de esqueleto** (típico en la escena
     de sillón/cámara ancha donde `yolo11n-pose` no detecta a la persona).
  3. Sin ninguna de las dos → se confirma por defecto.

  Un evento no confirmado se descarta (solo queda logueado). El `metadata`
  guarda `fall_temporal_score` y/o `clip_confirmation_margin` y `confirmed`.

Números: el filtro CLIP solo, calibrado sobre 445 eventos de 183 clips
(`vision/calibrate_fall_confirmation.py`), retiene 82.0% del recall y elimina
67% de los FP. El pipeline con el temporal como 2ª etapa
(`vision/evaluate_falls.py --temporal`, 5 datasets): recall end-to-end igual
que geométrico solo (133/158), FP de alerta dura 185 → 91 — ver
`fusion/README.md` para el desglose.

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

## CLIP y clasificador temporal de caída son opcionales

Si `vlm/` no está disponible (no se pudo importar) o no se generaron los
modelos ONNX (`vlm/export_onnx.py`), el runner sigue funcionando solo con
pose estimation — loggea una advertencia y no clasifica actividad ni
confirma caídas con CLIP. La confirmación CLIP de caídas además necesita
`fall_confirmation_features.npy` / `fall_confirmation_probe.npz` (mismo
`export_onnx.py`).

El **clasificador temporal de caída** (`fusion/models/fall_temporal_int8.onnx`,
generado por `fusion/train_fall_temporal.py` + `export_fall_temporal_onnx.py`)
es la 2ª etapa preferida cuando está presente; si no, `_confirm_fall_event`
cae directamente a CLIP. Si tampoco hay CLIP, cada evento `fall` de evidencia
completa se emite tal cual sale de la heurística geométrica (los de evidencia
parcial siguen yendo a sospecha + escalada por inmovilidad, eso no depende de
ningún modelo). Ver `../fusion/README.md` y `../vlm/README.md` para generar
los modelos.

El esqueleto de `fusion/` (ver "Fusión con esqueleto" arriba) es opcional de
la misma manera: sin él, la clasificación de actividad sigue siendo CLIP
solo (`fusion_mode: "clip_only"`), no un modo degradado distinto — es
literalmente el comportamiento de antes de que existiera `fusion/`.

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
pip install -r ../fusion/requirements.txt   # opcional — solo si se quiere fusión con esqueleto, ver "Fusión con esqueleto" abajo
```

## Cómo probarlo

```
python -m runner                          # usa cameras.yaml en la raíz
python -m runner --config otra_config.yaml
```

Ctrl+C (o `SIGTERM`) detiene todas las cámaras de forma prolija.

## Sincronización a la nube (`cloud_sync.py`, 2026-10-01)

Thread de fondo opcional (`CloudSyncThread`, arrancado desde `main.py` solo si `cameras.yaml`
tiene una sección `cloud:`) que lee eventos locales no sincronizados
(`events/store.py::EventStore.unsynced`) y los manda en batch al backend `cloud/` — ver
`cloud/README.md`. Solo hace conexiones salientes (nunca expone un puerto), y ante un corte de
conexión no pierde nada: los eventos siguen en `events/events.db` hasta que se confirma el envío
(RNF6/RNF7 del informe). Sin esa sección en `cameras.yaml`, el runner funciona exactamente igual
que antes — 100% local.

## Fuera de alcance por ahora

- Multi-persona por cámara (mismo límite que `vision/`).
- Balanceo de carga entre cámaras si el CPU no alcanza para todas en
  paralelo — asume que el hardware objetivo (Pi 5) soporta la cantidad de
  cámaras configuradas.
