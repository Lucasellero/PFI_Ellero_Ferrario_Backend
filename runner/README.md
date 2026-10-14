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
├── overlay_state.py     # lee/escribe runner_state/overlay.json (toggle de overlay compartido con dashboard/)
├── preview_file.py      # ruta y borrado de la vista previa runner_state/frames/{id}.jpg (ver "Frame preview")
├── door_events.py       # entradas y salidas de casa por la zona `puerta` (ver "Entradas y salidas de casa")
├── cloud_sync.py        # thread opcional: envía eventos a la nube y purga los ya enviados
└── tests/               # pruebas con pytest (ver "Pruebas")
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

### Una posible caída por minuto (`fall_grouping.py`, 2026-10-14)

Después de la segunda etapa, cada cámara publica **como mucho una posible caída por minuto**.
Dentro del minuto siguiente a la última caída publicada, otra del mismo nivel o menor no se
guarda ni viaja a la nube (queda solo en el log: "no publicada: ya se publicó una hace menos de
un minuto"). Una caída **confirmada** después de una **sospecha** se publica siempre: lo más
grave nunca se tapa. La vigilancia de `vision/` después de una sospecha no cambia: si la persona
queda inmóvil, la alerta `immobility` con `trigger: post_fall_sospecha` sale igual.

Motivo: en una prueba con el video `P09T05C05` de Toyota Smarthome corriendo como cámara de un
hogar, la app recibió 16 sospechas en unos 12 minutos, 7 de ellas en 26 segundos (la misma
persona moviéndose en el sillón). Con el agrupado habrían sido 6. Eran sospechas, no alertas:
no mandaron notificación, pero llenaban la lista de eventos.

Las mediciones de `vision/README.md` ("Actualización 2026-10-13") son **sin** este agrupado: el
recall por clip no cambia (una caída real sigue teniendo su primer evento), pero la cantidad de
sospechas y de falsas alertas publicadas sería menor. No se volvió a medir.

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
archivo igual que ya hace para el frame preview (`runner_state/frames/{id}.jpg`). Se
escribe en las transiciones de despertar/dormir (para que el dashboard
reaccione al toque) y además cada `_FRAME_PREVIEW_EVERY_N` frames como
heartbeat mientras está despierta. `dashboard/` expone ese estado en
`GET /api/motion` (ver `dashboard/README.md`) y lo pinta como un punto verde
(despierta) o el texto "SIN MOVIMIENTO" (dormida) sobre cada cámara — ver
"Indicador de movimiento" en `dashboard/README.md`. Una cámara con
`motion_gate_enabled: false` no escribe archivo, así que no aparece en
`/api/motion` y el dashboard no muestra nada para ella, en vez de asumir un
estado que no corresponde.

## Entradas y salidas de casa (`door_events.py`, 2026-10-13)

**Estado: encendidas por defecto.** Una cámara que tiene dibujada la zona `puerta` en
`cameras.yaml` (la puerta de calle, no una interior) emite dos eventos de actividad:
`LeaveHome` y `EnterHome`. No usa ningún modelo: es geometría sobre la posición de la persona
que YOLO ya calcula en cada frame (el punto medio de las caderas, o el centro del recuadro si
las caderas no se ven bien). Se apagan en todas las cámaras con `puerta_activada: false`; del
lado de la línea base la opción equivalente es `ENTRADAS_Y_SALIDAS_ACTIVADAS=0` (ver
`cloud/README.md`). La línea base solo usa estos eventos para un desvío: que falten salidas o
entradas (desde el 2026-10-13, ya el primer día; ver `baseline/README.md`, "Actualización
2026-10-13 (8)"). Nunca genera una alerta por el horario de una salida.

- **Salida**: la persona se deja de ver, lo último que se supo de ella (con una tolerancia de
  `puerta_memoria_s`, 2 s) es que estaba en la zona `puerta`, y pasan `puerta_ausencia_min_s`
  (60 s) sin que vuelva a aparecer. El evento lleva la hora en que se la vio por última vez.
- **Entrada**: después de 60 s o más sin ver a nadie, la primera detección cae en la zona
  `puerta`.

No son salida ni entrada: desaparecer lejos de la puerta (se fue a otro ambiente), perderse unos
segundos en la puerta y reaparecer, o reaparecer lejos de la puerta. Con la cámara dormida por
el gate de movimiento no hay nadie en cuadro, así que la ausencia sigue corriendo. Al arrancar
no se inventa una entrada: la ausencia se cuenta desde el primer frame.

```yaml
cameras:
- id: living
  source: rtsp://...
  zones:
    puerta: [[0.85, 0.2], [1.0, 0.2], [1.0, 0.95], [0.85, 0.95]]
puerta_activada: true       # opcional, es el default
puerta_ausencia_min_s: 60   # opcional
puerta_memoria_s: 2         # opcional
```

Sin zona `puerta` (o con `puerta_activada: false`) no se emite nada. Los eventos son `activity`
con `activity_label` `LeaveHome` / `EnterHome`, `confirmed: true` y `source: "door_zone"`; no
pasan por el clasificador ni por su deduplicación, y viajan a la nube como cualquier otro. La
línea base los sigue como `cam_salir_casa` / `cam_entrar_casa` (ver `baseline/README.md`,
"Actualización 2026-10-13 (6)").

**Validado con video solo en parte** (ver "Validación con video" abajo): con Toyota Smarthome,
donde se entra y se sale de cuadro, no por una puerta de calle. La lógica está probada con
trayectorias sintéticas (`tests/test_door_events.py`). El sistema es para un solo residente:
no distingue quién cruza. La sensibilidad de la línea base a perder entre 10 % y 30 % de estos
eventos está medida en `baseline/README.md`.

### Validación con video (2026-10-13)

`evaluate_door_events.py`, sobre Toyota Smarthome completo (16.115 clips recortados, fuera del
repo) y los 4 videos continuos de `videos/ToyotaSmarthome_FP`. Valores por defecto del
detector: 60 s de ausencia, 2 s de memoria. No se cambió ninguno.

**Qué se puede validar y qué no.** En ese dataset `Enter` y `Leave` son entrar y salir **del
ambiente que ve la cámara**, no de la casa por la puerta de calle. Para el detector es el mismo
problema (alguien aparece o desaparece en una zona), pero la zona validada es "por dónde se
entra y se sale de cuadro". No hay ningún video de una puerta de calle.

**Zona `puerta` por cámara, fijada antes de medir.** Con los participantes de ajuste (los de
entrenamiento del protocolo cross-subject del dataset, menos el 7 y el 9: 3, 4, 6, 12, 13, 15,
17, 19, 25) se toma dónde estaba la persona en la primera detección de cada clip `Enter` y en
la última de cada clip `Leave`. La zona es el rectángulo del percentil 10 al 90 de esos puntos
en cada eje, más 0,05 de margen. Queda en [`toyota_smarthome_puerta.yaml`](toyota_smarthome_puerta.yaml).
Se mide con los participantes de prueba del mismo protocolo (2, 10, 11, 14, 16, 18, 20); el 7 y
el 9 se reservan para los videos continuos.

| Cámara | Zona (x, y; 0–1) | Parte del cuadro | Clips de ajuste |
|---|---|---|---|
| c01 | x 0,87–1,00 · y 0,62–0,85 | 3 % | 44 |
| c02 | x 0,24–0,41 · y 0,14–0,27 | 2 % | 22 |
| c03 | x 0,46–0,85 · y 0,17–0,53 | 14 % | 53 |
| c04 | x 0,16–0,77 · y 0,13–0,71 | **36 %** | 141 |
| c05 | x 0,15–0,69 · y 0,23–0,45 | 12 % | 76 |
| c06 | x 0,00–0,81 · y 0,40–0,68 | 23 % | 26 |
| c07 | x 0,23–0,99 · y 0,15–0,46 | 23 % | 49 |

En varias cámaras la gente entra y sale de cuadro por lugares muy distintos y el rectángulo
sale grande (c04, c06, c07). Una puerta de calle dibujada por un instalador sería más chica.

**1. Entradas y salidas reales** (clips `Enter` / `Leave` de los participantes de prueba). Los
clips duran pocos segundos y el detector necesita 60 s sin nadie, así que esa ausencia se
agrega: 61 s sin detección después de cada `Leave` y antes de cada `Enter`.

| | Detectadas | c01 | c03 | c04 | c05 | c06 | c07 |
|---|---|---|---|---|---|---|---|
| Entradas | **84,2 %** (112/133) | 2/2 | 18/25 | 44/44 | 18/20 | 7/13 | 23/29 |
| Salidas | **86,0 %** (92/107) | 0/2 | 17/18 | 38/40 | 14/16 | 10/12 | 13/19 |

No hay clips de prueba de la c02. En 2 clips `Leave` YOLO no detectó a nadie.

**2. Entradas y salidas inventadas** (764 clips de las otras 29 actividades, participantes de
prueba, hasta 30 por actividad; 3,20 h de video).
- **Tal cual, 0 eventos.** Un clip recortado no puede dar ninguno: nunca hay 60 s de
  ausencia. Tampoco apareció ninguno dentro de los clips largos.
- **Peor caso** (61 s sin detección antes y después de cada clip, como si la persona hubiera
  aparecido y desaparecido justo ahí): entrada inventada en 44,2 % (338/764), salida en 48,6 %
  (371/764), alguna de las dos en 50,0 % (382/764). **Es una cota superior, no una tasa**:
  mide cuántas veces el clip empieza o termina con la persona dentro de la zona.

| Peor caso | c01 | c02 | c03 | c04 | c05 | c06 | c07 |
|---|---|---|---|---|---|---|---|
| Entrada inventada | 5/95 | 1/60 | 82/126 | 119/136 | 27/167 | 104/118 | 0/62 |
| Salida inventada | 8/95 | 1/60 | 91/126 | 126/136 | 37/167 | 106/118 | 2/62 |

Depende casi por completo de la zona: en c03, c04 y c06 la zona se superpone con el lugar
donde se hace la actividad (cocina, sillón) y entre 65 % y 93 % de los clips empiezan o
terminan adentro; en c01, c02 y c07, casi ninguno. O sea: **si la persona se deja de ver 60 s
mientras está parada en la zona, el detector lo cuenta como una salida**. Con una zona chica y
apartada eso pasa poco; con una zona que pisa el lugar de estar, pasa seguido.

**3. Video continuo** (4 videos sin recortar de los participantes 7 y 9, 84 minutos, con las
anotaciones originales del dataset; el detector corre como en producción, sin agregar nada).

| Video | Minutos | Frames sin detección | Ausencias anotadas (`Leave` → `Enter`) | De 60 s o más | Eventos emitidos |
|---|---|---|---|---|---|
| P07T05C05 | 19,7 | 50 % | 5 (10,8 a 65,8 s) | 1 | 0 |
| P07T10C02 | 26,5 | 1 % | 0 | 0 | 0 |
| P09T05C05 | 18,3 | 22 % | 4 (1,0 a 18,0 s) | 0 | 0 |
| P09T06C05 | 19,5 | 22 % | 3 (19,4 a 29,0 s) | 0 | 0 |

- **Inventados: 0 en 84 minutos**, con hasta la mitad de los frames sin detección (la escena
  de sillón donde `yolo11n-pose` pierde a la persona).
- **Detectados: 0 de 1.** La única ausencia de 60 s o más (65,8 s) no dio ni salida ni
  entrada. No se miró por qué (si la última detección cayó fuera de la zona o si la ausencia
  que vio el detector fue más corta que la anotada). Un caso no alcanza para una tasa.
- Las otras 11 ausencias duran menos de 60 s y, como corresponde, no dieron eventos. En este
  departamento casi todas las salidas de cuadro son a otro ambiente y duran menos de un minuto.

**Qué queda sin validar:** una puerta de calle real; si 60 s y 2 s son los valores correctos
(no se barrieron); salidas y entradas reales en video continuo (hay un solo caso); visitas o
más de una persona. La tasa de inventados en un hogar depende de dónde se dibuje la zona y de
cuánto pierda YOLO a la persona ahí: la medición 2 da la cota y la 3, un caso favorable.

```
python -m runner.evaluate_door_events <carpeta con los .mp4 recortados> --untrimmed videos/ToyotaSmarthome_FP --annotations <carpeta Annotation>
```
Tarda cerca de una hora con GPU; las trayectorias quedan en `runner_state/` y la segunda
corrida es inmediata.

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

Cada worker, mientras su cámara está despierta, guarda 1 de cada 5 frames como
`../runner_state/frames/<cam_id>.jpg`
(escritura atómica vía archivo temporal + `replace`), que expone
`dashboard/` en `GET /api/frame/{cam_id}` para mostrar la cámara en vivo en
el panel. Si YOLO detectó a alguien en ese frame, el preview incluye el
overlay del esqueleto (bounding box + keypoints + huesos, vía
`vision/src/pose_pipeline.draw_pose()`) — así se puede ver en el dashboard
qué está detectando el pipeline, no solo la imagen cruda de la cámara.

**Excepción — cámara en foco:** mientras una cámara es la que está en vista
única del dashboard (`CameraWorker._is_focused()`, estado compartido vía
`focus_state.py` y `POST /api/focus`), el guardado pasa a ser en **cada**
frame en vez de 1 de cada 5. La pose ya corre sobre todos los frames
mientras la cámara está despierta (ver abajo), así que el único costo extra
es encode+escritura del JPEG — permite que la vista única muestre un stream
fluido con overlay (`GET /api/frame-stream/{cam_id}` en `dashboard/`, ver
"Vista única fluida" en `dashboard/README.md`). Con YOLO en CPU el "fluido"
es relativo (~150-300ms/frame); recién se nota fluido de verdad con GPU
(CUDA) disponible.

Dibujar ese overlay es opcional: cada worker consulta
`overlay_state.read_overlay_enabled()` antes de generar el preview, que lee
`../runner_state/overlay.json`. El dashboard escribe ese mismo archivo desde
`POST /api/overlay` (ver `dashboard/README.md`) — como `dashboard/` y
`runner/` corren en procesos separados, el archivo es la forma más simple de
que el toggle de la UI afecte a todas las cámaras ya en ejecución sin que un
proceso importe código del otro.

### La vista previa no queda en disco (2026-10-13)

`runner_state/frames/<cam_id>.jpg` es la única imagen del hogar que la
controladora escribe en disco. El informe (capítulo 3, Viabilidad Legal) declara que no se conserva,
así que ahora solo existe mientras la cámara se está analizando
(`preview_file.py`, llamado desde `camera_worker.py`):

- **Se borra cuando el runner se detiene.** Cada worker la borra en el
  `finally` de `run()`, que corre con Ctrl+C, con `SIGTERM`, cuando la fuente
  termina y cuando hay un error.
- **Se borra cuando la cámara se duerme** por el gate de movimiento
  (`_sleep()`), y **mientras está dormida no se vuelve a escribir**. Antes una
  cámara dormida seguía guardando 1 de cada 5 frames (y todos si estaba en
  foco); si eso se mantenía, el borrado duraba cinco frames. Vuelve a
  escribirse cuando la cámara despierta.
- **Se borra al arrancar**, por si la corrida anterior no llegó a limpiar
  (corte de luz, proceso matado). Con el gate activo la cámara arranca dormida,
  así que sin esto el panel mostraría una imagen vieja como si fuera actual.
- Se borra también el temporal `<cam_id>.tmp.jpg` si quedó uno a medio
  escribir. No se toca ningún otro archivo.

Consecuencias para el panel, verificadas (ver `dashboard/README.md`): con el
runner detenido o la cámara dormida, `GET /api/frame/{cam_id}` responde 404 y
la celda muestra su placeholder; el stream de la vista única espera sin cortar
la conexión y retoma solo cuando la cámara despierta. Una cámara que arranca
dormida no muestra imagen hasta el primer movimiento: para revisar el encuadre
durante la instalación hay que moverse delante de la cámara o poner
`motion_gate_enabled: false`.

**Límite conocido en Windows:** `dashboard/` detiene el runner con
`terminate()`, que en Windows mata el proceso sin ejecutar el `finally`. Por
eso `POST /api/stop` borra también la vista previa de cada cámara de
`cameras.yaml`. En Linux (la Pi) `terminate()` es `SIGTERM` y el runner limpia
solo. Lo que no se puede cubrir es un corte de luz: la imagen queda en disco
hasta el próximo arranque.

**La vista previa ya no vive en `frames/`.** Antes se escribía en
`frames/<cam_id>.jpg`, la misma carpeta de las imágenes de demo versionadas en
git. Como `cameras.yaml` tiene una cámara con `id: webcam`, el runner pisaba
`frames/webcam.jpg` en cada corrida y, con el borrado, además lo eliminaba. Ahora
se escribe en `runner_state/frames/`, que git ignora, y `frames/` queda solo con
las imágenes de demo: el runner no la toca más.

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

```yaml
cloud:
  url: https://monitor-api.ejemplo.com   # base del backend, sin /devices/events
  api_key: "la-api-key-del-kit"          # la muestra una sola vez cloud/alta_kit.py
  interval_s: 30                         # segundos entre sincronizaciones (default 30)
  retencion_local_dias: 7                # ver "Retención local de eventos" (default 7)
```

### Retención local de eventos (2026-10-13)

El informe (capítulo 3, Viabilidad Legal) declara que la controladora borra los eventos una vez
que están en la nube. Lo hace este mismo thread: después de cada sincronización llama a
`_purge_if_due`, que **una vez por hora como mucho** (`_PURGE_INTERVAL_S`; la primera, al
arrancar) borra los eventos que la nube confirmó hace más de `retencion_local_dias` días
(`EventStore.purge_synced`, ver `events/README.md`). No corre en cada ciclo de 30 segundos.

- `retencion_local_dias` va dentro de `cloud:`, default **7**. Admite decimales. Un valor de 0
  o negativo hace que el runner no arranque (`ValueError` al leer la config): 0 borraría cada
  evento apenas se envía, y se prefirió rechazarlo antes que adivinar la intención.
- **La purga nunca borra un evento que no se envió.** Si no hay conexión, o la nube responde
  409 porque el kit todavía no está vinculado, los eventos quedan pendientes hasta que se
  puedan enviar.
- **Hogar dado de baja:** si el hogar del kit se dio de baja en la nube (`DELETE /hogares/{id}`,
  ver `cloud/README.md`), la nube responde **410** y `sync_once` descarta ese lote sin enviarlo
  (`EventStore.discard`). Es el único caso en que se borra un evento no enviado. La cámara
  sigue analizando, pero lo que genera se descarta en el ciclo siguiente (30 segundos), así
  que no se acumula nada. Cuando el kit se vuelve a vincular, la nube deja de responder 410 y
  todo sigue normal, sin heredar eventos del hogar anterior.
- **Sin sección `cloud:` no se purga nada**: el thread no existe, así que una instalación 100%
  local conserva todo su historial como antes.
- La arquitectura no cambia: los eventos se siguen guardando en SQLite antes de sincronizar y
  el panel local sigue leyendo esa base.

De paso se corrigió un error que ya estaba en este thread: guardaba su `Event` de parada en
`self._stop`, el nombre de un método interno de `threading.Thread` en Python 3.12, y eso hacía
fallar `join()` al cerrar el runner con la nube configurada. Ahora se llama `_stop_event`, como
en `camera_worker.py`.

### Corte y reconexión con la nube, medido (2026-10-13)

Prueba de RNF6 y RNF7 de punta a punta, automatizada en
`tests/test_cloud_sync_corte_reconexion.py`. No simula la red: levanta el backend `cloud/` real
(FastAPI + uvicorn, `tests/backend_nube_local.py`) en un proceso aparte contra un SQLite
temporal, y `cloud_sync.py` le habla por HTTP. "Backend apagado" es ese proceso detenido, con
el puerto cerrado. Lo único reemplazado es la verificación del token de Firebase que usa la app
para vincular el kit. Nada toca AWS ni la base de producción.

Cada caso compara, evento por evento (un número en `metadata`), lo que hay en la base local
de la controladora con lo que devuelve `GET /hogares/{id}/eventos`, y además cuenta las filas
directo en el SQLite de la nube.

| Caso | Eventos generados | Pendientes durante el corte | Llegaron a la nube | Duplicados | Marcados como enviados |
|---|---|---|---|---|---|
| Backend apagado, 3 intentos de envío, después se enciende | 60 | 60 | 60 | 0 | 60 |
| El backend se cae entre dos lotes (lote = 200) y se generan 30 eventos más durante el corte | 480 | 280 | 480 | 0 | 480 |
| Kit sin vincular (409), después vinculado sin residente (409), después con residente | 40 | 40 | 40 | 0 | 40 |
| Thread `CloudSyncThread` corriendo durante el corte, sin reiniciarlo | 25 | 25 | 25 | 0 | 25 |

En los cuatro casos no se perdió ningún evento, cada uno llegó una sola vez y quedó marcado
como enviado (`cloud_synced_at`). Repetir la sincronización después no envió nada más. En el
primer caso, la caída y la inmovilidad grave del lote aparecen en `GET /hogares/{id}/alertas`.
En el del kit sin vincular, la nube no guardó ninguna fila mientras respondía 409.

**Límite encontrado, no cubierto por la prueba automatizada.** "Una sola vez" vale mientras la
controladora reciba la respuesta de la nube. Si la nube guarda el lote pero la respuesta se
pierde (corte justo después del `commit`, o más de 15 segundos de demora), la controladora no
lo marca como enviado y lo manda de nuevo: `POST /devices/events` no reconoce un evento
repetido. Se verificó a mano contra el backend local: 10 eventos con la respuesta descartada
quedaron como 20 filas en la nube. La entrega es "al menos una vez", no "exactamente una vez".
No se pierde nada, pero puede duplicarse un lote. Corregirlo requiere que cada evento lleve
un identificador propio y que la nube ignore los que ya tiene; no está hecho.

**Corregido el mismo día (2026-10-13, segunda ronda).** Cada evento lleva ahora un
identificador único (`MonitoringEvent.event_id`) que `events/store.py` genera al guardarlo en
la controladora y que no cambia en los reenvíos; la nube no vuelve a guardar un evento cuyo
identificador ya recibió de ese kit (ver "Ingestión idempotente" en `cloud/README.md`).
`cloud_sync.py` no cambió: sigue reenviando lo que no pudo marcar como enviado, y la nube
responde 201 con los repetidos contados en `duplicados`, así que la controladora los marca y
deja de insistir.

Quedó como prueba automatizada (`test_respuesta_perdida_no_deja_filas_duplicadas`), contra el
backend real en local: 250 eventos en dos lotes; la nube guarda el primero (200) y la respuesta
se descarta, dos veces seguidas; la controladora sigue con los 250 pendientes. Al volver la
conexión reenvía el primer lote y manda el segundo.

| | Antes del cambio (verificación a mano, 10 eventos) | Ahora (250 eventos, 2 respuestas perdidas) |
|---|---|---|
| Filas en la nube | 20 | 250 |
| Filas duplicadas | 10 | 0 |
| Eventos perdidos | 0 | 0 |
| Alertas en `/alertas` (1 caída + 1 inmovilidad enviadas) | no medido | 2 |

Las cuatro pruebas de corte y reconexión de la tabla de arriba siguen pasando sin cambios.
La protección depende de que la controladora mande `event_id`: una con el código anterior a
este cambio no lo manda y sigue como antes (se guarda todo, sin detectar repetidos).

Tampoco se midió qué pasa si se corta la energía de la controladora mientras escribe en su
SQLite.

## Pruebas

```
pip install pytest
pytest runner/tests        # desde la raíz del repo
```

No necesitan YOLO, PyTorch ni cámaras: `tests/conftest.py` reemplaza `vision/` por un detector
falso, y todo se escribe en carpetas temporales (nunca en `runner_state/` ni en `events/events.db`).
Cubren (2026-10-13):

- `test_cloud_sync_retencion.py`: la purga usa los días configurados, corre una vez por hora
  como mucho, no borra lo que no se pudo enviar, `retencion_local_dias` se lee de `cloud:` con
  default 7, y sin sección `cloud:` el runner no purga nada. Con un 410 de la nube los eventos
  pendientes se descartan; con cualquier otro error (409 incluido) se conservan.
- `test_vista_previa.py`: la vista previa se borra al detenerse el runner, al terminar o fallar
  la fuente y al dormirse la cámara; dormida no se escribe (ni en foco) y vuelve al despertar.
- `test_door_events.py` (25 pruebas, ver "Entradas y salidas de casa"): salida, entrada, ida y
  vuelta, pérdida breve en la puerta, desaparecer o reaparecer lejos de la puerta, arranque,
  salida informada una sola vez, video en loop, zona de forma cualquiera, configuración
  inválida, de dónde sale la posición, y el ciclo de la cámara: por defecto guarda los eventos
  si tiene la zona (también dormida por el gate); con `puerta_activada: false` o sin zona
  `puerta` no guarda ninguno.
- `test_fall_grouping.py` (8 pruebas, ver "Una posible caída por minuto"): las 16 sospechas
  de la prueba del 2026-10-14 quedan en 6, una racha larga publica una por minuto, una
  confirmada después de una sospecha se publica siempre, y lo mismo en el ciclo de la cámara.
- `test_evaluate_door_events.py` (8 pruebas, ver "Validación con video"): cómo se ajusta la
  zona, qué participantes ajustan y cuáles miden, y cómo se agrega la ausencia que le falta a
  un clip recortado. Sin YOLO ni videos.
- `test_cloud_sync_corte_reconexion.py` (5 pruebas, ver "Corte y reconexión con la nube,
  medido"): es la única que levanta otro proceso (el backend `cloud/` en un puerto local
  libre), así que además necesita las dependencias de `cloud/requirements.txt`. Tarda cerca de
  medio minuto. La quinta (respuesta perdida, 2026-10-13) comprueba que un lote reenviado no
  deja filas duplicadas.

## Fuera de alcance por ahora

- Multi-persona por cámara (mismo límite que `vision/`).
- Balanceo de carga entre cámaras si el CPU no alcanza para todas en
  paralelo — asume que el hardware objetivo (Pi 5) soporta la cantidad de
  cámaras configuradas.
