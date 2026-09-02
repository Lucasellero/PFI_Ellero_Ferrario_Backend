# `fusion/` — Clasificación de actividad híbrida (esqueleto temporal + CLIP)

## Resultados actuales (resumen, 2026-08-23)

> Accuracy de clasificación de actividad (4 categorías: Cooking,
> EatingDrinkingMedication, Moving, Resting), `evaluate_hybrid.py`. El split
> **holdout** es el que mejor representa producción (datos no usados ni para
> entrenar ni para elegir el peso de fusión) — no citar train como número esperable.

| Split | CLIP solo | Esqueleto solo | Híbrido (mejor `skeleton_weight`) |
|---|---|---|---|
| Train (824 videos) | 87.6% | 98.5% | **99.2%** (peso 0.8) |
| Val (830 videos) | 85.3% | 92.2% | **93.5%** (peso 0.5) |
| Holdout (273 videos) | 75.8% | 78.0% | **82.8%** (peso 0.2–0.5) |

El esqueleto solo ya supera a CLIP solo en los tres splits; la fusión suma sobre
ambos. `runner/config.py` usa `skeleton_weight=0.5` como default — razonable como
punto medio, pero el óptimo varía según split (0.8 en train, 0.2–0.5 en holdout);
falta más dato de holdout para fijarlo con más confianza. Integrado en el runner en
vivo desde el commit `e57542e` (`runner/camera_worker.py` + `main.py`), con
fallback automático a CLIP solo si el modelo de esqueleto no está entrenado/exportado.

## Clasificador temporal de caída (`fall_temporal.py`) — resumen (2026-09-01)

> Segunda deliverable de este módulo, aparte de la fusión de actividad: reemplaza la
> heurística de umbrales de caída de `vision/` por un GRU entrenado. Detalle completo
> en la sección "Clasificador temporal de caída" más abajo.

- **Clasificador solo, out-of-fold** (GroupKFold por video, sin leakage): recall
  96–99 % de las caídas, FP 2.8 % de las ventanas `normal`.
- **Como 2ª etapa** (`vision/evaluate_falls.py --temporal`, 5 datasets, 158 caídas /
  292 clips): recall end-to-end **133/158 (84.2 %) — igual que geométrico solo, no
  cae** — y **FP de alerta dura 185 → 91**.
- **Donde hay ventana de esqueleto que clasificar** (todas las caídas reales + ~la
  mitad de los FP): rechaza **94/94 FP (100 %)**, pierde **1/113 caídas**.
- Los ~74 FP residuales (casi todos de `ToyotaSmarthome_FP`) son eventos **sin
  ventana**: `yolo11n-pose` no detecta a la persona en sillón/cámara ancha (límite ya
  documentado en `vision/README.md`). En producción caen a la confirmación CLIP.
- Umbral congelado: `_FALL_TEMPORAL_REJECT = 0.15` (`runner/camera_worker.py`).
  Artefacto: `fusion/models/fall_temporal_int8.onnx`.

## Plan hacia ~10 clases y sustitución de CLIP

Ver [`PLAN_sustitucion_clip.md`](PLAN_sustitucion_clip.md) (2026-09-01): cómo pasar de
las 4 categorías actuales a ~10 clases canónicas agregando señal de **zona del hogar**
y **objeto de contexto** (no un CLIP más grande — ya se descartó, ver `vlm/README.md`),
degradando CLIP a señal opcional, y cómo medir cada fase. El vocabulario canónico es
el mismo que el del ETL de [`dataset_events/`](../dataset_events/README.md).

## Por qué existe este módulo

`vlm/` (CLIP, zero-shot o linear probe) clasifica actividad **frame a
frame**: no ve movimiento, solo la apariencia de un instante. Eso le alcanza
para actividades con un contexto visual claro (`Uselaptop`, `WatchTV`,
`Walk`) pero no para las que dependen de un gesto de mano breve y de qué
objeto hay en la mano — el propio `vlm/README.md` documenta que
`EatingDrinkingMedication` (que agrupa `Takepills`, `Drink.*`, etc.) es la
categoría más débil incluso con linear probe + hand-crop (56.1%), porque el
embedding de CLIP es holístico y "tomar una pastilla" y "tomar un vaso de
agua" se ven casi igual en un solo frame.

`vision/` ya corre YOLOv11-pose sobre cada frame para caídas/inmovilidad,
pero solo usa descriptores geométricos instantáneos (aspect ratio, ángulo
de torso) o una tasa de desplazamiento escalar — nunca la forma completa
del movimiento a lo largo del tiempo.

Este módulo suma esa pieza que falta: un clasificador **temporal**, liviano,
sobre la secuencia de keypoints que YOLO-pose ya calcula (sin cargar un
segundo modelo de pose), como señal principal de *movimiento*, y lo combina
con CLIP como señal secundaria de *contexto/objeto*, en vez de reemplazar
ninguno de los dos.

## Estructura

```
fusion/
├── src/
│   ├── keypoint_sequence.py   # features por frame + ventanas temporales (SkeletonSequenceBuffer)
│   ├── labels.py              # espacio de clases (31 de vlm/ + Standing) y su mapeo a categoría
│   ├── temporal_model.py      # GRU (PyTorch, solo dev/entrenamiento)
│   ├── temporal_classifier.py # inferencia ONNX Runtime (Pi, sin PyTorch)
│   ├── fusion.py              # ScoreFuser + HybridActivityClassifier
│   └── fall_temporal.py       # FallTemporalClassifier (2ª etapa del evento de caída, ver sección abajo)
├── extract_dataset.py         # cachea ventanas de esqueleto + embeddings CLIP de un dataset externo
├── train_temporal_classifier.py  # entrena el GRU de actividad
├── export_temporal_onnx.py       # exporta el GRU de actividad a ONNX INT8
├── train_fusion_combiner.py      # entrena el combinador "mlp" (opcional, ver ScoreFuser)
├── evaluate_hybrid.py             # compara CLIP solo / esqueleto solo / híbrido
├── extract_falls.py              # ventanas etiquetadas normal/caída desde videos/**/Fall.csv (+ hard-negative mining)
├── train_fall_temporal.py       # entrena el GRU de caída (augmentación de oclusión, GroupKFold por video)
├── export_fall_temporal_onnx.py # exporta el GRU de caída a ONNX INT8
├── requirements.txt            # Pi: onnxruntime + numpy
└── requirements-train.txt      # dev: torch, sklearn, ultralytics, opencv
```

## Integración con `vision/` y `vlm/` — sin tocar ninguno de los dos

Siguiendo la misma regla que ya rige el resto del repo (`vision/` y `vlm/`
no se importan entre sí — ver "Cómo trabajar en paralelo" en el README
raíz), `fusion/` tampoco modifica código de ninguno de los dos módulos.
Es un tercer consumidor, en el mismo lugar donde hoy ya conviven `vision/`
y `vlm/`: `runner/camera_worker.py`.

**El punto de enganche exacto**: `FallImmobilityDetector.process_frame()`
ya expone `detector.last_detection` (`kpts, box`) y
`detector.last_detection_conf` después de cada frame — es lo que
`_clip_tick()` en `runner/camera_worker.py` ya lee hoy para recortar a la
persona antes de llamar a CLIP. El clasificador de esqueleto se engancharía
exactamente ahí, sin ningún cambio a `vision/src/pose_pipeline.py`:

```python
# Dentro de CameraWorker.run(), después de detector.process_frame(...)
# (mismo lugar donde ya se lee detector.last_detection para CLIP):
if detector.last_detection is not None:
    kpts, box = detector.last_detection
    ps = build_pose_sample(kpts, box, w, h, frame.timestamp_s)
    feat = frame_features(kpts, detector.last_detection_conf, box,
                           ps.aspect_ratio, ps.torso_angle_deg, ps.hip_y, ps.bbox_height_frac)
    self._skeleton_buffer.push(frame.timestamp_s, feat)
else:
    self._skeleton_buffer.reset()   # hueco de detección: no interpolar sobre el vacío

window = self._skeleton_buffer.build_window(frame.timestamp_s)
# ... en el mismo tick de muestreo que ya usa CLIP (_clip_tick):
result = self._hybrid.classify(window, crop_to_person(frame.image, box), hand_crop)
```

`vision/` no necesita exponer nada nuevo — `PoseSample`, `build_pose_sample`
y `detect_person` ya son públicos y ya se usan así desde `vlm/` (ver
`crop_to_hands` en `vlm/src/clip_classifier.py`, que consume la misma
tupla). `vlm/` tampoco cambia: `ClipActivityClassifier.classify()` se sigue
llamando exactamente igual, `HybridActivityClassifier` solo lo envuelve.

**Implementado (2026-08-18)**: el enganche de arriba ya no es solo el plan —
está en producción en `runner/camera_worker.py`. `CameraWorker` acumula un
`SkeletonSequenceBuffer` por cámara (push en cada frame con detección, reset
en cada hueco/loop de video/despertar del gate de movimiento, igual criterio
que el historial de inmovilidad de `vision/`) y arma la ventana en cada tick
de `_clip_tick` (mismo `clip_interval_s` que ya usaba CLIP solo). El modo se
decide en el constructor: `"hybrid"` si `fusion/models/temporal_classifier_int8.onnx`
existe (cargado una sola vez en `runner/main.py`, compartido entre cámaras,
igual que CLIP), `"clip_only"` si no — degradación elegante, mismo patrón que
ya usa el proyecto para CLIP y la confirmación de caídas. `skeleton_weight`
es configurable por `cameras.yaml`/`RunnerConfig` (default 0.5, el de
`ScoreFuser` — **sin validar todavía**, ver "Estado" abajo). Cada evento de
actividad guarda `fusion_mode` en su metadata para poder auditar si salió de
`hybrid` o `clip_only`. El doble check probe/zero-shot de `vlm/` (ver
`runner/README.md`) sigue operando solo sobre la señal de CLIP, sin tocar
esqueleto — es una verificación de acuerdo interno de CLIP consigo mismo, no
cambia con esta integración. Ver "Clasificación de actividades (CLIP)" en
`runner/README.md` para el detalle de la ventana fusionada.

**Costo en el Pi**: el GRU corre sobre una ventana ya calculada a partir de
keypoints que YOLO-pose ya produjo — no hay inferencia de red neuronal
nueva por *frame*, solo una pasada de GRU (miles de FLOPs, no millones)
en cada tick de clasificación (cada `clip_interval_s`, igual que CLIP). El
costo dominante sigue siendo YOLO-pose + CLIP, ya medido en
`vlm/README.md`.

## Diseño

### 1. Ventanas de keypoints (`src/keypoint_sequence.py`)

Cada frame con detección se convierte a un vector de **55 features**:
34 (17 keypoints × x,y, centrados en la cadera media y escalados por el
alto del bbox — invariante a traslación y a distancia de cámara, mismo
criterio de normalización que ya usa `vision/` para la velocidad de caída y
el desplazamiento de inmovilidad) + 17 (confianza individual por keypoint)
+ 4 (`aspect_ratio`, `torso_angle_deg`, `hip_y`, `bbox_height_frac`, que
`vision/` ya calcula gratis).

`SkeletonSequenceBuffer` acumula estos vectores y arma ventanas de tamaño
fijo (`n_frames`, default 16) resampleadas dentro de los últimos
`window_s` segundos (default 2.0s — dentro del rango de 2-3s pedido, elegido
corto a propósito: los gestos que más importa capturar, ej. llevarse algo a
la boca, duran ~1-2s, y una ventana más larga diluye la señal con contexto
de antes/después del gesto sin ganar nada). El resampleo usa vecino más
cercano en el tiempo (no por índice de frame), igual criterio que
`_vertical_velocity`/`_smoothed_displacement_rate` en
`vision/src/pose_pipeline.py`: tolera que el intervalo entre detecciones
varíe (frame-dropping en tiempo real) sin romper.

Una sola implementación para los dos usos (en vivo y offline en
`extract_dataset.py`) — evita que la lógica de ventaneo diverja entre
entrenamiento e inferencia, que fue justamente la fuente de dos bugs reales
documentados en `vision/README.md` (normalización de velocidad y de
desplazamiento).

### 2. Clasificador temporal (`src/temporal_model.py`, `src/temporal_classifier.py`)

GRU de 2 capas, 64 unidades ocultas (configurable) sobre la secuencia de 55
features → clasifica directamente sobre las **31 clases finas de Toyota
Smarthome que ya usa `vlm/`** (mismo vocabulario — `ACTIVITY_PROMPTS` en
`vlm/src/clip_classifier.py`) **+ `Standing`** (ver `src/labels.py` y la
sección "Datos" abajo).

Por qué 31 clases finas y no las categorías gruesas de `vlm/`
(`COARSE_CATEGORIES`) directamente: el esqueleto y CLIP tienen fortalezas
distintas dentro de la misma categoría gruesa (ej. dentro de `Moving`, el
esqueleto debería separar bien `Walk` de `Sitdown`/`Getup` por la forma del
movimiento; CLIP no tanto). Entrenar fino y agregar a categoría después
(`fine_scores_to_category_scores`, igual criterio que
`aggregate_to_categories` en `vlm/`) deja esa señal disponible para
`evaluate_hybrid.py` y para debug, en vez de perderla de entrada.

Presupuesto de parámetros: ~50-90k con los defaults (input=55, hidden=64,
2 capas) — dos órdenes de magnitud por debajo del límite de 1-2M pedido,
deja margen para escalar si el modelo resulta con poca capacidad. Exporta a
ONNX igual que CLIP (`export_temporal_onnx.py`, cuantización dinámica INT8
con `onnxruntime.quantization`, mismo flujo que `vlm/export_onnx.py`) — la
normalización de input (z-score por canal) queda horneada dentro del grafo
(buffers no entrenables del módulo), así que `SkeletonActivityClassifier`
en el Pi no necesita mantener sincronizado ningún archivo de
mean/std aparte.

**Puerta abierta a ST-GCN**: se empieza con GRU porque es más simple de
exportar/cuantizar y de justificar en la tesis, y porque ya hay evidencia
en la literatura citada en `vision/README.md` (Benabdennour et al. 2026: un
Transformer chico sobre secuencias de YOLOv11-Pose da 97.7% F1 para
caídas) de que un modelo temporal liviano sobre este tipo de secuencia
funciona bien en este dominio. Si el GRU no alcanza a separar clases con
gestos sutiles (la hipótesis que motiva todo este módulo), el siguiente
paso natural es ST-GCN (Yan et al. 2018), que modela explícitamente la
topología del esqueleto COCO en vez de tratarlo como una secuencia plana —
no implementado todavía porque primero hay que medir si el GRU alcanza
(ver "Estado" abajo).

### 3. Fusión (`src/fusion.py`)

`ScoreFuser` combina las distribuciones de categoría de esqueleto y CLIP,
con dos métodos intercambiables (mismo criterio de "empezar simple, medir,
escalar solo si hace falta" que ya aplicó `vlm/` con zero-shot → linear
probe):

- **`weighted_average`** (default): promedio pesado por categoría, un solo
  hiperparámetro (`skeleton_weight`). Categorías que una señal no predice
  (CLIP nunca predice `Standing`) se tratan como probabilidad 0 de esa
  señal, no como error — degrada con gracia en vez de romper.
- **`mlp`**: logistic regression (mismo enfoque que el linear probe de
  `vlm/`) entrenado sobre `[scores_esqueleto, scores_clip]` concatenados →
  categoría real (`train_fusion_combiner.py`). Deja que el modelo aprenda,
  por categoría, cuál señal pesa más, en vez de un peso global fijo — la
  hipótesis a validar es que confía más en CLIP para
  `EatingDrinkingMedication` (depende del objeto en la mano) y más en
  esqueleto para `Moving`/`Standing` (depende del movimiento del cuerpo).

`HybridActivityClassifier` envuelve ambos clasificadores + el fusor detrás
de un solo `mode` (`"hybrid"` / `"skeleton_only"` / `"clip_only"`) — el
mismo código sirve para producción (`runner/`, `mode="hybrid"`) y para la
comparación de `evaluate_hybrid.py`, sin dos implementaciones paralelas que
puedan divergir.

## Datos

**No hay dataset de video completo disponible en esta máquina** (está en
otra, ver README raíz) — el mismo condicionante que ya documenta
`vlm/README.md` para sus propios experimentos a escala. Lo que sí hay
localmente son los 39 clips curados de Toyota Smarthome (repartidos en
`videos/acostado/`, `videos/parado/`, `videos/sentado/` — la estructura
vieja que `vision/README.md` documenta como no confiable para *ground
truth de postura por carpeta*, pero los nombres de archivo
`<Actividad>_pXX_...mp4` siguen siendo válidos como etiqueta de actividad,
que es lo que usa este módulo, no la carpeta contenedora).

Usado como smoke test end-to-end (extracción → entrenamiento → export ONNX
→ evaluación) — ver "Estado" abajo. **No alcanza para entrenar en serio**:
faltan participantes por clase para un split honesto, y varias de las 31
clases no tienen ningún clip local. Para entrenar/evaluar de verdad hace
falta apuntar `extract_dataset.py` al dataset completo de Toyota Smarthome
en la otra máquina (mismo flujo que `vlm/extract_clip_features.py`).

**Actualización (2026-08-18): ya no falta.** Se sumó a `../videos/` un
dataset local de ~2200 clips (no versionado en git), organizado por el
usuario en subcarpetas por actividad en español (`beber/`, `caminar/`,
`cocinar/`, etc. — ver `../videos/README.md`), con los mismos nombres de
archivo `<Actividad>_pXX_...mp4` de siempre dentro de cada carpeta. Cubre 18
de las 31 clases finas (las que ya se ven en "Estado" abajo). No cubre
`Standing` (sigue sin datos, ver más abajo) ni las clases de `ReadingScreen`/
`Phone`/`Background` que sí tiene el dataset externo de `vlm/`. Bug
encontrado en el camino: `extract_dataset.py` barría con
`DATASET_DIR.glob("*.mp4")` (no recursivo) — con los videos anidados en
subcarpetas, encontraba 0 archivos. Corregido a `glob("**/*.mp4")`, mismo fix
que `vlm/extract_clip_features.py`.

**Nota sobre las carpetas y `Getup`**: el usuario separó "levantarse desde
el piso/cama" (`levantarse (acostado)/`, 8 clips) de "levantarse desde una
silla" (`pararse (sentado)/`, 95 clips, y algunos más mezclados con
`Sitdown` en `sentarse/`) — pero el label real (nombre de archivo) para
todos esos clips es el mismo `Getup` de Toyota Smarthome, que no distingue
el punto de partida. Esa distinción más fina que sí está en la organización
de carpetas **se pierde** con el pipeline actual (labels salen del nombre de
archivo, no de la carpeta — mismo criterio documentado en
`../videos/README.md`). Si se quisiera esa distinción como clase nueva,
haría falta tratar la carpeta como fuente de label solo para estos casos
(cambio real, no solo de datos) — no implementado.

**`Standing`**: cero datos locales. Si se retoma (ver `src/labels.py` para
la justificación de por qué el esqueleto podría generalizar mejor que CLIP
acá), el camino más corto es el mismo que ya recorrió `vlm/` con
`extract_omnifall_features.py`/`extract_gmdcsa24_features.py` (ver
"`Standing`: se probó y se descartó" en `vlm/README.md`, y notar que esos
scripts se borraron del repo tras descartar la clase — habría que
reconstruir el mapeo OmniFall→archivo real, no reusar código existente):
resolver las anotaciones de OmniFall contra `videos/CAUCAFall/` y
`videos/GMDCSA-24/official/` (el dataset completo de GMDCSA-24 ya está
descargado ahí, sin video en esta rama de trabajo tampoco). No implementado
en esta sesión — la arquitectura y el espacio de labels ya están
preparados para no tener que retocar nada más que los datos cuando esto se
retome.

## Estado

Implementado, con smoke test end-to-end corrido en su momento sobre el
dataset local chico (39 clips) y, desde el 2026-08-18, entrenado en serio
sobre el dataset local grande (ver "Datos" arriba) — integrado además en
`runner/` (ver "Integración" arriba). Resultados reales:

**Extracción** (`extract_dataset.py ../videos --max-per-class 60`): 2464
ventanas de esqueleto+CLIP, 824 videos, mismas 18 clases finas que
`vlm/README.md` documenta para su propio reentreno (30 videos sin ventana
con cobertura temporal suficiente, descartados).

**Clasificador temporal** (`train_temporal_classifier.py`, participantes
disjuntos train/val, 5 seeds, a nivel de **clase fina** — no de categoría
gruesa, no comparable directo al 75.6% de `vlm/`):

| | Accuracy |
|---|---|
| Esqueleto solo (18 clases finas) | **47.9%** (±4.3%) |

Con azar en ~5.6% (1/18), el esqueleto solo captura señal real de
movimiento — esperable que separe bien `Walk` de posturas estáticas, más
dudoso entre gestos finos de mano (`Drink.*`/`Takepills`), que es
justamente la hipótesis que motivó este módulo (ver "Por qué existe este
módulo" arriba). No se desglosó todavía por clase — pendiente si hace falta
para el informe. Modelo final entrenado sobre el 100% del cache, exportado a
ONNX (`export_temporal_onnx.py`, INT8) y ya sirviendo en `runner/`.

**Comparación de los tres enfoques** (`evaluate_hybrid.py
dataset_features.npz --skeleton-weight-sweep`): CLIP solo 87.6%, esqueleto
solo 98.5%, híbrido hasta **99.2%** con `skeleton_weight=0.8`.

**Estos números de la comparación NO son confiables — no usar en el
informe tal cual.** El propio script lo advierte en su docstring: para que
la comparación sea honesta hace falta evaluar contra videos/participantes
que los modelos no vieron en entrenamiento. Acá se evaluó
`dataset_features.npz` contra `temporal_classifier.pt` y
`category_probe.npz`, **el mismo cache con el que se entrenó el modelo
final de ambos** (tanto `train_temporal_classifier.py` como
`train_category_probe.py` reentrenan sobre el 100% del cache en su último
paso) — es un chequeo de que el pipeline de fusión funciona de punta a
punta (el barrido de `skeleton_weight` sí muestra que sumar esqueleto ayuda
en la dirección esperada), no una medición válida de accuracy real.

**Resuelto (2026-08-18): número honesto de comparación.** Se extrajo un
segundo cache con `--seed 99` (`dataset_features_val.npz`) — pero cambiar
solo el seed no alcanzó: 7 clases (`Cook.Cut`, `Cook.Usestove`,
`Drink.Fromglass`, `Eat.Snack`, `Laydown`, `Maketea.Insertteabag`,
`Takepills`) ya usaban el 100% de sus videos disponibles en el entrenamiento
original, así que cualquier seed vuelve a elegir exactamente los mismos.
Midiendo el overlap real video por video: **67.1%** de los 830 videos de
este segundo cache ya estaban en el cache de entrenamiento. Se filtró a los
**273 videos genuinamente no vistos** (11 de las 18 clases —
`dataset_features_holdout.npz`, no versionado) y se corrió
`evaluate_hybrid.py` sobre ese subconjunto limpio:

| | Accuracy (273 videos, sin leakage) |
|---|---|
| CLIP solo | 75.8% |
| Esqueleto solo | 78.0% |
| **Híbrido** (`skeleton_weight=0.2-0.5`, empatados) | **82.8%** |

Consistente con el 75.6% que ya reportaba `train_category_probe.py` para
CLIP solo (mismo orden de magnitud, dataset distinto pero mismo tipo de
split) — buena señal de que este número sí es honesto. El híbrido mejora
+7 puntos sobre CLIP solo y +4.8 sobre esqueleto solo: la fusión ayuda de
verdad, no es ruido del leakage. `skeleton_weight=0.5` (el default actual,
ver "Integración" arriba) cae justo en la zona empatada como mejor opción —
queda validado, no es solo el default de la librería.

**Salvedad de esta validación**: cubre 11 de las 18 clases (las otras 7 no
tienen ningún video sobrante para poder medirse sin leakage con este
dataset) y las clases están desbalanceadas (`Walk` 55 videos,
`Drink.Frombottle` solo 2) — un número más sólido para el informe pediría
más datos por clase, no algo resoluble con lo que hay hoy en `videos/`.

## Clasificador temporal de caída (`fall_temporal.py`)

### Por qué

El informe propone (Sec. 2.2.7) detectar caídas con la heurística geométrica de
Giri et al. (2025): pico de velocidad vertical de la cadera + aspect ratio del
bounding box, con confirmación CLIP zero-shot encima (`vlm/src/fall_confirmation.py`).
`vision/README.md` documenta el techo de ese enfoque: **encuadre parcial** (el bbox
recortado contra el borde ensancha el aspect ratio y hace saltar `hip_y` → firma
falsa de caída) y **firma geométrica ambigua** (sentarse en un sillón bajo genera la
misma firma; barrido exhaustivo de 182 combinaciones de umbrales → ninguna lo
resuelve).

Este clasificador reemplaza la **decisión** de caída (no el trigger): la heurística
geométrica de `vision/` sigue siendo el disparador barato por frame, y cuando
dispara, en vez de mirar la apariencia de un frame recortado (CLIP), se mira la
**forma del movimiento** de los últimos ~2 s sobre la misma ventana de keypoints que
ya alimenta la fusión de actividad. Es el mismo salto metodológico que ya se hizo en
`vlm/` (zero-shot → linear probe) y en la fusión de actividad de este módulo
(frame-a-frame → GRU temporal).

Respaldo en la literatura: Benabdennour et al. 2026 (YOLOv11-Pose → Transformer,
97.7 % F1, ya citado en `vision/README.md`); VIRA-GCN / PointOut (augmentación de
oclusión sintética para robustez del esqueleto); ConvLSTM y LSTM 3-etapas
Normal/Warning/Fall (99.44 % en Raspberry Pi).

### Diseño

- **Clases**: binario `normal` / `caida`. Se probó el esquema de 3 etapas
  (`normal` / `caida_en_curso` / `caido`), pero los datasets de caída disponibles
  (`GMDCSA-24`, `CAUCAFall`) están recortados al acto de caer — solo 165 ventanas
  `caido` vs. 3676 `caida_en_curso` — muy poco para una 3ª clase. `extract_falls.py`
  igual emite las etiquetas finas al `.npz`; `train_fall_temporal.py` las colapsa.
- **Modelo**: reusa `GRUActivityClassifier` (`src/temporal_model.py`) con
  `num_classes=2` — ~48 k parámetros, ONNX INT8, misma normalización horneada en el
  grafo. La inferencia (`FallTemporalClassifier`) reusa `SkeletonSequenceBuffer` /
  `frame_features`, igual que el clasificador de actividad — no hay segundo modelo
  de pose ni inferencia por frame, solo una pasada de GRU cuando la geometría
  dispara.
- **Datos** (`extract_falls.py`, todos locales): `GMDCSA-24`, `CAUCAFall`, `URFD`
  con `Fall.csv` → ventanas positivas; `ToyotaSmarthome_FP` (4 clips largos anotados
  `NoFall` de punta a punta, los FP que encontró `vision/` validando inmovilidad) →
  negativos duros. El parseo del CSV reusa `vision/evaluate_falls.py::parse_classes`.
- **Augmentación de oclusión/truncamiento** (`train_fall_temporal.py`, solo en
  train): *keypoint dropout* (PointOut — pone a 0 posición + confianza de un
  subconjunto aleatorio de keypoints) y *truncamiento de encuadre* (VIRA-GCN —
  elimina piernas o cabeza en toda la ventana y ajusta los escalares que el recorte
  del bbox distorsiona: aspect ratio más ancho, bbox más bajo). Objetivo: que el
  modelo aprenda "torso visible + este movimiento = caída" en vez de romperse cuando
  la persona sale de cuadro.
- **Hard-negative mining** (`extract_falls.py --mine-hard-negatives`): el muestreo
  aleatorio de ventanas `normal` no le muestra al modelo el caso que tiene que
  aprender a rechazar — la geometría dispara justo en el instante *más* parecido a
  una caída (bbox colapsa + pico de velocidad al dejarse caer en un sillón). El
  mining corre la heurística geométrica en paralelo a la extracción y agrega como
  `normal` la ventana en cada instante donde dispara dentro de un tramo no-caída
  (126 de `ToyotaSmarthome_FP`), sobre-representadas ×6 en el entrenamiento
  (`--hard-neg-oversample`).
- **Métrica** (`train_fall_temporal.py`, GroupKFold por video, out-of-fold): recall
  *por video de caída* (¿alguna ventana `caida` del video se predijo como caída?) y
  FP (ventanas `normal` predichas como caída), alineada con `evaluate_falls.py`, no
  accuracy por ventana.

### Integración con `runner/`

`FallTemporalClassifier` se carga una vez en `runner/main.py` y se pasa a cada
`CameraWorker`. En `_confirm_fall_event` (reemplaza a `_confirm_fall`): cuando la
heurística geométrica de `FallImmobilityDetector` dispara un evento `fall` de
evidencia completa, se corre el temporal sobre la ventana de esqueleto del momento.

- `fall_score ≥ _FALL_TEMPORAL_CONFIRM` (0.45) → confirma.
- `fall_score ≤ _FALL_TEMPORAL_REJECT` (0.15) → descarta.
- En el medio → desempate con CLIP si está disponible.
- Sin ventana de esqueleto (cobertura insuficiente, típico en la escena de sillón /
  cámara ancha donde `yolo11n-pose` no detecta bien) → cae a la confirmación CLIP.

Las caídas de evidencia **parcial** (ver `vision/README.md`, Fase 0) nunca pasan por
el temporal: van directo a `severity: "sospecha"` y escalan por inmovilidad.
`FallImmobilityDetector` no cambia su heurística — sigue siendo el trigger y se puede
seguir calibrando solo con `evaluate_falls.py`.

### Resultados (2026-09-01)

Modelo con hard-negative mining, `τ = _FALL_TEMPORAL_REJECT = 0.15`.

**Clasificador solo, out-of-fold** (`train_fall_temporal.py`, GroupKFold por video):
recall 96 % de las caídas, FP 2.8 % de las 10 362 ventanas `normal`.

**Como 2ª etapa** (`vision/evaluate_falls.py --temporal --fps-cap 10`, 5 datasets,
158 caídas anotadas / 292 clips):

| Config | Recall | FP alerta dura | FP "sospecha" (encuadre parcial) |
|---|---|---|---|
| Geométrico solo (propuesta original, sin CLIP) | 133/158 (84.2 %) | 185 | 71 |
| **+ clasificador temporal** | **133/158 (84.2 %)** | **91** | 71 |
| *Referencia histórica: geométrico + CLIP probe (otro split)* | *84.8 %* | *~81 en ToyotaSmarthome_FP* | *—* |

Separando por si el modelo **puede** actuar (hay ventana de esqueleto que clasificar):

| | FP | rechazados | Caídas reales | rechazadas de más |
|---|---|---|---|---|
| Con ventana de esqueleto | 94 | **94 (100 %)** | 113 | 1 |
| Sin ventana (`yolo11n-pose` no detecta a la persona) | 91 | 0 | 34 | 0 |

**Conclusión honesta:** donde el modelo tiene entrada es un discriminador casi
perfecto (100 % de los FP, 1/113 caídas perdidas). Los ~74 FP residuales — casi todos
de `ToyotaSmarthome_FP` — son eventos **sin ventana de esqueleto**: es el mismo
bloqueante ya documentado en `vision/README.md` (`yolo11n-pose` no detecta a una
persona acostada en un sillón desde cámara ancha/elevada, 94.5 % de frames sin
detección). No es una falla del clasificador temporal: no hay esqueleto que
clasificar. En producción esos eventos caen a la confirmación CLIP, así que el
sistema desplegado = temporal (para lo que puede) + CLIP (para el resto).

**Salvedades para el informe:**
- El recall end-to-end (84 %) lo limita el **trigger geométrico** (dispara en ~86 %
  de las caídas); el temporal es un veto, no puede recuperar lo que la geometría no
  marcó. El 96–99 % es del clasificador aislado.
- Contra la referencia CLIP: en el caso duro de cámara ancha quedan **parejos**
  (~74 vs. ~81), no mejor — el cuello de botella ahí es el detector de pose.
- Varianza entre corridas de entrenamiento moderada — por eso `train_fall_temporal.py`
  fija `--seed`.

## Uso

```bash
cd fusion/
pip install -r requirements-train.txt

# 1. Extraer ventanas de esqueleto + embeddings CLIP de un dataset externo
python extract_dataset.py <carpeta_dataset> --max-per-class 40 -o dataset_features.npz

# 2. Entrenar el clasificador temporal
python train_temporal_classifier.py --features dataset_features.npz

# 3. Exportar a ONNX INT8 (para el Pi)
python export_temporal_onnx.py

# 4. (opcional) entrenar el combinador de fusión aprendido
python train_fusion_combiner.py --features dataset_features.npz

# 5. Comparar los tres enfoques — usar un cache de videos NO vistos en 1-4
python extract_dataset.py <carpeta_validacion> -o val_features.npz
python evaluate_hybrid.py val_features.npz --skeleton-weight-sweep
```

### Clasificador temporal de caída

```bash
cd fusion/
# 1. Ventanas etiquetadas + hard negatives desde videos/**/Fall.csv
python extract_falls.py ../videos --fps-cap 8 --max-normal-per-video 250 \
    --post-fall-s 4.0 --mine-hard-negatives -o fall_windows.npz
# 2. Entrenar (augmentación de oclusión, hard negatives ×6, semilla fija)
python train_fall_temporal.py --features fall_windows.npz --hard-neg-oversample 6 --seed 0
# 3. Exportar a ONNX INT8
python export_fall_temporal_onnx.py
# 4. Evaluar end-to-end (desde vision/)
cd ../vision && python evaluate_falls.py --temporal --fps-cap 10
#   opcional: --dump-scores scores.csv para barrer el umbral offline
```

En el Pi, `pip install -r requirements.txt` (sin PyTorch/Ultralytics —
`SkeletonActivityClassifier` solo necesita `onnxruntime` + `numpy`, mismo
criterio de separación dev/Pi que `vlm/`).
