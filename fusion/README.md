# `fusion/` — Clasificación de actividad híbrida (esqueleto temporal + CLIP)

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
│   └── fusion.py              # ScoreFuser + HybridActivityClassifier
├── extract_dataset.py         # cachea ventanas de esqueleto + embeddings CLIP de un dataset externo
├── train_temporal_classifier.py  # entrena el GRU
├── export_temporal_onnx.py       # exporta el GRU a ONNX INT8
├── train_fusion_combiner.py      # entrena el combinador "mlp" (opcional, ver ScoreFuser)
├── evaluate_hybrid.py             # compara CLIP solo / esqueleto solo / híbrido
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

Implementado y con smoke test end-to-end corrido sobre el dataset local
chico (ver más abajo el resultado real, incluida su salvedad — dataset
demasiado chico para que el número signifique generalización real, igual
que la primera versión de `vlm/`, medida sobre 39 videos, tampoco
generalizó — ver "El 65.2% no generalizaba" en `vlm/README.md`).

**Pendiente, antes de tener un resultado que valga para la tesis:**
extraer sobre el dataset completo en la otra máquina, entrenar en serio
(`train_temporal_classifier.py`, con más de una clase por participante),
elegir `skeleton_weight` (o entrenar `train_fusion_combiner.py`) contra un
split de validación real, y recién ahí correr `evaluate_hybrid.py` con
matrices de confusión que valgan para comparar los tres enfoques en el
informe.

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

En el Pi, `pip install -r requirements.txt` (sin PyTorch/Ultralytics —
`SkeletonActivityClassifier` solo necesita `onnxruntime` + `numpy`, mismo
criterio de separación dev/Pi que `vlm/`).
