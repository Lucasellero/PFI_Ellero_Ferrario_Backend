# `vlm/` — Clasificación semántica de actividades

## Qué hace

Consulta un VLM local open-source de forma **periódica** (no frame a frame —
ver Sec. 2.1.6 del informe, el costo computacional de un VLM es mucho mayor
al de pose estimation) sobre frames muestreados de un video, y genera una
descripción textual + etiqueta de actividad (ej. "cocinando", "mirando
televisión", "leyendo").

## Dos modos de uso

- **Batch** (`vlm/main.py`): procesa un archivo o todo `../videos/` de una
  sola vez, muestreando cada `--interval` segundos. Pensado para el dataset
  grabado.
- **En vivo** (vía `../runner/`): `runner/camera_worker.py` importa
  `ClipActivityClassifier` directamente y lo corre sobre cada cámara
  configurada en `cameras.yaml`, en ventanas de muestreo cada
  `clip_interval_s` segundos, en paralelo con la detección de pose. En este
  modo cada frame se recorta a la persona detectada (`crop_to_person`), se
  promedian varias muestras por ventana, y hay umbral de confianza y
  deduplicación de actividades repetidas — ver "Clasificación de
  actividades (CLIP)" en `../runner/README.md`.
  Requiere haber generado los modelos ONNX primero (ver Setup abajo) — si no
  están, `runner/` sigue funcionando pero sin clasificación de actividad.

## Entrada

Un archivo de video en `../videos/<nombre>.mp4` (modo batch) o un frame en
vivo de `../runner/` (modo en vivo). Se muestrea un frame cada N segundos
(configurable) en vez de procesar todos los frames.

## Salida

`MonitoringEvent` con `event_type = "activity"` (ver
`../events/schema.py`), `video_timestamp_s` = momento del frame muestreado,
y `metadata` con al menos `description` (texto libre del VLM) y, si se logra
clasificar en una categoría cerrada, `activity_label`.

En modo en vivo `activity_label` es la **categoría gruesa** (ver "Precisión
y evaluación" abajo) y el metadata incluye además `fine_label` (la clase
fina ganadora), `confidence` (de la categoría, promedio de la ventana),
`samples` (cuántos frames se promediaron), `ongoing` (`true` si es el
heartbeat de una actividad que continúa, no un cambio), `scores`
(distribución sobre las 15 clases finas, incluida `Background`) y
`category_scores` (distribución agregada por categoría).

Los eventos se persisten vía `../events/store.py`, igual que en `vision/`.

## Decisión de modelo: CLIP (OpenAI, ViT-B/32)

Se eligió CLIP en lugar de un VLM generativo (LLaVA, Moondream, etc.) por las
siguientes razones:

- **Velocidad**: corre bien en CPU sin cuantización; ViT-B/32 es el punto de
  entrada más liviano. Para mayor precisión, `--model ViT-L/14`.
- **Zero-shot sin fine-tuning**: los prompts en lenguaje natural
  ("a photo of a person reading a book") permiten clasificar directamente las
  14 actividades del dataset Toyota Smarthome sin ajuste adicional.
- **Sin servidor externo**: se descarga al primer uso y queda cacheado
  localmente (`~/.cache/clip/`), cumpliendo el requisito de privacidad.
- **Trade-off aceptado**: CLIP no genera texto libre — la `description` en
  `metadata` es el prompt textual que ganó la clasificación, no una descripción
  generada. Si en el futuro se necesita texto generativo, se puede cambiar el
  clasificador sin tocar `main.py`.

## Precisión y evaluación (`evaluate_activity.py`)

Los videos llevan la actividad real como prefijo del nombre de archivo
(`Walk_...`, `Laydown_...`), lo que permite medir la clasificación
cuantitativamente. Desde la raíz del proyecto:

```bash
python -m vlm.evaluate_activity                 # usa ../videos/ (39 videos, dataset chico)
python -m vlm.evaluate_activity <carpeta>        # dataset externo más grande
```

Resultado inicial (2026-07-16, 39 videos, 112 ventanas de 3 muestras), antes
de las mejoras descriptas más abajo:

| Enfoque | Ventanas correctas |
|---|---|
| Frame completo, 1 muestra por ventana (enfoque original) | 25.0% |
| Recorte a la persona + promedio de 3 muestras | 40.2% |
| + pad-a-cuadrado antes del resize (clase fina) | 45.5% |
| + agregación a categorías gruesas (zero-shot) | 65.2% |

Las dos primeras mejoras, con costo cero en runtime del Pi:

1. **Recorte a la persona** (`crop_to_person` en `src/clip_classifier.py`,
   reutiliza la detección de pose de `vision/`): elimina el sesgo del fondo
   fijo — con el frame completo, una TV visible empuja todo hacia `WatchTV`.
2. **Pad-a-cuadrado** (+5.3 pts): el resize directo a 224×224 deformaba la
   imagen; un recorte de persona parada (~1:3) quedaba aplastado e
   irreconocible. Ahora se rellena a cuadrado con el color medio de CLIP
   (neutro tras normalizar) antes del resize.

También hay una clase negativa `Background` ("an empty room...") que absorbe
recortes sin persona reconocible en vez de forzar una actividad con
confianza baja; el runner descarta esas ventanas.

### El 65.2% no generalizaba: evaluación contra el dataset completo

El 65.2% se midió sobre 39 videos curados a mano, todos de pocos
participantes. Al evaluar la misma agregación zero-shot contra una muestra
grande y disjunta del dataset Toyota Smarthome completo (no versionado en
git, ver "Escalar la evaluación" abajo), separando por participante para que
train/val no compartan sujeto, la precisión cayó a **39.6-48.3%** según
cuántos videos se usaran — el número original estaba sobreajustado a un
dataset chico, no reflejaba generalización real.

El diagnóstico por clase mostró un patrón consistente, no ruido aleatorio:
`Takepills`, `Drink.Fromcan`/`Drink.Fromcup` y `Maketea.Insertteabag` fallan
casi al 0% (CLIP las confunde con `Walk`, `WatchTV` o entre sí), mientras que
`Walk`, `Uselaptop`, `WatchTV`, `Usetablet` y `Laydown` sostienen bien. La
causa: son gestos de mano breves, indistinguibles desde un frame estático a
distancia de cámara de habitación — la misma limitación que señala la
literatura citada en el informe (Sec. 2.2.4: Reilly; Chakraborty et al. 2025,
LLAVIDAL — los VLMs generalistas sin especialización de dominio fallan en
actividades de baja movilidad). Se probó **CLIP ViT-B/16** (~4x cómputo) para
ver si un encoder más grande alcanzaba: dio 52.6% vs 51.6% de ViT-B/32 en el
mismo fold — mejora nula, confirmando que el problema es estructural
(falta de información en el frame), no de capacidad del modelo.

### Categorías fusionadas y linear probe

Dos cambios en conjunto, ambos con costo de inferencia idéntico al zero-shot
(mismo encoder ONNX, solo cambia qué se hace con el embedding):

1. **`Takepills`, `Drink.Fromcan`, `Drink.Fromcup` y `Maketea.Insertteabag`
   se fusionaron en una sola categoría** (`EatingDrinkingMedication` en
   `COARSE_CATEGORIES`, `src/clip_classifier.py`). `Takepills` tenía
   categoría propia por relevancia clínica, pero si nunca se predice bien no
   aporta esa señal igual — reportar una distinción que en la práctica es
   ruido es peor que no reportarla.
2. **Linear probe entrenado sobre los embeddings de CLIP** (congelados, sin
   fine-tuning del encoder) para clasificar directamente a nivel de
   categoría gruesa, en vez de zero-shot + agregación. Scripts:
   `extract_clip_features.py` (corre YOLO+CLIP sobre una carpeta externa y
   cachea embeddings a `.npz`) y `train_category_probe.py` (entrena
   logistic regression con split por participante, evalúa por video, y
   exporta `models/category_probe.npz`). `ClipActivityClassifier` lo carga
   automáticamente si el archivo existe — si no, cae al comportamiento
   zero-shot de siempre. El probe no tiene clase `Background` (no hay datos
   de "sin persona" para entrenarla), así que solo se usa cuando el
   zero-shot no cree que el recorte esté vacío.

Resultado, evaluado por video (participantes de validación nunca vistos en
entrenamiento, promedio de 5 splits distintos):

| Cobertura del dataset | Videos usados | Zero-shot | Linear probe |
|---|---|---|---|
| ~15% (`--max-per-class 40`) | 477 | 45.7% (±5.1) | 61.0% (±2.4) |
| ~50% (`--max-per-class 120`) | 1099 | 48.3% (±3.3) | **63.3%** (±2.2) |

Escalar de 15% a 50% del dataset dio solo +2.3 puntos — rendimientos
decrecientes, se frenó el escalado ahí en vez de seguir a costo creciente
por una ganancia cada vez más chica.

Por categoría (con el 50% del dataset, promedio 5 seeds):

| Categoría | n | Zero-shot | Linear probe |
|---|---|---|---|
| Cooking | 246 | 28.5% | **88.6%** |
| EatingDrinkingMedication | 359 | 15.0% | 39.3% |
| Moving | 480 | 68.8% | 67.5% |
| ReadingScreen | 266 | 77.1% | 68.8% |
| Resting | 204 | 45.1% | 56.9% |

`EatingDrinkingMedication` sigue siendo la más débil incluso fusionada y con
el probe (39.3%): el límite no era solo "demasiadas clases finas parecidas",
es que el embedding de CLIP es holístico (describe la escena completa, no
hace zoom a las manos) y esta categoría depende enteramente de qué hace la
persona con las manos. `Moving`/`ReadingScreen` bajaron un poco con el probe
respecto al zero-shot — dentro de margen, no una regresión real (confirmado
por el std bajo entre seeds).

**Trabajo futuro, no implementado:** la vía más prometedora para
`EatingDrinkingMedication` es sumar un crop de manos (usando los keypoints de
muñeca que `vision/pose_pipeline.py` ya calcula) como feature adicional al
embedding de cuerpo completo, en vez de seguir escalando datos o probando
modelos CLIP más grandes — ninguna de esas dos vías movió la aguja en esta
categoría específica.

### Escalar la evaluación a un dataset externo

`evaluate_activity.py`, `extract_clip_features.py` y `train_category_probe.py`
aceptan apuntar a cualquier carpeta con archivos `<Actividad>_pXX_...mp4`
(mismo patrón que Toyota Smarthome) fuera de `../videos/` — útil porque el
dataset curado del repo (39 videos) es demasiado chico para medir
generalización o entrenar nada. `evaluate_activity.py` además soporta
`--per-class N --seed S` para iterar tipo cross-validation sobre muestras
distintas de una carpeta grande. Estas carpetas externas no se versionan en
git (son de cada desarrollador), así que los resultados de esta sección no
son reproducibles solo con lo que hay en el repo — sí lo son los scripts y
los pesos entrenados (`models/category_probe.npz`, aunque tampoco se
versiona por ser un artefacto derivado, igual que el resto de `models/`).

## Setup

### En la máquina de desarrollo (una sola vez)

```bash
cd vlm/
pip install -r requirements-export.txt
python export_onnx.py
```

Genera `vlm/models/` con:
- `clip_image_encoder_int8.onnx` (~85 MB, INT8 cuantizado para ARM)
- `text_features.npy` (embeddings pre-computados, ~28 KB)
- `labels.json`

Si además se corrió `extract_clip_features.py` + `train_category_probe.py`
(ver "Precisión y evaluación" arriba), `vlm/models/` también tiene
`category_probe.npz` (pesos del linear probe, unos pocos KB) — se usa
automáticamente si está presente, si no `ClipActivityClassifier` cae al
zero-shot. Copiar `vlm/models/` completo al Pi antes de correr.

### En el Pi 5

```bash
cd vlm/
pip install -r requirements.txt   # solo onnxruntime + opencv, sin PyTorch
python main.py --video <archivo>.mp4
```

**Por qué esta separación:** PyTorch pesa ~1.5 GB instalado en ARM y solo se necesita para exportar el modelo. En el Pi solo se usa `onnxruntime`, que pesa ~30 MB y es ~2-4x más rápido en CPU ARM que PyTorch.

### Estimación de rendimiento en Pi 5

| Componente | Tiempo por inferencia | Cores usados |
|---|---|---|
| YOLO11n-pose (vision/) | ~150-300ms/frame | 2-4 |
| CLIP ONNX INT8 (vlm/) | ~300-600ms/frame | 2 (limitado) |

CLIP muestrea cada 10s (default), dejando el CPU libre para YOLO entre clasificaciones.

## Fuera de alcance por ahora

- Optimización para hardware embebido (Raspberry Pi).
- Usar un VLM vía API cloud (contradice el requisito de privacidad del
  informe — todo el procesamiento debe ser local).
