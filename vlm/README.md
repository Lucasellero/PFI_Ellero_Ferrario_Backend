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

### Vocabulario ampliado a las ~31 clases de Toyota Smarthome

El catálogo original de Toyota Smarthome tiene ~31 actividades; el código
solo definía prompts para 14, así que `extract_clip_features.py` y
`evaluate_activity.py` (que filtran por `label in LABEL_TO_CATEGORY`)
descartaban en silencio cualquier video con una etiqueta no cubierta
(`Getup`, `Leave`, `Usetelephone`, `Eat.Attable`, `Cook.Usestove`, `Pour.*`,
`Makecoffee.*`, `Maketea.Boilwater`, `Cook.Cleandishes`, `Cook.Cut`,
`Cutbread`, `Drink.Frombottle`, `Drink.Fromglass`, `Eat.Snack`), aunque
existieran en el dataset. Se agregaron prompts para las 17 clases
faltantes y se extendió `COARSE_CATEGORIES`:

- `Cooking` suma las variantes de preparación de comida sin bebida
  (`Cook.Cleandishes`, `Cook.Cut`, `Cook.Usestove`, `Cutbread`).
- `EatingDrinkingMedication` suma el resto de comer/beber/servir
  (`Drink.Frombottle`, `Drink.Fromglass`, `Eat.Attable`, `Eat.Snack`,
  `Maketea.Boilwater`, `Makecoffee.Pourgrains`, `Makecoffee.Pourwater`,
  `Pour.Frombottle`, `Pour.Fromcan`, `Pour.Fromkettle`) — mismo criterio que
  ya agrupaba `Maketea.Insertteabag` ahí, no en `Cooking`.
- `Moving` suma `Getup` y `Leave` (transiciones/locomoción, como `Sitdown`/`Enter`).
- Nueva categoría `Phone` con `Usetelephone` sola: a diferencia de
  `Takepills` (fusionado a priori en `EatingDrinkingMedication`), acá no se
  asume de entrada que sea indistinguible — se mide primero.

### Hand-crop: feature adicional para el linear probe

La vía identificada como más prometedora para `EatingDrinkingMedication` (la
categoría más floja, 39.3% con el probe) era sumar un crop de manos, en vez
de seguir escalando datos o probando modelos CLIP más grandes (ninguna de
esas dos vías había movido la aguja). Implementado en
`src/clip_classifier.py::crop_to_hands`: usa los keypoints de muñeca (índices
COCO 9 y 10) que `vision/src/pose_pipeline.py` ya calcula, recorta la zona
alrededor de las muñecas sobre el frame completo (no el crop de persona), y
lo codifica con el mismo encoder ONNX. El embedding resultante se concatena
al embedding de cuerpo completo (`[body, hand]`, 1024-dim) **solo como input
del linear probe** — el zero-shot con prompts de texto sigue siendo 512-dim,
sin cambios.

Cuando no hay un crop de manos válido (ninguna muñeca predicha, o llamadas
sin pose como `vlm/main.py` en modo batch), se usa el body embedding
duplicado como fallback, tanto en `extract_clip_features.py` (entrenamiento)
como en `ClipActivityClassifier.classify()` (inferencia) — mismo tratamiento
en ambos lados para no alimentar al probe con una distribución de input
distinta a la de entrenamiento.

### Resultado tras ampliar vocabulario + hand-crop

Extracción sobre el dataset completo (`D:\DESCARGAS\mp4`, `--max-per-class
120 --frames-per-video 5`): 8882 muestras, 1916 videos, las 31 clases.
Entrenamiento evaluado por video, participantes de validación nunca vistos
(promedio de 5 splits):

| | Zero-shot | Linear probe (cuerpo+manos) |
|---|---|---|
| Global (7 categorías) | 47.7% (±2.3) | 58.4% (±3.6) |

El número global no es comparable al 63.3% anterior: son 7 categorías en vez
de 6, con más subclases finas por categoría (más difícil por diseño, no una
regresión). El desglose por categoría es lo que importa:

| Categoría | n | Zero-shot | Linear probe | Linear probe (antes, 14 clases) |
|---|---|---|---|---|
| Cooking | 485 | 22.3% | 79.0% | 88.6% |
| EatingDrinkingMedication | 828 | 41.8% | **44.7%** | 39.3% |
| Moving | 831 | 74.0% | 61.5% | 67.5% |
| Phone (nueva) | 166 | 2.4% | 48.8% | — |
| ReadingScreen | 268 | 63.8% | 62.7% | 68.8% |
| Resting | 205 | 40.5% | 53.2% | 56.9% |

**El hand-crop movió la aguja donde tenía que hacerlo**:
`EatingDrinkingMedication` mejoró de 39.3% a 44.7% (+5.4 pts) a pesar de que
la categoría ahora tiene 14 subclases en vez de 4 (mucho más difícil por
composición) — es la primera mejora real en esta categoría desde que se
fusionaron las subclases, confirmando que la señal de manos ayudaba donde el
embedding holístico de cuerpo completo no alcanzaba.

**`Phone` resultó separable**: 48.8% con el probe (vs. 2.4% zero-shot, que
prácticamente nunca la predice) — muy por encima del 16.7% de azar entre 6
categorías, y sin el colapso a ~0% que motivó fusionar `Takepills` en su
momento. Se mantiene como categoría propia.

**`Cooking`, `Moving`, `ReadingScreen` y `Resting` bajaron con el probe**
respecto a la tabla anterior — pero cada una absorbió subclases nuevas y más
difíciles (`Cook.Cleandishes`/`Cook.Cut`/`Cook.Usestove`/`Cutbread` en
Cooking; `Getup`/`Leave` en Moving), así que la caída es el costo esperado
de ampliar el vocabulario, no una regresión del hand-crop — Moving y Cooking
seguían dominadas por zero-shot/probe con buen margen sobre el azar.

`EatingDrinkingMedication` seguía siendo la categoría más floja en términos
absolutos tras el hand-crop original (44.7%): recortar alrededor de la
muñeca no siempre aísla lo que la mano sostiene, y con cámara de habitación
a media distancia el objeto en la mano sigue siendo pequeño.

### Bug de coordenadas en `crop_to_hands` y filtro de confianza por-muñeca

Antes de seguir iterando sobre el 44.7% se encontró un **bug real** en
`crop_to_hands`: `detect_person` devuelve los keypoints en coordenadas de
**píxel** (`.xy` de ultralytics), pero la función los multiplicaba por
`w, h` como si fueran normalizados a [0, 1] — verificado corriendo el modelo
sobre un frame real y comparando `.xy` contra `.xyn`. El resultado: para
casi cualquier coordenada de píxel real (>1), el cálculo del recorte daba un
ancho negativo y la función devolvía `None` sistemáticamente. En la
práctica, **el hand-crop nunca se activó de verdad** en toda la extracción
anterior — siempre cayó al fallback de duplicar el embedding de cuerpo, así
que el 44.7% medido antes no reflejaba señal de manos real. Se corrigió
usando las coordenadas de píxel directamente, sin la multiplicación.

Aprovechando que había que retocar esta función, se sumó un **filtro de
confianza por-muñeca**: `detect_person` ahora también devuelve
`keypoints_conf` (confianza individual por keypoint, que YOLO-pose ya
calculaba pero se descartaba — antes solo se usaba el promedio de *todo* el
cuerpo para aceptar/rechazar el frame entero). Con la persona lejos o de
espaldas, la muñeca puede tener una posición predicha pero de baja
confianza (adivinada, no observada); `crop_to_hands` ahora descarta esas
muñecas igual que si no hubieran sido predichas, en vez de recortar sobre
una posición poco confiable.

Resultado tras re-extraer el dataset completo con ambas correcciones y
reentrenar:

| Categoría | Antes (44.7%, con el bug) | Después |
|---|---|---|
| **EatingDrinkingMedication** | 44.7% | **56.1%** |
| Cooking | 79.0% | 88.9% |
| Moving | 61.5% | 64.2% |
| Phone | 48.8% | 63.9% |
| ReadingScreen | 62.7% | 60.9% |
| Resting | 53.2% | 62.9% |
| Standing | 94.6%* | 97.3%* |

*Ver salvedad de generalización de `Standing` más abajo — no cambia con
esta ronda de cambios, sigue siendo la misma limitación.

El salto de +11.4 puntos en `EatingDrinkingMedication` confirma que el bug
era la causa principal del techo anterior: la señal de manos, cuando
realmente se calcula, ayuda.

**Se probó también centrar el recorte en muñeca+objeto sostenido** (un
segundo YOLO general sobre COCO, buscando taza/pastillero/cubiertos cerca de
la muñeca) — descartado antes de dejarlo en producción: el recorte de
persona en Toyota Smarthome tiene una mediana de solo ~101×233px, así que un
objeto sostenido ahí adentro mide 15-30px, y midiendo la tasa de activación
real sobre 120 videos, `find_held_object_box` encontraba un objeto en apenas
15.6% de los frames — no aportaba lo suficiente para justificar un tercer
modelo cargado en el Pi (sin medir en hardware real) ni la carga extra de
documentación en el informe. Se revirtió por completo: no queda código de
esto en el repo.

**En producción**: dado que 56.1% sigue siendo la categoría más floja y es
clínicamente sensible (medicación), `runner/camera_worker.py` sigue sin
reportarla con el mismo umbral que al resto — le exige una confianza más
alta (`_EATING_CONFIRM_THRESHOLD`) antes de afirmarla, y si no la alcanza
degrada a la segunda categoría más probable o a un evento con
`confirmed: false` (el dashboard lo muestra con un texto genérico en vez de
nombrarla con seguridad). Ver "Clasificación de actividades (CLIP)" en
`runner/README.md` para el detalle. Es un ajuste de umbral/presentación en
el runner, no un cambio al modelo entrenado.

### `Standing`: parado inactivo, vía OmniFall + CAUCAFall

Ninguna de las 31 clases de Toyota Smarthome cubre "persona parada sin hacer
nada" (`Getup` es la transición de incorporarse, no un estado). Se buscaron
datasets públicos de ADL en entornos domésticos que la cubrieran y se
descartó bajar uno nuevo: se usa
[OmniFall](https://huggingface.co/datasets/simplexsigil2/omnifall) (CC
BY-SA 4.0, 2025), que no redistribuye video propio sino **anotaciones por
segmento armonizadas a una taxonomía de 16 clases de postura** sobre 8
datasets de laboratorio ya existentes — dos de ellos, `CAUCAFall` y
`GMDCSA24`, ya están en `videos/` de este repo.

`labels/caucafall.csv` de OmniFall resuelve 1:1 contra los videos que ya
tenemos: el `path` de cada fila (ej. `adl/PickupobjectS3`) mapea a
`videos/CAUCAFall/CAUCAFall/Subject.{subject}/{Carpeta}/{Activity}S{subject}.avi`,
así que no hizo falta descargar nada — implementado en
`extract_omnifall_features.py`. `labels/GMDCSA24.csv` en cambio referencia
una estructura (`Subject_1/ADL/...`) que no está en nuestro
`videos/GMDCSA-24/` (solo tenemos el subset de caídas de ese dataset) — para
sumarlo habría que descargar el dataset completo; queda pendiente, no
implementado. Los otros 6 datasets que unifica OmniFall (`UP-Fall`,
`CMDFall`, `Le2i`, `EDF`, `OCCU`, `MCFD`) tampoco están en el repo — mismo
motivo.

De las 16 clases de OmniFall, sobre `caucafall.csv` (258 segmentos) solo dos
tenían volumen y aportaban algo nuevo:

- **`Standing`** (32 segmentos): nueva clase — mayormente los primeros 1-3s
  de los clips de caída, antes de que la persona caiga, más algunas pausas
  en `Walk`/`Pickupobject`/`Hop`. Sin volumen suficiente hubiera sido
  descartada (como se hizo con `Kneeling`/`Squatting`, 0 segmentos en este
  dataset pese a existir una carpeta `Kneel` — OmniFall etiqueta esos clips
  como `Other`/`StandUp`, no como postura de rodillas).
- **`StandUp`** (16 segmentos, del final de los clips de `Kneel`): mismo
  gesto que la clase `Getup` ya existente — se suman como refuerzo de esa
  clase en vez de crear una redundante (ver `OUTPUT_LABEL` en
  `extract_omnifall_features.py`).

El resto (`Walk`, `Fall`, `Fallen`, `Sitting`, etc.) ya está cubierto por
Toyota Smarthome o por el detector geométrico de `vision/`, así que no se
extrae. Nueva categoría en `COARSE_CATEGORIES`: `Standing`.

Resultado (probe reentrenado sobre Toyota Smarthome + estos 60 frames,
`train_category_probe.py --extra-features omnifall_features.npz`):

| Categoría | n | Zero-shot | Linear probe |
|---|---|---|---|
| Standing (nueva) | 37 | 0.0% | **94.6%** |

**Con salvedad importante**: ese 94.6% probablemente sobreestima cuánto
generaliza. Las muestras de `Standing` vienen todas de `CAUCAFall` (una
sola instalación de cámara, fondo y calibración consistentes en las 10
personas grabadas), mientras que el resto de las categorías mezcla
participantes y cámaras de Toyota Smarthome — es plausible que el probe esté
aprendiendo a distinguir "es un frame de CAUCAFall" más que "la persona está
parada", dado lo fácil que separa esta clase comparado con el resto. No hay
forma de descartar esto sin datos de `Standing` de otra fuente (por eso
`GMDCSA24` completo, con su propia instalación de cámara, sería el primer
candidato para validar esta clase de verdad antes de confiar en el número).
Las demás categorías fluctuaron un poco respecto a la tabla anterior
(`Moving` 61.5%→54.6%, `ReadingScreen` 62.7%→54.0%, `Resting` 53.2%→61.8%)
— es ruido del split aleatorio (agregar los 60 frames de CAUCAFall cambia el
orden/composición del array de videos únicos, así que el mismo `--seed` ya
no reproduce exactamente la misma partición train/val de antes), no un
efecto de esta clase nueva sobre las demás.

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

## Confirmación de caídas (`src/fall_confirmation.py`)

Además de clasificar actividades, `vlm/` expone un segundo uso de CLIP:
confirmar (o descartar) los eventos de caída que dispara el detector
geométrico de `vision/`. Ese detector (velocidad vertical + aspect ratio) no
distingue una caída real de sentarse/acostarse a propósito en algunos setups
de cámara — ver "Detección de caída como evento" en `vision/README.md` para
el detalle completo (por qué recalibrar los umbrales geométricos no alcanza,
y cómo se calibró este filtro). `FallConfirmationClassifier` reusa el mismo
encoder ONNX que `ClipActivityClassifier` (vía `encode_image()`, no carga un
segundo modelo) con un set de prompts distinto ("caída accidental" vs.
"postura voluntaria"). Lo integra `runner/camera_worker.py` — ver
"Confirmación de caídas con CLIP" en `runner/README.md`. `vision/main.py`
(modo batch) no lo usa: el detector geométrico ahí queda puro, para que
`vision/` se pueda calibrar/evaluar de forma independiente (ver límites de
módulo en el README raíz).

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
- `fall_confirmation_features.npy` (embeddings de texto para confirmación de
  caídas, ~4 KB — ver sección de arriba)

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
