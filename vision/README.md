# `vision/` — Pose estimation, clasificación de postura y detección de caídas/inmovilidad

## Resultados actuales (resumen, 2026-08-23)

> Estado vigente hoy en producción (`FallImmobilityDetector` + `FallConfirmationClassifier`
> con el probe entrenado). El detalle completo, con la historia de cómo se llegó a estos
> números, está más abajo — este resumen es solo para no tener que leer todo el documento.

- **Postura (parado/sentado/acostado)**, solo `aspect_ratio`: recall parado 90.0%,
  sentado 82.1%, acostado 86.6%; accuracy por video (voto mayoritario) 79.5%.
- **Detección geométrica de caídas + confirmación CLIP (probe entrenado)**: **84.8%
  recall / 30 FP** sobre 500 eventos out-of-fold (292 clips), dominando al filtro
  zero-shot (82.3% recall / 116 FP) en las dos métricas a la vez.
- **Filtro CLIP en validación end-to-end** (heurística + confirmación + tracking de
  inmovilidad, sobre ~80 min de video doméstico continuo real, los 4 clips más
  jodidos de `ToyotaSmarthome_FP`): **elimina 269 de 270 falsos positivos geométricos
  (99.6%)**. Solo sobrevive 1 evento, confirmado con margen holgado.
- **Inmovilidad/sedentarismo**: bug de historial corregido (`9356c15`, la ventana de
  2.5s de velocidad de caída se compartía indebidamente con el tracking de
  inmovilidad; ahora tiene ventana propia de 15s). Pendiente, no bug: en 3 de esos 4
  clips largos la inmovilidad no se dispara porque la persona nunca está 180s
  realmente quieta en el clip completo (más actividad real que en el tramo corto
  usado para calibrar).
- **Bloqueante identificado, no resuelto**: `yolo11n-pose` no detecta bien a una
  persona acostada en sillón visto desde cámara ancha/elevada (94.5% de frames sin
  ninguna detección) — limita la rama "acostado" de inmovilidad en ese ángulo de
  cámara específico. Es un límite del modelo de pose, no de la lógica del detector. Desde
  2026-10-18, la histéresis de detección recupera parte de esos frames para inmovilidad
  (ver "Actualización 2026-10-18").

## Actualización 2026-09-01 — guardas de encuadre + clasificador temporal de caída

Iteración para subir precisión (feedback post-presentación). CPU-only en la Pi 5.
Dos cambios, ambos **encima** de la heurística geométrica (que no se toca):

1. **Guardas de encuadre (Fase 0, en este módulo).** Cada evento `fall` lleva
   `evidence: "full" | "partial"` según `PoseSample.framing_ok` (bbox no toca el
   borde superior/izq/der **y** hombros+cadera visibles con confianza). Un evento
   `partial` — la persona parcialmente fuera de cuadro, donde el aspect ratio y
   `hip_y` son poco confiables — **no se descarta ni se confirma con CLIP** (recorte
   truncado): se emite como `severity: "sospecha"` y `_track_postfall` lo escala a
   `severity: "alta"` (`trigger: "post_fall_sospecha"`) si la persona queda inmóvil
   `IMMOBILITY_WINDOW_POSTFALL_S` = 45 s. Alguien que pasa caminando por el borde
   sigue moviéndose y nunca escala. En `evaluate_falls.py` los eventos `partial`
   **recuperan +10–12 caídas reales** de "solo se ve el torso" que la geometría
   "full" perdía. Detalle en "Detección de caída como evento" abajo.

2. **Clasificador temporal de caída (vive en `fusion/`, ver `fusion/README.md`).**
   Reemplaza la *decisión* de caída basada en umbrales por un GRU entrenado sobre la
   ventana de keypoints (augmentación de oclusión + hard-negative mining). Es la 2ª
   etapa del evento en `runner/camera_worker.py::_confirm_fall_event`, en reemplazo
   de la confirmación CLIP (CLIP queda de fallback para eventos sin ventana de
   esqueleto). `evaluate_falls.py --temporal` lo corre como 2ª etapa.

**Resultado end-to-end** (`evaluate_falls.py --temporal --fps-cap 10`, 5 datasets,
158 caídas / 292 clips): recall **133/158 (84.2 %) — igual que geométrico solo** —
y **FP de alerta dura 185 → 91**. Donde hay ventana de esqueleto rechaza **94/94 FP
(100 %)** y pierde **1/113 caídas**; los ~74 FP residuales de `ToyotaSmarthome_FP`
son eventos **sin ventana** (el bloqueante de `yolo11n-pose` de arriba — no hay
esqueleto que clasificar) y en producción caen a CLIP. El detector geométrico y su
calibración (`FALL_VERTICAL_VELOCITY=0.12`, `FALL_ASPECT_RATIO=1.4`) **no cambian**.

## Actualización 2026-10-13 — la cadena completa, medida de punta a punta

`evaluate_falls.py --temporal --clip --fps-cap 10`. Hasta ahora el script corría la heurística
y, con `--temporal`, el clasificador temporal; los eventos sin ventana de esqueleto se contaban
como alerta con la nota "en producción caen a CLIP". Con `--clip` (nuevo) se aplica la misma
decisión que `runner/camera_worker.py::_confirm_fall_event`: evidencia parcial → sospecha;
evidencia completa → el temporal confirma (≥ 0,45) o descarta (≤ 0,15), y si queda en el medio
o no hubo ventana decide el probe de CLIP (`vlm/models/fall_confirmation_probe.npz`). El script
cuenta además las sospechas, las que escalan a alerta por inmovilidad
(`trigger: post_fall_sospecha`) y las horas de video. No se cambió ningún umbral.

Los mismos 292 clips de siempre (5 datasets, 158 con caída anotada).

| | |
|---|---|
| Caídas detectadas con alerta | **120 de 158 (75,9 %)** |
| Caídas que quedan solo como sospecha (evidencia parcial) | 12 de 158 (7,6 %) |
| Caídas sin ningún evento | 26 de 158 (16,5 %) |
| Caídas con algún evento (alerta o sospecha) | 132 de 158 (83,5 %) |
| **Falsas alertas finales** | **9** (las 9 de evidencia completa; ninguna sospecha escaló) |
| Sospechas | 94: 19 dentro de una caída anotada, 71 en tramos sin caída, 4 fuera de los tramos anotados |
| Sospechas que escalaron a alerta por inmovilidad | 0 |
| Horas de video | 2,08 h (1,88 h anotadas como sin caída) |
| **Falsas alertas por hora de video** | **4,33** |
| Latencia promedio | 1,92 s |

Una sospecha no es una alerta: se guarda con `confirmed: false` y no se avisa
(`cloud/app/alertas.py`); solo avisa si escala.

Por dataset:

| Dataset | Horas | Falsas alertas | Por hora | Sospechas en tramos sin caída |
|---|---|---|---|---|
| CAUCAFall (100 clips) | 0,28 | 6 | ≈ 21 | 0 |
| GMDCSA-24 (79 clips) | 0,15 | 0 | 0 | 0 |
| URFD (70 clips) | 0,11 | 2 | ≈ 18 | 7 |
| ToyotaSmarthome (39 clips cortos, sin caídas) | 0,14 | 1 | ≈ 7 | 0 |
| ToyotaSmarthome_FP (4 videos continuos, sin caídas) | 1,40 | **0** | **0** | 64 (≈ 46 por hora) |

**El 4,33 por hora no es la tasa esperable en una casa.** Dos tercios del video (1,40 h) son
los 4 videos domésticos continuos, y ahí hay 0 falsas alertas. Las 9 salen de 0,68 h de clips
cortos de los datasets de caídas, que son actividades actuadas una atrás de otra (sentarse,
agacharse, arrodillarse, acostarse): mucho más densas que la vida real. Lo que sí queda de los
videos continuos es la cantidad de **sospechas: 64 en 1,40 h**. No avisan, pero se guardan, y
ninguna escaló.

Qué hizo cada etapa con los 432 eventos de la heurística:

| | Eventos | En una caída anotada | En un tramo sin caída | Fuera de los tramos |
|---|---|---|---|---|
| Evidencia parcial → sospecha | 94 | 19 | 71 | 4 |
| Evidencia completa, con ventana de esqueleto | 213 | | | |
| — el temporal confirma | 117 | 112 | 0 | 5 |
| — el temporal descarta | 94 | 0 | 94 | 0 |
| — indeciso → CLIP | 2 | 1 | 0 | 1 |
| Evidencia completa, sin ventana → CLIP | 125 | 34 | 91 | 0 |
| CLIP: de los 127 que le llegan, confirma / descarta | 40 / 87 | | | |

De los 91 eventos sin ventana en tramos sin caída (74 de `ToyotaSmarthome_FP`), CLIP dejó pasar
9. Contra la medición del 2026-09-01 (solo temporal): las falsas alertas de evidencia completa
bajan de 91 a 9 y las caídas con algún evento pasan de 133 a 132.

**Salvedades:**
- **Las dos segundas etapas se entrenaron con estos mismos clips.** El clasificador temporal y
  el probe de CLIP que están desplegados se entrenaron sobre todos los eventos de estos 292
  clips (cada script evalúa fuera de muestra y al final reentrena con todo), incluidos los 4
  videos de `ToyotaSmarthome_FP` como negativos difíciles. Esta medición es la del sistema
  desplegado sobre datos que ya vio: es optimista. Las cifras fuera de muestra de cada etapa
  por separado siguen siendo las de arriba (probe de CLIP: 84,8 % / 30 FP; temporal: 96–99 %
  de las caídas con ventana). No hay un dataset de caídas apartado para medir la cadena entera.
- `--fps-cap 10`, como la corrida del 2026-09-01. En producción los frames que se analizan
  dependen del hardware.
- No se simula el gate de movimiento del runner ni el paso por la nube.
- "Caída detectada" es por clip: algún evento dentro del tramo anotado.
- `Kaggle` sigue fuera (su `Fall.csv` está pendiente).

## Actualización 2026-10-18 — histéresis de detección; `yolo11s-pose` no se adopta

### Histéresis de detección

Ataca el bloqueante de arriba (persona acostada en un sillón vista desde cámara ancha/elevada:
YOLO la pierde buena parte del tiempo). Bajar el umbral global de caja la recupera, pero
multiplica las cajas espurias. `FallImmobilityDetector._detect` usa en cambio dos umbrales:

- una detección normal (caja ≥ `PERSON_BOX_CONFIDENCE` = 0,25, como siempre) con caja ≥
  `TRACK_CREATE_BOX_CONFIDENCE` = 0,5 fija un **ancla**;
- sin detección normal en el frame, una caja **débil** (≥ `TRACK_KEEP_BOX_CONFIDENCE` = 0,05)
  mantiene el seguimiento solo si solapa con el ancla (IoU ≥ 0,5), hasta 3 s sin ninguna caja
  aceptada y como mucho 60 s desde la última vez que se fijó el ancla.

Las detecciones débiles alimentan **solo** el tracking de inmovilidad (con la tasa de
desplazamiento medida por caja, que puede sostener la cuenta de quietud pero no resetearla) y
`person_present`, que usa el gate de movimiento del runner para no dormir la cámara con alguien
acostado. Nunca entran a la heurística de caída: `last_detection` sigue en `None`.

`sweep_detection_hysteresis.py --model <modelo> --ref yolo11m-pose.pt`, sobre 32 clips de
`acostado`/`acostarse` (2.100 frames a 5 fps) y, como control de cajas espurias, 40 de `mirar
tele` y 60 de `caminar`. "Coincide con la referencia" = la caja débil aceptada solapa (IoU ≥ 0,3)
con donde `yolo11m-pose` ve a la persona. "Fantasma" = cuánto sobrevive el seguimiento si la
persona desaparece y solo quedan cajas débiles de otro clip (400 pruebas).

| `yolo11n-pose` | Frames con persona (acostado) | Cajas espurias cada 1000 frames (tele / caminar) |
|---|---|---|
| Umbral global 0,25 (sin cambios) | 44,8 % de frames con alguna caja; 41,7 % que pasan el filtro de keypoints | 34,6 / 55,7 |
| Umbral global 0,05 | 92,8 % | 236,4 / 376,5 |
| **Histéresis integrada** | **41,7 % + 15,8 % débil = 57,5 %** | ninguna nueva (solo acepta cajas sobre el ancla) |

Con la histéresis integrada, el 89 % de las cajas débiles verificables coinciden con la
referencia (93/104), y el seguimiento fantasma dura como máximo 29,2 s (más de 5 s en el 0,5 %
de las pruebas). Variantes que aceptan más frames (IoU 0,3, o 15 s de hueco) dejan seguimientos
fantasma de hasta ~70 s. La primera versión (crear el ancla con 0,25 y encadenar débil con
débil) llegaba a 82 % de frames, pero en el clip más largo era un seguimiento colgado de una
planta.

Con `yolo11s-pose` la histéresis casi no suma: 43,4 % → 47,5 % (75 % de coincidencia, fantasma
máximo 4,8 s), porque ese modelo, o detecta a la persona, o no le da ni una caja débil.

**Efecto sobre caídas: ninguno**, como se buscaba. `evaluate_falls.py --temporal --clip --fps-cap
10` con la histéresis da exactamente los números de la actualización 2026-10-13 (120/158 con
alerta, 9 falsas alertas). El efecto sobre la inmovilidad de punta a punta (el bloqueante de los 3
de 4 clips de `ToyotaSmarthome_FP`) **no está medido**.

### `yolo11s-pose` + tolerancia de huecos + umbral de CLIP: se midió y empeora

Con `yolo11s-pose`, dos cambios en `runner/camera_worker.py` bajaban las falsas alarmas sobre
clips cortos de acostarse/levantarse (12 → 7): que el buffer de esqueleto tolere huecos de
detección de hasta 0,6 s (`_SKELETON_GAP_TOLERANCE_S`) y exigir logit ≥ 0,3 al probe de CLIP
(`_FALL_CLIP_MIN_PROBE_LOGIT`). En la cadena completa, sobre los mismos 292 clips de arriba
(`evaluate_falls.py --temporal --clip --fps-cap 10 --pose-model yolo11s-pose.pt
--gap-tolerance-s 0.6 --clip-min-logit 0.3`):

| | Vigente (`yolo11n`) | `yolo11s` + 0,6 s + 0,3 |
|---|---|---|
| Caídas con alerta | 120 de 158 | 121 de 158 |
| Solo como sospecha | 12 | 8 |
| Sin ningún evento | 26 | 29 |
| **Falsas alertas finales** | **9** | **27** |
| — en `ToyotaSmarthome_FP` (1,40 h continuas) | **0** | **17** |
| — CAUCAFall / URFD / ToyotaSmarthome | 6 / 2 / 1 | 5 / 3 / 2 |

Detecta lo mismo y triplica las falsas alertas, casi todas en los videos domésticos continuos,
que son los más parecidos a producción. **No se adopta**: `cameras.yaml` sigue con
`yolo11n-pose.pt` y las dos constantes quedan en 0 (el comportamiento de siempre). Con
`yolo11n-pose`, la tolerancia sola tampoco servía (14 → 17 sobre los clips de acostarse).

Causa probable, sin verificar: el clasificador temporal y el probe de CLIP se entrenaron con
esqueletos y eventos de `yolo11n` (incluidos estos 4 videos como negativos), así que con
`yolo11s` reciben entradas distintas de las que conocen. Para volver a probar `yolo11s` habría
que re-entrenar las dos etapas con su salida (`fusion/extract_falls.py --pose-model
yolo11s-pose.pt`) y repetir esta medición.

`evaluate_falls.py` suma `--pose-model`, `--gap-tolerance-s` y `--clip-min-logit` para replicar
esas opciones del runner; sin ellas corre igual que antes.

## Estructura

```
vision/
├── main.py               # pipeline de eventos (fall/immobility) → events/store.py
├── src/
│   ├── capture.py          # lee frames de un archivo de video (VideoFile)
│   ├── pose_pipeline.py    # YOLOv11n-pose + descriptores (PoseSample) + FallImmobilityDetector
│   └── posture.py          # classify_posture(): parado / sentado / acostado a partir de un PoseSample
├── evaluate_falls.py                    # recall/FP contra Fall.csv (dev, no productivo)
├── calibrate_falls.py                   # calibración original (GMDCSA-24), ver historia abajo
├── recalibrate_fall_thresholds.py       # grid search de umbrales con cache de descriptores
├── extend_descriptor_cache.py           # suma clips nuevos a fall_descriptors_cache.pkl sin recomputar todo
├── calibrate_fall_confirmation.py       # prototipo/calibración del filtro CLIP de confirmación (dev, ver vlm/src/fall_confirmation.py)
├── extract_fall_confirmation_embeddings.py  # cachea embeddings ONNX por evento geométrico, para entrenar un linear probe (experimento en curso, ver abajo)
├── train_fall_confirmation_probe.py     # entrena el linear probe sobre ese cache, con GroupKFold por video (experimento en curso, ver abajo)
├── build_caucafall_groundtruth.py       # genera videos/CAUCAFall/Fall.csv desde sus etiquetas por frame
└── build_kaggle_multicam_groundtruth.py # genera videos/Kaggle/Fall.csv desde kaggle_data.csv
```

`pose_pipeline.py` es la base compartida: expone `detect_person()` (corre
YOLO sobre un frame) y `build_pose_sample()` (calcula los descriptores
geométricos: aspect ratio, ángulo de torso, posición de cadera). Tanto
`FallImmobilityDetector` (eventos) como `posture.classify_posture()`
(clasificación de postura) se construyen sobre esos mismos descriptores,
sin duplicar la lógica de keypoints.

## Qué hace — dos cosas distintas, en dos etapas

### 1. Clasificación de postura (parado / sentado / acostado) — calibrada

`src/posture.py` clasifica cada frame usando **solo `aspect_ratio`** del
bounding box, con dos cortes:

```
aspect_ratio > 1.13  → acostado
aspect_ratio > 0.58  → sentado
si no                → parado
```

**Cómo se llegó a estos números, y una corrección importante en el
camino:**

1. Se corrió el pipeline sobre 39 videos cortos, organizados en su momento
   en subcarpetas `acostado/`/`parado/`/`sentado/` clasificadas a mano por
   un conocido, usadas como ground truth para medir accuracy.
2. Un primer clasificador combinando `aspect_ratio` + ángulo de rodilla
   daba 80.2% de accuracy, pero el ángulo de rodilla resultó muy ruidoso al
   estar sentado (piernas ocluidas por mesa/sillón, YOLO estima mal esos
   keypoints) — se descartó.
3. Un clasificador de dos cortes usando solo `aspect_ratio`, optimizado por
   *accuracy total*, llegó a 92.7% pero sacrificaba la clase minoritaria
   "parado" (recall 24.6%) porque "sentado" era el 86% del dataset.
   Optimizando por **recall promedio entre clases** en cambio se llegó a
   los umbrales actuales (recall parado=90.0%, sentado=82.1%,
   acostado=86.6%).
4. Con esos umbrales, la accuracy por video (voto mayoritario) daba 79.5%
   (31/39) — se investigaron los 8 videos "mal clasificados" extrayendo un
   frame de cada uno y mirándolo, y **ninguno era un error real del
   modelo**: varios clips etiquetados "sentado" (`Cook.Cleanup`,
   `Drink.Fromcan`, `Maketea.Insertteabag`) mostraban a la persona **parada**
   cocinando — la carpeta agrupaba por tipo de actividad doméstica, no por
   postura real. Otros (`Sitdown`, `Takepills`, `Laydown`) capturaban la
   transición entre posturas, no una postura estable.

**Conclusión:** la clasificación manual por carpeta no era ground truth
confiable de postura. Se eliminó esa estructura de `videos/` (ver
`../videos/README.md`) — de acá en adelante, la referencia para evaluar
este pipeline es el modelo y su matriz de confusión sobre casos verificados
a mano, no una categorización previa sin validar. Los umbrales de
`posture.py` quedan como están (el proceso de calibración fue válido, el
problema era solo la métrica de evaluación posterior); si hace falta
recalibrar con un dataset más confiable, repetir este proceso con
etiquetas verificadas.

### 2. Detección de caída como evento — calibrada

`src/pose_pipeline.py` (`FallImmobilityDetector`) detecta caídas partiendo
del enfoque de Giri et al. (2025) citado en el informe (Sec. 2.2.7):
velocidad vertical de la cadera + aspect ratio del bounding box.

**Cómo se llegó a los umbrales actuales, y una corrección importante en el
camino:**

1. Se armaron cuatro datasets de clips de caída reales, `videos/fall1/`
   a `videos/fall4/` (16+25+21+17 = 79 clips en total, distintos tipos:
   lateral/adelante/atrás, desde parado/sentado/caminando). Script de
   calibración: `calibrate_falls.py` (loggea los descriptores frame a frame
   a un CSV, no es parte del pipeline productivo).
2. **Primer intento** (umbrales originales, tomados directo de la
   literatura): exigir que velocidad vertical, aspect ratio *y* ángulo de
   torso superen sus umbrales **en el mismo frame** — 0/16 eventos
   detectados. Investigando el porqué se encontraron dos bugs, no solo un
   problema de umbrales:
   - La velocidad vertical se calculaba contra la muestra *más vieja* del
     historial (hasta 5s atrás), no contra una muestra reciente — eso
     diluye cualquier pico de velocidad real casi a cero.
   - Aún corrigiendo eso, exigir las tres condiciones en el mismo frame es
     físicamente incorrecto: la velocidad pica **durante** la caída
     (~0.3-1s) pero el aspect ratio "acostado" recién se estabiliza
     **después**, cuando el cuerpo termina de asentarse en el piso. Son dos
     fases del mismo evento, no un instante único.
3. **Diseño final**, calibrado por barrido de umbrales sobre los 41 clips:
   detección en dos fases — (1) un pico de velocidad vertical > 0.25 (en
   ventana de 0.4s) marca una posible caída en curso; (2) si dentro de los
   1.5s siguientes el aspect ratio supera 1.1 (postura final acostada, muy
   cerca del umbral sentado→acostado de `posture.py` = 1.13, tiene sentido
   físico), se confirma el evento — con cooldown de 3s para no reportar la
   misma caída varias veces mientras el cuerpo termina de moverse en el
   piso. Resultado: **41/41 clips detectados**, 45 eventos totales (algunos
   clips más largos tienen más de una caída real).
4. El ángulo de torso se sacó de la condición: en caídas laterales no llega
   a valores altos, y exigirlo hacía perder ~2/16 casos sin mejorar la
   precisión de los que sí detectaba. Se sigue guardando en `metadata` del
   evento como dato de contexto, solo que ya no filtra la detección.

**Evaluación con ground truth real:** `fall3/` y `fall4/` (y versiones
corregidas de `fall1/`/`fall2/`) vienen con un `Fall.csv` por carpeta —
anotado a mano por quien armó el dataset, con el segundo exacto de inicio/
fin de cada caída y de las actividades previas (`Sitting[0 to 3.4]`,
`Walking[0 to 1.8]`, etc.). Importante: **este CSV es solo una herramienta
de evaluación en desarrollo, no algo que el sistema tenga en producción**
— en el entorno real el detector nunca sabe de antemano cuándo va a caer
alguien, es justo lo que tiene que predecir solo. `evaluate_falls.py` usa
estos CSV para medir contra el timestamp real (no solo "¿detectó algo en
algún momento del clip?") y para detectar falsos positivos dentro de los
tramos ya etiquetados como no-caída del mismo clip. Sobre 78 casos
evaluables (1 quedó sin rango de tiempo parseable en el CSV original), con
los umbrales originales:

- Recall: 91% (71/78)
- Falsos positivos: 1/78 (dentro de tramos explícitamente no-caída)
- Latencia promedio: 1.2s desde el inicio real de la caída hasta el evento

De los 7 casos no detectados: 5 son caídas **hacia la cama** estando ya
sentado en el borde (la cama está a una altura similar a la posición final
acostada, así que no hay el descenso brusco de cadera que sí hay al caer al
piso — la premisa física de `vertical_velocity` no aplica igual). 1 es una
oclusión real (la persona se saca la remera y la cuelga en un perchero
antes de caer, YOLO pierde el tracking de forma intermitente). 1 es un
artefacto del cooldown: un evento disparado antes del inicio real de la
caída consumió los 3s de cooldown y tapó la detección correcta que venía
después. No se fuerza más la calibración para cazar estos casos — el
riesgo es meter falsos positivos nuevos (confundir acostarse/sentarse
voluntario con una caída es, según la literatura — ver `vlm/README.md` y
el paper de Benabdennour et al. 2026 abajo —, el modo de falla más común en
todo el campo, no algo específico de esta heurística).

**Recalibración multi-dataset (bug de distancia de cámara):** el sistema
está pensado para funcionar como app de **una sola cámara en un hogar**, con
setups (ángulo/distancia) que varían de instalación en instalación — no una
cámara fija conocida de antemano. Al evaluar los umbrales de arriba sobre
`videos/CAUCAFall/` (100 clips de otro dataset, con distancia de cámara
variable por sujeto — ver `build_caucafall_groundtruth.py`), el recall cayó
a 28% y los falsos positivos subieron a 19/50. La causa: `hip_y` en
`build_pose_sample()` se normalizaba contra el **alto del frame**, no contra
el tamaño de la persona — el mismo movimiento real de cadera da una
velocidad calculada distinta según qué tan lejos esté la cámara (más lejos
→ se subestima → se pierden caídas; más cerca → se sobreestima → dispara
con movimientos normales).

Fix: `_vertical_velocity()` ahora normaliza por el alto del bbox de la
persona (`bbox_height_frac`), no por el alto del frame — invariante a
distancia de cámara. Con eso, `recalibrate_fall_thresholds.py` recalibró
`FALL_VERTICAL_VELOCITY` y `FALL_ASPECT_RATIO` por grid search sobre 179
clips (GMDCSA-24 completo + CAUCAFall completo, incluyendo los 51 clips sin
ninguna caída — antes ni se evaluaban como fuente de FP, ver fix en
`evaluate_falls.py`). El script cachea los descriptores por frame en
`fall_descriptors_cache.pkl` para poder barrear ~70 combinaciones de
umbrales sin correr YOLO de nuevo por cada una.

Resultado con los umbrales nuevos (`FALL_VERTICAL_VELOCITY=0.12`,
`FALL_ASPECT_RATIO=1.4`, en unidades de "alturas de persona/segundo"):

| Dataset | Recall antes | Recall después | FP antes | FP después |
|---|---|---|---|---|
| GMDCSA-24 (78 casos) | 91% | **96.2%** (75/78) | 1/78 | 2/79 |
| CAUCAFall (50 casos) | 28% | **68.0%** (34/50) | 19/50 | 16/100 |
| Combinado (128 casos) | 66.4% | **85.2%** (109/128) | 20 | 18/179 |

Mejoró en ambos datasets a la vez (no fue trade-off): la causa real era el
bug de normalización, no una tensión inherente entre "andar bien en un
setup" y "generalizar a otros". Latencia promedio subió de 1.2s a 2.3s (el
aspect ratio más exigente tarda un poco más en confirmarse), aceptable para
este caso de uso.

**Dataset pendiente, no incluido todavía en la calibración:** `videos/Kaggle/`
(Multiple Cameras Fall Dataset, Auvinet et al. — 24 escenas grabadas
simultáneamente con 8 cámaras en una habitación amoblada como living/casa,
no institucional). Ground truth en `kaggle_data.csv` (frame de inicio/fin
por cámara, `build_kaggle_multicam_groundtruth.py` lo convierte al formato
`Fall.csv`).

Se probó con chute01 (8 cámaras, un solo evento de caída) y dio recall
2/8 — mucho peor que GMDCSA-24/CAUCAFall, pero **no se confía en ese
número**: la ventana de caída anotada da ~0.25s (contra >1s típico en los
otros datasets) al convertir frame→segundo con el FPS que reporta el propio
`.avi` (120). No se pudo confirmar si ese FPS es real o si el contenedor
quedó mal etiquetado en el re-upload de Kaggle (se buscaron frames
duplicados como evidencia de un FPS inflado 4x y no aparecieron, así que
tampoco se puede descartar sin más investigación). Sumado a que estos
`.avi` tienen bastante corrupción de frames (warnings de FFmpeg tipo
`ac-tex damaged`) y que son mucho más largos que los de los otros dos
datasets (hasta ~4200 frames vs. ~100-300, carísimo correr YOLO-pose frame
a frame en CPU sobre las 24 escenas × 8 cámaras completas), se decidió
**dejarlo afuera de la calibración por ahora** en vez de confiar en un
timing dudoso. `videos/Kaggle/Fall.csv.pending` guarda lo ya generado (para
no recalcularlo) pero no se llama `Fall.csv` a propósito — así
`evaluate_falls.py`/`recalibrate_fall_thresholds.py`, que buscan
`videos/**/Fall.csv` recursivamente, no lo mezclan sin querer en las
métricas agregadas. Para retomarlo: confirmar el FPS real (ej. contando
frames en un tramo de duración conocida por otro medio) antes de volver a
generar el `Fall.csv` real.

**Trabajo futuro (no implementado):** para resolver justamente la
ambigüedad "cayó vs. se acostó/sentó voluntariamente" — el modo de falla
que domina tanto nuestros 7 casos perdidos como la literatura — se podría
combinar esta detección geométrica con `vlm/` (CLIP): cuando YOLO dispara
un evento de caída, pedirle a CLIP zero-shot una clasificación puntual de
ese frame con prompts tipo `"a person having an accidental fall"` vs.
`"a person intentionally lying down"` / `"a person sitting down calmly"`,
como filtro de confirmación (no reemplazo). CLIP no necesita entrenamiento
para esto — es zero-shot, solo hace falta agregar y calibrar los prompts
nuevos, igual que se hizo con las 14 actividades actuales. El costo
(~300-600ms de latencia en el Pi) es aceptable porque solo correría cuando
YOLO ya disparó, no en cada frame.

Referencia externa relevante para esta línea de trabajo: A. Benabdennour
et al., "Real-Time Human Fall Detection From Video Using YOLOv11 With Pose
Estimation," IEEE Access, vol. 14, 2026 — combinan YOLOv11-Pose con un
Transformer entrenado sobre ~7.000 videos (97.7% F1). Confirma varias
decisiones de acá (la señal principal es la caída vertical de la cadera,
ventana temporal de ~2s) y aporta dos ideas aplicables sin entrenar nada:
filtrar detecciones con confianza de keypoints baja, y elegir la persona
del bounding box más grande si hay más de una en el frame — ambas ya
incorporadas a `detect_person()`.

**Actualización 2026-09-01 — el "trabajo futuro" de arriba está hecho, en dos
partes:**

*Parte 1 — guardas de encuadre (Fase 0, este módulo).* `build_pose_sample()`
calcula por frame `bbox_touches_edge` (el bbox llega a `FRAME_EDGE_MARGIN_FRAC`
del borde superior/izq/der — el inferior no cuenta, los pies de alguien parado lo
tocan normalmente) y `torso_visible` (hombros+cadera COCO 5,6,11,12 con confianza
≥ `TORSO_KEYPOINT_MIN_CONF`). `PoseSample.framing_ok` combina ambos. Cuando
`_detect_fall` dispara, el evento lleva `metadata.evidence`: `"full"` si
`framing_ok`, `"partial"` si no. Un evento `partial` es una caída donde la firma
geométrica es poco confiable (persona truncada contra el borde) — **no se
descarta**: `runner/camera_worker.py` lo emite como `severity: "sospecha"` /
`confirmed: False` sin pasarlo por CLIP (recorte truncado → CLIP no sirve), y
`_track_postfall` en este módulo lo escala a un evento `immobility` de
`severity: "alta"` (`trigger: "post_fall_sospecha"`) si la persona queda inmóvil
`IMMOBILITY_WINDOW_POSTFALL_S` = 45 s (ventana corta: posible emergencia, no
sedentarismo), o lo cancela si se mueve. Reusa `_smoothed_displacement_rate` /
`_immobility_history`. `evaluate_falls.py` reporta recall/FP desglosado por
`evidence`: los eventos `partial` **recuperan +10–12 caídas reales** de "solo se
ve el torso" que la geometría "full" perdía, a costa de FP que son sospechas, no
alertas duras.

*Parte 2 — clasificador temporal entrenado (vive en `fusion/`).* En vez del
filtro CLIP zero-shot/probe, cuando la geometría dispara un evento de evidencia
completa se consulta un GRU entrenado sobre la ventana de keypoints
(`fusion/src/fall_temporal.py`, ver `fusion/README.md` para diseño, augmentación
de oclusión, hard-negative mining y números). Es exactamente el "enfoque tipo
Transformer entrenado" que este README daba por fuera de alcance — resultó viable
con los datasets locales (`GMDCSA-24` + `CAUCAFall` + `URFD` + `ToyotaSmarthome_FP`
como negativos duros) porque el modelo (GRU ~48 k params) es mucho más chico que
el de Benabdennour y la ventana de keypoints ya la produce `fusion/`.

`evaluate_falls.py --temporal` corre el pipeline completo (geometría + veto
temporal). Sobre 5 datasets (158 caídas / 292 clips): recall **133/158 (84.2 %),
igual que geométrico solo**, FP de alerta dura **185 → 91**. Donde hay ventana de
esqueleto que clasificar el temporal rechaza **94/94 FP** y pierde **1/113
caídas**; los ~74 FP residuales de `ToyotaSmarthome_FP` son eventos **sin ventana**
— el mismo bloqueante de `yolo11n-pose` documentado arriba (persona en sillón
desde cámara ancha, 94.5 % de frames sin detección) — y en producción caen a la
confirmación CLIP. `FallImmobilityDetector` y sus umbrales geométricos **no
cambian**: sigue siendo el trigger y se calibra/evalúa solo con `evaluate_falls.py`
(sin `--temporal`).

### 3. Inmovilidad prolongada — calibrada para el caso sedentario, sin validar para el caso post-caída

El informe (Sec. 2.1.2) enmarca la inmovilidad prolongada como la secuela
clínica de una caída no detectada — el "síndrome post-caída": la persona
cae, no logra levantarse, queda inmóvil en el piso. Ninguno de los datasets
de caída disponibles (`videos/GMDCSA-24/`, `videos/CAUCAFall/`) tiene ese
escenario grabado — los clips terminan pocos segundos después del impacto,
no minutos después. Por eso se separó el evento en dos ramas según la
postura al momento de reportarlo (`posture.classify_posture()` sobre las
últimas `IMMOBILITY_POSTURE_VOTE_N` muestras, por mayoría — evita que una
clasificación de postura ruidosa de un solo frame decida la severidad):

- **`posture: "sentado"` → `severity: "baja"`** (sedentarismo prolongado,
  ej. quedarse leyendo o mirando TV mucho rato sin moverse): **calibrado con
  datos reales**, ver abajo.
- **`posture: "acostado"` → `severity: "alta"`** (posible caída no
  detectada, persona inmóvil en el piso): la lógica de detección es la
  misma, pero **sin dataset propio para validar la duración real** de este
  escenario — extrapolación razonable del umbral calibrado para el caso
  sedentario, no una medición. Mismo tipo de gap que `videos/Kaggle/` para
  caídas (ver arriba): documentado en vez de forzar una calibración sin
  evidencia.
- **`posture: "parado"`**: no se reporta ningún evento — quedarse parado e
  inmóvil por minutos no es un patrón contemplado (ni sedentarismo ni
  post-caída), fuera de alcance por ahora.

**Calibración del caso sedentario** (`calibrate_immobility.py`): usa el
dataset **untrimmed** de Toyota Smarthome (grabación continua sin recortar,
con anotación `event,start_frame,end_frame` por clip — distinto del dataset
trimmed en `videos/`, ver `../videos/README.md`). A diferencia de los clips
trimmed (~1-2s), estos duran varios minutos, así que contienen huecos reales
sin ningún evento etiquetado mientras la persona sigue en cuadro —
concretamente, tramos acotados por `Sit_down` al principio y `Get_up` al
final, sin otro evento en el medio: la persona se sienta, no hace nada
catalogado por un rato, se levanta. Eso es sedentarismo real medido, no un
artefacto de anotación (los huecos entre `Leave` y el siguiente `Enter`, en
cambio, son la persona fuera de cuadro, y se excluyen).

Se procesaron 10 de estos tramos (64s a 281s, de 7 participantes distintos)
más 9 tramos `Walk` (≥2.5s) de los mismos clips como contraste, muestreando
a ~2fps (cada 12 frames de los 25fps nativos) por costo de cómputo. Resultado
(1990 muestras "quieto", 168 "caminando"), desplazamiento de keypoints
normalizado por segundo transcurrido entre muestras (no por frame — ver
abajo):

| | mediana | p90 | p95 |
|---|---|---|---|
| Quieto (sentado) | 0.0056 | 0.0172 | 0.0299 |
| Caminando | 0.0606 | 0.1343 | 0.1658 |

Separación clara (10x en la mediana). Con umbral **0.02**: 8.1% de frames
"quieto" lo superan por ruido puntual (acomodarse, dar vuelta una hoja) y
8.9% de frames "caminando" quedan por debajo (frames de transición) —
tomado frame a frame, no es un umbral perfecto, pero el valor coincide con
el que ya estaba puesto (tomado de la literatura sin validar antes de
esto).

**Dos correcciones a la lógica, no solo al valor del umbral:**

1. **Normalización por tiempo** (`_smoothed_displacement_rate`): la versión
   anterior comparaba desplazamiento crudo entre frames consecutivos sin
   dividir por `dt` — mismo bug ya encontrado y corregido en la velocidad de
   caída (`_vertical_velocity`, ver arriba). Con frame dropping en tiempo
   real (`runner/`, ver `runner/README.md`), el intervalo entre muestras
   procesadas varía, así que el desplazamiento necesita normalizarse por
   segundo para no depender de cuántos frames se saltean.
2. **Tolerancia a blips puntuales**: con ~8% de frames "quieto" superando el
   umbral por ruido, resetear la cuenta completa de inmovilidad ante *un
   solo* frame por encima del umbral (como hacía la versión anterior) casi
   nunca dejaría acumular la ventana completa. Ahora el desplazamiento se
   promedia sobre `IMMOBILITY_SAMPLE_WINDOW_S` (2s, usando la muestra más
   cercana a ese tiempo atrás en el historial, igual criterio que
   `_vertical_velocity`) en vez de comparar frame a frame.

**Ventana** (`IMMOBILITY_WINDOW_S`): se subió de 30s a **180s (3 min)**. El
umbral de 30s original no tenía respaldo — con eso, cualquiera que se quede
sentado leyendo un párrafo dispara el evento. 180s surge de la conversación
de diseño (evento de baja severidad, no una emergencia instantánea) y de
que los tramos "quieto" medidos llegan hasta 281s reales sin interrupción,
así que 180s es alcanzable sin depender de tramos al límite de lo
disponible.

**Validación end-to-end** (`FallImmobilityDetector` completo, frame a frame,
no subsampleado, sobre 12 tramos reales de Toyota Smarthome untrimmed):

| Resultado | Tramos | Detalle |
|---|---|---|
| Disparó correctamente (>180s) | 4/4 | `P15T06C05` (262s→180.04s), `P15T12C01` (484s→413.6s, ver bug abajo), `P07T10C01` (459s→180.04s), `P06T14C01` (267s→180.00s) — todos `posture: "sentado"`, `severity: "baja"` |
| No disparó antes de tiempo (<180s) | 4/4 | `P07T01C05` (152s), `P11T12C03` (138s), `P10T01C05` (64s), `P10T07C05` (129s) — cero falsos positivos prematuros |
| Derivó en falso positivo de **caída** en vez de probar inmovilidad | 4/12 | `P07T10C02`, `P09T05C05`, `P07T05C05`, `P09T06C05` — ver hallazgo aparte abajo |

**8 de 8 tramos que sí llegaron a probar la lógica de inmovilidad se
comportaron exactamente como se esperaba** (4 disparos correctos, 4
correctamente contenidos). Detección de persona 99.6-100% en todos, así que
el resultado no está maquillado por pérdida de tracking.

`P15T12C01` (484s) encontró un bug real en el camino: la versión inicial de
esta calibración emitía el evento igual con `posture: "parado"`
(contradiciendo lo de arriba, que dice que no debería reportarse) porque la
exclusión nunca se había implementado, solo documentado. Al revisar el
frame real en el momento del disparo, la persona estaba **sentada** en una
mesa — error del clasificador de postura (`posture.py`, recall de sentado
documentado en 82.1%, no 100%): con este ángulo de cámara y las piernas
tapadas por la mesa, el aspect ratio cae del lado de "parado". Se corrigió
`_track_immobility` para no emitir mientras la postura por mayoría sea
"parado" (sin resetear el conteo — si la postura pasa a sentado/acostado
más tarde sin que la persona se haya movido, ahí sí se reporta). Con el fix,
este mismo tramo dispara correctamente a los 413.6s con `posture: "sentado"`
en vez de a los 376.8s con `"parado"`.

**Sanity-check de la rama `acostado`** (cola de los últimos 2s de 7 clips de
`GMDCSA-24/fall3` y `fall4`, caídas al piso con cuerpo visible, no en
silla): 6/7 con detección de persona ~100% y `classify_posture()` leyendo
`acostado` de forma consistente en esos frames finales. Un clip
(`fall3/04.mp4`) con detección más floja (41/59, 69%) y postura mezclada
`parado`/`acostado`, sin investigar todavía — candidato a revisar antes de
confiar del todo en la rama de severidad alta. Esto valida que la detección
y clasificación de postura funcionan bien cuando el cuerpo está visible en
el piso (a diferencia de estar tapado con una manta, ver más abajo el
intento fallido con `videos` `Lay_down` de Toyota Smarthome) — pero **no
valida la duración real**: los clips de caída duran 3-10s, muy lejos de los
180s de la ventana, así que si el contador de inmovilidad se sostiene
180s seguidos en un escenario post-caída real sigue sin comprobarse.

**Hallazgo aparte, no de este módulo pero encontrado validando acá — y no
es un caso aislado:** de los 12 tramos "quieto" probados, **4 (un tercio)**
no llegaron a disparar inmovilidad porque **el detector de caídas disparó
primero**, siempre falso positivo (verificado visualmente en varios: la
persona está sentada, no cayendo):

| Clip | Falsos positivos de caída | Nota |
|---|---|---|
| `P07T10C02` | 1 en 460s | — |
| `P09T05C05` | 20 en 212s | persona sentada en el piso/otomana baja (bbox ancho y bajo, parecido a "acostado") |
| `P07T05C05` | 4 en 220s | `torso_angle_deg` inusualmente alto (82-86°) en los disparos, distinto patrón al de `P09T05C05` |
| `P09T06C05` | 8 en 211s | `aspect_ratio` muy pegado al umbral (1.40-1.46 vs. `FALL_ASPECT_RATIO=1.4`) |

Los umbrales de caída (recall 85.2%/96.2%, ver arriba) están calibrados y
validados contra `GMDCSA-24`/`CAUCAFall`, no contra este setup de cámara.
Esto quedó documentado como el gap más urgente a resolver en `vision/` — ver
la sección siguiente, "Recalibración con el setup de Toyota Smarthome como
contraste", para el proceso completo que lo cerró.

**Actualización (2026-08-13): revalidado con el filtro CLIP ya integrado —
el "secuestro" por falsos positivos de caída prácticamente desaparece, pero
eso no alcanzó para que la inmovilidad se disparara en la mayoría de estos
4 clips.** `vision/validate_immobility_with_clip.py` (nuevo) corre el
pipeline completo (heurística geométrica + confirmación CLIP + tracking de
inmovilidad) sobre los 4 clips completos de `videos/ToyotaSmarthome_FP/`
(1100-1591s cada uno, no solo el sub-tramo corto de la tabla de arriba —
esos límites exactos no habían quedado guardados en ningún script
versionado, así que esta corrida no es una reproducción exacta de la
original, es una nueva medición más completa sobre el clip entero):

| Clip | Caídas geométricas crudas | Sobreviven CLIP | Inmovilidad |
|---|---|---|---|
| `P07T10C02` | 3 | 0 | disparó @238.3s, `sentado`/`baja` (correcto) |
| `P09T05C05` | 82 | 0 | nunca se disparó |
| `P07T05C05` | 124 | 0 | nunca se disparó |
| `P09T06C05` | 61 | 1 | nunca se disparó |

**El filtro CLIP elimina 269 de 270 caídas geométricas falsas (99.6%) sobre
~80 minutos de video doméstico continuo real** — el mismo orden de magnitud
que ya mostró el 30 FP/500 eventos out-of-fold (ver arriba), pero acá medido
end-to-end sobre clips que antes disparaban decenas o cientos de veces cada
uno. La única que sobrevive (`P09T06C05` @150.52s, margin 0.182) es
holgadamente confirmada por el modelo — candidata a revisar el frame en
persona en algún momento, pero es 1 evento, no un patrón.

Lo que **no** se resolvió: la inmovilidad solo se disparó correctamente en
1 de los 4 clips. No es evidencia de que el fix no sirva — es evidencia de
que la hipótesis original ("el fall FP tapa la inmovilidad") no explica el
comportamiento de estos 3 clips en particular: si la persona nunca se queda
quieta 180s seguidos en el clip COMPLETO (a diferencia del tramo corto y
aislado con el que se armó la tabla de arriba), suprimir los falsos
positivos de caída no la va a hacer quedarse quieta — el gap no era el
"secuestro", era que estos clips (elegidos justamente como negativos duros
por tener mucho movimiento ambiguo) tienen más actividad real de la que
tenía el tramo corto original. Sin los límites exactos de esos tramos no se
puede aislar si el disparo tardío/ausente es por eso o por otra causa
distinta — queda documentado como el próximo paso si se retoma esto, no
como cerrado.

**Diagnóstico de por qué (2026-08-13): se probó restringir el desplazamiento
a hombros+cadera en vez de los 17 keypoints, y el aparente arreglo se
deshace al retunear el umbral correctamente — explorado, sin ganancia
neta.** La hipótesis (planteada por el usuario del proyecto): una persona
sentada durante mucho tiempo no está 100% quieta — mueve las manos (comer,
usar el celular, gesticular) sin cambiar de postura — y
`_smoothed_displacement_rate` promedia sobre los 17 keypoints COCO,
incluyendo manos/codos, así que ese movimiento normal infla el
desplazamiento medido y nunca deja acumular los 180s de la ventana.

`vision/diagnose_immobility_displacement.py` (nuevo) reprodujo, frame a
frame y a framerate nativo (no subsampleado — una primera pasada a ~2fps
dio números que no se sostuvieron al pasar a framerate completo, mismo tipo
de error de resolución temporal que otros hallazgos de esta sesión) la
racha continua más larga de "quietud" en los 3 clips que nunca dispararon,
comparando los 17 keypoints ("full", el método actual) contra solo
hombros+cadera ("core", índices COCO 5,6,11,12):

| Clip | racha `full` | racha `core` (umbral 0.02) |
|---|---|---|
| `P09T05C05` | 80s | 82s (sin cambio real) |
| `P07T05C05` | 79s | 79s (sin cambio) |
| `P09T06C05` | 145s | **210s** (cruza los 180s) |

Con el umbral actual (0.02, calibrado para `full`), restringir a `core`
parecía arreglar 1 de los 3 casos. Pero 0.02 nunca se calibró para `core` —
se reusó tal cual. `vision/calibrate_immobility_core.py` (nuevo) repitió la
calibración original (10 tramos "quieto" + 9 "moviendo" reales de Toyota
Smarthome untrimmed, mismo dataset de `calibrate_immobility.py`) calculando
`core` en paralelo a `full`:

| | mediana quieto | mediana moviendo | ratio separación |
|---|---|---|---|
| full | 0.0056 | 0.0606 | 10.8x |
| core | 0.0036 | 0.0550 | 15.2x |

`core` separa mejor en términos relativos (15.2x vs 10.8x), pero ambas
distribuciones están desplazadas hacia abajo — el umbral que balancea error
en "quieto" vs. error en "moviendo" para `core` no es 0.02, es **~0.014**
(7.1%/7.7% de error, comparable a como está calibrado `full` hoy). Repitiendo
la racha de `P09T06C05` con ese umbral bien calibrado (no el reusado): **baja
de 210s a 161s — vuelve a quedar por debajo de los 180s.** La mejora
observada con 0.02 no venía de mirar menos keypoints, venía de estar usando
sin querer un umbral más laxo para esa variante — al retunear correctamente,
la ganancia desaparece.

**Conclusión: no hay una combinación (subconjunto de keypoints, umbral) que
arregle este caso sin resignar precisión en otro lado — es un trade-off
genuino, no una mejora limpia.** De los 3 clips, solo uno estaba cerca del
límite (145-161s vs. 180s) con cualquiera de los dos métodos; los otros dos
(`P09T05C05` 80-82s, `P07T05C05` 79s) ni se acercan, sea cual sea el
criterio — ahí la persona genuinamente mueve el cuerpo entero cada
poco más de un minuto, no es un problema de qué keypoints mirar. No se
integró ningún cambio a producción; queda explorado y documentado, no
resuelto — alargar `IMMOBILITY_SAMPLE_WINDOW_S` (hoy 2s) para diluir más los
gestos de mano es una idea alternativa no probada, con el mismo riesgo de
trade-off (también diluiría el arranque de un movimiento real).

**Validación visual con `evaluation/annotate_video.py` (2026-10-01):**
primera vez que se mide la tasa de detección de `yolo11n-pose` sobre el
clip COMPLETO (1100-1600s) de estos 4 videos — antes solo se había medido
sobre tramos cortos aislados (99.6-100%, ver tabla de arriba, que es de
otros clips). Sobre el clip entero:

| Clip | Detección (clip completo) | Hueco máx |
|---|---|---|
| `P07T10C02` (dispara inmovilidad) | 99.2% | 1.6s |
| `P09T05C05` | 77.8% | 9.7s |
| `P09T06C05` | 77.7% | 23.1s |
| `P07T05C05` | 50.6% | 11.7s |

El único clip con detección casi perfecta es el único que dispara
inmovilidad — a primera vista sugiere que los huecos de tracking (no la
actividad real) podrían ser la causa. Se revisaron a mano los frames antes/
durante/después de los dos huecos más grandes (`P09T06C05` @959-982s,
`P07T05C05` @1128-1141s): en ambos casos **la persona no está en el cuadro**
ni antes ni después del hueco en ese punto — se fue de la habitación, no es
que esté quieta y el modelo la pierda (a diferencia del bloqueante de la
rama `acostado` documentado arriba, que es sobre una persona presente y
quieta que el modelo no detecta). Esto confirma, con evidencia visual
directa, la conclusión ya alcanzada arriba por otro camino (medición de
rachas): esto 3 clips tienen más actividad/ausencia real de la que asumía
el tramo corto original, no es un artefacto de tracking.

**Datos reales nuevos para la rama `acostado` (2026-08-13): se recuperó la
anotación untrimmed oficial y se encontraron 214 segmentos `Lay_down` largos
— la primera validación de esta rama con minutos reales de "acostado
quieto", no solo segundos.** Hasta ahora la rama de severidad "alta"
(posible caída no detectada) no tenía ninguna validación de duración real —
solo un sanity-check de postura sobre los últimos 2s de clips de caída (ver
arriba), muy lejos de los 180s de la ventana. `calibrate_immobility.py`
había usado el dataset untrimmed de Toyota Smarthome para el caso
sedentario, pero la anotación cruda (`smarthome_CS_51.json`, tuplas
`[label, start_frame, end_frame]` por video) no había quedado guardada en
ningún archivo del proyecto — se recuperó del repo oficial
([dairui01/Toyota_Smarthome](https://github.com/dairui01/Toyota_Smarthome/blob/main/pipline/data/smarthome_CS_51.json),
mapeo de labels en `pipline/data/Action_list` del mismo repo: `25=Sit_down`,
`28=Get_up`, `32=Lay_down`).

Buscando `Lay_down -> próximo evento anotado` (mismo criterio que
`Sit_down -> Get_up` para el caso sedentario) entre los 536 videos ya
descargados: **214 segmentos**, varios de varios minutos —
`P25T05C05`/`P25T05C04` (misma escena real, 2 cámaras sincronizadas) con
~519s/518s hasta `Get_up`, `P18T03C04` con 277s. Se revisaron frames de los
candidatos más largos a ojo: cuerpo visible, sin la oclusión por manta que
arruinó el intento anterior con clips `Lay_down` cortos (ver arriba) — la
diferencia es que esos eran clips *trimmed* (1.7-7.5s, la acción de
acostarse en sí) mientras que estos son gaps *untrimmed* (el tiempo que la
persona se queda acostada después). **Matiz importante:** son personas
acostadas en un sillón, no en el piso después de una caída — no es el
escenario clínico exacto del informe (Sec 2.1.2), pero sí es la primera
evidencia real de que el mecanismo (clasificación de postura +
desplazamiento) se sostiene —o no— durante minutos reales, no solo segundos.

`vision/validate_immobility_acostado.py` (nuevo) corrió el pipeline
completo (heurística geométrica + confirmación CLIP + tracking de
inmovilidad) frame a frame, a framerate nativo, sobre los 3 casos más
largos, empezando 3s antes del `Lay_down` anotado para capturar la
transición completa:

| Clip | Ventana | Caídas geométricas crudas | Sobreviven CLIP | Inmovilidad |
|---|---|---|---|---|
| `P25T05C05` | 527s | 7 (t=1-283s) | 1 | nunca se disparó |
| `P25T05C04` (misma escena, otra cámara) | 527s | 3 (t=4-12s) | **3 (todas)** | nunca se disparó |
| `P18T03C04` | 279s | 1 (t=2.3s) | 0 | nunca se disparó |

Tres resultados, ninguno el esperado:

1. **Las caídas geométricas falsas se agrupan en la transición de
   acostarse** (los primeros ~12s de la ventana en los 3 casos) — el acto de
   acostarse en sí produce el patrón geométrico de dos fases (velocidad +
   postura ensanchada) que dispara el heurístico. Ya sabíamos que sentarse
   podía disparar esto (ver "Recalibración..." abajo); acostarse en un
   sillón hace lo mismo.
2. **Hallazgo nuevo, no relacionado con inmovilidad pero real: el filtro
   CLIP fue inconsistente entre dos cámaras sincronizadas del MISMO evento.**
   `P25T05C05` y `P25T05C04` son la misma persona acostándose en el mismo
   sillón al mismo tiempo. En una vista de cámara, CLIP filtró 6 de 7 caídas
   falsas. En la otra vista del mismo evento real, **las 3 sobrevivieron**
   (margin 0.189-0.435, confirmadas con margen cómodo, no al límite). No se
   investigó la causa (ángulo, recorte de persona, iluminación) — queda como
   gap de robustez conocido: la validación de CLIP hecha hasta ahora
   (84.8% recall/30 FP, ver arriba) es sobre eventos ya recortados de los
   datasets de entrenamiento/evaluación, no sobre variación de cámara
   explícita.
3. **La inmovilidad NO se disparó en ninguno de los 3 casos**, pese a ser
   ventanas de 279-527s reales de "acostado" anotado. Diagnosticado a
   fondo (ver siguiente sección): la causa NO fue la misma que en el caso
   sedentario (mover manos/cabeza) — fue un bug real de código (ya
   corregido) más un problema de detección más profundo que ese bug no
   alcanza a tapar.

**Diagnóstico y fix (mismo día): bug real encontrado en `_smoothed_displacement_rate`
(corregido), pero el bloqueante de fondo es que YOLO11n-pose no detecta bien
a alguien acostado en un sillón visto desde cámara ancha/elevada — no es un
problema de código, es un límite del modelo.**

Repitiendo el diagnóstico full-vs-core (`diagnose_immobility_acostado.py`,
nuevo, mismo método que `diagnose_immobility_displacement.py`) sobre estos 3
clips: `P25T05C05`/`P25T05C04` muestran baja detección explicada por la
manta (revisado a ojo con capturas a distintos tiempos: la persona ya está
tapada a los 30s de acostarse y se queda así el resto de la ventana — mismo
patrón ya documentado arriba, no es nuevo). Pero `P18T03C04` —confirmado
SIN manta, cuerpo visible en capturas a los 10s/60s/120s/180s/250s— dio una
racha offline de **213s de quietud** (por encima de los 180s), y sin
embargo el detector end-to-end real nunca disparó. Esa contradicción llevó
a un bug real:

`self._history` (la ventana de 2.5s usada por `_smoothed_displacement_rate`
para inmovilidad) está dimensionada para `_vertical_velocity` de caída, que
solo necesita 0.4s de lookback. Pero `IMMOBILITY_SAMPLE_WINDOW_S` es 2.0s —
casi todo ese presupuesto. Cualquier hueco de detección más largo que 2.5s
(nada anómalo en video doméstico real: cambio de ángulo, blur, oclusión
parcial breve) vacía `self._history`, `_smoothed_displacement_rate` devuelve
`None`, y `_track_immobility` lo trata como "moviéndose" — resetea el
contador. **Corregido:** se separó un historial propio para inmovilidad
(`self._immobility_history`, constante nueva `IMMOBILITY_HISTORY_SECONDS =
15.0`, `vision/src/pose_pipeline.py`), independiente del de caída, que
tolera huecos de hasta 15s sin resetear el contador de quietud.

**Validado sin regresión:** se re-corrió `validate_immobility_with_clip.py`
(los 4 clips sedentarios de `ToyotaSmarthome_FP`) con el fix — resultado
IDÉNTICO al de antes, `P07T10C02` dispara exactamente en el mismo segundo
(238.3s) y los otros 3 siguen sin disparar. Esperable y bueno: esos clips
tienen detección casi continua (99.6-100%, ya medido), así que el fix no
tenía nada que corregir ahí — confirma que no rompió nada donde ya
funcionaba.

**Pero el fix NO alcanzó para `P18T03C04`.** Revisando los huecos reales de
detección en ese clip: no son huecos breves — son **4 huecos de 17 a 103s**
cada uno, cubriendo casi toda la ventana de 279s. 15s de tolerancia no
alcanza para huecos de esa magnitud (ni tendría sentido subir la tolerancia
tanto: un hueco de 100s podría perfectamente ser alguien que se levantó y
volvió). Diagnosticando por qué hay huecos tan grandes pese a que el cuerpo
está visible a ojo en TODOS los frames revisados: se corrió YOLO sin el
filtro de confianza sobre una muestra del clip —
`vision/diagnose_detection_lying.py` (nuevo)— y **94.5% de los frames no
tienen NINGUNA detección de persona**, ni siquiera de baja confianza (no es
un problema de `MIN_KEYPOINT_CONFIDENCE`, es que YOLO no encuentra el
bounding box). No es oclusión (ya descartada a ojo) — es que `yolo11n-pose`
(la variante liviana elegida para poder correr en la Raspberry Pi) no
reconoce bien a una persona acostada en un sillón vista desde una cámara
ancha, elevada y lejos, donde ocupa una fracción chica del frame con un
bounding box horizontal poco común en los datos de entrenamiento típicos de
modelos de pose (mayoría personas paradas/caminando). Contraste: el
sanity-check de la rama `acostado` sobre clips de CAÍDA real (arriba) dio
99.6-100% de detección — ahí la persona cae al piso, más cerca de cámara,
con el cuerpo extendido de forma más prominente; acá es una persona en
reposo normal en un mueble, lejos, en una cámara de vigilancia de ambiente
completo.

**Conclusión: el fix de historial es real, correcto, y queda integrado
(sin regresión, corrige un bug genuino) — pero no resuelve el caso que lo
motivó.** El bloqueante de fondo para la rama `acostado` no es una cuestión
de umbral, ventana de suavizado, ni ahora tampoco de historial: es que el
modelo de pose liviano no detecta confiablemente a una persona en reposo en
un mueble desde una cámara de ambiente completo. Mejorar esto necesitaría
un modelo de pose más grande (yolo11s/m-pose en vez de nano) — con el
costo de cómputo que eso implica en una Raspberry Pi, fuera de alcance para
resolver en esta sesión — o repensar la estrategia de detección para esta
situación específica (ej. no depender de keypoints completos para
"¿hay alguien ahí, quieto?", bastaría con la presencia/tamaño del bounding
box aunque la confianza de keypoints individuales sea baja).

No se integró ningún cambio de umbrales/keypoints a producción — solo el
fix de historial, que es una corrección de bug, no un ajuste de
calibración. Queda documentado como la primera evidencia real (aunque
parcial: sillón, no piso) de que la rama `acostado` tiene un problema más
profundo que la rama `sentado` (que sí es "la persona se mueve", medido y
confirmado arriba) — acá el problema es anterior, es de detección, no de
movimiento. Más el hallazgo aparte sobre robustez de CLIP entre cámaras que
no se había visto antes.

### Recalibración con el setup de Toyota Smarthome como contraste (resuelto con un filtro de CLIP, no con umbrales)

**Paso 1 — sumar los 4 clips como contraste y re-correr el grid search.** Se
armó `videos/ToyotaSmarthome_FP/Fall.csv` con los 4 clips completos de la
tabla de arriba (P07T10C02, P09T05C05, P07T05C05, P09T06C05, ~1100-1600s
cada uno, anotados como `NoFall` de punta a punta — no tienen ninguna caída
real, son actividades cotidianas) y se extendió
`fall_descriptors_cache.pkl` con `extend_descriptor_cache.py` (183 clips en
total). Al correr `recalibrate_fall_thresholds.py` con estos clips sumados,
apareció un bug de calibración aparte: el rango de `aspect_candidates` del
grid search (`[1.0, 1.05, ..., 1.2]`) **no cubría el umbral realmente
desplegado en producción (`FALL_ASPECT_RATIO=1.4`)** — se corrigió el rango
a `[1.0, ..., 1.6]` antes de confiar en cualquier resultado nuevo.

**Paso 2 — el grid search por sí solo no alcanzaba a mostrar el problema
real.** Con el rango corregido, `(0.12, 1.4)` seguía siendo la mejor
combinación del barrido — pero el score del grid search topea a 1 falso
positivo por clip (para rankear rápido), lo que esconde la frecuencia real.
Corriendo el detector completo (con cooldown real, sin ese tope) sobre los 4
clips enteros: **3, 82, 124 y 61 falsos positivos** respectivamente — mucho
peor que la tabla de arriba, que solo medía un tramo corto de cada clip, no
el clip completo.

**Paso 3 — barrido exhaustivo sin el tope de 1 FP/clip, para confirmar que
no es un problema de calibración fina.** Repitiendo el grid search sobre las
mismas 182 combinaciones de umbrales pero contando falsos positivos reales
(no topeados), **ninguna combinación resuelve el problema**: `(0.12, 1.4)`
sigue siendo la mejor (85.2% recall, 91 FP en los 4 clips de 109 FP
totales — el 83% de todos los falsos positivos del sistema vienen de estos 4
clips), y cualquier otra combinación o mantiene ~90-108 FP en estos clips, o
empieza a perder recall real sin arreglar el problema. **Conclusión
cuantitativa:** la heurística geométrica (velocidad vertical + aspect ratio)
tiene un límite estructural con este ángulo de cámara — sentarse en un
sillón bajo/piso mientras se mira TV genera la misma firma geométrica que
una caída (bbox ancho y bajo + movimientos de brazo/torso que cruzan el
umbral de velocidad). No es una cuestión de calibración fina de umbrales.

**Paso 4 — filtro de confirmación con CLIP zero-shot (la vía que ya estaba
anotada como trabajo futuro más arriba).** Cuando el detector geométrico
dispara un evento de caída, se le pregunta a CLIP zero-shot si el recorte de
la persona en ese frame se parece más a "una caída accidental" o a "una
postura voluntaria" (prompts en `vlm/src/fall_confirmation.py`), y solo se
confirma el evento si CLIP coincide. Prototipado y calibrado en
`calibrate_fall_confirmation.py`: corre el detector completo sobre los 183
clips y, en cada evento geométrico, recorta la persona y consulta CLIP
(ViT-B/32 fp32 vía el paquete `clip`/PyTorch, no el ONNX de producción —
ver el hallazgo del paso 5 sobre por qué esto importa), guardando los
scores en `fall_confirmation_scores.csv` (445 eventos evaluados). El margen
`clip_fall_score - clip_notfall_score` separa claramente los eventos reales
(mediana ≈0) de los falsos positivos (mediana ≈-0.024).

**Paso 5 — la cuantización INT8 desplaza el umbral: hay que calibrar contra
el modelo real, no el prototipo.** El umbral elegido con el prototipo fp32
(-0.02) se probó también contra el encoder ONNX INT8 real (el que corre en
producción, `clip_image_encoder_int8.onnx`) reclasificando los mismos 445
eventos con `ClipActivityClassifier`/`FallConfirmationClassifier`
(`fall_confirmation_onnx_check.csv`). El resultado fue distinto al
esperado: con -0.02 sobre márgenes ONNX el recall real caía a 71.1% (vs. el
82.8% calibrado en el prototipo) — la cuantización corre los embeddings lo
suficiente como para invalidar un umbral calibrado en fp32. Se recalibró el
umbral directamente contra los márgenes ONNX reales, encontrando
`CONFIRMATION_MARGIN_THRESHOLD = -0.028` en
`vlm/src/fall_confirmation.py`:

| | Recall combinado (128 casos) | FP totales (183 clips) | FP en los 4 clips de Toyota Smarthome |
|---|---|---|---|
| Solo heurística geométrica | 85.2% (109/128) | 288 | 270 |
| + filtro CLIP (ONNX, umbral final) | 82.0% (105/128, -3.2pp) | 96 (-67%) | 81 (-70%) |

Los 3-4 casos de recall perdidos son mayormente de `GMDCSA-24` (el dataset
mejor calibrado, 96.2% de partida); `CAUCAFall` no pierde recall con este
umbral. **Trade-off aceptado:** en un producto de monitoreo doméstico, la
fatiga de alertas por falsos positivos repetidos (hasta 124 en un solo clip
antes del filtro) es un riesgo de adopción tan real como perder una caída
puntual, y el detector geométrico solo ya tenía blind spots conocidos
(caídas hacia la cama, oclusión) que este filtro no toca ni empeora.

**Dónde vive el filtro:** `vision/` queda con el detector puramente
geométrico (`FallImmobilityDetector`, sin cambios de umbral — `(0.12, 1.4)`
se mantiene, confirmado como el mejor punto del grid search) para poder
seguir calibrándolo/evaluándolo de forma independiente vía
`evaluate_falls.py`, sin depender de `vlm/`. El filtro de confirmación
(`FallConfirmationClassifier`) vive en `vlm/` y se integra en
`runner/camera_worker.py` — la única pieza que ya tenía permiso de importar
`vision/` y `vlm/` juntos (ver "Cómo trabajar en paralelo" en el README
raíz). `vision/main.py` (modo batch) y los scripts de calibración de este
README siguen midiendo solo la heurística geométrica, a propósito.

**Intento descartado, documentado para no repetirlo:** se probó usar clips
`Lay_down` de Toyota Smarthome (persona tapada con manta en un sillón, no
en el piso) como sustituto de "acostado post-caída". La detección de
persona fue prácticamente nula (1-22% según el clip) — pero al revisar los
frames a ojo, es oclusión por manta (solo se ve la cabeza), no una
limitación del modelo con posturas acostadas en general. No sirve como
proxy del escenario post-caída (cuerpo visible) y no se debe reintentar sin
buscar clips `Lay_down` donde el cuerpo quede visible.

**Experimento en curso, sin terminar (2026-08-07): linear probe entrenado
en vez de zero-shot para el filtro de confirmación.** Inspirado en Raza et
al. (2022, ICPRS) — YOLO como clasificador *entrenado* fall/no-fall en vez
de heurística — y en que `vlm/` ya mejoró la clasificación de actividades
reemplazando zero-shot por un linear probe entrenado sobre los mismos
embeddings de CLIP (ver `vlm/README.md`). La idea: en vez de comparar el
embedding de la persona contra el margen de `FALL_PROMPTS`/`NOTFALL_PROMPTS`
(zero-shot), entrenar una regresión logística sobre esos mismos embeddings
— sin sumar ningún modelo nuevo al Pi, reusa el encoder ONNX que `vlm/` ya
carga.

Scripts nuevos en `vision/`: `extract_fall_confirmation_embeddings.py`
(corre el detector geométrico + encoder ONNX sobre `videos/**/Fall.csv` y
cachea embeddings + etiqueta en `fall_confirmation_embeddings.npz`, análogo
a `vlm/extract_clip_features.py`) y `train_fall_confirmation_probe.py`
(entrena con `GroupKFold` por video para medir recall/FP *out-of-fold*, sin
leakage, análogo a `vlm/train_category_probe.py`).

Corrida parcial sobre 179/183 clips (GMDCSA-24 + CAUCAFall, sin los 4 clips
largos de `ToyotaSmarthome_FP`, cortada por tiempo — cada uno tarda mucho en
CPU frame a frame): **71.1% recall (91/128), 4 FP** out-of-fold, contra el
82.0% recall / ~15 FP estimado del filtro zero-shot en el mismo subset (el
96 FP documentado arriba es sobre los 183 clips completos, la mayoría de
Toyota Smarthome). **No es un resultado confiable todavía:** el subset
usado para entrenar el probe tiene 150 eventos de caída real contra solo 16
de no-caída, porque los negativos informativos (los falsos positivos tipo
"sentado en sillón bajo") son justamente los que vienen de los 4 clips de
Toyota Smarthome excluidos — el probe casi no vio el caso que tiene que
aprender a rechazar. El recall bajo probablemente es artefacto de ese
desbalance, no una limitación real del método. **Pendiente:** volver a
correr `extract_fall_confirmation_embeddings.py` (sin `--skip-toyota`) hasta
completar los 183 clips, reentrenar con `train_fall_confirmation_probe.py`,
y recién ahí comparar en serio contra el filtro zero-shot antes de decidir
si reemplazarlo. `vlm/models/fall_confirmation_probe.npz` ya se generó con
el subset parcial pero **no está integrado en `FallConfirmationClassifier`**
— `vlm/src/fall_confirmation.py` sigue sin cambios, el filtro en producción
sigue siendo el zero-shot ya validado (82.0% recall, -67% FP).

Solo se trackea la primera persona detectada por frame — el informe excluye
explícitamente el caso multi-residente (Sec. 1.2.2).

**Actualización (2026-08-09): intento de completar los 183 clips, cortado por
costo de cómputo; hallazgo de una fuente de negativos más barata.** Se
intentó correr `extract_fall_confirmation_embeddings.py` sin `--skip-toyota`
sobre los 4 clips largos de `ToyotaSmarthome_FP` (pendiente señalado arriba).
Un solo clip (640×480, 25fps, ~28-40k frames) tardó más de 3 horas en CPU sin
terminar — muy por encima de lo estimado (25-65 min) — y se abortó. La causa
probable: YOLOv11n-pose + el encoder ONNX de CLIP, corridos frame a frame
sin límite de threads, sobre-suscriben los cores (oversubscription) en vez
de paralelizar de forma eficiente, un patrón confirmado también más abajo.
**Antes de reintentar, hay que limitar threads (`OMP_NUM_THREADS`/
`ORT_NUM_THREADS`, session options de onnxruntime) o agregar frame-skip
(el detector usa timestamps reales, no índices de frame, así que es seguro
saltear frames) — correrlo tal cual de nuevo va a volver a tardar horas.**

Se agregó `extract_fall_confirmation_embeddings.py --max-toyota-clips=N` para
poder cortar la corrida antes de los clips largos, y guardado incremental
(`save()` después de cada video, no solo al final) para no perder progreso
si se corta.

Hallazgo aparte, más barato que terminar los 4 clips largos: los 39 clips
cortos del subset Toyota Smarthome en `videos/ToyotaSmarthome/*.mp4`
(dataset de actividades, no de caídas — no tenían `Fall.csv`) **sí generan
eventos geométricos falsos**, en particular los clips `Laydown_*` (acostarse
en cama/sillón): 14 eventos de 12 clips. Se generó
`videos/ToyotaSmarthome/Fall.csv` marcando las 39 actividades como `NoFall`
completo, para que
`extract_fall_confirmation_embeddings.py` las levante como negativos. Con
esto, el cache pasó de 150 caída real / 16 no-caída (≈9:1) a **110/36
(≈3:1)** — mejora real en el desbalance, con costo de cómputo mínimo (clips
de 1-2s en su mayoría) frente a los 4 clips largos.

Un primer reentreno sobre un cache parcial (cortado a mitad de GMDCSA-24,
positivos incompletos) dio 56.2% recall — número descartado, no comparable,
porque varios clips de caída real quedaron sin ningún evento registrado y
no pueden contar para el recall.

**Resultado final (2026-08-10), con GMDCSA-24 completo:** 218 clips
procesados (179 de antes + los 39 nuevos de Toyota, sin los 4 largos),
195 eventos geométricos, **150 caída real / 38 no-caída** (≈4:1, vs. el
≈9:1 original). Reentrenado con `train_fall_confirmation_probe.py`:
**77.3% recall (99/128), 4 FP** out-of-fold — mejora real de +6.2 puntos
sobre el 71.1% anterior, manteniendo los mismos 4 FP, agregando *solo* los
39 clips cortos de Toyota (sin necesitar los 4 clips largos que costaron
horas). Sigue por debajo del zero-shot en producción (82.0% recall / 96 FP
sobre los 183 clips completos, no directamente comparable en FP porque es
un subset distinto y más grande), pero la brecha se achicó bastante y
confirma que el desbalance de negativos era la causa real del mal
resultado inicial, no una limitación del método. `vlm/models/fall_confirmation_probe.npz`
quedó sobreescrito con este resultado — **sigue sin estar integrado en
`FallConfirmationClassifier`**, el filtro en producción sigue siendo el
zero-shot, sin cambios. Si en el futuro se consigue correr los 4 clips
largos (con frame-skip o límite de threads, ver arriba), vale reentrenar
de nuevo antes de decidir si reemplazar el zero-shot.

**Actualización (2026-08-11): GPU, dataset URFD, intento fallido de mejorar
la geometría para caídas hacia adelante, y hallazgo de que la geometría ya
llegó a su techo razonable.**

*GPU sin usar (bug encontrado y arreglado):* `FallImmobilityDetector` nunca
movía el modelo YOLO a GPU pese a haber una disponible (RTX 3050, CUDA) —
corría en CPU por default. Explica buena parte de la lentitud de intentos
anteriores con los 4 clips largos de `ToyotaSmarthome_FP` (3+ horas sin
terminar). Con `device=None` autodetectando CUDA, el mismo clip pasó a
~10 min. Con esto se pudieron procesar los 4 clips largos completos por
primera vez en varias corridas de recalibración.

*Dataset nuevo:* se sumó `videos/URFD/` (UR Fall Detection Dataset — 30
`fall-XX` + 40 `adl-XX`, cámara fija), elegido específicamente porque sus
secuencias ADL son "casi-caídas" diseñadas a propósito (sentarse,
agacharse). `vision/build_urfd_groundtruth.py` genera su `Fall.csv` a partir
de las etiquetas oficiales por frame. También se reorganizó `videos/` para
que cada dataset viva en su propia subcarpeta (`ToyotaSmarthome/` en vez de
clips sueltos en la raíz) — sin impacto en el código, que ya usaba globs
recursivos.

*Hallazgo real: la geometría no detecta caídas hacia adelante.* Diagnóstico
sobre clips aislados de CAUCAFall mostró que 9 de 10 sujetos "Fall forward"
generan **cero** eventos geométricos — la velocidad vertical sí dispara
(picos >1.8, muy por encima del umbral 0.12), pero `aspect_ratio` nunca
cruza 1.4 porque en una caída hacia la cámara la persona termina encogida
(bbox angosto), no ensanchada como en una caída lateral. Es un techo
estructural del heurístico de dos fases, no un problema de datos.

*Intento de arreglarlo, con dos vueltas de bugs de metodología antes de un
resultado confiable:*
1. Se probó una condición alternativa (OR con `aspect_ratio`): el bbox se
   derrumba en altura vs. una línea de base "de pie" (`height_drop_ratio`).
   Sola, no sirve — cualquier sentada/agachada normal también derrumba el
   bbox, dispara todo el tiempo.
2. Se agregó un segundo requisito: además del drop, la cadera tiene que
   terminar cerca del piso en términos absolutos (`floor_hip_y`), no solo
   relativo al bbox — sentarse en una silla deja la cadera a media altura,
   una caída real termina en el piso. El barrido inicial (`score()` capeando
   FP a 1 por clip) dio una mejora aparente: 86.1%→87.3% recall por
   46→55 FP. Validado con `evaluate_falls.py` (conteo real, sin capear) el
   número real fue 340 FP, no 55 — un solo clip largo (~19min) disparó
   **124 veces**, ~cada 9s en promedio. El capeo a 1 FP por clip en el
   barrido era ciego a que una condición puede repetirse sin límite dentro
   del mismo video, y eso es exactamente lo que pasa en video doméstico
   continuo. Se corrigió `score()` para contar cada evento.
3. Con el conteo corregido, los números seguían sin cerrar (159 predichos
   vs. 340 reales para los mismos umbrales) — la causa fue otro bug,
   `MAX_FRAMES_PER_CLIP = 10_000` en `recalibrate_fall_thresholds.py`
   (agregado como tope de seguridad contra videos corruptos) cortaba los
   clips largos a los primeros 400s de sus 1100-1600s reales, subestimando
   sus FP por ~3x sin ningún aviso. Subido a 100_000 (~66min) y reconstruido
   el cache completo.

*Resultado final, con cache completo y conteo real:* el baseline actual
(vel=0.12, asp=1.4, sin las condiciones nuevas) da **86.1% recall, 325 FP**
sobre los 292 clips (la mayoría de esos FP viene de los 4 clips largos de
Toyota — siempre generaron esa cantidad, nunca se había medido bien porque
no se procesaban completos). Ninguna combinación con `height_drop`/
`floor_hip_y` domina al baseline: todas suman FP por cada punto de recall
ganado (ej. 87.3% recall cuesta 340 FP, +15 FP por +1.2pp). Dado que el
criterio del proyecto es priorizar bajar falsos positivos sobre subir
recall (una alarma doméstica que grita en falso repetidamente pierde
confianza, ver riesgo de "boy who cried wolf"), **no se integró ningún
cambio a producción** — `FALL_HEIGHT_DROP_RATIO = None` en
`pose_pipeline.py` deja la condición nueva escrita pero desactivada,
documentada con este resultado. El heurístico geométrico queda exactamente
como estaba antes de esta sesión.

**Conclusión — dónde seguir a partir de acá:** el heurístico geométrico ya
está razonablemente cerca de su techo con una sola cámara 2D: geométricamente,
"arrodillarse" o "sentarse cerca del piso" y "estar tirado por una caída
real" se ven casi idénticos en un frame aislado, y seguir agregando
condiciones geométricas solo mueve el mismo trade-off recall/FP, no lo
mejora. Los dos ejes con margen real, para retomar:

1. **El filtro de confirmación (CLIP), no la geometría.** El heurístico
   geométrico solo necesita ser un disparador razonable — la decisión fina
   de "¿esto es realmente una caída?" la tiene que hacer algo que vea
   contexto visual (postura completa, entorno, si hay una silla debajo),
   que es justo lo que la geometría no puede. Ahí hay más margen: mejores
   prompts para el zero-shot, o seguir sumando negativos duros al linear
   probe experimental (ya se sumó URFD, ver arriba).
2. **Confirmación por inmovilidad sostenida, en vez de postura instantánea.**
   No se llegó a probar esta sesión. En vez de decidir en el instante del
   "drop" de altura, esperar 1-2s y ver si la persona se queda quieta en el
   piso vs. se reacomoda y sigue con lo suyo — separaría mejor "se arrodilló
   un segundo" de "se cayó y no se levantó", sin el disparo repetido que
   causó la explosión de FP en el intento de este punto (2).

**Actualización, más tarde el mismo día (2026-08-12): se probaron ambas
ideas del punto anterior — la de inmovilidad falló peor todavía, la de CLIP
dio el mejor resultado de toda la sesión.**

*Inmovilidad sostenida (punto 2), probada y descartada:* exigir que la
persona se quede quieta 1-2s después del "drop" antes de confirmar, en vez
de mejorar el trade-off, lo empeoró mucho más — con 2s de inmovilidad el
recall se derrumbó a 27.8% (bajaba los FP a 58, pero tirando 3 de cada 4
caídas reales). Causa: justo después de una caída real la pose estimada es
ruidosa (la persona se acomoda, keypoints con jitter), así que la condición
de "quietud" se rompe seguido incluso en caídas reales; varios clips además
son más cortos que la ventana de confirmación pedida. Implementado en
`recalibrate_fall_thresholds.py` (`immobility_confirm_s`) por si se retoma
con una versión menos estricta (ej. promedio de desplazamiento en vez de
"ningún frame se puede mover"), pero no se integró a producción.

*CLIP con más negativos duros (punto 1), resultado fuerte:* con el fix de
GPU de esta misma sesión (`FallImmobilityDetector` no usaba CUDA pese a
haber una disponible — ver arriba), finalmente se pudo correr
`extract_fall_confirmation_embeddings.py` con los 4 clips largos de
`ToyotaSmarthome_FP` incluidos (antes: 3+ horas sin terminar en CPU; con
GPU: ~10 min cada uno). Cache resultante: 509 eventos geométricos (antes
195), **177 caída real / 323 no-caída** (vs. el 150/38 anterior) — la
mayoría de los 323 son el ruido geométrico real de los clips largos
(coincide con los 325 FP geométricos medidos más arriba con el heurístico
sin cambios). Reentrenado con `train_fall_confirmation_probe.py`:
**84.8% recall (134/158), 30 FP** out-of-fold sobre 500 eventos etiquetados
— mejor resultado de la sesión: supera el recall del zero-shot en
producción (82.0%) entrenado sobre un conjunto de negativos mucho más duro
que el de intentos anteriores (323 eventos "no-caída" reales de video
continuo, no clips cortos limpios). Que falle en solo 30/323 sugiere que el
enfoque "CLIP ve el contexto visual completo" generaliza mucho mejor que
seguir afinando la geometría (ver los dos intentos fallidos de arriba).
`vlm/models/fall_confirmation_probe.npz` quedó sobreescrito con este
resultado — **sigue sin estar integrado en `FallConfirmationClassifier`**,
decisión pendiente para una sesión con más calma, no a esta hora. Si se
integra, evaluar contra el zero-shot con el mismo conjunto de evaluación
(actualmente no son directamente comparables en FP: el zero-shot se midió
sobre 183 clips, el probe sobre 292 con datasets adicionales).

**Actualización (2026-08-13): comparación justa hecha, probe integrado a
producción.** `vision/compare_confirmation_filters.py` (nuevo) resuelve el
pendiente de arriba: recalcula el margen zero-shot sobre exactamente el
mismo cache de 509 eventos/292 clips que usó el probe (los embeddings de
imagen ya estaban cacheados en `fall_confirmation_embeddings.npz`, no hizo
falta re-extraer nada), y compara ambos con el mismo criterio de recall/FP.
Zero-shot sobre este conjunto: 82.3% recall, 116 FP. Probe (out-of-fold):
**84.8% recall, 30 FP**. El probe domina en los dos ejes a la vez —no hay
trade-off que resolver según el criterio de "menos FP por sobre más
recall"— así que se integró directo: `FallConfirmationClassifier`
(`vlm/src/fall_confirmation.py`) ahora usa el probe si encuentra
`fall_confirmation_probe.npz` en `vlm/models/`, y cae al zero-shot
(`fall_confirmation_features.npy`) si no está presente (p. ej. una
instalación que nunca corrió `train_fall_confirmation_probe.py`). No
requirió cambios en `runner/camera_worker.py`: la interfaz `confirm(crop) ->
(bool, margin)` se mantuvo igual, solo cambió qué modelo calcula el margen
por dentro.

Pendiente real ahora: `vlm/models/` está en `.gitignore` (se regenera en la
máquina de desarrollo, no se versiona) — antes de desplegar a la Raspberry
Pi hay que asegurarse de copiar `fall_confirmation_probe.npz` junto con los
`.onnx`, no solo lo que genera `export_onnx.py` (que solo produce el
`.npy` del zero-shot, el probe se genera aparte con
`train_fall_confirmation_probe.py`).

**Actualización (2026-08-13): se probó la versión "soft" de inmovilidad
sostenida (punto 2 de la lista de arriba) — también descartada, peor que la
versión strict ya rechazada.** La idea pendiente era: en vez de cancelar el
candidato apenas UN frame se mueve más que el umbral ("strict", ver arriba,
2s de ventana → 27.8% recall), promediar el desvío de hip_y/altura de bbox
en TODA la ventana antes de decidir, para tolerar el jitter puntual de
keypoints justo después de una caída real sin dejar de exigir quietud
sostenida. Implementado en `simulate()` (`immobility_soft=True`,
`vision/recalibrate_fall_thresholds.py`) y barrido en
`vision/sweep_immobility_soft.py` (nuevo) sobre ventanas de 0.5-3.0s y
combinaciones de `immobility_hip_thresh`/`immobility_height_ratio`, todo
sobre el mismo cache de 292 clips.

Resultado: el promediado es **peor** que el strict a ventana equivalente, no
mejor. Ej. a 1.0s: strict daba 53.2% recall/93 FP, soft da 53.8% recall/
**141 FP** — prácticamente el mismo recall pero 52% más falsos positivos.
Ningún punto del frente de Pareto (17 combos no dominados) se acerca al
baseline sin inmovilidad (86.1% recall/325 FP). La explicación: promediar
no solo tolera el jitter post-caída (el efecto buscado), también tolera
mejor el movimiento real dentro de la ventana si el resto de los frames se
quedan quietos — exactamente lo que pasa seguido en video doméstico
continuo (alguien se agacha, se mueve un poco, se vuelve a quedar quieto),
así que termina confirmando más falsos positivos sin ganar recall real.

**Conclusión sobre el punto 2: la confirmación por inmovilidad sostenida
—tanto strict como soft— queda cerrada, no hay una versión de esta idea que
mejore el trade-off.** El problema de fondo no es el criterio de
cancelación (instantáneo vs. promediado) sino que la señal geométrica
post-caída es intrínsecamente ambigua entre "se cayó y no se levantó" y
"se movió normalmente" con solo hip_y/bbox_height_frac — para distinguirlas
hace falta ver la escena, que es lo que ya está resolviendo mejor el punto 1
(CLIP, ver arriba, 84.8% recall/30 FP, ya integrado a producción). No queda
ningún eje pendiente de la lista original del 2026-08-11: los dos puntos se
probaron a fondo, uno se integró (CLIP) y el otro se cierra definitivamente
(inmovilidad, ambas variantes).

## Entrada

Un archivo de video en `../videos/<nombre>.mp4` (o `.avi`/`.mov`), o toda
una subcarpeta si en el futuro se organiza `videos/` de nuevo
(`--folder <subcarpeta>`).

## Salida

`main.py` guarda `MonitoringEvent` (ver `../events/schema.py`) vía
`../events/store.py`, con:
- `video_id` = nombre de archivo sin extensión
- `video_timestamp_s` = segundos dentro del video (no timestamp de reloj)
- `event_type` = `fall` o `immobility`
- `metadata` = descriptores que motivaron la detección

## Setup

```
cd vision
py -m venv .venv && .venv\Scripts\activate
pip install -r requirements.txt -r ../events/requirements.txt
```

El primer uso descarga automáticamente los pesos `yolo11n-pose.pt`
(pre-entrenados, ~6MB) desde los servers de Ultralytics — no hay que
entrenar ni descargar nada a mano. Ver nota en la raíz del repo.

## Cómo probarlo

```
python main.py --video <nombre>.mp4   # un solo archivo
python main.py                        # todo videos/
```

La inspección visual (video anotado) es responsabilidad de `../evaluation/`,
no de este módulo.

## Fuera de alcance por ahora

- Ingesta RTSP / cámara en vivo (fase futura).
- Soporte multi-persona.
- Optimización para hardware embebido (Raspberry Pi).
