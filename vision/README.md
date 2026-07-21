# `vision/` — Pose estimation, clasificación de postura y detección de caídas/inmovilidad

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
incorporadas a `detect_person()`. Un enfoque tipo Transformer entrenado
queda fuera de alcance por ahora (necesita miles de videos etiquetados que
no tenemos).

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
**No se tocaron los umbrales de caída para arreglar esto** — hacerlo sin el
mismo proceso de calibración con datos reales (`recalibrate_fall_thresholds.py`
con casos de "sentado bajo"/estos 4 clips como contraste) arriesgaría la
calibración ya validada sin el mismo rigor. **Este es el gap más urgente a
resolver de los que quedan abiertos en `vision/`** — más que sumar más
muestras de inmovilidad (que ya tiene 8/8 de validación) — porque compromete
una funcionalidad que ya se daba por calibrada y evaluada.

**Intento descartado, documentado para no repetirlo:** se probó usar clips
`Lay_down` de Toyota Smarthome (persona tapada con manta en un sillón, no
en el piso) como sustituto de "acostado post-caída". La detección de
persona fue prácticamente nula (1-22% según el clip) — pero al revisar los
frames a ojo, es oclusión por manta (solo se ve la cabeza), no una
limitación del modelo con posturas acostadas en general. No sirve como
proxy del escenario post-caída (cuerpo visible) y no se debe reintentar sin
buscar clips `Lay_down` donde el cuerpo quede visible.

Solo se trackea la primera persona detectada por frame — el informe excluye
explícitamente el caso multi-residente (Sec. 1.2.2).

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
