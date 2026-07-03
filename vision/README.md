# `vision/` — Pose estimation, clasificación de postura y detección de caídas/inmovilidad

## Estructura

```
vision/
├── main.py               # pipeline de eventos (fall/immobility) → events/store.py
└── src/
    ├── capture.py          # lee frames de un archivo de video (VideoFile)
    ├── pose_pipeline.py    # YOLOv11n-pose + descriptores (PoseSample) + FallImmobilityDetector
    └── posture.py          # classify_posture(): parado / sentado / acostado a partir de un PoseSample
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
evaluables (1 quedó sin rango de tiempo parseable en el CSV original):

- **Recall: 91% (71/78)**
- **Falsos positivos: 1/78** (dentro de tramos explícitamente no-caída)
- **Latencia promedio: 1.2s** desde el inicio real de la caída hasta el evento

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

### 3. Inmovilidad prolongada — sin calibrar

Desplazamiento de keypoints por debajo de un umbral durante una ventana
larga (default 30s). Los umbrales (`IMMOBILITY_WINDOW_S`,
`IMMOBILITY_MOVEMENT_THRESHOLD`) siguen siendo un punto de partida de la
literatura, sin el mismo proceso de calibración que caídas/postura — falta
un dataset de clips largos con inmovilidad real para validar.

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
