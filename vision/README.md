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

### 2. Detección de caída/inmovilidad como evento — siguiente etapa

`src/pose_pipeline.py` (`FallImmobilityDetector`) detecta, siguiendo el
enfoque de Giri et al. (2025) citado en el informe (Sec. 2.2.7):

- **Caída**: velocidad vertical de la cadera + aspect ratio + ángulo de
  torso superan umbrales en una ventana corta de frames — es decir,
  necesita ver la **transición** parado/sentado → acostado.
- **Inmovilidad prolongada**: desplazamiento de keypoints por debajo de un
  umbral durante una ventana larga (default 30s).

**Hallazgo importante:** los clips cortos de `videos/` (~1-2s) están
centrados en una postura ya alcanzada, no capturan el momento de la caída
— por eso el detector de caídas da 0 eventos sobre ellos, y es esperable,
no un bug. Hay un dataset con videos más largos y la secuencia completa
(parado → cayendo → acostado) para esta etapa — pendiente de usar acá.

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
