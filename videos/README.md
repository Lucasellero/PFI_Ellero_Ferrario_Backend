# `videos/`

Dataset de actividades cotidianas: `.mp4` planos, directo en `videos/`. El
nombre de archivo (sin extensión) es el `video_id` usado en
`events/schema.py`.

**Excepción:** varias subcarpetas con datasets externos de caídas, usados
para calibrar y evaluar `FallImmobilityDetector` (ver `vision/README.md`,
sección "Detección de caída como evento"). Cada uno trae su propio
`Fall.csv` (algunos generados por un script conversor, ver abajo) con el
segundo exacto de inicio/fin de cada caída — ground truth usado solo para
evaluar en desarrollo (`evaluate_falls.py`), no algo que el sistema tenga
disponible en producción:

- **`GMDCSA-24/fall1/` a `fall4/`** (16+25+21+17 = 79 clips, ~2-15s cada
  uno): caídas simuladas reales, mismo encuadre de cámara fijo. `Fall.csv`
  anotado a mano por quien armó el dataset.
- **`CAUCAFall/`** (100 clips: 10 sujetos × 10 actividades, 5 tipos de
  caída + 5 sin caída — Walk/Sit down/Hop/Kneel/Pick up object): entorno
  doméstico no controlado, distancia/ángulo de cámara variable por sujeto.
  Viene con etiqueta de clase por frame (`classes.txt` + `.txt` por frame,
  formato YOLO); `vision/build_caucafall_groundtruth.py` deriva el
  `Fall.csv` a partir de esas etiquetas (no viene con uno propio).
- **`Kaggle/`** (Multiple Cameras Fall Dataset, Auvinet et al. — 24 escenas
  grabadas con 8 cámaras simultáneas en una habitación amoblada tipo
  living, no institucional). Ground truth en `kaggle_data.csv` (frame de
  inicio/fin por cámara, bajado aparte de Kaggle porque `kagglehub` no trae
  los archivos sueltos del visor). **Pendiente, no incluido en la
  calibración todavía** — el FPS que reporta el `.avi` (120) da ventanas de
  caída sospechosamente angostas (~0.25s) y no se pudo confirmar si es
  correcto; hay además bastante corrupción de frames. Ver
  `vision/README.md` para el detalle y qué falta para retomarlo.
  `vision/build_kaggle_multicam_groundtruth.py` genera el `Fall.csv` cuando
  se resuelva; por ahora el archivo generado se llama
  `Fall.csv.pending` para que no se mezcle solo en las métricas.

Se mantienen separados del resto porque son datasets distintos, no parte
del dataset de actividades de Toyota Smarthome descripto abajo.

Sigue el patrón `Actividad_pXX_rXX_vXX_cXX.mp4` (actividad, participante,
repetición, video, cámara) — parece un subset del dataset **Toyota
Smarthome**.

## Por qué está plano y no organizado por postura

Hasta hace poco había subcarpetas `acostado/`/`parado/`/`sentado/`,
clasificadas manualmente por un tercero (no verificadas). Se usaron como
ground truth proxy para calibrar el clasificador de postura de `vision/`
(ver `vision/README.md`), pero al revisar visualmente los casos que ese
proceso marcaba como "error" se encontró que varios videos estaban en la
carpeta equivocada — por ejemplo, clips de `Cook.Cleanup`/`Drink.Fromcan`/
`Maketea.Insertteabag` etiquetados "sentado" mostraban a la persona
**parada** en la cocina, y clips como `Sitdown`/`Takepills` capturaban la
transición entre posturas, no una postura estable.

Conclusión: esa clasificación manual no es confiable como ground truth de
postura y generaba una falsa sensación de que el modelo estaba fallando
cuando en realidad estaba prediciendo correctamente. Por eso se sacó la
estructura de carpetas — **la referencia para evaluar el pipeline de acá en
adelante es el modelo mismo y su matriz de confusión sobre casos
verificados a mano (ver `vision/README.md`), no una categorización manual
previa sin validar.**

`legacy_folder_labels.csv` en esta carpeta guarda qué carpeta tenía
originalmente cada video (por si sirve de referencia histórica), pero no
debe tratarse como ground truth — es exactamente la clasificación que se
determinó poco confiable.

## Qué hay en el dataset

Actividades cotidianas grabadas con múltiples cámaras (`_c01` a `_c07`):
`Cook.Cleanup`, `Cook.Stir`, `Drink.Fromcan`, `Drink.Fromcup`,
`Maketea.Insertteabag`, `Readbook`, `Sitdown`, `Takepills`, `Uselaptop`,
`Usetablet`, `WatchTV`, `Enter`, `Walk`, `Laydown`. Es exactamente el tipo
de actividad sedentaria fina que `vlm/` tiene que distinguir más adelante
(misma postura, actividad distinta — ver Sec. 2.2.3 del informe).

Este dataset de actividades no tiene etiqueta de "caída" — son clips cortos
(~1-2s) que no capturan la transición completa. Para eso se usan los
datasets de caída descriptos arriba.

Los archivos de video no se versionan en git (ver `.gitignore`), solo este
README, `legacy_folder_labels.csv` y los `Fall.csv`/`kaggle_data.csv` de
ground truth (livianos, a diferencia de los videos).
