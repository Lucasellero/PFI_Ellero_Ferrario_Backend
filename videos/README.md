# `videos/`

Cada subcarpeta es un dataset de origen distinto. El nombre de archivo (sin
extensión) es el `video_id` usado en `events/schema.py` — no depende de en
qué subcarpeta esté el archivo.

- **`ToyotaSmarthome/`** (39 clips cortos, ~1-2s en su mayoría): dataset de
  actividades cotidianas, ver "Qué hay en el dataset" más abajo. Hasta
  2026-08-11 estaban sueltos directo en `videos/`; se movieron a su propia
  subcarpeta para que `videos/` siga el mismo patrón que el resto de los
  datasets (uno por subcarpeta) — ver "Por qué no está organizado por
  postura" más abajo sobre por qué **no** se reintrodujo una estructura de
  carpetas por postura/actividad.

Los datasets de caídas, usados para calibrar y evaluar
`FallImmobilityDetector` (ver `vision/README.md`, sección "Detección de
caída como evento"). Cada uno trae su propio `Fall.csv` (algunos generados
por un script conversor, ver abajo) con el segundo exacto de inicio/fin de
cada caída — ground truth usado solo para evaluar en desarrollo
(`evaluate_falls.py`), no algo que el sistema tenga disponible en
producción:

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

- **`URFD/`** (UR Fall Detection Dataset, Kepski & Kwolek — 70 clips: 30
  `fall-XX` + 40 `adl-XX`, cámara fija única `cam0`, ~3-15s cada uno):
  incluida específicamente porque las secuencias `adl-XX` cubren "casi
  caídas" (sentarse, agacharse, acostarse) etiquetadas a propósito como no-
  caída — el tipo de negativo duro que faltaba (ver `vision/README.md`,
  hallazgo con Toyota `Laydown_*`). `Fall.csv` generado con
  `vision/build_urfd_groundtruth.py`, a partir de las etiquetas por frame de
  `_meta/urfall-cam0-falls.csv` (columna `label`: `-1` antes de la caída,
  `0`/`1` durante y después) bajado del sitio oficial junto con los videos
  (`_meta/`, solo el CSV de metadata se versiona, liviano). El inicio de la
  ventana `Fall` es el primer frame con `label != -1`; el fin es el final
  del clip (la persona queda en el piso, no hay "se levanta" en estas
  secuencias cortas). Licencia CC BY-NC-SA 4.0, solo uso académico —
  https://fenix.ur.edu.pl/~mkepski/ds/uf.html.
- **`ToyotaSmarthome_FP/`** (4 clips completos, ~1100-1600s cada uno,
  `P07T10C02`/`P09T05C05`/`P07T05C05`/`P09T06C05` del dataset untrimmed de
  Toyota Smarthome): no son un dataset de caídas — son clips de actividades
  cotidianas (sentado mirando TV, leyendo) sin ninguna caída real, usados
  como **contraste** porque el detector de caídas disparaba falsos
  positivos repetidos en ellos (persona sentada en el piso/sillón bajo,
  bbox parecido a "acostado" — encontrado validando inmovilidad, ver
  `vision/README.md`). `Fall.csv` marca el clip entero como `NoFall`.

**`ToyotaSmarthome/Fall.csv` (2026-08-09):** cubre los 39 clips cortos de
actividades de esta subcarpeta, marcando cada uno como `NoFall` completo —
igual criterio que `ToyotaSmarthome_FP/Fall.csv`. Se generó porque, al
probar si estos clips cortos (1-2s en su mayoría) servían como negativos
baratos para el linear probe de confirmación de caídas (ver
`vision/README.md`), los clips `Laydown_*` resultaron sí disparar eventos
geométricos falsos — una fuente de negativos mucho más barata que los 4
clips largos de `ToyotaSmarthome_FP` (~horas en CPU). Detalle completo del
experimento y su resultado parcial en `vision/README.md`.

Los archivos de `ToyotaSmarthome/` siguen el patrón
`Actividad_pXX_rXX_vXX_cXX.mp4` (actividad, participante, repetición, video,
cámara) — parece un subset del dataset **Toyota Smarthome**.

## Por qué no está organizado por postura

Hasta hace tiempo `ToyotaSmarthome/` tenía subcarpetas
`acostado/`/`parado/`/`sentado/` dentro suyo, clasificadas manualmente por
un tercero (no verificadas). Se usaron como ground truth proxy para
calibrar el clasificador de postura de `vision/` (ver `vision/README.md`),
pero al revisar visualmente los casos que ese proceso marcaba como "error"
se encontró que varios videos estaban en la carpeta equivocada — por
ejemplo, clips de `Cook.Cleanup`/`Drink.Fromcan`/`Maketea.Insertteabag`
etiquetados "sentado" mostraban a la persona **parada** en la cocina, y
clips como `Sitdown`/`Takepills` capturaban la transición entre posturas,
no una postura estable.

Conclusión: esa clasificación manual no es confiable como ground truth de
postura y generaba una falsa sensación de que el modelo estaba fallando
cuando en realidad estaba prediciendo correctamente. Por eso se sacó esa
estructura de carpetas — **la referencia para evaluar el pipeline de acá en
adelante es el modelo mismo y su matriz de confusión sobre casos
verificados a mano (ver `vision/README.md`), no una categorización manual
previa sin validar.** La subcarpeta `ToyotaSmarthome/` (2026-08-11) agrupa
por dataset de origen, igual que `CAUCAFall/`/`URFD/`/etc. — no reintroduce
la clasificación por postura que se descartó.

`ToyotaSmarthome/legacy_folder_labels.csv` guarda qué carpeta de postura
tenía originalmente cada video (por si sirve de referencia histórica), pero
no debe tratarse como ground truth — es exactamente la clasificación que se
determinó poco confiable.

## Qué hay en el dataset (`ToyotaSmarthome/`)

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
README, `ToyotaSmarthome/legacy_folder_labels.csv` y los
`Fall.csv`/`kaggle_data.csv` de ground truth (livianos, a diferencia de los
videos).
