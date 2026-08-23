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
  anotado a mano por quien armó el dataset. Es un subset (solo caídas) del
  dataset GMDCSA-24; el `fall1-4` es la fuente que ya se usaba para calibrar
  el detector geométrico de `vision/`, sin tocar.
- **`GMDCSA-24/official/`** (2026-08-13, 81 caídas + 79 ADL, 4 actores en 3
  ambientes domésticos — dataset completo, [Zenodo, DOI
  10.5281/zenodo.11216408](https://zenodo.org/records/11216408), CC BY 4.0):
  se bajó para validar si la categoría `Standing` de `vlm/` (parado sin
  hacer nada) generalizaba a una cámara distinta a la de entrenamiento — no
  generalizó (0% recall) y la categoría se sacó por completo, ver
  "`Standing`: se probó y se descartó" en `vlm/README.md`. Queda acá sin
  usarse desde `vlm/` por ahora; es un candidato para completar el dataset
  de caídas de `vision/` (que documentaba el GMDCSA-24 completo como
  pendiente) si se retoma esa evaluación más adelante — no se integró en
  esta ronda, es trabajo aparte.
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

## Dataset local categorizado por actividad, en la raíz de `videos/` (2026-08-18)

Además de `ToyotaSmarthome/` (39 clips curados), se sumó directo en
`videos/` un dataset mucho más grande (~2200 clips, no versionado en git,
igual que el resto) organizado por el usuario en subcarpetas por actividad,
en español:

| Carpeta | Clips | Actividad real (por nombre de archivo) |
|---|---|---|
| `caminar/` | 730 | `Walk` |
| `beber/` | 521 | `Drink.Frombottle`/`Drink.Fromcan`/`Drink.Fromcup`/`Drink.Fromglass` |
| `cocinar/` | 288 | `Cook.Cleandishes`/`Cook.Cleanup`/`Cook.Cut`/`Cook.Stir`/`Cook.Usestove` |
| `sentarse/` | 207 | `Getup`/`Sitdown` (mezclados) |
| `comer/` | 139 | `Eat.Attable`/`Eat.Snack` |
| `mirar tele/` | 112 | `WatchTV` |
| `pararse (sentado)/` | 95 | `Getup` |
| `tomar pastillas/` | 46 | `Takepills` |
| `acostarse/` | 30 | `Laydown` |
| `levantarse (acostado)/` | 8 | `Getup` |
| `acostado/` | 2 | `Laydown` |

Los nombres de archivo dentro de cada carpeta siguen el patrón
`<Actividad>_pXX_...mp4` de siempre (ej. `Walk_p13_r07_v23_c07.mp4` dentro de
`caminar/`) — **la carpeta es solo organización del usuario, el label real
sigue siendo el nombre de archivo**, mismo criterio que ya establece "Por
qué no está organizado por postura" arriba: no se reintroduce la
clasificación manual por carpeta como ground truth, esta vez es agrupación
por actividad (no por postura) y coincide con el nombre de archivo en todos
los casos revisados.

**`Getup` aparece en 3 carpetas distintas** (`levantarse (acostado)/`,
`pararse (sentado)/`, y mezclado con `Sitdown` en `sentarse/`) — el usuario
separó "levantarse desde el piso/cama" de "levantarse desde una silla", una
distinción más fina que la que tiene el vocabulario de Toyota Smarthome
(`Getup` a secas, sin importar el punto de partida). Con los scripts
actuales (`vlm/extract_clip_features.py`, `fusion/extract_dataset.py`) esa
distinción se pierde — los tres grupos se entrenan como la misma clase
`Getup`. Ver la nota correspondiente en `fusion/README.md`, sección "Datos",
si se quiere retomar esto como clase nueva.

Usado el 2026-08-18 para reentrenar `vlm/models/category_probe.npz` (ver
"Reentrenamiento con dataset local categorizado por carpetas" en
`vlm/README.md`) y para entrenar por primera vez en serio
`fusion/models/temporal_classifier.pt` (ver "Estado" en `fusion/README.md`)
— exactamente el dataset grande que ambos README documentaban como
pendiente/en otra máquina.

Los archivos de video no se versionan en git (ver `.gitignore`), solo este
README, `ToyotaSmarthome/legacy_folder_labels.csv` y los
`Fall.csv`/`kaggle_data.csv` de ground truth (livianos, a diferencia de los
videos).
