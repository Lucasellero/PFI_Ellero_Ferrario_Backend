# `videos/`

Dataset de actividades cotidianas: `.mp4` planos, directo en `videos/`. El
nombre de archivo (sin extensión) es el `video_id` usado en
`events/schema.py`.

**Excepción:** `videos/fall1/` a `videos/fall4/` (16+25+21+17 = 79 clips)
sí son subcarpetas — son clips de caídas reales, usados para calibrar y
evaluar `FallImmobilityDetector` (ver `vision/README.md`, sección
"Detección de caída como evento"). `fall3/` y `fall4/` incluyen además un
`Fall.csv` por carpeta con el segundo exacto de inicio/fin de cada caída y
de las actividades previas (Sitting/Walking/Standing) — ground truth
anotado a mano, usado solo para evaluar en desarrollo (`evaluate_falls.py`),
no algo que el sistema tenga disponible en producción. Se mantienen
separados del resto porque son un dataset distinto (caídas simuladas de
~2-15s), no parte del dataset de actividades de Toyota Smarthome descripto
abajo.

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
(~1-2s) que no capturan la transición completa. Para eso se usan
`videos/fall1/` y `videos/fall2/` (ver arriba).

Los archivos de video no se versionan en git (ver `.gitignore`), solo este
README y `legacy_folder_labels.csv`.
