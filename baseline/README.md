# `baseline/` — Línea base de comportamiento y detección de desvíos

## Qué hace

Consume eventos ya almacenados y construye una línea base de comportamiento mediante clustering
dinámico (ver Sec. 2.1.7 y 2.2.5 del informe, enfoque de Snoun et al. 2024 — clustering con
ventana móvil y "behavioral drift"), para luego detectar desvíos significativos respecto al
patrón habitual.

**Nota de honestidad académica**: el texto completo de Snoun et al. 2024 ("A Deep Learning Based
System For a Long-term Elderly Behavioral Drift Detection", SN Computer Science) está detrás de
un paywall — solo se pudo confirmar el enfoque a nivel de abstract (clustering dinámico iterativo
con ventana móvil, comparando cada día observado contra el patrón de rutina del período anterior
vía un score de similitud). Lo que implementa este módulo es una **interpretación propia fiel a
ese enfoque documentado**, no una reproducción de sus fórmulas exactas (no publicadas en acceso
abierto). Aclarar esto en el informe si se cita este módulo como implementación de ese paper.

## Fuente de datos (2026-09-05)

Se desarrolló y validó directamente contra
[`dataset_events/`](../dataset_events/README.md) (CASAS, 319k eventos reales, 80 residentes) —
no contra eventos sintéticos ni contra la salida de `vision/`/`vlm/` todavía. Motivo: al momento
de esta iteración, mejorar la clasificación de actividad de la cámara (más de las 4 categorías
actuales, ver [`fusion/PLAN_sustitucion_clip.md`](../fusion/PLAN_sustitucion_clip.md)) estaba
bloqueado porque el dataset de video grande necesario para medirlo no estaba disponible en la
máquina de desarrollo — así que se adelantó `baseline/`, que no tenía ese bloqueo.

El diseño es agnóstico de la fuente por construcción (ver "Diseño" abajo): el mismo código sirve
el día de mañana con eventos reales de cámara (cuando tenga más de 4 clases), sin reescritura,
solo cambiando el adapter de entrada — el objetivo de comparación "desvíos detectables con
sensores vs. solo con cámara" que plantea `fusion/PLAN_sustitucion_clip.md` sigue pendiente hasta
que exista esa fuente alternativa.

Usa la clase **canónica** de CASAS (`actividad`, 19 clases — `dataset_events/activity_map.py`),
no la gruesa: es lo que `dataset_events/README.md` ya documenta como el nivel que debe consumir
este módulo.

## Diseño

### 1. Vector de "día" (`src/features.py`)

Por `(hogar, fecha)`: para cada una de las 19 clases canónicas, duración total en minutos y
cantidad de eventos → vector de 38 dims (`build_day_vectors()`). Los días con menos de
`COBERTURA_MIN_HORAS` (4h) de actividad etiquetada se marcan `cobertura_baja` y se excluyen tanto
de construir la rutina como de evaluarse — no se inventa actividad en huecos sin dato (mismo
criterio que el `sin_dato` del ETL de `dataset_events/build_table.py`).

Simplificación consciente: la duración de un evento se atribuye entera al día de su inicio, aunque
cruce la medianoche (ej. dormir 23:40→07:00 cuenta todo para el día de las 23:40) — no se
descompuso por franja horaria en esta primera iteración porque los tres escenarios de validación
(medicación, baño nocturno, salir de casa) ya se resuelven con duración/conteo por clase sin esa
finura (`bano_nocturno` ya es una clase canónica separada de `bano` en el ETL, la sensibilidad
horaria de esa distinción específica viene resuelta de antes).

### 2. Clustering dinámico multivariado (`src/dynamic_clustering.py`)

Por residente, día a día, causal (nunca mira el futuro):

1. **Ventana móvil**: los `window_days` días previos (default 30) del mismo residente, sin
   incluir el día evaluado.
2. **Reclustering iterativo**: `KMeans(k)` (default `k=2`) sobre esa ventana — se reajusta en
   cada paso, así que si la rutina cambia de forma sostenida los clusters de referencia se corren
   solos con la ventana en vez de comparar para siempre contra un patrón fijo.
3. **Distancia** del día evaluado (features estandarizadas con `StandardScaler` ajustado sobre la
   misma ventana) al centroide más cercano.
4. **z-score contra el historial propio del residente** (cada persona tiene volumen/rutina de
   base muy distinto, la distancia cruda no es comparable entre hogares).
5. `z_score > threshold_z` (default 2.5) → posible desvío. Guarda además qué 3 dimensiones más
   contribuyeron a la distancia (`top_contribuyentes`), para poder decir *qué* cambió.

### 3. Desvío por clase individual (`src/class_deviations.py`) — agregado tras medir

**Hallazgo real durante la validación** (ver "Estado" abajo): el detector multivariado nunca
marcaba el escenario "sin medicación" en ningún punto del barrido, aunque `medicacion_n`/
`medicacion_min` sí aparecían como el contribuyente principal de la distancia. Causa: con 38
dimensiones, un cambio real pero concentrado en una sola clase de bajo volumen diario (minutos de
medicación son chicos frente a `trabajo`/`ver_tv`/`dormir`) queda diluido por el ruido cotidiano
normal de las otras 37 clases — el "día completo" no se ve tan distinto en agregado.

Se agregó un segundo detector, complementario (no reemplaza al multivariado): evalúa cada clase
de `TRACKED_CLASSES` (`medicacion`, `salir_casa`, `bano_nocturno` — las 3 clínicamente relevantes
de los escenarios de validación) **por separado**, contra su propia mediana/MAD histórica en la
misma ventana móvil, marcando `ausencia` (hoy en 0, cuando históricamente ocurre casi todos los
días), `ausencia_sostenida` (varios días seguidos en 0) o `exceso` (muy por encima de lo habitual).

Se probó primero con un conjunto más amplio de clases (sumando `comer`/`beber`/`dormir`) y un
umbral más laxo — dio 93% de recall pero **52% de falsos positivos**, inaceptable. Causa
diagnosticada: el conteo de eventos de clases como `dormir` es ruidoso por el corte de día
(un evento que cruza la medianoche puede contarse como 0 eventos en el día calendario que
"debería" tenerlo) — genera ausencias falsas que no son un desvío real. Se acotó `TRACKED_CLASSES`
a las 3 clases realmente necesarias y se subió `frac_historica_min` hasta bajar el FP a un rango
razonable (ver barrido en "Estado").

Un día cuenta como desvío si **cualquiera de los dos detectores** lo marca (`evaluate_deviations.py`,
columna `combinado`).

### 4. Qué no se toca en esta iteración

- `events/schema.py` no se modifica — `baseline/` no escribe eventos de vuelta al store de
  `MonitoringEvent` todavía. La salida vive en sus propios artefactos
  (`baseline/synthetic_eval/*.parquet`, `grid_results.csv`), no como un nuevo `EventType`. Se
  evita tocar el contrato compartido antes de validar el enfoque.

## Validación: desvíos sintéticos (no hay ground truth real de "día anómalo" en CASAS)

Mismo patrón que ya usa el resto del proyecto para medir sin dataset etiquetado de antemano
(hard-negative mining en `fusion/extract_falls.py`, grid search en
`vision/recalibrate_fall_thresholds.py`): perturbar copias de datos reales de forma controlada y
medir recall/FP contra esas perturbaciones.

- **`inject_synthetic_deviations.py`**: sobre los 7 hogares recomendados por
  `dataset_events/README.md` (`hh113`, `hh115`, `hh117`, `hh120`, `hh109`, `hh101`, `mn83`),
  genera 3 escenarios sobre días reales elegidos al azar — `sin_medicacion` (medicación puesta a
  0), `bano_nocturno_excesivo` (conteo subido bien por encima del histórico), `semana_sin_salir`
  (7 días consecutivos con `salir_casa`/`entrar_casa` en 0) — más una muestra de 20 días reales
  sin tocar por hogar (`real_sin_tocar`) como negativos. Los días perturbados nunca contaminan la
  ventana de referencia de otros días (`exclude_from_reference` en ambos detectores) ni el
  historial de z-score del residente — se evalúan contra la rutina real, no una rutina ya
  corrompida por la propia inyección.
- **`evaluate_deviations.py`**: corre ambos detectores sobre ese set, barre
  `(window_days ∈ {14,21,30}, k ∈ {2,3,4}, threshold_z ∈ {2.0,2.5,3.0,3.5})`, reporta recall/FP
  del multivariado solo, del detector por clase solo, y del combinado — mismo framing recall/FP
  que `vision/evaluate_falls.py`.

## Estado (2026-09-05)

Implementado y validado con desvíos sintéticos sobre CASAS. 202 días evaluados (59 positivos —
6 `sin_medicacion`, 7 `bano_nocturno_excesivo`, 49 días de `semana_sin_salir`, repartidos en 7
semanas — y 121 negativos `real_sin_tocar`; 22 quedaron sin score por historial insuficiente).

**Mejor combinación del barrido** (`window_days=30, k=2, threshold_z=2.5`, defaults actuales de
`DriftConfig`; `min_window_days=7` — período de silencio de 7-10 días decidido con el usuario,
ver "Período de aprendizaje" abajo):

| Detector | Recall global | FP | `sin_medicacion` | `bano_nocturno_excesivo` | `semana_sin_salir` |
|---|---|---|---|---|---|
| Multivariado solo | 16.7% | 2.3% | 0% | 28.6% | 16.3% |
| Por clase solo | 76.7% | 9.2% | 75.0% | 14.3% | 85.7% |
| **Combinado** | **86.7%** | **11.5%** | **75.0%** | **42.9%** | **93.9%** |

(Con `min_window_days=14` el combinado daba 89.8%/10.7% — el recorte a 7 días de silencio pierde
~3pp de recall a cambio de que el sistema empiece a opinar antes sobre una persona nueva.)

Lecturas honestas:

- El detector multivariado por sí solo es débil para estos 3 escenarios — captura mejor cambios
  repartidos en varias clases a la vez ("la rutina en general se ve distinta"), no ausencias/
  excesos concentrados en una sola clase. Sigue siendo la pieza que cubriría desvíos que el
  detector por clase no está mirando (cambios de horario generalizados, redistribución de tiempo
  entre actividades sin que ninguna llegue a 0 o exceso).
- El detector por clase es el que hace la mayor parte del trabajo en estos 3 escenarios — por
  diseño: son justo el tipo de señal sparse/concentrada para la que se creó.
- `bano_nocturno_excesivo` es el más flojo (50% combinado) — **solo 7 muestras**, no alcanza para
  confiar en el número; además la métrica de exceso depende del MAD histórico de una clase que
  ya es esporádica de por sí (varios residentes tienen 0 baños nocturnos la mayoría de las
  noches), lo que hace más ruidoso detectar "más de lo normal" cuando "lo normal" ya es cero la
  mayoría de los días.
- FP del 10.7% sobre días reales sin tocar: aceptable como punto de partida, pero no bajo — antes
  de producción convendría una segunda vuelta de calibración con más días negativos (acá solo 121)
  y, si se define el sesgo hacia menos FP que ya rige la calibración de caídas del proyecto, un
  barrido más fino alrededor de `threshold_z`/`excess_k` priorizando bajar FP sobre subir recall.

Ver `synthetic_eval/grid_results.csv` (no versionado, se regenera con `evaluate_deviations.py`)
para el barrido completo.

**Nota**: estos números son del estado con 3 escenarios (`medicacion`/`bano_nocturno`/
`salir_casa`); la sección "Frecuencia/omisión de eventos" más abajo los actualiza al sumar
`comer`/`deficit` y recalibrar — usar esa tabla como la vigente, esta queda como registro de cómo
se llegó ahí.

## Período de aprendizaje (arranque en frío)

Todo lo de arriba se validó contra CASAS, que ya tiene meses de historial por residente. Una
persona nueva monitoreada por cámara real arranca sin ningún historial — el sistema no puede (ni
debe) opinar sobre "lo normal" de alguien que recién empieza a ver. Decisión de producto (con el
usuario, 2026-09-05): **7-10 días de silencio total** antes de que cualquier detector empiece a
emitir desvíos — es lo que fija `min_window_days` en `DriftConfig`/`ClassDeviationConfig` (7).
Acortar el período de silencio de 14 a 7 días le cuesta al sistema ~3pp de recall combinado
(89.8%→86.7%, ver tabla de arriba) a cambio de empezar a opinar antes — trade-off aceptado.

`timing_deviations.py` usa una ventana en **cantidad de eventos** (`window_events`/
`min_eventos_historial`), no en días — no hay una traducción exacta a "7-10 días" porque cada
clase ocurre con frecuencia distinta (medicación ~2/día, beber varias veces/día). Pendiente
alinear este detector al mismo criterio de período de silencio en días cuando se calibre en serio
(ver "Horario habitual" abajo, estado parcial).

## Horario habitual (`src/timing_deviations.py`) — estado parcial, en curso

**Estado al 2026-10-13: implementado, pero sus alertas están apagadas por defecto** en el
proceso diario (ver "Actualización 2026-10-13 (3)" y "(4)"): medido con validación cruzada por
hogar, con las clases de cámara avisa en dos de cada tres días normales. Lo que sigue es el
registro de cómo se desarrolló.

Detector adicional (2026-09-05, agregado a pedido del usuario): además de cuánto/cuántas veces y
la forma general del día, mira **a qué hora** ocurre cada actividad, aprendiendo el horario
habitual de cada `(hogar, actividad)` de su propio historial (clustering circular de horarios en
hasta `k_time_slots` franjas, para no promediar desayuno+almuerzo+cena como si fueran una sola
franja). No perturba nada del dataset real — el único "toqueteo" (correrle el horario a una copia
de eventos) es, igual que en los otros dos detectores, un banco de pruebas para medir si el
mecanismo funciona, no la lógica de producción.

**Hallazgo real** (`evaluate_timing_deviations.py`, corriendo horarios ±3h sobre eventos reales de
7 hogares): la señal es fuerte para actividades con horario genuinamente disciplinado y floja para
las que naturalmente ocurren "a cualquier hora":

| Clase | Recall | FP |
|---|---|---|
| `medicacion` | **55.6%** | **2.1%** |
| `comer` | 42.9% | 21.4% |
| `dormir` | 38.1% | 23.2% |
| `salir_casa` | 38.1% | 12.5% |
| `beber` | 16.7% | 6.3% |

Mezclando las 5 clases parejo, el mejor punto del barrido combinado da solo 38% de recall con
11-15% FP — muy por debajo de lo que se ve por clase. Pendiente (próxima sesión): acotar
`DEFAULT_TRACKED_CLASSES` a las clases donde el horario realmente importa (`medicacion` es la más
clara; `comer`/`dormir`/`salir_casa` con calibración por-clase en vez de parámetros compartidos)
en vez de un único conjunto de parámetros para las 5 — mismo criterio que ya se aplicó en
`class_deviations.py` para acotar `TRACKED_CLASSES`.

## Frecuencia/omisión de eventos (`src/class_deviations.py`, extendido 2026-09-05)

Se agregó un tercer tipo al detector por clase, simétrico a `exceso`: **`deficit`** — la clase
ocurrió hoy, pero bastante menos veces de lo habitual (`0 < hoy < mediana - excess_k * sigma`),
distinto de `ausencia` (=0). Pensado para "se saltó una comida" sin dejar de comer nada en todo el
día: CASAS fusiona desayuno+almuerzo+cena en una sola clase `comer` (~3 eventos/día típico), así
que un día con `comer_n=1` no es `ausencia` y probablemente tampoco mueve lo suficiente la
duración total para que `dynamic_clustering.py` lo note (mismo problema de dilución que ya se
había diagnosticado con `medicacion`).

Se sumó `comer` a `TRACKED_CLASSES` (antes solo `medicacion`, `salir_casa`, `bano_nocturno`) y se
agregó el escenario **`comida_salteada`** a `inject_synthetic_deviations.py` (baja `comer_n` a un
valor bajo, no a 0, manteniendo proporcional la duración por evento). Se recalibró
`(frac_historica_min, excess_k, dias_consecutivos_ausencia)` con esta clase sumada — el barrido
ahora vive en `evaluate_deviations.py` (antes se corría a mano fuera del repo).

**Resultado** (`frac_historica_min=0.9, excess_k=4.0, dias_consecutivos_ausencia=2`, defaults
actuales de `ClassDeviationConfig`), sobre 200 días (73 positivos, 127 negativos — incluye el
escenario `comida_camara_ausente` sumado más abajo en "Vocabulario de cámara", que también corre
en modo sensor):

| Detector | Recall global | FP | `sin_medicacion` | `bano_nocturno_excesivo` | `comida_salteada` | `comida_camara_ausente` | `semana_sin_salir` |
|---|---|---|---|---|---|---|---|
| Multivariado solo | 12.3% | 1.6% | 0% | 42.9% | 14.3% | 20.0% | 8.2% |
| Por clase solo | 72.6% | 5.5% | 100% | 14.3% | 14.3% | 80.0% | 85.7% |
| **Combinado** | **78.1%** | **7.1%** | **100%** | **57.1%** | **28.6%** | **80.0%** | **85.7%** |

Mejora el FP (7.1% vs. 8.8% antes de sumar `comida_camara_ausente`) — el recall global bajó un
poco (78.1% vs. 85.1%) porque este escenario nuevo es más difícil en promedio, no porque haya
empeorado la calibración de los demás (compararlos escenario por escenario contra la versión
anterior, arriba en el historial de commits, confirma que ninguno empeoró). `comida_salteada`
sigue siendo el más débil (28.6%, 7 casos de prueba, no alcanza para un número definitivo).

## Tendencia gradual (`src/trend_deviations.py`, nuevo 2026-09-05)

Los 3 detectores anteriores comparan **un punto en el tiempo** (un día, un evento) contra su
historial reciente — ninguno detecta un cambio **lento y sostenido a lo largo de semanas**, que es
el caso clínico que más enfatiza el informe (Sec. 2.1.2). Este detector agrega los vectores-día a
nivel de semana ISO y compara la mediana de un bloque de `n_semanas_recientes` (default 3) contra
la mediana del bloque de `n_semanas_base` (default 8) inmediatamente anterior — mismo criterio
robusto (mediana + MAD) que `class_deviations.py`, aplicado a semanas en vez de días. No asume de
antemano qué dirección (`sube`/`baja`) es clínicamente mala por clase — queda para una capa de
mensajes futura.

`TRACKED_CLASSES` por defecto: `salir_casa`, `visitas`, `dormir`, `descansar` — las primeras tres
por señal directa de mobilidad/aislamiento/sueño; `visitas` se sumó de forma oportunista para
cubrir aislamiento social (no era una prioridad en sí misma, pero cuesta ~0 agregarla al mismo
detector).

**Solo se pudo validar sobre 3 hogares** (`hh113` ~70 semanas, `hh115` ~43, `hh117` ~34) — los
otros 4 hogares recomendados por `dataset_events/README.md` solo tienen ~9 semanas de historial,
insuficiente para separar un bloque base de 8 + reciente de 3 con margen.

**Validación** (`inject_gradual_trend.py`): sobre cada `(hogar, clase)` con historial suficiente,
se aplica un factor sostenido (×0.4 o ×2.0, al azar) a un bloque real de 3 semanas consecutivas —
no un día suelto — y se mide si `score_trend_deviations` lo marca en la semana exacta donde
termina ese bloque.

**Resultado** (12 positivos, 12 negativos — muestra muy chica, un evento por `hogar×clase`,
**no una calibración robusta, un primer punto de partida**):

| Clase | Detectado |
|---|---|
| `dormir` | 3/3 |
| `salir_casa` | 2/3 |
| `descansar` | 1/3 |
| `visitas` | 0/3 |

Recall global 50%, FP 8.3% (mejor punto del barrido: `n_semanas_recientes=3, n_semanas_base=8,
threshold_z=2.0`). `visitas` no detectó ningún caso — mismo patrón que `beber`/`bano_nocturno` en
los otros detectores: pocos eventos por semana hacen que la mediana/MAD histórica sea poco
informativa (a menudo 0). Se mantiene en `TRACKED_CLASSES` documentado como señal floja, no se
sacó — igual criterio de transparencia que el resto del módulo.

**Bug real encontrado y corregido en el camino**: la primera versión de `exclude_from_reference`
sacaba los días perturbados de *toda* la base de datos antes de armar las semanas — incluida la
propia semana que se quería evaluar como "reciente", así que el detector nunca podía ver la señal
que se le estaba pidiendo detectar (0% de recall en la primera corrida). Corregido para que la
exclusión solo afecte qué semanas pueden usarse como referencia **base**, nunca la evaluada.

## Vocabulario de cámara (2026-09-05) — reconciliación de dos documentos que no coincidían

Al revisar con el usuario qué desvíos son realmente detectables **sin sensor dedicado, solo con
cámara**, se encontró que `fusion/PLAN_sustitucion_clip.md` (el plan de subir de 4 a ~10 clases
de actividad) y `dataset_events/activity_map.py` (el mapeo canónico→grueso que ya usaba este
módulo) tenían vocabularios distintos y nunca se habían reconciliado:

- `activity_map.py` fusionaba `salir_casa`+`entrar_casa` en un estado **`fuera_casa`** con
  duración — no tiene sentido: la cámara nunca ve "afuera", solo el instante de cruzar la puerta.
  **Corregido**: quedan como dos eventos puntuales propios, `salir_casa`/`entrar_casa`.
- Vocabulario final acordado (9 clases + `otro`, ver `COARSE` en `activity_map.py`): `dormir`,
  `descanso_sofa`, `ocio_sentado`, `cocina`, `comida_mesa`, `comida`, `aseo`, `salir_casa`,
  `entrar_casa`. Se sacaron `visitas` (multi-residente fuera de alcance, ver `vision/README.md`)
  y `tareas_hogar` (no estaba en el vocabulario acordado) — ambas quedan mapeadas a `otro`.
  `bano_nocturno` se fusiona con `aseo` ("de noche" es derivable del horario del evento, no
  necesita ser una clase de escena aparte). `moverse` queda fuera de este mapeo (no tiene análogo
  en los sensores de CASAS, solo existe del lado de `vlm/`).
- **`comer` → `comida_mesa`, separado de `beber`/`medicacion` → `comida`**: idea del usuario —
  comer dura más y pasa en una zona fija (la mesa), a diferencia de un trago o una pastilla
  (gesto breve, cualquier zona). **Depende de la Fase 1 de zonas del hogar
  (`fusion/PLAN_sustitucion_clip.md`), todavía no implementada** — sin esa señal la cámara no
  puede confirmar "está en la mesa", así que `comida_mesa` es aspiracional, no algo para producción
  todavía.

### Por qué importa: `medicacion` no es alcanzable por cámara sola

Los 3 detectores de este módulo estaban calibrados usando `medicacion` como clase canónica de
CASAS — un sensor separado de comer/beber. Pero **ninguna cámara única puede distinguir tomar una
pastilla de tomar un trago de agua** (ya documentado en `vlm/README.md`: gesto de <2s, casi
siempre ocluido). Todo lo detectado como "`sin_medicacion`" en las secciones anteriores es válido
solo si algún día se suma un sensor dedicado (ej. un pastillero inteligente) — no es lo que la
cámara puede dar hoy ni con el plan de zonas.

### `features.add_camera_columns()` y `class_deviations.CAMERA_TRACKED_CLASSES`

Se agregó una función que suma las columnas canónicas del vector-día a nivel de clase gruesa
(prefijo `cam_*`, sin pisar las columnas canónicas — las dos conviven en el mismo DataFrame) y un
segundo conjunto de clases trackeadas, `CAMERA_TRACKED_CLASSES = ("cam_comida", "cam_comida_mesa",
"cam_salir_casa", "cam_aseo")`, para poder correr el mismo `class_deviations.py` en modo "sin
sensor". `evaluate_deviations.py --camera` corre esta variante.

**Dos bugs reales encontrados y corregidos armando esto:**
1. Las columnas `cam_*` se calculaban una sola vez, antes de inyectar los desvíos sintéticos — un
   escenario que perturbaba una columna canónica (ej. `semana_sin_salir` sobre `salir_casa_n`) no
   se reflejaba en su columna `cam_*` correspondiente, que quedaba con el valor viejo. Corregido:
   `add_camera_columns()` se recalcula recién después de inyectar todos los escenarios, no antes.
2. El escenario `sin_medicacion` original solo zeroeaba `medicacion`; para la versión cámara se
   necesitaba un escenario nuevo (`comida_camara_ausente`) que zerea **comer+beber+medicación
   juntos** — la única forma realista de que la cámara vea "no hubo actividad de comida" (si solo
   se zerea `medicacion`, la persona sigue comiendo/bebiendo normal y la cámara no nota nada raro).

**Resultado** (`frac_historica_min=0.8, excess_k=4.0, dias_consecutivos_ausencia=2` — elegido
sobre el mejor punto del barrido priorizando menos FP, mismo criterio de todo el módulo):

| Escenario | Recall modo cámara | Recall modo sensor (referencia) |
|---|---|---|
| `semana_sin_salir` | 83.7% | 85.7% |
| `comida_camara_ausente` (comer+beber+medicación, las 3 a la vez) | 71.4% | 80.0% |
| `sin_medicacion` (residual, vía `cam_comida` diluido) | 50.0% | 100% |
| `bano_nocturno_excesivo` | 14.3% | 57.1% |
| `comida_salteada` | 14.3% | 28.6% |
| **Global** | **67.1%** | **10.0% FP** |

Lecturas honestas: `semana_sin_salir` casi no pierde nada al pasar a cámara (salir/entrar de casa
es un evento limpio, no depende de fusionar clases). `comida_camara_ausente` funciona razonable
como sustituto realista de "sin_medicacion" cuando la ausencia es total (no comió/bebió/medicó
nada en absoluto). Pero cuando el cambio es más sutil — solo faltó la medicación, o solo una
comida — la señal se diluye mucho al fusionar clases (50%→ y 14.3% respectivamente) o al fusionar
`bano_nocturno` con `aseo` (14.3%): es el mismo problema de dilución de siempre, ahora a nivel de
vocabulario en vez de a nivel de dimensiones del multivariado.

## Cross-validation por hogar (2026-09-05) — número final, reemplaza a los de arriba

Todos los resultados de las secciones anteriores tienen un problema real que el usuario señaló:
(1) se calibraron sobre solo **7 de los 80 hogares** disponibles en CASAS (los "recomendados" por
`dataset_events/README.md` por calidad de datos, sin más motivo que eso), y (2) el barrido de
`ClassDeviationConfig` elegía los parámetros **mirando los mismos datos** con los que después se
medía el resultado — sesgo optimista clásico, no muy distinto de reportar accuracy de train.

**Corrección aplicada:**

1. **Más hogares**: `inject_synthetic_deviations.py` ahora usa los **80 hogares** del parquet por
   defecto (antes, los 7 "recomendados" de `dataset_events/README.md`). De los 80, **21 pasan el
   mínimo de historial** (`MIN_DIAS_POR_HOGAR=60`) — la mayoría de los 80 hogares de CASAS tienen
   solo semanas de datos (ver "Cobertura por hogar" en `dataset_events/README.md`), no meses. 640
   etiquetas totales (220 positivas) sobre 4362 días, contra las 200/73 de antes.
2. **`cross_validate()` en `evaluate_deviations.py` (`--cv N`)**: k-fold **por hogar** (mismo
   criterio que `GroupKFold` por video en `vision/`/`fusion/`, no k-fold por día — evita que días
   del mismo hogar terminen repartidos entre entrenamiento y test). En cada fold: el barrido de
   `ClassDeviationConfig` se corre solo con los hogares de *entrenamiento* del fold, y el resultado
   se mide solo sobre los hogares de *test* (que nunca influyeron en la elección de parámetros).
   Las predicciones de test de los 5 folds se juntan (pooled) para un recall/FP "out-of-fold"
   único, más robusto que promediar tasas de folds de tamaño desigual.

**Resultado** (5 folds, `--cv 5`):

| | Recall (antes, mismo dataset) | **Recall out-of-fold** | FP (antes) | **FP out-of-fold** |
|---|---|---|---|---|
| Modo sensor | 78.1% | **71.4%** | 7.1% | **14.2%** |
| Modo cámara | 67.1% | **70.5%** | 10.0% | **14.3%** |

El FP casi se **duplica** en modo sensor al corregir el sesgo (7.1%→14.2%) — es la cifra real, la
de antes estaba inflada por calibrar y medir con los mismos hogares. El recall se mantiene
razonablemente estable en los dos modos. Curiosamente el modo cámara con más hogares mejora un
poco su recall respecto al número anterior (67.1%→70.5%) — con 21 hogares en vez de 7 hay más
variedad de patrones reales de "ausencia total de comida", el escenario donde mejor rinde.

**Estabilidad de la calibración entre folds**: los parámetros elegidos en cada fold coinciden
bastante (`frac_historica_min=0.8, excess_k=3.0` en 3 de 5 folds del modo sensor; `frac_historica_min=0.6,
excess_k=4.0` en 3 de 5 del modo cámara) — no es una calibración inestable que cambia
completamente según qué hogares se usen, buena señal de que el método generaliza. Esos valores
mayoritarios quedaron como default final de `ClassDeviationConfig` (modo sensor) — para modo
cámara, instanciar `ClassDeviationConfig(frac_historica_min=0.6, excess_k=4.0)` explícitamente
(no hay un dataclass separado todavía para el modo cámara).

**Por escenario, out-of-fold, modo sensor**: `sin_medicacion` 88.9%, `comida_camara_ausente` 90.0%,
`semana_sin_salir` 77.6%, `bano_nocturno_excesivo` 52.4%, `comida_salteada` 20.0% (la más floja se
mantiene floja, consistente con lo ya documentado).

**Lo que sigue sin resolverse**: 14% de FP es alto para un producto real, incluso siendo ahora el
número honesto — antes de citar esto en el informe como resultado final conviene una vuelta más
sesgada explícitamente hacia bajar FP (el barrido interno ya prioriza recall dentro del límite de
FP, pero el límite mismo, `max_fp=0.15`, se puede bajar). `synthetic_eval/grid_results*.csv`
quedan como el barrido "en la misma data" (útil para explorar el espacio de parámetros), pero
**el número a citar es el de esta sección**, no los de las tablas anteriores.

### Actualización 2026-09-20 — separar semana/fin de semana, y por qué bajar `--max-fp` no alcanza

Antes de tocar el diseño, se probó simplemente bajar el límite de FP en la selección de
parámetros (`--cv 5 --max-fp 0.08`) — **empeoró** el resultado real: 76.0% recall / **23.9% FP**
out-of-fold (peor que el 14.2% de arriba). Causa: con folds chicos, "el mejor FP visto en
entrenamiento" es una estimación ruidosa — un límite más estricto en el barrido no garantiza un FP
más bajo en datos nunca vistos, solo elige un punto distinto (a veces peor) de una curva
recall/FP que en sí no cambió. Confirma que hacía falta la mejora de fondo, no solo mover el
umbral de selección.

**Cambio real**: `class_deviations.py` ahora puede comparar cada día contra el historial **del
mismo tipo de día** (semana vs. fin de semana) en vez de mezclar ambos — la hipótesis de Snoun et
al. 2024 (usar contexto como día/feriado, no solo el valor crudo). Nuevo campo
`ClassDeviationConfig.separar_fin_de_semana` (default `True`), con fallback automático a la
ventana completa si no hay suficientes días del mismo tipo todavía. Se agregó al barrido de
`evaluate_deviations.py` (`SEPARAR_FINDE_GRID`) en vez de fijarlo a mano, para que el
cross-validation elija el valor que realmente generaliza en cada modo — resultó que **ayuda en
modo sensor pero no en modo cámara** (con clases ya fusionadas como `cam_comida`, partir encima la
muestra en dos la vuelve más ruidosa todavía).

**Resultado, mismo protocolo que arriba (`--cv 5`, ahora con `separar_fin_de_semana` en el
barrido)**:

| | Recall out-of-fold (antes) | **Recall out-of-fold (ahora)** | FP out-of-fold (antes) | **FP out-of-fold (ahora)** |
|---|---|---|---|---|
| Modo sensor | 71.4% | **73.3%** | 14.2% | **13.9%** |
| Modo cámara | 70.5% | **70.0%** | 14.3% | **15.5%** |

Mejora chica pero real en modo sensor (+1.9pp recall, −0.3pp FP) — 4 de 5 folds eligieron
`separar_fin_de_semana=True`. En modo cámara el cross-validation eligió `False` en 4 de 5 folds
(confirma la hipótesis: no ayuda ahí) y el resultado quedó prácticamente sin cambios (dentro del
ruido esperable entre corridas). **Conclusión**: la separación semana/finde es una mejora
incremental real para el modo con sensor, no la solución al FP alto — sigue siendo ~14-15% en
ambos modos.

### Actualización 2026-09-20 (2) — exigir persistencia en `exceso`/`deficit`

Siguiente candidato probado: `ClassDeviationConfig.dias_consecutivos_exceso_deficit` — exigir que
`exceso`/`deficit` se sostengan varios días seguidos antes de contar como desvío (mismo patrón que
ya tenía `ausencia_sostenida`, extendido a los otros dos tipos), sumado al barrido
(`DIAS_CONSECUTIVOS_EXCESO_DEFICIT_GRID = (1, 2, 3)`, con `1` = comportamiento original).

| | Recall out-of-fold (antes) | **Recall out-of-fold (ahora)** | FP out-of-fold (antes) | **FP out-of-fold (ahora)** |
|---|---|---|---|---|
| Modo sensor | 73.3% | 72.8% | 13.9% | **13.1%** |
| Modo cámara | 70.0% | 69.5% | 15.5% | **13.8%** |

Mejora el FP en los dos modos (−0.8pp sensor, **−1.7pp cámara**) a costa de un recall casi
idéntico (−0.5pp ambos) — trade-off favorable, sobre todo en cámara. Pero **8 de 10 folds (ambos
modos) eligieron `dias_consecutivos_exceso_deficit=1`** — es decir, la mayoría de las veces el
cross-validation **no** encontró que exigir persistencia ayudara; la mejora observada viene más de
que el barrido más grande encontró mejores combinaciones de los otros parámetros (`frac_historica_min`/
`excess_k`) que ya estaban, no de la persistencia en sí. **Conclusión**: implementado y sumado al
barrido (no hace daño, ayuda marginalmente en algunos folds), pero no es la palanca que resuelve el
FP — dos ideas de "ajustar mejor los mismos parámetros" (semana/finde, persistencia) ya se
probaron y ninguna bajó el FP de forma sustancial. El límite parece estar en el diseño mismo del
detector (mediana+MAD por clase), no en su calibración.

### Actualización 2026-09-20 (3) — diagnóstico manual de FP concretos, y por qué exigir persistencia en `ausencia` NO ayudó

Se abrieron los 48 casos concretos marcados como desvío sobre días reales sin tocar (con la
config de mayoría de folds: `frac=0.8, excess_k=3.0, separar_fin_de_semana=True`). Dos hallazgos:

1. **21 de 48 (44%) son `ausencia` de un solo día en clases casi-diarias** (ej. `bano_nocturno`
   con mediana=1 y presente el 95-100% de los días históricos). Matemáticamente esperable: si algo
   pasa el 95% de los días, no pasar 1 de cada ~20 es ruido normal, no un desvío — el detector
   dispara con un solo día faltante, y solo la variante `ausencia_sostenida` exige racha.
2. **Un caso (`hh117`, 2011-12-03/04) donde 3 clases distintas se dispararon el mismo día**
   (medicación, baño nocturno, salir de casa) — no parece ruido: probablemente a esa persona le
   pasó algo real esos días (viaje, enfermedad). El día se sampleó como negativo (`real_sin_tocar`)
   sin verificar que fuera realmente "normal" — un límite de la metodología de validación, no del
   detector: parte del "FP" medido en toda esta sección puede en realidad ser el detector
   acertando sobre desvíos reales de CASAS que la muestra aleatoria de negativos no filtró.

**Hipótesis probada con el usuario**: exigir que `ausencia` también necesite racha (no solo
`ausencia_sostenida`) — nuevo campo `ClassDeviationConfig.ausencia_requiere_racha`, sumado al
barrido (`AUSENCIA_REQUIERE_RACHA_GRID`). Decisión tomada con el usuario: medirlo con
cross-validation en vez de asumir el trade-off a mano.

**Resultado: los 10 folds (5 sensor + 5 cámara) eligieron `ausencia_requiere_racha=False`** — cero
folds encontraron que ayudara. El costo en recall (perder `sin_medicacion`, hoy 88.9%, que depende
de detectar una ausencia de un solo día) es mayor que el beneficio en FP que se esperaba. Números
finales sin cambio respecto al punto anterior: **72.8%/13.1% sensor, 69.5%/13.8% cámara**.

**Conclusión de todo este ciclo de intentos** (semana/finde, persistencia en exceso/deficit,
persistencia en ausencia): se probaron 3 hipótesis de calibración con evidencia, no a ojo; una
ayudó un poco (semana/finde en modo sensor), dos no. El FP se movió de ~14-15% a ~13-14%, una
mejora real pero chica. El diagnóstico manual del punto 2 de arriba sugiere que el próximo paso
con más potencial no es otra variación de umbral/persistencia sobre el mismo esquema
mediana+MAD, sino **verificar si una fracción de los "FP" medidos son en realidad desvíos reales
de CASAS mal etiquetados como negativos** — repetir el diagnóstico manual sobre una muestra más
grande de los 48 casos, revisando a mano (o cruzando contra otra fuente, si existiera) si esos días
fueron realmente atípicos para esa persona.

### Actualización 2026-10-08 — diagnóstico escalado a TODOS los FP (no 48 a mano), y un hallazgo nuevo

Automaticé el criterio del diagnóstico manual del punto anterior (`diagnose_fp_manual.py`, nuevo:
para cada día `real_sin_tocar` marcado como desvío, cuenta cuántas clases distintas de
`TRACKED_CLASSES`/`CAMERA_TRACKED_CLASSES` dispararon ese mismo día) y lo corrí sobre **todos**
los FP del set de 80 hogares (no una muestra de 48), con la config de mayoría de folds:

| Modo | Días negativos | FP | 2+ clases el mismo día (hipótesis: desvío real mal muestreado) | 1 sola clase (ruido esperable) | Solo multivariado (ninguna clase individual disparó) |
|---|---|---|---|---|---|
| Sensor | 420 | 53 (12.6%) | 5 (9.4%) | 34 (64.2%) | 14 (26.4%) |
| Cámara | 420 | 63 (15.0%) | 7 (11.1%) | 56 (88.9%) | 0 (0.0%) |

**Lectura del primer hallazgo** (confirma y acota el punto 2 de la sección anterior): el caso real
de `hh117` que motivó la hipótesis no era único — aparecen 4-6 más con el mismo patrón (2+ clases
disparando el mismo día, ver `synthetic_eval/fp_manual_sensor.csv` para el detalle). Pero es una
fracción chica (9-11% de los FP, no "una fracción grande" como se esperaba) — **"perdonar" estos
casos bajaría el FP de 12.6% a ~11.4% en modo sensor, una mejora real pero menor, no la que
resolvería el problema de fondo.** La mayoría del FP (64-89%) sigue siendo ruido genuino de clases
casi-diarias, como ya se había diagnosticado.

**Hallazgo nuevo, no visto en el diagnóstico de 48 casos**: en modo sensor, **26.4% del FP viene
exclusivamente del detector multivariado** — ningún detector por clase individual disparó ese día,
solo `dynamic_clustering.py`. Esto no aparece en modo cámara (0%) porque ahí el multivariado ni se
corre (ver "Diseño" arriba, `evaluate_camera` no lo usa). Siguiente candidato más prometedor que
seguir ajustando `class_deviations.py` (ya se agotaron 4 variaciones, ver arriba): medir con
cross-validation si sacar el multivariado del `combinado` (o solo subirle `threshold_z`) en modo
sensor mejora el trade-off recall/FP — la tabla de la sección "Cross-validation por hogar" ya
sugiere que el multivariado aporta poco recall propio (12.3% solo, contra 72.6% del detector por
clase) a cambio de FP propio no despreciable.

**Medido (2026-10-08, `evaluate_deviations.py --cv 5 --metric por_clase`): la hipótesis NO se
sostiene.** Sacar el multivariado del combinado empeoró las dos métricas out-of-fold, no solo una:

| | Combinado (actual) | Solo por clase (sin multivariado) |
|---|---|---|
| Recall | 72.8% | 71.4% (−1.4pp) |
| FP | 13.1% | **16.1% (+3.0pp, peor)** |

Contraintuitivo: el multivariado aporta casi nada de recall propio, pero sacarlo también *empeoró*
el FP. Explicación más probable: no es que el multivariado agregue ruido que haya que filtrar —
al elegir `ClassDeviationConfig` sin mirarlo, el barrido por fold eligió configuraciones más
sueltas (`excess_k=2.5` en 2 de 5 folds, contra `3.0` en la mayoría de folds del combinado) que
ajustan mejor en entrenamiento pero generalizan peor. El multivariado funciona como un ancla que
empuja a elegir una calibración del detector por clase más conservadora, aunque su propio aporte
directo de recall/FP sea chico. **Conclusión: no tocar el combinado — van 4 variaciones de
calibración probadas con evidencia (semana/finde, persistencia en exceso/deficit, persistencia en
ausencia, sacar el multivariado) y solo una ayudó un poco.** El próximo candidato real es el
ajuste estadístico de "qué tan raro es raro" (ver abajo), no otra variación sobre qué se combina.

### Actualización 2026-10-13 — pruebas para el informe: ruido del clasificador y período de aprendizaje

Corridas en una máquina recién clonada: CASAS se bajó de nuevo de Zenodo (`labeled_data.zip`,
mismo md5 que publica el repositorio) y se regeneró con `etl.py` + `build_table.py`. Da los
mismos 319.229 eventos y 80 hogares del manifiesto versionado, y el set de desvíos
(`inject_synthetic_deviations.py`, semilla 0) es el mismo de siempre: 4362 días, 640 etiquetas
(220 positivas), 21 hogares.

#### Antes de medir: `--cv 5` ya no reproduce los números citados

`evaluate_deviations.py --cv 5`, tal como está en el repo, **no da 72,8 %/13,1 % ni
69,5 %/13,8 %**. Da 71,0 %/14,2 % (sensor) y 67,3 %/16,0 % (cámara). Motivo: el 2026-10-08, al
probar el método binomial, sus candidatos quedaron dentro del barrido de `cross_validate`
(`_grid_configs`) y cambió el orden en que se recorren los candidatos, que es el que decide los
empates de recall. El 71,0 %/14,2 % es el mismo que la "Actualización 2026-10-08" informa para
el barrido con binomial; en modo cámara ese barrido nunca se había medido.

Con el barrido del 2026-09-20 (solo candidatos `umbral`, en el orden de entonces) se reproducen
los dos números exactos: **72,8 %/13,1 % (158 de 217, 49 de 373) y 69,5 %/13,8 % (153 de 220,
58 de 420)**. Todo lo de esta sección usa ese criterio, para que sea comparable.

| Criterio de selección en cada partición | Sensor | Cámara |
|---|---|---|
| "documentado": solo `umbral`, orden del 2026-09-20 | 72,8 % / 13,1 % | 69,5 % / 13,8 % |
| Solo `umbral`, orden actual de `_grid_configs` | 72,8 % / 13,1 % | 68,2 % / 14,0 % |
| "actual": `umbral` + `binomial` (lo que corre hoy `--cv 5`) | 71,0 % / 14,2 % | 67,3 % / 16,0 % |

La fila del medio muestra que en modo cámara el resultado depende de un empate: con los mismos
candidatos, cambiar solo el orden mueve 1,3 puntos de detección. Es una medida de cuánto ruido
tiene el propio protocolo con 21 hogares.

`cv_grid.py` (nuevo) hace la misma validación cruzada evaluando cada candidato una sola vez
sobre todos los hogares, en vez de una vez por partición, y permite elegir el criterio
(`CRITERIOS`: `"documentado"` o `"actual"`). Da lo mismo que `cross_validate` porque los
detectores tratan cada hogar por separado. Comprobado de dos formas: `verify_cv_grid.py`
(barrido reducido, particiones idénticas en los dos modos) y 12 corridas completas de
`cross_validate` con el barrido actual (4 de la prueba de ruido, 8 de la de período de
aprendizaje), que coinciden al decimal con el criterio `"actual"` de `cv_grid.py`. `cross_validate`
no se tocó, salvo el parámetro opcional `min_window_days` (`--min-window-days`).

**Resuelto el mismo día**, ver "Actualización 2026-10-13 (2)" más abajo: `--cv 5` volvió a usar
por defecto el barrido de la fila "documentado". En el código ese criterio se llama ahora
`"umbral"` y el "actual" de esta sección, `"con_binomial"` (`--binomial`).

#### Prueba 1 — línea base con el error real del clasificador de la cámara

Hasta acá el modo cámara se evaluó con etiquetas limpias de CASAS: como si la cámara nunca
confundiera una actividad con otra. `evaluate_classifier_noise.py` (nuevo) contamina esas
etiquetas con la matriz de confusión medida del clasificador (`vlm/README.md`, "Matriz de
confusión de esta corrida", 2026-09-20, 6 categorías) antes de armar las columnas `cam_*` del
vector de día, y repite la validación cruzada por hogar en modo cámara.

**Cómo se mapeó.** Cada clase canónica de CASAS se asoció a la categoría del clasificador que
la contiene (según las clases finas de `vlm/src/clip_classifier.py::COARSE_CATEGORIES`). Por
cada evento se sortea la categoría predicha con la fila de la matriz de su categoría real. Si
acierta, el evento queda en su clase de cámara de siempre (`activity_map.COARSE`); si falla,
pasa a la clase de cámara de la categoría predicha, con el mapeo que ya usa producción
(`camera_events.CAMERA_ACTIVITY_MAP`).

| Clase canónica de CASAS | Categoría del clasificador (fila de la matriz) | Acierto publicado | Clase de cámara si acierta |
|---|---|---|---|
| `cocinar`, `lavar_platos` | `Cooking` | 80 % | `cocina` |
| `comer` | `EatingDrinkingMedication` | 57 % | `comida_mesa` |
| `beber`, `medicacion` | `EatingDrinkingMedication` | 57 % | `comida` |
| `descansar` | `Resting` | 67 % | `descanso_sofa` |
| `ver_tv` | `Resting` (incluye `WatchTV`) | 67 % | `ocio_sentado` |
| `leer`, `trabajo` | `ReadingScreen` | 64 % | `ocio_sentado` |
| `telefono` | `Phone` | 59 % | `ocio_sentado` |

| Categoría predicha por error | Clase de cámara a la que va el evento |
|---|---|
| `Cooking` | `cocina` |
| `EatingDrinkingMedication` | `comida` (nunca `comida_mesa`: sin acierto no hay zona de mesa confirmada) |
| `Resting` | `descanso_sofa` |
| `ReadingScreen`, `Phone` | `ocio_sentado` |
| `Moving` | `moverse` (no está entre las clases que se siguen: el evento se pierde) |

**Clases que quedaron sin ruido**, porque el clasificador no tiene una categoría equivalente:
`dormir`, `aseo`, `bano`, `bano_nocturno`, `salir_casa`, `entrar_casa`, `tareas_hogar`,
`visitas` y `otro`. Hoy la cámara no produce ninguna de esas clases (ver "Vocabulario de
cámara"), así que para ellas no hay error medido que aplicar. Consecuencia: de las cuatro
clases que sigue el modo cámara, `cam_comida` y `cam_comida_mesa` reciben ruido;
`cam_salir_casa` y `cam_aseo` no. `Moving` tampoco tiene equivalente en CASAS: aparece solo
como destino de un error.

**Cuánto ruido es.** De los 162.534 eventos de los 21 hogares, 84.659 (52,1 %) tienen
equivalente. De esos, entre 28,5 % y 28,7 % cambian de etiqueta según la semilla. En cada
corrida se pierden cerca de 7.000 comidas reales (pasan a otra clase) y aparecen cerca de
6.800 comidas falsas (otra actividad confundida con comer). El desvío se inyecta antes que el
ruido: un día sin ninguna comida real puede igual mostrar comidas. Con la semilla 0, de los 21
días de `comida_camara_ausente` solo 4 quedan realmente en cero; los otros 17 muestran entre 1
y 20 comidas falsas.

**Resultado** (validación cruzada por hogar, 5 particiones, criterio "documentado", 220
positivos y 420 negativos; el ruido se repitió con 3 semillas, la inyección y las particiones
son las mismas en todas las corridas):

| | Sin ruido | Con ruido, semilla 0 | Semilla 1 | Semilla 2 | Rango con ruido |
|---|---|---|---|---|---|
| **Detección global** | **69,5 %** | 65,5 % | 65,5 % | 65,0 % | **65,0 – 65,5 %** |
| **Falsos positivos** | **13,8 %** | 15,0 % | 17,4 % | 15,0 % | **15,0 – 17,4 %** |
| `semana_sin_salir` (147 días) | 78,9 % | 78,2 % | 78,2 % | 76,9 % | 76,9 – 78,2 % |
| `comida_camara_ausente` (21) | 95,2 % | 66,7 % | 71,4 % | 66,7 % | 66,7 – 71,4 % |
| `bano_nocturno_excesivo` (21) | 52,4 % | 38,1 % | 33,3 % | 52,4 % | 33,3 – 52,4 % |
| `comida_salteada` (21) | 14,3 % | 23,8 % | 23,8 % | 19,0 % | 19,0 – 23,8 % |
| `sin_medicacion` (10) | 30,0 % | 20,0 % | 10,0 % | 20,0 % | 10,0 – 20,0 % |

Lectura:

- La detección global baja entre 4,0 y 4,5 puntos y los falsos positivos suben entre 1,2 y 3,6.
- Casi toda la pérdida está en `comida_camara_ausente`: de 20 de 21 días detectados pasa a 14
  o 15. Es el escenario que depende de ver un cero, y las comidas falsas lo tapan.
- `semana_sin_salir` casi no cambia. Esperable: no recibe ruido. Pero también significa que
  esta prueba **no mide** el error de detectar entradas y salidas, que hoy la cámara no hace.
- `bano_nocturno_excesivo` tampoco recibe ruido y aun así se mueve entre 33 % y 52 %. Ese
  movimiento no es efecto directo del ruido: al cambiar los datos de comida, cada partición
  elige otros parámetros, y con 21 casos cada día pesa 4,8 puntos. Lo mismo vale para la suba
  de `comida_salteada` (3 a 5 días detectados de 21). No leer esas dos filas como mejora o
  empeoramiento.
- Con el criterio "actual" el rango con ruido es más ancho: 62,7 – 68,6 % de detección y
  12,9 – 14,0 % de falsos positivos (sin ruido: 67,3 %/16,0 %), con `comida_camara_ausente` en
  42,9 – 47,6 %.

Límites de la prueba: el ruido es independiente entre eventos (una cámara real se equivoca en
rachas, con la misma persona y la misma escena); la matriz es la del probe sin contexto de
zona, la única publicada (el de producción, con zona, acierta algo más en `Cooking`); y supone
que cada actividad de CASAS genera un solo evento de cámara.

#### Prueba 2 — cuántos días de historial hacen falta

`evaluate_learning_period.py` (nuevo) repite la validación cruzada variando `min_window_days`
(7, 14, 21 y 30) en los dos detectores. Criterio "documentado".

| Modo | Días de aprendizaje | Detección | Falsos positivos | Etiquetas sin evaluar (de 640) |
|---|---|---|---|---|
| Sensor | 7 | 72,8 % (158 de 217) | 13,1 % (49 de 373) | 50 |
| Sensor | 14 | 72,1 % (155 de 215) | 16,6 % (55 de 331) | 94 |
| Sensor | 21 | 72,3 % (149 de 206) | 16,4 % (46 de 280) | 154 |
| Sensor | 30 | no medible: 8 de 16 y 9 de 42 | | 582 |
| Cámara | 7 | 69,5 % (153 de 220) | 13,8 % (58 de 420) | 0 |
| Cámara | 14 | 72,3 % (159 de 220) | 15,2 % (64 de 420) | 0 |
| Cámara | 21 | 67,3 % (148 de 220) | 13,3 % (56 de 420) | 0 |
| Cámara | 30 | 60,5 % (133 de 220) | 12,4 % (52 de 420) | 0 |

Por escenario (detección; entre paréntesis, días evaluados cuando no son todos):

| Modo | Días | `semana_sin_salir` | `comida_camara_ausente` | `sin_medicacion` | `bano_nocturno_excesivo` | `comida_salteada` |
|---|---|---|---|---|---|---|
| Sensor | 7 | 78,9 % | 90,0 % (20) | 88,9 % (9) | 47,6 % | 30,0 % (20) |
| Sensor | 14 | 75,5 % | 89,5 % (19) | 88,9 % (9) | 57,1 % | 36,8 % (19) |
| Sensor | 21 | 75,5 % | 93,8 % (16) | 87,5 % (8) | 52,9 % (17) | 38,9 % (18) |
| Cámara | 7 | 78,9 % | 95,2 % | 30,0 % | 52,4 % | 14,3 % |
| Cámara | 14 | 80,3 % | 90,5 % | 30,0 % | 61,9 % | 28,6 % |
| Cámara | 21 | 78,9 % | 76,2 % | 10,0 % | 42,9 % | 28,6 % |
| Cámara | 30 | 70,1 % | 57,1 % | 20,0 % | 42,9 % | 33,3 % |

Lectura, con dos advertencias sobre cómo cuenta el protocolo:

- **Los dos modos no cuentan igual los días que caen dentro del período de aprendizaje.** En
  modo sensor una etiqueta sin score del multivariado se saca de la cuenta ("sin evaluar"); en
  modo cámara se queda y cuenta como no detectada (o como no falso positivo). Con 30 días, 223
  de las 640 etiquetas (36 positivas, 187 negativas) caen antes de que el detector pueda
  opinar. Buena parte de la caída del modo cámara a 30 días (69,5 % → 60,5 %) es eso: días en
  los que el sistema todavía está callado, no días que evaluó mal. La baja de falsos positivos
  tiene la misma causa.
- **Sensor con 30 días no se puede medir con este protocolo.** `window_days` también es 30, así
  que el detector exige los 30 días previos completos, y los días con desvío inyectado se
  excluyen de la referencia: cualquier día con uno inyectado en los 30 anteriores queda sin
  score. Quedan 58 etiquetas de 640. Medirlo bien requiere agrandar `window_days` junto con el
  período de aprendizaje, que es otro experimento.
- Con esas salvedades: entre 7 y 21 días el modo sensor detecta lo mismo (72,1 – 72,8 %) y
  con 14 o 21 tiene más falsos positivos (13,1 % → 16,4 – 16,6 %), sobre menos días evaluados.
  En modo cámara, 14 días da la mejor detección (72,3 %) con algo más de falsos positivos
  (15,2 %); la diferencia con 7 días son 6 días detectados de 220 y 6 falsos positivos de 420,
  del orden de lo que mueve un empate (ver la tabla de criterios, arriba).
- **Conclusión que los datos sostienen:** esperar más de 7 días no mejora la detección de
  forma medible y retrasa el arranque. El período de silencio de 7 días queda como está. Lo
  que estos datos no permiten afirmar es que 14 sea mejor que 7.

### Actualización 2026-10-13 (2) — `--cv 5` vuelve a dar los números citados por defecto

Decisión tomada con el usuario a partir del hallazgo de la sección anterior: el comportamiento
por defecto de `evaluate_deviations.py --cv 5` es otra vez el barrido del 2026-09-20, y los
candidatos del método binomial pasan a una opción explícita.

| Comando | Barrido | Sensor | Cámara (`--camera`) |
|---|---|---|---|
| `evaluate_deviations.py --cv 5` (**default, número a citar**) | Solo `ausencia_metodo="umbral"`, 384 candidatos, `frac_historica_min` como bucle de más afuera | **72,8 % / 13,1 %** | **69,5 % / 13,8 %** |
| `evaluate_deviations.py --cv 5 --binomial` | Lo anterior más los candidatos binomiales, 672 en total, en el orden del experimento del 2026-10-08 | 71,0 % / 14,2 % | 67,3 % / 16,0 % |

Qué cambió en el código:

- `_grid_configs(min_window_days=None, binomial=False)`: sin `binomial` genera solo los
  candidatos de umbral, en el orden que tenía `cross_validate` el 2026-09-20. Ese orden no hay
  que tocarlo: con empate de recall gana el primer candidato, y en modo cámara reordenar los
  mismos candidatos mueve el resultado de 69,5 %/13,8 % a 68,2 %/14,0 % (ver la tabla de
  criterios de la sección anterior).
- `cross_validate(..., binomial=False)` y la opción `--binomial` de la línea de comandos.
- El método binomial en sí (`ausencia_metodo="binomial"` en `ClassDeviationConfig`) no cambió:
  sigue implementado y se puede medir con `--binomial`. Lo único que cambia es que ya no
  compite por defecto. El barrido simple sin `--cv` tampoco cambió (nunca incluyó binomial).
- `cv_grid.py` sigue el mismo criterio: `cross_validate_from_grid(grid, binomial=False)`, y
  `CRITERIOS` pasó a llamarse `"umbral"` (antes `"documentado"`) y `"con_binomial"` (antes
  `"actual"`). Los barridos guardados (`synthetic_eval/cv_grid_*.pkl`) siguen sirviendo.

**Verificación** (`verify_cv_grid.py --completo`, 2026-10-13):

- `cross_validate` real, con el barrido completo por defecto, en los dos modos: sensor 72,8 %
  / 13,1 % (158 de 217, 49 de 373) y cámara 69,5 % / 13,8 % (153 de 220, 58 de 420). Son los
  cuatro conteos de referencia; el script falla si alguno difiere.
- `cv_grid.py` contra `cross_validate`, barrido reducido: particiones idénticas en 8 casos
  (sensor y cámara, 7 y 21 días de aprendizaje, con y sin `binomial`).
- `evaluate_classifier_noise.py` y `evaluate_learning_period.py`, vueltos a correr después del
  cambio, dan las mismas tablas de la sección anterior.

No se volvió a correr `--cv 5 --binomial` completo después del cambio (media hora por modo): esa
rama del código no se modificó, sus 12 corridas completas de la mañana coinciden con `cv_grid.py`
y el barrido reducido con `binomial` da igual por los dos caminos.

Los números de las tablas de la "Actualización 2026-10-08" que comparan contra "72,8 %/13,1 %
(actual)" siguen siendo válidos: esa referencia era, y vuelve a ser, la del barrido por defecto.

### Actualización 2026-10-13 (3) — detector de horario medido con validación cruzada, y pérdida de eventos de puerta

Segunda ronda del mismo día, sobre dos puntos que la primera dejó flojos.

#### Detector de horario: acotar clases y calibrar por clase NO se adoptó

Motivo de la prueba: por el camino de producción el detector de horario dispara dos alertas
sobre un día real sin modificar (`cloud/README.md`, "La línea base por el camino de producción,
medida"). Se probó lo que proponía "Pendiente": acotar las clases que sigue a las de horario
disciplinado y calibrarlo por clase. Criterio acordado: adoptar solo si baja los falsos
positivos sin perder detección.

`evaluate_timing_cv.py` (nuevo). Mismo banco de pruebas que `evaluate_timing_deviations.py` (a
3 eventos reales por hogar y clase se les corre el horario 3 horas; 8 sin tocar son los
negativos), con tres cambios: los 21 hogares con historial suficiente en vez de 7, tres semillas
de inyección juntas, y validación cruzada por hogar en 5 particiones (los parámetros se eligen
con unos hogares y se miden con otros). Barrido por clase: `window_events` 40/60/90,
`k_time_slots` 1/2/3, `excess_k` 2,0 a 5,0 (54 candidatos). Dos vocabularios: el de producción
(`cam_*`, armado desde CASAS con el mapeo de `evaluate_classifier_noise.py`, etiquetas limpias)
y el canónico de CASAS (`DEFAULT_TRACKED_CLASSES`).

Además de la tasa por etiqueta se midió la **falsa alarma por día**: de 315 días reales sin
tocar (5 por hogar y semilla), cuántos reciben al menos una alerta de horario. Es lo que ve un
familiar.

Variantes, todas sobre las mismas etiquetas:

- **Actual:** todas las clases con `TimingConfig()` (ventana 60, 2 franjas, `excess_k` 3,0).
- **Misma detección:** por clase, los parámetros de menos falsos positivos entre los que en
  entrenamiento detectan al menos lo mismo que la actual. Es la prueba directa del criterio.
- **Calibrado:** por clase, la mayor detección con falsos positivos ≤ 5 % en entrenamiento (si
  ninguno llega, el de menos falsos positivos).
- **Acotado:** como calibrado, pero una clase solo se sigue si en entrenamiento llega a falsos
  positivos ≤ 5 % con detección ≥ 30 %. Los dos umbrales se fijaron antes de medir.

Vocabulario de producción (692 positivos, 1.683 negativos):

| Variante | Detección | Falsos positivos | Días normales con alguna alerta |
|---|---|---|---|
| Actual | 51,3 % (355) | 36,8 % (620) | 65,7 % (207 de 315) |
| Misma detección | 47,0 % (325) | 36,6 % (616) | 63,2 % (199) |
| Calibrado | 15,8 % (109) | 11,9 % (200) | 23,2 % (73) |
| Acotado | 0 % | 0 % | 0 % |

| Clase (detección / falsos positivos) | Actual | Calibrado |
|---|---|---|
| `cam_cocina` | 63 % / 43 % | 26 % / 13 % |
| `cam_comida` | 47 % / 33 % | 8 % / 8 % |
| `cam_descanso_sofa` | 49 % / 33 % | 19 % / 16 % |
| `cam_ocio_sentado` | 45 % / 39 % | 10 % / 10 % |

Vocabulario canónico de CASAS (801 positivos, 1.964 negativos):

| Variante | Detección | Falsos positivos | Días normales con alguna alerta |
|---|---|---|---|
| Actual | 44,8 % (359) | 21,9 % (430) | 58,1 % (183 de 315) |
| Misma detección | 42,9 % (344) | 19,2 % (378) | 48,3 % (152) |
| Calibrado | 17,6 % (141) | 6,6 % (129) | 16,8 % (53) |
| Acotado (queda solo `medicacion`) | 4,0 % (32) | 0,6 % (11) | 0,6 % (2) |

| Clase (detección / falsos positivos) | Actual | Calibrado |
|---|---|---|
| `medicacion` | 67 % / 10 % | 40 % / 5 % |
| `comer` | 55 % / 18 % | 35 % / 9 % |
| `dormir` | 40 % / 32 % | 3 % / 5 % |
| `salir_casa` | 39 % / 22 % | 11 % / 6 % |
| `beber` | 35 % / 22 % | 11 % / 7 % |

Lectura:

- **Ninguna variante cumple el criterio.** La pensada para no perder detección tampoco lo
  logra fuera de los hogares con los que se eligió: en producción pierde 4,3 puntos de
  detección sin bajar los falsos positivos (36,8 % → 36,6 %); en el canónico baja 2,7 puntos de
  falsos positivos y pierde 1,9 de detección. Las otras dos bajan mucho los falsos positivos,
  pero porque detectan mucho menos.
- **Decisión: el detector quedó como estaba** (`TimingConfig()` y las mismas clases, en
  `baseline/run_live.py` y en `cloud/app/baseline_runner.py`). No se cambió ningún default.
- **Lo que la medición sí deja claro, y es peor de lo que se sabía:** con las clases que la
  cámara produce hoy, el detector de horario marca al menos una alerta en **dos de cada tres
  días normales** (65,7 %). Ninguna clase de cámara tiene horario disciplinado: con el criterio
  de "acotado" no queda ninguna (en las 5 particiones y con todos los hogares juntos). La única
  clase que califica es `medicacion` (40 % de detección con 5 % de falsos positivos fuera de
  muestra), y la cámara no la distingue (ver "Vocabulario de cámara").
- **Queda una decisión de producto que este criterio no cubre:** seguir emitiendo alertas de
  horario en producción con esa tasa de falsas alarmas, o dejar de emitirlas y resignar el
  51 % de detección de corrimientos de 3 horas. Los datos apuntan a lo segundo, pero es perder
  detección y por eso no se aplicó.
- `cam_moverse` no tiene equivalente en CASAS: no se pudo medir.

Límites: etiquetas limpias (sin el error del clasificador), un solo tamaño de corrimiento (3
horas), y un negativo cuenta como falso positivo si cualquier evento de esa clase en ese día
queda marcado, que es como se comporta el detector (una alerta por evento).

**Lo que sí se adoptó, porque no cambia ningún resultado:**

- `score_timing_deviations(..., evaluate_only={(hogar, fecha)})`: evalúa solo los eventos de
  esos días y usa el resto del historial como referencia. Cada evento se compara solo contra
  los anteriores, así que el resultado para esos días es idéntico. El proceso diario lo usa
  con el día evaluado. Comprobado: salida idéntica a la del código anterior en dos hogares
  completos (138 y 384 eventos marcados, dos configuraciones), idéntica con y sin
  `evaluate_only`, y las mismas 6 alertas en la prueba por el camino de producción. El proceso
  diario pasó de 42 segundos por hogar a menos de 2 (60 días de historial, misma PC).
- `timing_statistics()`: la parte cara (el clustering) separada del umbral, para probar varios
  `excess_k` sin repetirla. `TimingConfig.por_clase` permite parámetros distintos por clase;
  está vacío por defecto y nada lo usa en producción.

#### Sensibilidad a entradas y salidas perdidas

La prueba de ruido no le aplica error a `salir_casa`/`entrar_casa` porque no hay un detector
de puerta medido. Sin un error medido, `evaluate_door_event_loss.py` (nuevo) hace un análisis
de sensibilidad con tres niveles **supuestos**: cada evento de `salir_casa` y de `entrar_casa`
se pierde, de forma independiente, con probabilidad 10 %, 20 % o 30 %. El resto queda limpio.
Validación cruzada por hogar en modo cámara, barrido por defecto, 220 positivos y 420
negativos; 3 semillas por nivel (la inyección y las particiones no cambian).

| Eventos de puerta perdidos | `semana_sin_salir` (147 días) | Detección global | Falsos positivos |
|---|---|---|---|
| 0 % (referencia) | 78,9 % (116) | 69,5 % (153) | 13,8 % (58) |
| 10 % | 76,9 – 80,3 % (113 – 118) | 68,2 – 70,9 % (150 – 156) | 14,0 – 16,0 % (59 – 67) |
| 20 % | 76,2 – 80,3 % (112 – 118) | 67,3 – 71,4 % (148 – 157) | 14,0 – 16,2 % (59 – 68) |
| 30 % | 76,2 – 77,6 % (112 – 114) | 66,8 – 69,5 % (147 – 153) | 14,0 – 16,0 % (59 – 67) |

Rangos: mínimo y máximo entre las 3 semillas. Con 30 % de pérdida quedan unas 5.380 de las
7.716 salidas, y entre 91 y 102 de los 1.929 días con alguna salida quedan sin ninguna.

Lectura:

- `semana_sin_salir` aguanta: con 30 % de eventos perdidos baja entre 1,3 y 2,7 puntos (2 a 4
  días de 147). Se detecta por ausencia sostenida, y para que un día normal parezca "sin
  salir" hay que perder todas sus salidas, cosa que con 30 % pasa en 1 de cada 20 días.
- El costo aparece en los falsos positivos: suben entre 0,2 y 2,4 puntos (1 a 10 días de 420),
  por los días normales que quedan sin ninguna salida registrada. No crece de forma ordenada
  con la pérdida: dentro de cada nivel la semilla mueve tanto como el nivel.
- Con 10 % y 20 % el efecto sobre la detección no se distingue del ruido entre semillas (el
  rango incluye el valor de referencia).

Límites, que pesan más que en la prueba de ruido: los niveles son supuestos, no medidos; las
pérdidas son independientes entre eventos, y una cámara real falla de forma sistemática (si no
ve la puerta, pierde todas); y no se modelan entradas o salidas inventadas, que atacarían
justo lo que hace robusto al escenario (ver un cero). Hoy la cámara no produce estos eventos
(ver "Vocabulario de cámara"): esto acota cuánto error toleraría un detector de puerta, no
dice que exista.

### Actualización 2026-10-13 (4) — las alertas del detector de horario quedan apagadas por defecto

Decisión del usuario sobre lo que la sección anterior dejó abierto: el proceso diario deja de
emitir desvíos originados por el detector de horario. Se resigna su detección de corrimientos
de horario (51,3 % en la medición de arriba) a cambio de no avisar en dos de cada tres días
normales (65,7 %).

- **En la nube** (`cloud/app/baseline_runner.py`, el camino de producción): el detector de
  horario solo corre si la variable de entorno `ALERTAS_DE_HORARIO_ACTIVADAS` vale `1` (también
  `true`, `si`, `yes`). Sin la variable, o con cualquier otro valor, está apagado.
- **En local** (`run_live.py`): mismo criterio, con la opción `--timing`. Apagado sin ella.
- **No se borró nada**: `src/timing_deviations.py`, `evaluate_timing_deviations.py` y
  `evaluate_timing_cv.py` siguen igual y se pueden seguir midiendo.
- **Los otros tres detectores no cambian**: multivariado (clustering), por clase (por
  actividad) y tendencia siguen emitiendo como antes.
- **El horario habitual de la pantalla de Reportes no se toca**: lo calcula `cloud/app/rutina.py`
  con su propia cuenta (mediana y MAD de la primera comida), que nunca usó este detector.
- Los `routine_deviation` de horario que ya estuvieran guardados no se borran ni se modifican.

**No cambia ningún número de validación cruzada.** `evaluate_deviations.py --cv 5` **no incluye
el detector de horario** y nunca lo incluyó: el 72,8 %/13,1 % (sensor) es el detector por clase
combinado con el multivariado, y el 69,5 %/13,8 % (cámara) es el detector por clase solo.
Tampoco incluyen el detector de tendencia. Los cuatro escenarios con los que se miden esos
números (ausencias, excesos y déficits de un día, y una semana sin salir) no corren ningún
horario, así que el detector de horario no tenía nada que detectar ahí. Por lo mismo, tampoco
cambian las pruebas de ruido del clasificador, de período de aprendizaje ni de pérdida de
eventos de puerta. Los únicos números del detector de horario son los de
`evaluate_timing_deviations.py` y `evaluate_timing_cv.py`, que se miden aparte.

Lo que sí cambia es lo que emite el proceso diario, medido en la prueba por el camino de
producción (`cloud/README.md`): el día sin comidas pasa de 4 alertas a 2 (las dos que provoca
el desvío inyectado) y el mismo día sin modificar pasa de 2 alertas a ninguna.

### Actualización 2026-10-13 (5) — validación cruzada con el vocabulario que producción puede poblar

**Ningún número anterior de este README cambia.** `--cv 5` y `--cv 5 --camera` no se tocaron
(72,8 % / 13,1 % y 69,5 % / 13,8 %); esto es una medición nueva, en un script aparte
(`evaluate_production_vocab.py --paso1`), con su propio set de desvíos. La medición con el
vocabulario de producción (con entradas y salidas) y sus parámetros está en la actualización (6);
la variante sin puerta, evaluada y no adoptada, en la (7).

**Por qué**: el 69,5 % / 13,8 % del modo cámara se mide con `CAMERA_TRACKED_CLASSES`
(`cam_comida`, `cam_comida_mesa`, `cam_salir_casa`, `cam_aseo`). La controladora emite cinco
clases (`camera_events.CAMERA_ACTIVITY_MAP`: `cocina`, `comida`, `descanso_sofa`,
`ocio_sentado`, `moverse`), y de las cuatro evaluadas solo coincide `cam_comida`. Además, el
escenario que más pesa en ese número es `semana_sin_salir` (147 de 220 días positivos), que
depende de `cam_salir_casa`. **El 69,5 % / 13,8 % describe un vocabulario que producción no
puebla hoy.**

| Dónde | Clases |
|---|---|
| Clasificador de la controladora (`activity_label`) | `Cooking`, `EatingDrinkingMedication`, `Resting`, `ReadingScreen`, `Phone`, `Moving` (`Background` se descarta) |
| `camera_events.CAMERA_ACTIVITY_MAP` → `LIVE_TRACKED_CLASSES` | `cam_cocina`, `cam_comida`, `cam_descanso_sofa`, `cam_ocio_sentado` (`ReadingScreen` + `Phone`), `cam_moverse` |
| Proceso diario (`cloud/app/baseline_runner.py`, `run_live.py`) | las 5 de `LIVE_TRACKED_CLASSES`, en los cuatro detectores (el de horario apagado por defecto). Desde la actualización (6) son 7: se suman `cam_salir_casa` y `cam_entrar_casa` (solo ausencia sostenida) |
| `evaluate_deviations.py --cv 5` (sensor) | `medicacion`, `salir_casa`, `bano_nocturno`, `comer` + multivariado sobre las 19 canónicas |
| `evaluate_deviations.py --cv 5 --camera`, `cv_grid.py`, `evaluate_classifier_noise.py`, `evaluate_learning_period.py`, `evaluate_door_event_loss.py` | `cam_comida` (acá solo beber + medicación), `cam_comida_mesa`, `cam_salir_casa`, `cam_aseo` |
| `evaluate_timing_cv.py --vocab camara` | `cam_cocina`, `cam_comida`, `cam_descanso_sofa`, `cam_ocio_sentado` |
| `evaluate_timing_deviations.py` / `evaluate_trend_deviations.py` | canónicas de CASAS (`comer`, `beber`, `medicacion`, `dormir`, `salir_casa` / `salir_casa`, `visitas`, `dormir`, `descansar`) |
| `evaluate_production_vocab.py` (nuevo) | `cam_cocina`, `cam_comida`, `cam_descanso_sofa`, `cam_ocio_sentado` |
| Indicadores de la app (`cloud/app/rutina.py`) | `comida` (`cam_comida`, horario), `sedentarismo` (`cam_descanso_sofa` + `cam_ocio_sentado`), `cocina`, `movilidad` (`cam_moverse`) |

Dos diferencias más dentro del mismo nombre de clase:
- `cam_comida` en `features.add_camera_columns()` es beber + medicación (comer va a
  `cam_comida_mesa`); en producción comer, beber y medicación son una sola clase.
- Ver televisión es `ocio_sentado` en `activity_map.COARSE` y `descanso_sofa` en producción
  (`WatchTV` está dentro de `Resting`).

**Qué se midió** (`evaluate_production_vocab.py`): mismo protocolo que `--cv 5 --camera` — 21
hogares, 5 particiones por hogar con semilla 0, mismo barrido de `ClassDeviationConfig`, tope
de 15 % de falsos positivos al elegir, solo el detector por clase — con las cuatro clases de
producción que tienen equivalente en CASAS, armadas por el camino de producción
(`CANONICAL_TO_CATEGORY` → `CAMERA_ACTIVITY_MAP`). Desvíos inyectados: `dia_sin_comida`,
`dia_sin_cocinar`, `exceso_descanso` y `exceso_ocio_sentado` (los excesos suben los minutos a
max(1,5 × percentil 95, mediana + 180 min) sin cambiar la cantidad de episodios). 79 días
positivos (21 + 21 + 20 + 17) y 420 negativos. La semilla de inyección 0 es la del protocolo; 1
y 2 se corrieron para ver cuánto se mueve.

| Medición (semilla 0; entre paréntesis, semillas 1 y 2) | Detección | Falsos positivos | Sin comida | Sin cocinar | Exceso descanso | Exceso ocio sentado |
|---|---|---|---|---|---|---|
| Validación cruzada, señal de producción (episodios por día) | **48,1 %** (51,9 / 54,4) | **9,5 %** (13,8 / 14,5) | 85,7 % | 85,7 % | 10,0 % | 0,0 % |
| Validación cruzada, minutos por día | 68,4 % (67,1 / 72,2) | 15,2 % (16,7 / 13,8) | 85,7 % | 76,2 % | 60,0 % | 47,1 % |
| Producción tal cual, detector por clase (`ClassDeviationConfig()`) | 49,4 % (50,6 / 57,0) | 18,3 % (18,3 / 17,1) | 85,7 % | 71,4 % | 5,0 % | 29,4 % |
| Producción tal cual, multivariado sobre las columnas `cam_*` | 13,9 % (17,7 / 11,4) | 2,1 % (2,1 / 2,1) | 0,0 % | 14,3 % | 25,0 % | 17,6 % |
| Producción tal cual, los dos juntos | 57,0 % (55,7 / 62,0) | 19,0 % (19,5 / 17,6) | 85,7 % | 71,4 % | 30,0 % | 35,3 % |
| Indicadores de la app (regla de `rutina.py`: sedentarismo o cocina) | 24,1 % (21,5 / 25,3) | 0,2 % (0,7 / 0,0) | 0,0 % | 9,5 % | 50,0 % | 41,2 % |

Lecturas:
- **Las ausencias de un día se detectan bien** (día sin comida 85,7–95,2 %, día sin cocinar
  71,4–85,7 % según la semilla), casi siempre por la clase correcta.
- **El detector por clase de producción no ve el exceso de descanso ni de ocio sentado.** Mira
  la cantidad de episodios (`_n`), no los minutos: contando solo las alertas de la clase del
  desvío, detecta 0 de 20 y 0 de 17 con la semilla 0 (el 10 % de la tabla son alertas de otra
  clase que coincidieron ese día). Con minutos por día sube a 45 % y 41 %, a costa de más
  falsos positivos por `exceso` (2,6 % → 10,2 % de los días normales).
- **Con los parámetros por defecto que corre producción, los falsos positivos son 17–19 %**, por
  encima del 13,8 % citado para el modo cámara y del tope de 15 %.
- En falsos positivos por clase (validación cruzada, episodios, semilla 0): `cam_cocina` 3,6 %,
  `cam_descanso_sofa` 3,6 %, `cam_comida` 2,4 %, `cam_ocio_sentado` 1,2 %. Por tipo: `ausencia`
  5,2 %, `exceso` 2,6 %, `ausencia_sostenida` 1,9 %, `deficit` 0,2 %.
- Los indicadores de la app casi no dan falsos positivos, pero no muestran el día sin comida
  (el indicador es el horario de la primera comida: sin comida no hay valor) y marcan el día sin
  cocinar solo en 2 de 21 hogares (la regla exige caer 3 sigmas bajo la mediana, con piso de
  15 min).

Límites: 79 positivos son pocos (cada escenario tiene 17–21 días, un día vale ~5 puntos);
etiquetas limpias de CASAS, sin el error del clasificador; `cam_moverse` no tiene equivalente en
CASAS y no se midió; el detector de tendencia no se evaluó (los desvíos son de un día); el
exceso se inyectó solo en minutos, que es el supuesto desfavorable para el conteo.

### Actualización 2026-10-13 (6) — el sistema tal como queda: entradas y salidas, parámetros y pruebas de robustez

> **Superada el mismo día por la actualización (8)** en dos cosas: los parámetros de producción
> (se volvió a votar con los días de puerta en el barrido: la puerta dispara desde el primer
> día, `excess_k` 2,5 y exceso/déficit sostenido 2 días) y el filtro
> `solo_ausencia_sostenida_en_puerta`, que ya no existe (la regla de puerta está dentro del
> detector por clase). El vocabulario y las entradas y salidas encendidas siguen igual. Los
> números de esta sección no se tocaron: son los de la puerta fija en 2 días y `excess_k` 5,0.
> Las tablas del sistema tal como corre hoy están en la (8).

> **Estado final (decisión del 2026-10-13): este es el vocabulario y estos son los parámetros
> de producción.** Las entradas y salidas de casa quedan **encendidas por defecto**
> (`puerta_activada` en `cameras.yaml` y `ENTRADAS_Y_SALIDAS_ACTIVADAS` en el proceso diario;
> las dos se pueden apagar), `cam_salir_casa` / `cam_entrar_casa` están en
> `LIVE_TRACKED_CLASSES` y los parámetros son los de `class_deviations.PRODUCTION_PARAMS`. Los
> números de esta sección se reproducen con `evaluate_production_vocab.py --con-puerta`. La
> versión sin puerta, que por unas horas fue la elegida, es la variante evaluada y no adoptada
> de la actualización (7).
>
> **Diferencia entre lo medido y lo que emite el proceso diario.** De las clases de puerta el
> proceso diario solo deja pasar un tipo de desvío, la **ausencia sostenida**: nunca por
> horario (aunque se enciendan las alertas de horario), ni por un solo día sin salir, ni por
> exceso, déficit o tendencia (`camera_events.solo_ausencia_sostenida_en_puerta` y
> `sin_clases_de_puerta`). Las tablas de esta sección se midieron **antes** de esa restricción,
> con el detector por clase sin filtrar, y no se tocaron. Con la restricción, medido en la misma
> corrida (3 semillas):
>
> | | Global | Semana sin salir | Falsos positivos |
> |---|---|---|---|
> | Por clase, como en las tablas | 70,9 % (481/678) | 80,0 % (353/441) | 16,3 % (206/1260) |
> | Por clase, solo ausencia sostenida en puerta | 64,2 % (435/678) | 70,7 % (312/441) | 12,9 % (162/1260) |
> | Los dos detectores, como en las tablas | 73,3 % (497/678) | 80,0 % (353/441) | 17,7 % (223/1260) |
> | **Los dos detectores, solo ausencia sostenida en puerta (lo que emite producción)** | **67,4 %** (457/678) | 71,9 % (317/441) | **14,2 %** (179/1260) |
> | Los dos detectores con error del clasificador, como en las tablas | 62,5 % (1272/2034) | 78,7 % (1041/1323) | 17,9 % (677/3780) |
> | Los dos detectores con error del clasificador, solo ausencia sostenida en puerta | 57,0 % (1159/2034) | 71,6 % (947/1323) | 14,4 % (545/3780) |
>
> La restricción cuesta unos 8 puntos en la semana sin salir (sobre todo el primer día de cada
> semana, que antes se marcaba como ausencia de un día) y baja los falsos positivos unos 3,5
> puntos, a 14,2 %: bajo el tope de 15 %. Los demás desvíos no cambian.

**Ningún número anterior de este README cambia.** `--cv 5` y `--cv 5 --camera` no se tocaron
(72,8 % / 13,1 % y 69,5 % / 13,8 %), ni `evaluate_deviations.py`, `cv_grid.py` ni los defaults
de `ClassDeviationConfig`. La medición del punto (5) se sigue obteniendo con
`evaluate_production_vocab.py --paso1`. Todo lo de abajo sale de
`evaluate_production_vocab.py --con-puerta`.

**Qué cambió en el código:**
- La controladora emite `LeaveHome` / `EnterHome` cuando la cámara tiene dibujada la zona
  `puerta` (`runner/door_events.py`, ver `runner/README.md`). `camera_events.py` las lleva a
  `cam_salir_casa` / `cam_entrar_casa` (`DOOR_ACTIVITY_MAP`) como eventos puntuales: cuentan
  como ocurrencias, no suman minutos y no alteran la duración de las otras actividades.
  `LIVE_TRACKED_CLASSES` pasa de 5 a 7 clases. Una cámara sin zona `puerta` no las emite: la
  clase queda en cero y su ausencia nunca es un desvío.
- El proceso diario (`cloud/app/baseline_runner.py`, `run_live.py`) usa
  `class_deviations.production_config()` en vez de `ClassDeviationConfig()`.
- `comida_mesa` y `aseo` siguen fuera de alcance: no se implementaron.
- El detector por clase sigue contando episodios, no minutos. El exceso de descanso o de ocio
  sentado no lo ve (abajo, medido); lo cubre el indicador de sedentarismo de `cloud/app/rutina.py`.

**Protocolo:** 21 hogares de CASAS, 5 particiones por hogar (semilla 0), 3 semillas de
inyección, el barrido de `ClassDeviationConfig` de siempre, tope de 15 % de falsos positivos al
elegir. Clases: `cam_cocina`, `cam_comida`, `cam_descanso_sofa`, `cam_ocio_sentado`,
`cam_salir_casa`, `cam_entrar_casa`. **`cam_moverse` no tiene equivalente en CASAS y no se
midió.** Desvíos por semilla: día sin comida (21), día sin cocinar (21), semana sin salir (147
días, 7 por hogar), exceso de descanso (20), exceso de ocio sentado (17) y 420 días normales.
Sumando las 3 semillas: 678 días con desvío y 1260 normales. La semana sin salir es el 65 % de
los días con desvío, así que domina la cifra global: mirar la tabla por desvío.

#### Parámetros de producción

Un único juego, elegido por voto: la combinación que más particiones de entrenamiento eligieron
entre las 15 (5 por semilla), sin usar sus hogares de prueba. La regla (más votada; en empate,
la de menos falsos positivos en entrenamiento) se fijó antes de mirar el resultado.

| Parámetro | Antes (default del dataclass) | Producción desde 2026-10-13 |
|---|---|---|
| `frac_historica_min` | 0,8 | **0,6** |
| `excess_k` | 3,0 | **5,0** |
| `dias_consecutivos_ausencia` | 2 | 2 |
| `separar_fin_de_semana` | sí | sí |
| `dias_consecutivos_exceso_deficit` | 1 | 1 |
| `ausencia_requiere_racha` | no | no |

El voto fue repartido: ganó con 4 de 15, seguido por una combinación con 3 y dos con 2. Las 15
particiones eligieron `dias_consecutivos_ausencia=2` y `ausencia_requiere_racha=no`, y 13 de 15
`frac_historica_min=0,6`; lo que varía es cómo tratar excesos y déficits.

| Medición (detector por clase, 3 semillas) | Detección | Falsos positivos |
|---|---|---|
| Validación cruzada, fuera de muestra (cada partición con los parámetros que eligieron las otras) | **68,9 %** (467/678) | **14,4 %** (182/1260) |
| Juego de producción, solo en las 4 particiones de prueba cuyo entrenamiento lo eligió | 74,3 % (136/183) | 15,6 % (53/340) |
| Juego de producción, sobre los 21 hogares | 70,9 % (481/678) | 16,3 % (206/1260) |
| Defaults anteriores, sobre los 21 hogares | 68,4 % (464/678) | 22,5 % (284/1260) |

Lectura: la cifra fuera de muestra es la primera (68,9 % / 14,4 %); es la del procedimiento,
no la de un juego fijo. El juego elegido da 16,3 % de falsos positivos sobre los 21 hogares,
**por encima del tope de 15 %**: baja desde 22,5 % sin perder detección, pero no llega al
tope. Esa tercera fila no es estrictamente fuera de muestra (cada hogar participó en elegir
parámetros en 4 de las 5 particiones). Hay combinaciones con 12 % sobre los 21 hogares
(`dias_consecutivos_exceso_deficit=2`), pero elegirlas mirando todos los hogares sería elegir
con los datos de prueba.

#### Medición final: detector de alertas

Juego de producción fijo, 3 semillas de inyección. Entre paréntesis, detectados / casos.

| | Global | Día sin comida | Día sin cocinar | Semana sin salir | Exceso de descanso | Exceso de ocio sentado | Falsos positivos |
|---|---|---|---|---|---|---|---|
| Detector por clase | 70,9 % (481/678) | 85,7 % (54/63) | 84,1 % (53/63) | 80,0 % (353/441) | 18,3 % (11/60) | 19,6 % (10/51) | 16,3 % (206/1260) |
| — semilla 0 | 68,6 % (155/226) | 90,5 % (19/21) | 85,7 % (18/21) | 76,9 % (113/147) | 10,0 % (2/20) | 17,6 % (3/17) | 17,6 % (74/420) |
| — semilla 1 | 71,2 % (161/226) | 81,0 % (17/21) | 76,2 % (16/21) | 81,0 % (119/147) | 30,0 % (6/20) | 17,6 % (3/17) | 14,0 % (59/420) |
| — semilla 2 | 73,0 % (165/226) | 85,7 % (18/21) | 90,5 % (19/21) | 82,3 % (121/147) | 15,0 % (3/20) | 23,5 % (4/17) | 17,4 % (73/420) |
| Multivariado sobre las columnas `cam_*` | 5,0 % (34/678) | 0,0 % (0/63) | 3,2 % (2/63) | 2,5 % (11/441) | 23,3 % (14/60) | 13,7 % (7/51) | 2,1 % (26/1260) |
| Los dos juntos (lo que emite el proceso diario) | 73,3 % (497/678) | 85,7 % (54/63) | 84,1 % (53/63) | 80,0 % (353/441) | 36,7 % (22/60) | 29,4 % (15/51) | 17,7 % (223/1260) |
| Validación cruzada del detector por clase (fuera de muestra) | 68,9 % (467/678) | 85,7 % (54/63) | 85,7 % (54/63) | 77,8 % (343/441) | 15,0 % (9/60) | 13,7 % (7/51) | 14,4 % (182/1260) |

- **Detección por la clase del desvío** (promedio de las 3 semillas, sin contar alertas de otra
  clase que coincidieron ese día): día sin comida 85,7 %, día sin cocinar 84,1 %, semana sin
  salir 76,9 %, exceso de descanso 3,3 %, exceso de ocio sentado 0 %. O sea: el 18–20 % de
  "detección" de los excesos en la tabla es casualidad, no el detector viendo el exceso.
- **Falsos positivos por clase** (promedio): `cam_cocina` 5,5 %, `cam_descanso_sofa` 3,9 %,
  `cam_ocio_sentado` 3,1 %, `cam_entrar_casa` 2,9 %, `cam_salir_casa` 2,4 %, `cam_comida` 2,0 %.
- **Falsos positivos por tipo** (promedio): `ausencia` 9,2 %, `exceso` 4,9 %,
  `ausencia_sostenida` 2,9 %, `deficit` 0,6 %.
- El detector de tendencia no se evaluó (los desvíos inyectados duran un día o una semana) y el
  de horario está apagado por defecto.

#### Medición final: indicadores de la app (`cloud/app/rutina.py`)

La misma inyección, con la regla de los indicadores de duración (mediana ± 3 sigmas robustos,
piso de 15 min). No son alertas: es lo que la pantalla Reportes marca como desvío.

| Indicador | Día sin comida | Día sin cocinar | Semana sin salir | Exceso de descanso | Exceso de ocio sentado | Falsos positivos |
|---|---|---|---|---|---|---|
| Sedentarismo (descanso + ocio sentado, exceso) | 0,0 % (0/63) | 0,0 % (0/63) | 0,2 % (1/441) | 48,3 % (29/60) | 43,1 % (22/51) | 0,9 % (11/1260) |
| Cocina (déficit) | 0,0 % (0/63) | 9,5 % (6/63) | 0,2 % (1/441) | 0,0 % (0/60) | 0,0 % (0/51) | 0,2 % (2/1260) |
| Cualquiera de los dos | 0,0 % (0/63) | 9,5 % (6/63) | 0,5 % (2/441) | 48,3 % (29/60) | 43,1 % (22/51) | 1,0 % (13/1260) |

El exceso de descanso y de ocio sentado, que el detector por clase no ve, lo marca el indicador
de sedentarismo en poco menos de la mitad de los casos, con 1 % de falsos positivos. No se midió
el indicador `comida` (es el horario de la primera comida: un día sin comida no tiene valor que
mostrar), ni `movilidad` (`cam_moverse`, sin equivalente en CASAS). No hay indicador de salidas.

#### Error del clasificador

Mismo juego de parámetros. Cada evento que la cámara clasificaría se re-etiqueta con la matriz
de confusión medida (`evaluate_classifier_noise.py`, 6 categorías); las entradas y salidas no
pasan por el clasificador y quedan igual. 3 semillas de inyección × 3 de ruido.

| | Global | Día sin comida | Día sin cocinar | Semana sin salir | Exceso de descanso | Exceso de ocio sentado | Falsos positivos |
|---|---|---|---|---|---|---|---|
| Por clase, sin ruido | 70,9 % (481/678) | 85,7 % (54/63) | 84,1 % (53/63) | 80,0 % (353/441) | 18,3 % (11/60) | 19,6 % (10/51) | 16,3 % (206/1260) |
| Por clase, con ruido | 59,8 % (1217/2034) | **20,1 %** (38/189) | **43,4 %** (82/189) | 78,5 % (1039/1323) | 18,9 % (34/180) | 15,7 % (24/153) | 16,9 % (640/3780) |
| — entre semillas de ruido | 59,4 – 60,6 % | 15,9 – 25,4 % | 39,7 – 46,0 % | 78,5 – 78,7 % | 18,3 – 20,0 % | 13,7 – 17,6 % | 16,7 – 17,1 % |
| Los dos juntos, sin ruido | 73,3 % (497/678) | 85,7 % (54/63) | 84,1 % (53/63) | 80,0 % (353/441) | 36,7 % (22/60) | 29,4 % (15/51) | 17,7 % (223/1260) |
| Los dos juntos, con ruido | 62,5 % (1272/2034) | 20,6 % (39/189) | 43,4 % (82/189) | 78,7 % (1041/1323) | 36,7 % (66/180) | 28,8 % (44/153) | 17,9 % (677/3780) |
| Indicadores de `rutina.py`, sin ruido | — | 0,0 % (0/63) | 9,5 % (6/63) | 0,5 % (2/441) | 48,3 % (29/60) | 43,1 % (22/51) | 1,0 % (13/1260) |
| Indicadores de `rutina.py`, con ruido | — | 0,0 % (0/189) | 3,2 % (6/189) | 0,5 % (6/1323) | 50,6 % (91/180) | 35,3 % (54/153) | 0,7 % (27/3780) |

**El día sin comida casi se deja de detectar (85,7 % → 20,1 %) y el día sin cocinar cae a la
mitad (84,1 % → 43,4 %).** El motivo es el mismo que en la prueba 1 del 2026-10-13: el
clasificador confunde otras actividades con comer o cocinar (según la matriz, entre 4 % y 15 %
de los episodios de las otras categorías se etiquetan `EatingDrinkingMedication`), así que un
día sin ninguna comida real igual muestra "comidas". La semana sin salir casi no cambia
(78,5 %) porque no depende del clasificador. Los falsos positivos no se mueven. El global baja
poco (70,9 → 59,8 %) solo porque la semana sin salir pesa el 65 %.

#### Período de aprendizaje

Mismo juego de parámetros, variando `min_window_days` en los dos detectores. 3 semillas.

| Días de historial | Global (por clase) | Día sin comida | Día sin cocinar | Semana sin salir | Falsos positivos (por clase) | Global (los dos juntos) | Falsos positivos (los dos juntos) |
|---|---|---|---|---|---|---|---|
| 7 (actual) | 70,9 % (481/678) | 85,7 % (54/63) | 84,1 % (53/63) | 80,0 % (353/441) | 16,3 % (206/1260) | 73,3 % (497/678) | 17,7 % (223/1260) |
| 14 | 66,8 % (453/678) | 77,8 % (49/63) | 71,4 % (45/63) | 78,0 % (344/441) | 13,2 % (166/1260) | 68,0 % (461/678) | 14,2 % (179/1260) |
| 21 | 64,0 % (434/678) | 68,3 % (43/63) | 63,5 % (40/63) | 76,9 % (339/441) | 10,7 % (135/1260) | 65,2 % (442/678) | 11,7 % (148/1260) |
| 30 | 53,2 % (361/678) | 58,7 % (37/63) | 47,6 % (30/63) | 63,9 % (282/441) | 8,3 % (105/1260) | 54,0 % (366/678) | 9,0 % (114/1260) |

Con más días de historial bajan las dos cosas. Parte de la baja no es del detector: los días
que caen antes de completar el historial no se evalúan y cuentan como no detectados (y como "sin
falso positivo"), igual que en la prueba de período de aprendizaje anterior. Con 14 días los
falsos positivos quedan bajo el tope (13,2 %) a cambio de 4 puntos de detección. No se cambió
el período: sigue en 7 días.

#### Pérdida de eventos de puerta

Mismo juego de parámetros. Cada salida y cada entrada se pierde con probabilidad 10, 20 o 30 %
(niveles **supuestos**: no hay un error medido del detector de puerta). 3 semillas de inyección
× 3 de pérdida. Solo eventos perdidos, no inventados.

| Eventos perdidos | Global (por clase) | Semana sin salir | Falsos positivos (por clase) | Global (los dos juntos) | Falsos positivos (los dos juntos) |
|---|---|---|---|---|---|
| 0 % | 70,9 % (481/678) | 80,0 % (353/441) | 16,3 % (206/1260) | 73,3 % (497/678) | 17,7 % (223/1260) |
| 10 % | 70,9 % (1442/2034) | 79,4 % (1050/1323) | 16,8 % (634/3780) | 73,3 % (1491/2034) | 18,2 % (688/3780) |
| 20 % | 70,5 % (1434/2034) | 78,9 % (1044/1323) | 17,7 % (668/3780) | 72,8 % (1481/2034) | 19,0 % (717/3780) |
| 30 % | 70,1 % (1425/2034) | 78,1 % (1033/1323) | 19,5 % (737/3780) | 72,6 % (1477/2034) | 20,6 % (780/3780) |

Perder eventos de puerta casi no baja la detección (80,0 → 78,1 % con 30 %): sin eventos, una
semana sin salir sigue pareciendo una semana sin salir. Lo que sube son los falsos positivos
(16,3 → 19,5 %), porque un día normal en el que se pierden todas las salidas parece un día sin
salir. No modela una cámara que directamente no ve la puerta ni entradas o salidas inventadas
(otra persona, o alguien que pasa por la zona y sale de cuadro), que serían el error contrario:
esconderían una semana sin salir.

#### Qué no se pudo validar sin videos

> Actualización del mismo día: el detector de puerta se midió con Toyota Smarthome
> (`runner/README.md`, "Validación con video") y la matriz de confusión se volvió a medir
> (actualización (8)). Lo de abajo es lo que se sabía al escribir esta sección.

- **El detector de puerta en sí.** Cuántas entradas y salidas reales detecta o inventa, y si
  60 s de ausencia y 2 s de tolerancia son los valores correctos. Todas las cifras de
  `cam_salir_casa` / `cam_entrar_casa` de arriba suponen que la controladora ve las mismas
  entradas y salidas que los sensores de CASAS, salvo en la prueba de pérdida.
- **Entradas y salidas inventadas**: no hay base para suponer una tasa.
- El detector cuenta a cualquiera que cruce. El sistema es para un solo residente, así que
  no se hizo nada para visitas ni para un segundo residente.
- **`cam_moverse`**: ni detección ni falsos positivos, por no tener equivalente en CASAS.
- **El error del clasificador en un hogar real**: la matriz de confusión es la medida sobre
  Toyota Smarthome, no sobre las cámaras del proyecto.
- Sí está probado sin videos: la lógica de la máquina de estados con trayectorias sintéticas
  (`runner/tests/test_door_events.py`), el adaptador (`baseline/tests/`) y el camino completo
  controladora → nube → alerta con SQLite (`cloud/tests/test_linea_base_puerta.py`).

### Actualización 2026-10-13 (7) — variante sin entradas ni salidas de casa: evaluada y NO adoptada

> **Estado final (decisión del 2026-10-13, posterior a esta medición): esta variante no se
> adoptó.** El sistema queda con las entradas y salidas encendidas y con los parámetros de la
> actualización (6). Lo de abajo es lo que daría el sistema con las entradas y salidas apagadas
> (`ENTRADAS_Y_SALIDAS_ACTIVADAS=0`): en ese caso el proceso diario usa las cuatro clases y los
> parámetros de esta sección (`class_deviations.PRODUCTION_PARAMS_SIN_PUERTA`). Los números no
> se tocaron y se reproducen con `evaluate_production_vocab.py` (sin argumentos). Donde esta
> sección dice "producción" o "el que queda" hay que leer "la variante sin puerta".

**Ningún número anterior de este README cambia**, incluidos los de la actualización (6) y
los históricos de `--cv 5` (72,8 % / 13,1 % y 69,5 % / 13,8 %). `evaluate_deviations.py`,
`cv_grid.py` y los defaults de `ClassDeviationConfig` no se tocaron. Todo lo de abajo sale de
`evaluate_production_vocab.py` (sin argumentos).

**Supuestos de esta variante:**
- **Vocabulario**: `cam_cocina`, `cam_comida`, `cam_descanso_sofa`, `cam_ocio_sentado` y
  `cam_moverse`. **`cam_moverse` no tiene equivalente en CASAS y no se midió**: todo lo de
  abajo es sobre las otras cuatro.
- **Entradas y salidas de casa: apagadas.**
- **Período de aprendizaje: 7 días.** No se cambió (tampoco en el sistema adoptado).
- El detector por clase sigue contando episodios, no minutos. El exceso de descanso o de ocio
  sentado no lo ve (abajo, medido); lo cubren los indicadores de `cloud/app/rutina.py`.
- `comida_mesa` y `aseo` fuera de alcance. El sistema es para un solo residente.

**Protocolo:** 21 hogares de CASAS, 5 particiones por hogar (semilla 0), 3 semillas de
inyección, el barrido de `ClassDeviationConfig` de siempre, tope de 15 % de falsos positivos al
elegir. Desvíos por semilla: día sin comida (21), día sin cocinar (21), exceso de descanso (20),
exceso de ocio sentado (17) y 420 días normales. Sumando las 3 semillas: 237 días con desvío y
1260 normales. Son pocos casos por desvío: un día vale 1,6 a 2 puntos.

#### Parámetros de producción

Mismo procedimiento que en (6): la combinación que más particiones de entrenamiento eligieron
entre las 15 (5 por semilla), sin usar sus hogares de prueba; en empate, la de menos falsos
positivos en entrenamiento. Son los de `class_deviations.production_config(con_puerta=False)`.

| Parámetro | Antes (default del dataclass) | Variante sin puerta |
|---|---|---|
| `frac_historica_min` | 0,8 | **0,6** |
| `excess_k` | 3,0 | **2,5** |
| `dias_consecutivos_ausencia` | 2 | 2 |
| `separar_fin_de_semana` | sí | **no** |
| `dias_consecutivos_exceso_deficit` | 1 | **2** |
| `ausencia_requiere_racha` | no | no |

**El voto empató 4 a 4** y lo definió la regla de desempate, fijada antes de mirar el
resultado:

| Combinación (`frac` / `excess_k` / finde separado / días de exceso) | Votos | Falsos positivos en entrenamiento |
|---|---|---|
| **0,6 / 2,5 / no / 2** (elegida) | 4 | 9,9 % |
| 0,6 / 3,0 / no / 1 | 4 | 14,0 % |
| 0,7 / 3,0 / no / 1 | 3 | 14,2 % |
| 0,6 / 2,5 / sí / 2 | 2 | 9,5 % |
| 0,6 / 4,0 / no / 1 | 1 | 13,5 % |
| 0,8 / 3,0 / no / 1 | 1 | 15,0 % |

Las 15 particiones eligieron `dias_consecutivos_ausencia=2` y `ausencia_requiere_racha=no`. La
elegida exige que un exceso o un déficit se sostenga 2 días (`dias_consecutivos_exceso_deficit=2`).
Esa misma opción se había descartado en (6) como alternativa, porque ahí se la habría elegido
mirando los 21 hogares; acá sale del voto de las particiones de entrenamiento y del desempate,
sin mirar los hogares de prueba. Su efecto es el esperable: un exceso de un solo día ya no
dispara, que es justo el caso que el detector por clase no veía de todos modos.

#### Medición final

Entre paréntesis, detectados / casos. 3 semillas de inyección.

| | Global | Día sin comida | Día sin cocinar | Exceso de descanso | Exceso de ocio sentado | Falsos positivos |
|---|---|---|---|---|---|---|
| **Validación cruzada, fuera de muestra** (detector por clase) | **51,5 %** (122/237) | 90,5 % (57/63) | 79,4 % (50/63) | 16,7 % (10/60) | 9,8 % (5/51) | **12,6 %** (159/1260) |
| — semilla 0 | 48,1 % (38/79) | 85,7 % (18/21) | 85,7 % (18/21) | 10,0 % (2/20) | 0,0 % (0/17) | 9,5 % (40/420) |
| — semilla 1 | 51,9 % (41/79) | 90,5 % (19/21) | 81,0 % (17/21) | 20,0 % (4/20) | 5,9 % (1/17) | 13,8 % (58/420) |
| — semilla 2 | 54,4 % (43/79) | 95,2 % (20/21) | 71,4 % (15/21) | 20,0 % (4/20) | 23,5 % (4/17) | 14,5 % (61/420) |
| **Juego fijo, sobre los 21 hogares** (detector por clase) | **49,8 %** (118/237) | 90,5 % (57/63) | 79,4 % (50/63) | 11,7 % (7/60) | 7,8 % (4/51) | **8,8 %** (111/1260) |
| — semilla 0 | 49,4 % (39/79) | 85,7 % (18/21) | 85,7 % (18/21) | 10,0 % (2/20) | 5,9 % (1/17) | 9,8 % (41/420) |
| — semilla 1 | 49,4 % (39/79) | 90,5 % (19/21) | 81,0 % (17/21) | 15,0 % (3/20) | 0,0 % (0/17) | 8,1 % (34/420) |
| — semilla 2 | 50,6 % (40/79) | 95,2 % (20/21) | 71,4 % (15/21) | 10,0 % (2/20) | 17,6 % (3/17) | 8,6 % (36/420) |
| Multivariado sobre las columnas `cam_*` | 14,3 % (34/237) | 12,7 % (8/63) | 7,9 % (5/63) | 21,7 % (13/60) | 15,7 % (8/51) | 2,1 % (27/1260) |
| **Juego fijo, los dos detectores juntos** (lo que emite producción) | **58,6 %** (139/237) | 90,5 % (57/63) | 79,4 % (50/63) | 33,3 % (20/60) | 23,5 % (12/51) | **10,6 %** (134/1260) |
| Juego fijo, solo en las 4 particiones de prueba cuyo entrenamiento lo eligió | 48,4 % (31/64) | | | | | 9,1 % (31/340) |
| Defaults anteriores, sobre los 21 hogares (detector por clase) | 52,3 % (124/237) | | | | | 17,9 % (226/1260) |
| La otra combinación empatada, sobre los 21 hogares | 54,4 % (129/237) | | | | | 16,6 % (209/1260) |

- La cifra fuera de muestra es la de la validación cruzada (51,5 % / 12,6 %); es la del
  procedimiento, no la de un juego fijo. La del juego fijo sobre los 21 hogares no es
  estrictamente fuera de muestra (cada hogar participó en elegir parámetros en 4 de las 5
  particiones).
- **Los falsos positivos quedan bajo el tope de 15 %** con el juego fijo, también con los dos
  detectores juntos (10,6 %).
- **El global es bajo porque casi la mitad de los casos (111 de 237) son excesos**, que el
  detector por clase no ve: detecta 90,5 % de los días sin comida y 79,4 % de los días sin
  cocinar, y de los excesos solo lo que coincide por casualidad. Contando solo alertas de la
  clase del desvío (promedio de las 3 semillas): día sin comida 90,5 %, día sin cocinar 77,8 %,
  exceso de descanso 1,7 %, exceso de ocio sentado 2,0 %.
- **Falsos positivos por clase** (promedio de las 3 semillas): `cam_cocina` 3,5 %,
  `cam_descanso_sofa` 3,1 %, `cam_ocio_sentado` 2,1 %, `cam_comida` 1,2 %. **Por tipo**:
  `ausencia` 5,1 %, `exceso` 2,8 %, `ausencia_sostenida` 1,2 %, `deficit` 0,2 %.
- El detector de tendencia no se evaluó (los desvíos inyectados duran un día) y el de horario
  está apagado por defecto.

#### Indicadores de la app (`cloud/app/rutina.py`)

La misma inyección, con la regla de los indicadores de duración (mediana ± 3 sigmas robustos,
piso de 15 min). No son alertas: es lo que la pantalla Reportes marca como desvío.

| Indicador | Global | Día sin comida | Día sin cocinar | Exceso de descanso | Exceso de ocio sentado | Falsos positivos |
|---|---|---|---|---|---|---|
| Sedentarismo (descanso + ocio sentado, exceso) | 21,1 % (50/237) | 0,0 % (0/63) | 0,0 % (0/63) | 48,3 % (29/60) | 41,2 % (21/51) | 0,3 % (4/1260) |
| Cocina (déficit) | 2,5 % (6/237) | 0,0 % (0/63) | 9,5 % (6/63) | 0,0 % (0/60) | 0,0 % (0/51) | 0,0 % (0/1260) |
| Cualquiera de los dos | 23,6 % (56/237) | 0,0 % (0/63) | 9,5 % (6/63) | 48,3 % (29/60) | 41,2 % (21/51) | 0,3 % (4/1260) |

El exceso de descanso y de ocio sentado lo marca el indicador de sedentarismo en algo menos de
la mitad de los casos, casi sin falsos positivos. No se midieron el indicador `comida` (es el
horario de la primera comida: un día sin comida no tiene valor que mostrar) ni `movilidad`
(`cam_moverse`).

#### Error del clasificador

Juego fijo. Cada evento se re-etiqueta con la matriz de confusión medida
(`evaluate_classifier_noise.py`, 6 categorías). 3 semillas de inyección × 3 de ruido.

| | Global | Día sin comida | Día sin cocinar | Exceso de descanso | Exceso de ocio sentado | Falsos positivos |
|---|---|---|---|---|---|---|
| Por clase, sin ruido | 49,8 % (118/237) | 90,5 % (57/63) | 79,4 % (50/63) | 11,7 % (7/60) | 7,8 % (4/51) | 8,8 % (111/1260) |
| Por clase, con ruido | **20,5 %** (146/711) | **29,1 %** (55/189) | **33,3 %** (63/189) | 12,2 % (22/180) | 3,9 % (6/153) | 11,9 % (448/3780) |
| — ruido semilla 0 | 22,8 % (54/237) | 31,7 % (20/63) | 36,5 % (23/63) | 15,0 % (9/60) | 3,9 % (2/51) | 12,1 % (152/1260) |
| — ruido semilla 1 | 20,3 % (48/237) | 33,3 % (21/63) | 33,3 % (21/63) | 6,7 % (4/60) | 3,9 % (2/51) | 12,5 % (157/1260) |
| — ruido semilla 2 | 18,6 % (44/237) | 22,2 % (14/63) | 30,2 % (19/63) | 15,0 % (9/60) | 3,9 % (2/51) | 11,0 % (139/1260) |
| Los dos juntos, sin ruido | 58,6 % (139/237) | 90,5 % (57/63) | 79,4 % (50/63) | 33,3 % (20/60) | 23,5 % (12/51) | 10,6 % (134/1260) |
| Los dos juntos, con ruido | **33,5 %** (238/711) | 32,8 % (62/189) | 33,3 % (63/189) | 38,9 % (70/180) | 28,1 % (43/153) | 13,2 % (498/3780) |
| Indicadores de `rutina.py`, sin ruido | 23,6 % (56/237) | 0,0 % (0/63) | 9,5 % (6/63) | 48,3 % (29/60) | 41,2 % (21/51) | 0,3 % (4/1260) |
| Indicadores de `rutina.py`, con ruido | 22,4 % (159/711) | 0,0 % (0/189) | 2,1 % (4/189) | 53,9 % (97/180) | 37,9 % (58/153) | 0,6 % (21/3780) |

**Es el resultado más importante de esta sección: con el error real del clasificador, el
detector de alertas pierde más de la mitad de lo que detecta** (49,8 % → 20,5 %; con los dos
detectores, 58,6 % → 33,5 %). El día sin comida cae de 90,5 % a 29,1 % y el día sin cocinar de
79,4 % a 33,3 %, y los falsos positivos suben de 8,8 % a 11,9 %. El motivo es el de siempre: el
clasificador confunde otras actividades con comer o cocinar (entre 4 % y 15 % de los episodios
de las otras categorías se etiquetan `EatingDrinkingMedication`), así que un día sin ninguna
comida real igual muestra "comidas". Los indicadores de `rutina.py` casi no cambian, porque
miran minutos acumulados y no si hubo o no un episodio.

#### Período de aprendizaje

Juego fijo, variando `min_window_days` en los dos detectores. 3 semillas.

| Días de historial | Global (por clase) | Día sin comida | Día sin cocinar | Exceso de descanso | Exceso de ocio sentado | Falsos positivos (por clase) | Global (los dos juntos) | Falsos positivos (los dos juntos) |
|---|---|---|---|---|---|---|---|---|
| **7 (el que queda)** | 49,8 % (118/237) | 90,5 % (57/63) | 79,4 % (50/63) | 11,7 % (7/60) | 7,8 % (4/51) | 8,8 % (111/1260) | 58,6 % (139/237) | 10,6 % (134/1260) |
| 14 | 42,2 % (100/237) | 74,6 % (47/63) | 68,3 % (43/63) | 10,0 % (6/60) | 7,8 % (4/51) | 7,6 % (96/1260) | 50,6 % (120/237) | 9,0 % (113/1260) |
| 21 | 37,1 % (88/237) | 63,5 % (40/63) | 61,9 % (39/63) | 10,0 % (6/60) | 5,9 % (3/51) | 6,7 % (85/1260) | 45,6 % (108/237) | 8,3 % (104/1260) |
| 30 | 30,0 % (71/237) | 50,8 % (32/63) | 50,8 % (32/63) | 8,3 % (5/60) | 3,9 % (2/51) | 5,2 % (66/1260) | 33,3 % (79/237) | 5,6 % (70/1260) |

Con más días de historial bajan las dos cosas. Parte de la baja no es del detector: los días
que caen antes de completar el historial no se evalúan y cuentan como no detectados (y como "sin
falso positivo"). Con 7 días los falsos positivos ya están bajo el tope, así que alargar el
período solo cuesta detección.

#### Qué no se pudo validar

- **`cam_moverse`**: ni detección ni falsos positivos, por no tener equivalente en CASAS.
- **El error del clasificador en un hogar real**: la matriz de confusión es la medida sobre
  Toyota Smarthome, no sobre las cámaras del proyecto. No hay videos en esta máquina.
- **Episodios y duraciones como los ve la cámara**: acá salen de las anotaciones de CASAS; en
  producción se reconstruyen por los huecos entre eventos (`camera_events.py`).
- Los indicadores `comida` y `movilidad` de `rutina.py`, y los detectores de tendencia y de
  horario.

### Actualización 2026-10-13 (8) — días de puerta votados, medición final del sistema y matriz de confusión nueva

> **Es la sección vigente.** Reemplaza a la (6) en los parámetros de producción y en las tablas
> del sistema tal como corre. Todo sale de
> `evaluate_production_vocab.py --con-puerta --matriz ../fusion/confusion_holdout_produccion.json`.
> Los números de la (6) y de la (7) no se tocaron: son lo que se midió con la puerta fija en 2
> días y con la variante sin puerta. `--cv 5` y `--cv 5 --camera` tampoco cambian (72,8 % /
> 13,1 % y 69,5 % / 13,8 %).

**Qué cambió en el código:**
- `ClassDeviationConfig` tiene dos parámetros nuevos, solo para las clases de puerta:
  `dias_consecutivos_ausencia_puerta` (cuántos días seguidos sin eventos hacen falta) y
  `puerta_dispara_un_dia` (si un solo día ya es un desvío). `score_class_deviations` recibe
  qué clases son de puerta (`clases_de_puerta`); de ellas solo sale la falta de eventos, nunca
  exceso ni déficit. Los defaults (2 días, un solo día no dispara) son lo que hacía el proceso
  diario hasta hoy, así que nada que no pase `clases_de_puerta` cambia.
- El filtro `camera_events.solo_ausencia_sostenida_en_puerta` se eliminó: la regla está ahora
  dentro del detector, y el barrido evalúa lo mismo que emite el proceso diario. En la (6) el
  barrido votaba con las clases de puerta sin filtrar y producción filtraba después.
- El primer día sin eventos de puerta sale con tipo `ausencia`; del segundo en adelante,
  `ausencia_sostenida`.
- `evaluate_production_vocab.py --con-puerta` barre además 1, 2, 3 y 4 días × un solo día
  sí / no: 384 × 8 = 3.072 candidatos. Cinco de las ocho combinaciones de puerta son el mismo
  detector (si un solo día dispara, la cantidad de días no cambia nada, y eso es lo mismo que
  pedir 1 día); entre candidatos idénticos gana el primero del barrido, que es "1 día".
- Mediciones nuevas en el mismo script: "detección del sistema" y la prueba del error del
  clasificador con una matriz leída de un archivo (`--matriz`).

**Protocolo:** el de siempre. 21 hogares de CASAS, 5 particiones por hogar (semilla 0), 3
semillas de inyección, tope de 15 % de falsos positivos al elegir, juego de producción = la
combinación más votada por las 15 particiones de entrenamiento (en empate, la de menos falsos
positivos en entrenamiento). Mismos desvíos que en la (6): por semilla, día sin comida (21),
día sin cocinar (21), semana sin salir (147 días, 7 por hogar), exceso de descanso (20), exceso
de ocio sentado (17) y 420 días normales. Sumando las 3 semillas: 678 días con desvío y 1260
normales. `cam_moverse` no tiene equivalente en CASAS y no se midió.

#### 1. El voto

| Parámetro | Votación anterior (6) | **Producción desde 2026-10-13 (8)** |
|---|---|---|
| `frac_historica_min` | 0,6 | 0,6 |
| `excess_k` | 5,0 | **2,5** |
| `dias_consecutivos_ausencia` | 2 | 2 |
| `separar_fin_de_semana` | sí | sí |
| `dias_consecutivos_exceso_deficit` | 1 | **2** |
| `ausencia_requiere_racha` | no | no |
| `dias_consecutivos_ausencia_puerta` | 2 (fijo, no se votaba) | **1** |
| `puerta_dispara_un_dia` | no (fijo) | no (con 1 día es redundante) |

**Las 15 particiones eligieron que la puerta dispare desde el primer día.** El resto del voto
fue repartido, como en la (6):

| `frac` / `excess_k` / finde separado / días de exceso / días de puerta | Votos | Falsos positivos en entrenamiento |
|---|---|---|
| **0,6 / 2,5 / sí / 2 / 1** (elegida) | 4 | 14,3 % |
| 0,6 / 5,0 / sí / 1 / 1 (la de la (6), con puerta en 1 día) | 3 | 13,8 % |
| 0,6 / 2,5 / no / 2 / 1 | 2 | 14,3 % |
| 0,6 / 5,0 / no / 1 / 1 | 2 | 14,4 % |
| 0,7 / 4,0 / sí / 1 / 1 | 2 | 15,0 % |
| 0,6 / 2,5 / sí / 3 / 1 | 1 | 12,9 % |
| 0,6 / 4,0 / sí / 1 / 1 | 1 | 14,4 % |

Las 15 eligieron `dias_consecutivos_ausencia=2` y `ausencia_requiere_racha=no`, y 13 de 15
`frac_historica_min=0,6`.

**Por qué gana 1 día, y qué cuesta cada opción.** Cada uno de los 7 días de una semana sin
salir cuenta como un caso. Un detector que espera N días seguidos no puede marcar los primeros
N−1: su techo en ese desvío es 100 %, 85,7 %, 71,4 % y 57,1 % para 1, 2, 3 y 4 días. El
procedimiento elige la mayor detección que entra en el tope de falsos positivos, así que
prefiere 1 día mientras entre. La misma validación cruzada, con los días de puerta fijados de
antemano (fuera de muestra, 3 semillas; es informativa, no eligió nada):

| Días seguidos sin salir | Global | Semana sin salir | Falsos positivos |
|---|---|---|---|
| **1 (elegido)**, o cualquiera con "un solo día dispara" | 68,7 % (466/678) | 77,8 % (343/441) | 15,2 % (192/1260) |
| 2 | 64,0 % (434/678) | 70,3 % (310/441) | 14,0 % (177/1260) |
| 3 | 57,7 % (391/678) | 60,8 % (268/441) | 13,8 % (174/1260) |
| 4 | 51,3 % (348/678) | 51,2 % (226/441) | 13,7 % (172/1260) |

Pedir más días baja poco los falsos positivos (15,2 → 13,7 %) y mucho la detección medida así.
**Lo que la tabla no dice** es si avisar el primer día sin salir es lo que se quiere: la
métrica cuenta días, no semanas. Un detector de 3 días marca igual todas las semanas sin salir
(a partir del tercer día) y en esta cuenta "pierde" dos días de cada una. Si lo que importa es
enterarse de la semana y no del primer día, el parámetro se puede subir sin tocar código
(`PRODUCTION_PARAMS["dias_consecutivos_ausencia_puerta"]`); eso sería una decisión de
producto, no el resultado del voto.

#### 2. Tablas finales del sistema tal como corre

Juego de producción fijo, 3 semillas de inyección. Entre paréntesis, detectados / casos.

| | Global | Día sin comida | Día sin cocinar | Semana sin salir | Exceso de descanso | Exceso de ocio sentado | Falsos positivos |
|---|---|---|---|---|---|---|---|
| **Validación cruzada del detector por clase (fuera de muestra)** | **68,7 %** (466/678) | 85,7 % (54/63) | 85,7 % (54/63) | 77,8 % (343/441) | 13,3 % (8/60) | 13,7 % (7/51) | **15,2 %** (192/1260) |
| — semilla 0 | 67,3 % (152/226) | 90,5 % (19/21) | 90,5 % (19/21) | 75,5 % (111/147) | 5,0 % (1/20) | 11,8 % (2/17) | 13,6 % (57/420) |
| — semilla 1 | 70,4 % (159/226) | 81,0 % (17/21) | 76,2 % (16/21) | 80,3 % (118/147) | 25,0 % (5/20) | 17,6 % (3/17) | 15,7 % (66/420) |
| — semilla 2 | 68,6 % (155/226) | 85,7 % (18/21) | 90,5 % (19/21) | 77,6 % (114/147) | 10,0 % (2/20) | 11,8 % (2/17) | 16,4 % (69/420) |
| Detector por clase, juego fijo, 21 hogares | 69,3 % (470/678) | 85,7 % (54/63) | 84,1 % (53/63) | 78,7 % (347/441) | 15,0 % (9/60) | 13,7 % (7/51) | 14,5 % (183/1260) |
| — semilla 0 | 67,7 % (153/226) | 90,5 % (19/21) | 85,7 % (18/21) | 76,9 % (113/147) | 5,0 % (1/20) | 11,8 % (2/17) | 15,5 % (65/420) |
| — semilla 1 | 69,9 % (158/226) | 81,0 % (17/21) | 76,2 % (16/21) | 79,6 % (117/147) | 25,0 % (5/20) | 17,6 % (3/17) | 13,6 % (57/420) |
| — semilla 2 | 70,4 % (159/226) | 85,7 % (18/21) | 90,5 % (19/21) | 79,6 % (117/147) | 15,0 % (3/20) | 11,8 % (2/17) | 14,5 % (61/420) |
| Multivariado sobre las columnas `cam_*` | 5,0 % (34/678) | 0,0 % (0/63) | 3,2 % (2/63) | 2,5 % (11/441) | 23,3 % (14/60) | 13,7 % (7/51) | 2,1 % (26/1260) |
| **Los dos juntos (lo que emite el proceso diario)** | **72,0 %** (488/678) | 85,7 % (54/63) | 84,1 % (53/63) | 78,7 % (347/441) | 35,0 % (21/60) | 25,5 % (13/51) | **16,2 %** (204/1260) |
| Juego fijo, solo en las 4 particiones de prueba cuyo entrenamiento lo eligió | 67,8 % (116/171) | | | | | | 14,1 % (45/320) |

- La cifra fuera de muestra es la primera (68,7 % / 15,2 %): **los falsos positivos quedan
  apenas por encima del tope de 15 %** con el que se eligió en entrenamiento. Es la del
  procedimiento, no la de un juego fijo. La del juego fijo sobre los 21 hogares no es
  estrictamente fuera de muestra (cada hogar participó en elegir en 4 de las 5 particiones).
- **Lo que emite el proceso diario, comparado con la (6):** de 67,4 % / 14,2 % a 72,0 % /
  16,2 %. Sube la detección de la semana sin salir (71,9 → 78,7 %, por el primer día) y
  suben 2 puntos los falsos positivos, que pasan el tope.
- **Detección por la clase del desvío** (promedio de las 3 semillas): día sin comida 85,7 %,
  día sin cocinar 84,1 %, semana sin salir 76,9 %, exceso de descanso 1,7 %, exceso de ocio
  sentado 0 %. El detector por clase sigue sin ver los excesos; lo que figura en la tabla es
  coincidencia con otra alerta del mismo día.
- **Falsos positivos por clase** (promedio): `cam_cocina` 4,8 %, `cam_descanso_sofa` 3,7 %,
  `cam_entrar_casa` 2,9 %, `cam_ocio_sentado` 2,5 %, `cam_comida` 1,9 %, `cam_salir_casa`
  1,9 %. **Por tipo**: `ausencia` 9,2 %, `exceso` 3,6 %, `ausencia_sostenida` 2,9 %, `deficit`
  0,2 %.
- El detector de tendencia no se evaluó y el de horario está apagado por defecto.

#### 3. Detección del sistema: alertas o indicadores de la app

Un día con desvío cuenta como detectado si lo marca una alerta (el detector por clase o el
multivariado) **o** el indicador de `cloud/app/rutina.py` que le corresponde: `cocina` para el
día sin cocinar y `sedentarismo` para los dos excesos. El día sin comida y la semana sin salir
no tienen indicador (`comida` es el horario de la primera comida: un día sin comidas no tiene
valor que mostrar; no hay indicador de salidas), así que solo cuentan por alerta. Un día normal
es un falso positivo si lo marca una alerta o cualquiera de los dos indicadores. La regla de
los indicadores es la de la (6): mediana ± 3 sigmas robustos, piso de 15 min. Un indicador no
avisa a nadie: es lo que la pantalla Reportes muestra como desvío.

| | Global | Día sin comida | Día sin cocinar | Semana sin salir | Exceso de descanso | Exceso de ocio sentado | Falsos positivos |
|---|---|---|---|---|---|---|---|
| Solo alertas (los dos detectores) | 72,0 % (488/678) | 85,7 % (54/63) | 84,1 % (53/63) | 78,7 % (347/441) | 35,0 % (21/60) | 25,5 % (13/51) | 16,2 % (204/1260) |
| Solo indicadores (cualquiera de los dos) | 8,7 % (59/678) | 0,0 % (0/63) | 9,5 % (6/63) | 0,5 % (2/441) | 48,3 % (29/60) | 43,1 % (22/51) | 1,0 % (13/1260) |
| **Sistema (alerta o indicador)** | **76,8 %** (521/678) | 85,7 % (54/63) | 84,1 % (53/63) | 78,7 % (347/441) | 65,0 % (39/60) | 54,9 % (28/51) | **16,9 %** (213/1260) |
| — semilla 0 | 76,1 % (172/226) | 90,5 % (19/21) | 85,7 % (18/21) | 76,9 % (113/147) | 55,0 % (11/20) | 64,7 % (11/17) | 17,9 % (75/420) |
| — semilla 1 | 76,5 % (173/226) | 81,0 % (17/21) | 76,2 % (16/21) | 79,6 % (117/147) | 75,0 % (15/20) | 47,1 % (8/17) | 16,4 % (69/420) |
| — semilla 2 | 77,9 % (176/226) | 85,7 % (18/21) | 90,5 % (19/21) | 79,6 % (117/147) | 65,0 % (13/20) | 52,9 % (9/17) | 16,4 % (69/420) |

| Subtotal | Solo alertas | Sistema |
|---|---|---|
| **Ausencias** (sin comida, sin cocinar, semana sin salir) | 80,1 % (454/567) | 80,1 % (454/567) |
| **Excesos** (descanso, ocio sentado) | 30,6 % (34/111) | **60,4 %** (67/111) |
| Global | 72,0 % (488/678) | 76,8 % (521/678) |
| Falsos positivos | 16,2 % (204/1260) | 16,9 % (213/1260) |

Los indicadores no agregan nada en las ausencias (el de cocina marca 6 días sin cocinar, los 6
ya tenían alerta) y duplican los excesos, de 30,6 % a 60,4 %, con 9 falsos positivos más
sobre 1260 días. La fila "solo indicadores" de la columna Global no dice mucho: la semana sin
salir es el 65 % de los casos y no tiene indicador.

#### 4. Período de aprendizaje

Juego de producción, variando `min_window_days` en los dos detectores. 3 semillas.

| Días de historial | Global (por clase) | Día sin comida | Día sin cocinar | Semana sin salir | Exceso de descanso | Exceso de ocio sentado | Falsos positivos (por clase) | Global (los dos juntos) | Falsos positivos (los dos juntos) |
|---|---|---|---|---|---|---|---|---|---|
| **7 (actual)** | 69,3 % (470/678) | 85,7 % (54/63) | 84,1 % (53/63) | 78,7 % (347/441) | 15,0 % (9/60) | 13,7 % (7/51) | 14,5 % (183/1260) | 72,0 % (488/678) | 16,2 % (204/1260) |
| 14 | 65,9 % (447/678) | 77,8 % (49/63) | 71,4 % (45/63) | 77,3 % (341/441) | 8,3 % (5/60) | 13,7 % (7/51) | 13,0 % (164/1260) | 67,3 % (456/678) | 14,4 % (181/1260) |
| 21 | 63,9 % (433/678) | 68,3 % (43/63) | 63,5 % (40/63) | 76,9 % (339/441) | 8,3 % (5/60) | 11,8 % (6/51) | 10,2 % (129/1260) | 65,2 % (442/678) | 11,7 % (148/1260) |
| 30 | 52,9 % (359/678) | 58,7 % (37/63) | 47,6 % (30/63) | 63,9 % (282/441) | 8,3 % (5/60) | 9,8 % (5/51) | 8,1 % (102/1260) | 53,7 % (364/678) | 8,8 % (111/1260) |

En la columna "los dos juntos" los excesos son, para 7 / 14 / 21 / 30 días: descanso 35,0 %
(21/60), 18,3 % (11/60), 16,7 % (10/60), 13,3 % (8/60); ocio sentado 25,5 % (13/51), 19,6 %
(10/51), 19,6 % (10/51), 13,7 % (7/51).

Con más días de historial bajan las dos cosas. Parte de la baja no es del detector: los días
que caen antes de completar el historial no se evalúan y cuentan como no detectados (y como
"sin falso positivo"). Con 14 días los dos detectores juntos quedan bajo el tope (14,4 %) a
cambio de 4,7 puntos de detección. No se cambió el período: sigue en 7 días. No se midieron
los indicadores de `rutina.py` con otro período (tienen el suyo, 7 días, y no usan
`min_window_days`).

#### 5. Error del clasificador, con la matriz anterior y con la medida el 2026-10-13

Cada evento que la cámara clasificaría se re-etiqueta al azar con una matriz de confusión; las
entradas y salidas no pasan por el clasificador y quedan igual. 3 semillas de inyección × 3 de
ruido. Dos matrices:

- **Anterior**: la de `evaluate_classifier_noise.py` (`vlm/README.md`, 2026-09-20; por
  muestra, validación cruzada por participante, sin zona ni umbral de confianza).
- **Nueva**: `fusion/confusion_holdout_produccion.json`, la decisión del runner por ventana
  sobre 863 videos que el clasificador no vio (`fusion/README.md`, "Matriz de confusión del
  clasificador en uso"). Se usan las 738 ventanas que emiten evento. En porcentaje por fila:

| Real \ Predicho | Cooking | EatingDrinkingMed. | Moving | Phone | ReadingScreen | Resting |
|---|---|---|---|---|---|---|
| Cooking | **95,8** | 4,2 | 0,0 | 0,0 | 0,0 | 0,0 |
| EatingDrinkingMedication | 5,6 | **61,7** | 4,4 | 14,5 | 7,7 | 6,0 |
| Moving | 7,1 | 1,2 | **72,6** | 8,9 | 4,8 | 5,4 |
| Phone | 2,8 | 13,9 | 13,9 | **63,9** | 5,6 | 0,0 |
| ReadingScreen | 0,0 | 4,2 | 1,4 | 2,8 | **86,1** | 5,6 |
| Resting | 0,0 | 1,4 | 0,0 | 1,4 | 8,3 | **88,9** |

| | Global | Día sin comida | Día sin cocinar | Semana sin salir | Exceso de descanso | Exceso de ocio sentado | Falsos positivos |
|---|---|---|---|---|---|---|---|
| Por clase, sin ruido | 69,3 % (470/678) | 85,7 % (54/63) | 84,1 % (53/63) | 78,7 % (347/441) | 15,0 % (9/60) | 13,7 % (7/51) | 14,5 % (183/1260) |
| Por clase, matriz anterior | 59,7 % (1214/2034) | 21,2 % (40/189) | 43,9 % (83/189) | 78,1 % (1033/1323) | 17,8 % (32/180) | 17,0 % (26/153) | 16,2 % (613/3780) |
| **Por clase, matriz nueva** | 64,0 % (1302/2034) | **46,6 %** (88/189) | **59,8 %** (113/189) | 78,4 % (1037/1323) | 19,4 % (35/180) | 19,0 % (29/153) | 17,8 % (673/3780) |
| — ruido semilla 0 | 63,7 % (432/678) | 46,0 % (29/63) | 57,1 % (36/63) | 78,2 % (345/441) | 21,7 % (13/60) | 17,6 % (9/51) | 17,4 % (219/1260) |
| — ruido semilla 1 | 63,4 % (430/678) | 47,6 % (30/63) | 58,7 % (37/63) | 77,8 % (343/441) | 18,3 % (11/60) | 17,6 % (9/51) | 19,1 % (241/1260) |
| — ruido semilla 2 | 64,9 % (440/678) | 46,0 % (29/63) | 63,5 % (40/63) | 79,1 % (349/441) | 18,3 % (11/60) | 21,6 % (11/51) | 16,9 % (213/1260) |
| Los dos juntos, sin ruido | 72,0 % (488/678) | 85,7 % (54/63) | 84,1 % (53/63) | 78,7 % (347/441) | 35,0 % (21/60) | 25,5 % (13/51) | 16,2 % (204/1260) |
| Los dos juntos, matriz anterior | 62,5 % (1272/2034) | 21,7 % (41/189) | 43,9 % (83/189) | 78,2 % (1035/1323) | 37,2 % (67/180) | 30,1 % (46/153) | 17,4 % (656/3780) |
| **Los dos juntos, matriz nueva** | 66,5 % (1353/2034) | 46,6 % (88/189) | 60,8 % (115/189) | 78,6 % (1040/1323) | 34,4 % (62/180) | 31,4 % (48/153) | **19,2 %** (726/3780) |
| — ruido semilla 0 | 66,5 % (451/678) | 46,0 % (29/63) | 58,7 % (37/63) | 78,5 % (346/441) | 36,7 % (22/60) | 33,3 % (17/51) | 18,9 % (238/1260) |
| — ruido semilla 1 | 65,5 % (444/678) | 47,6 % (30/63) | 58,7 % (37/63) | 77,8 % (343/441) | 31,7 % (19/60) | 29,4 % (15/51) | 20,5 % (258/1260) |
| — ruido semilla 2 | 67,6 % (458/678) | 46,0 % (29/63) | 65,1 % (41/63) | 79,6 % (351/441) | 35,0 % (21/60) | 31,4 % (16/51) | 18,3 % (230/1260) |
| Sistema (alerta o indicador), sin ruido | 76,8 % (521/678) | 85,7 % (54/63) | 84,1 % (53/63) | 78,7 % (347/441) | 65,0 % (39/60) | 54,9 % (28/51) | 16,9 % (213/1260) |
| Sistema, matriz anterior | 66,8 % (1358/2034) | 21,7 % (41/189) | 43,9 % (83/189) | 78,2 % (1035/1323) | 69,4 % (125/180) | 48,4 % (74/153) | 17,8 % (672/3780) |
| **Sistema, matriz nueva** | 71,3 % (1451/2034) | 46,6 % (88/189) | 60,8 % (115/189) | 78,6 % (1040/1323) | 68,9 % (124/180) | 54,9 % (84/153) | 19,5 % (737/3780) |
| — ruido semilla 0 | 71,8 % (487/678) | 46,0 % (29/63) | 58,7 % (37/63) | 78,5 % (346/441) | 73,3 % (44/60) | 60,8 % (31/51) | 19,2 % (242/1260) |
| — ruido semilla 1 | 70,5 % (478/678) | 47,6 % (30/63) | 58,7 % (37/63) | 77,8 % (343/441) | 66,7 % (40/60) | 54,9 % (28/51) | 20,7 % (261/1260) |
| — ruido semilla 2 | 71,7 % (486/678) | 46,0 % (29/63) | 65,1 % (41/63) | 79,6 % (351/441) | 66,7 % (40/60) | 49,0 % (25/51) | 18,6 % (234/1260) |
| Indicadores de `rutina.py`, matriz nueva | 7,9 % (161/2034) | 0,0 % (0/189) | 4,8 % (9/189) | 0,6 % (8/1323) | 49,4 % (89/180) | 35,9 % (55/153) | 0,6 % (21/3780) |

- **Con la matriz nueva el día sin comida se detecta 46,6 % de las veces, no 21,2 %**, y el
  día sin cocinar 59,8 %, no 43,9 %. La razón está en la columna `EatingDrinkingMedication`:
  en la matriz anterior entre 4 % y 15 % de los episodios de las otras categorías salían
  como comida; en la nueva, entre 1 % y 14 % (y solo `Phone` pasa de 4,2 %). Hay menos
  "comidas" inventadas en un día sin comidas.
- Sigue siendo la pérdida más grande del sistema: **con el error del clasificador se pierde
  casi la mitad de los días sin comida** (85,7 → 46,6 %).
- **Los falsos positivos suben más con la matriz nueva** (los dos juntos: 16,2 → 19,2 %; con
  la anterior, 17,4 %). No se investigó por qué.
- La semana sin salir no cambia: no pasa por el clasificador.
- La matriz nueva no es "más verdadera" que la anterior en todo: sale de videos no vistos pero
  de participantes y de un departamento que el clasificador ya conocía, y del clasificador sin
  la fusión con esqueleto (ver `fusion/README.md`). En un hogar real el error puede ser mayor.
  Tampoco modela las ventanas que no emiten ningún evento (14,5 % de los videos): acá cada
  evento de CASAS produce siempre un evento de cámara, con la etiqueta que sea.

#### Pérdida de eventos de puerta, con el juego nuevo

| Eventos perdidos | Global (por clase) | Semana sin salir | Falsos positivos (por clase) | Global (los dos juntos) | Falsos positivos (los dos juntos) |
|---|---|---|---|---|---|
| 0 % | 69,3 % (470/678) | 78,7 % (347/441) | 14,5 % (183/1260) | 72,0 % (488/678) | 16,2 % (204/1260) |
| 10 % | 69,2 % (1408/2034) | 78,0 % (1032/1323) | 15,3 % (580/3780) | 71,9 % (1463/2034) | 17,1 % (645/3780) |
| 20 % | 68,9 % (1401/2034) | 77,5 % (1025/1323) | 16,3 % (616/3780) | 71,4 % (1453/2034) | 17,9 % (676/3780) |
| 30 % | 68,5 % (1394/2034) | 76,6 % (1014/1323) | 18,3 % (693/3780) | 71,2 % (1449/2034) | 19,8 % (749/3780) |

Los niveles siguen siendo supuestos. La validación del detector de puerta con video
(`runner/README.md`, "Validación con video") mide cuántas entradas y salidas de cuadro detecta
e inventa sobre Toyota Smarthome, pero no da una tasa de pérdida para una puerta de calle real.

#### Qué no se pudo medir

- **`cam_moverse`**: sin equivalente en CASAS, ni detección ni falsos positivos.
- Los indicadores `comida` y `movilidad` de `rutina.py`, y los detectores de tendencia y de
  horario (apagado).
- **El error del clasificador con la fusión de esqueleto**: el modelo no está en la máquina
  donde se midió la matriz.
- **Cómo muestra la app un desvío `cam_salir_casa:ausencia`** (el tipo del primer día): el
  frontend está en otro repositorio y no se revisó.
- La variante sin puerta (7) no se volvió a medir: sus parámetros no cambian.

## Patrones identificados pero no priorizados (quedan documentados para no perderlos)

- **Secuencia/orden de la rutina** (ej. medicación sin haber desayunado antes) — el más complejo
  de los cuatro patrones discutidos con el usuario, necesita mining de secuencias, no solo
  estadística por clase/día o por semana.

## Cómo seguir (para quien retome el código)

Orden recomendado, de mayor a menor impacto — cada uno referencia el detalle en "Pendiente" abajo:

1. **Seguir bajando el FP (13.1%/13.8% hoy, ver "Actualización 2026-09-20 (3)" en
   "Cross-validation por hogar")**. Es lo primero: un sistema que avisa mal ~1 de cada 8 veces no
   es defendible para el informe ni para un usuario real. Cuatro intentos de calibración ya
   probados con evidencia (no a ojo) — ~~bajar `--max-fp` en la selección~~ (empeoró, 23.9% FP),
   ~~separar semana/fin de semana~~ (ayudó un poco en sensor), ~~persistencia en exceso/deficit~~
   (ayudó un poco en los dos modos), ~~persistencia también en ausencia~~ (probado con
   cross-validation, **los 10 folds la descartaron** — pierde más recall del que ahorra en FP). El
   FP bajó de ~14-15% a ~13-14%, mejora real pero chica — cuatro variaciones del mismo esquema
   (mediana+MAD por clase) ya se agotaron.
   - ~~Repetir el diagnóstico manual sobre una muestra más grande (o el set completo)~~ — **hecho
     2026-10-08** (`diagnose_fp_manual.py`, ver "Actualización 2026-10-08"): el efecto es real pero
     chico (9-11% del FP, no "una fracción grande") — "perdonar" esos días bajaría el FP de 12.6% a
     ~11.4% en modo sensor, no resuelve el problema de fondo. El límite sigue estando en el diseño
     del detector por clase (mediana+MAD), como se sospechaba.
   - ~~Sacar el multivariado del `combinado` en modo sensor~~ — **medido 2026-10-08, NO ayudó**
     (ver "Actualización 2026-10-08"): empeoró recall (−1.4pp) Y FP (+3.0pp) out-of-fold. El
     multivariado aporta poco recall propio pero ancla a una calibración del detector por clase más
     conservadora — sacarlo deja elegir parámetros más sueltos que generalizan peor. Van 4
     variaciones de calibración probadas con evidencia, solo 1 ayudó un poco.
   - ~~**Candidato siguiente**: test de significancia binomial para `ausencia`~~ — **medido
     2026-10-08, NO ayudó** (`ausencia_metodo="binomial"` en `ClassDeviationConfig`, sumado al
     barrido de `evaluate_deviations.py --cv 5`). Out-of-fold: **71.0% recall / 14.2% FP**, peor
     que el 72.8%/13.1% actual en las dos métricas. Causa: 4 de 5 folds eligieron `alpha=0.05` (el
     punto más suelto del barrido de alpha, ya se veía flojo en una prueba rápida previa:
     77.7%/14.8%) — mismo patrón que el experimento del multivariado: lo que el barrido elige
     mirando solo entrenamiento no siempre generaliza. **El umbral actual (`frac_historica_min`)
     queda sin cambios**, es mejor que la alternativa medida. El código de `ausencia_metodo` queda
     implementado y documentado (`src/class_deviations.py`) por si alguien quiere retomar la idea
     con otra variante (ej. acotar el barrido de `alpha` a valores más estrictos, `0.01-0.02`, que
     en la prueba rápida sin cross-validation se veían mejor — no se probó así con CV todavía).
     Van 5 variaciones de calibración probadas con evidencia en este módulo (semana/finde,
     persistencia en exceso/deficit, persistencia en ausencia, sacar el multivariado, binomial),
     solo 1 ayudó (un poco). El FP real del sistema parece estar cerca de su piso con este esquema
     de detector — una mejora grande necesitaría repensar el diseño, no otro parámetro.
2. **Subir la clasificación de actividad de `vlm/`/`fusion/` a las 9 clases que `baseline/` ya
   espera** (hoy la cámara solo da 4-6) — es el trabajo que deja este módulo con más margen de
   mejora real, más que seguir afinando `baseline/` en el vacío contra CASAS. Dos piezas, ambas ya
   diseñadas en `fusion/PLAN_sustitucion_clip.md` pero sin implementar:
   - **Fase 1 — zonas del hogar**: polígonos por cámara (`cama`, `sofá`, `mesa`, `cocina`, `baño`)
     para saber *dónde* está la persona, no solo qué gesto hace. Es lo que separaría
     `descanso_sofa` de `dormir` de `ocio_sentado` (hoy indistinguibles para el esqueleto solo) y
     lo que haría falta para validar `comida_mesa` de verdad (hoy aspiracional, ver "Vocabulario
     de cámara"). **Bloqueada** hasta que el dataset de video grande (~2200 clips) esté de vuelta
     en alguna máquina — sin eso no hay cómo medir si ayuda, solo se puede construir la
     infraestructura (`cameras.yaml` + lookup punto-en-polígono + enganche en
     `camera_worker.py`/`keypoint_sequence.py`).
   - **Fase 2 — contexto de objeto vía YOLO-COCO**: reusar YOLO (variante de detección, no pose)
     para saber qué mueble/electrodoméstico hay cerca de la persona (`bed`, `couch`,
     `dining table`, `tv`, `oven`, `refrigerator`, ya los reconoce sin entrenar nada nuevo) — es
     lo que el usuario pidió acá: que el sistema entienda el contexto (objeto/lugar), no solo el
     gesto, para deducir la actividad. Menos costoso que Fase 1 (no necesita anotar geometría por
     cámara), pero se solapa parcialmente con zona (una mesa ya implica "está en la zona mesa") —
     medir si suma sobre Fase 1 antes de dejarlo en producción, no dar por sentado que ambas
     aportan por separado.
   Una vez que cualquiera de las dos esté validada, conectar la salida real de `vlm/`+`fusion/` a
   `baseline/` deja de estar limitada a 4-6 clases de cámara.
3. El resto de "Pendiente" abajo, sin un orden estricto — son mejoras incrementales, no
   bloqueantes.

**Para orientarse rápido en el código**: `src/features.py` (vectores de día) es la base de todo;
`src/dynamic_clustering.py`, `src/class_deviations.py`, `src/trend_deviations.py` son los 3
detectores de nivel día/semana (mismo patrón: `Config` dataclass + función `score_*`);
`src/timing_deviations.py` es aparte (nivel evento, no día). Cada `evaluate_*.py` en la raíz de
`baseline/` es el arnés de prueba de su detector correspondiente — correrlos es la forma más
rápida de entender qué mide cada uno sin leer todo el código primero.

## Pendiente

- ~~Integrar con eventos reales de cámara~~ / ~~escribir la salida como un `EventType` nuevo~~ —
  **hecho 2026-10-01**, ver "Correr en producción" abajo. Lo que queda de este punto: la cámara
  en vivo hoy solo puebla 5 de las 9 clases del vocabulario (ver "Vocabulario de cámara"), no las
  9 con las que se hizo el cross-validation de arriba — falta medir cuánto de ese 72.8%/69.5%
  recall se sostiene con la cobertura real una vez que `fusion/PLAN_sustitucion_clip.md` (zona de
  mesa, etc.) suba esa cobertura.
- `baseline/src/camera_events.CAMERA_COBERTURA_MIN_HORAS` (1.0h, umbral de "hubo señal mínima ese
  día" para la cámara) es un punto de partida sin calibrar — no hay todavía datos reales de cámara
  para medir si es el valor correcto (ver docstring en `camera_events.py`).
- Segunda vuelta de calibración con sesgo explícito hacia menos FP (mismo criterio que
  `vision/README.md` para caídas), y más muestras por escenario — casi todos los escenarios de
  los 4 detectores tienen entre 3 y 12 casos de prueba, chico para un número definitivo.
- **Separar rutina de semana vs. fin de semana** en `src/features.py` — hoy el vector de día
  mezcla ambos como si fueran la misma distribución. Es justo el tipo de contexto (día de la
  semana) que Snoun et al. 2024 usa explícitamente y que acá no se implementó todavía — candidato
  más probable para bajar el FP sin sacrificar mucho recall (una diferencia real semana/finde hoy
  se lee como "desvío" cuando no lo es).
- `timing_deviations.py`: acotar `DEFAULT_TRACKED_CLASSES` a las clases con horario disciplinado
  (`medicacion` es la más clara) y calibrar por clase en vez de parámetros compartidos — ver
  "Horario habitual" arriba, estado parcial. **Medido el 2026-10-13 con validación cruzada por
  hogar, no se adoptó** (ver "Actualización 2026-10-13 (3)"): baja los falsos positivos solo a
  costa de la detección. Lo que quedó abierto es si conviene seguir emitiendo alertas de
  horario con las clases de cámara (66 % de los días normales reciben alguna). **Decidido el
  mismo día: quedan apagadas por defecto** (ver "Actualización 2026-10-13 (4)").
- Alinear el período de silencio de `timing_deviations.py` (event-count) y `trend_deviations.py`
  (semanas: la tendencia necesita ~11 semanas de historial antes de poder evaluarse, mucho más que
  los 7-10 días de los otros dos detectores) a un criterio único, documentado hacia el usuario
  final (ej. "todavía aprendiendo tu rutina" vs. "ya puedo detectar tendencias de varias semanas"
  como dos niveles de madurez distintos, no un solo estado binario).
- Franjas horarias en el vector de día (hoy solo duración/conteo diario total) si algún desvío
  futuro necesita distinguir "de mañana" vs "de noche" más allá de lo que ya separa
  `bano`/`bano_nocturno`.
- Patrones no priorizados en esta ronda: secuencia/orden de la rutina (ver arriba).

## Correr en producción (2026-10-01, actualizado — ver nota 2026-10-01 más abajo)

**Actualización 2026-10-01 (misma fecha, sesión posterior)**: el informe del PFI (capítulos 3/4)
aclara que la línea base corre del lado de la nube, no en la Pi (RNF6/RNF7). Lo de abajo
(`run_live.py` contra `events/events.db` local) queda como **herramienta de desarrollo/arnés de
prueba local**, no como el camino de producción — ese es ahora `cloud/app/baseline_runner.py` +
`cloud/run_baseline_daily.py` (ver `cloud/README.md`), que reusan exactamente el mismo
`baseline/src/camera_events.py` descripto acá, solo que leyendo eventos ya sincronizados en la
nube en vez de `events/events.db` local. Todo lo que sigue sobre el *diseño* del adaptador y los
4 detectores sigue vigente igual en los dos caminos.

Hasta acá todo lo de arriba corre contra CASAS/sintético vía los `evaluate_*.py`. Para que
`baseline/` opine sobre eventos reales de `runner/` (la cámara en vivo, no un dataset) hay dos
piezas nuevas:

- **`src/camera_events.py`**: adaptador que lee eventos `ACTIVITY` de `events/events.db` y arma
  el mismo vector-día que `features.build_day_vectors()`, reconstruyendo duración a partir de los
  gaps entre eventos consecutivos (el runner emite un evento al cambiar de actividad + heartbeats
  periódicos mientras continúa, no un intervalo con duración explícita). Necesita
  `MonitoringEvent.wall_clock_ts` (reloj real, no el `video_timestamp_s` relativo al archivo) —
  agregado a `events/schema.py` para esto. Vocabulario: ver `CAMERA_ACTIVITY_MAP` ahí, resumen en
  "Vocabulario de cámara" arriba (5 de 9 clases alcanzables hoy).
- **`run_live.py`**: corre los 4 detectores contra ese historial y escribe cada desvío encontrado
  como un evento `EventType.ROUTINE_DEVIATION` en `events/events.db` (idempotente: no duplica si
  se corre dos veces el mismo día). Evalúa **un día a la vez**, por default el de ayer:

  ```
  py run_live.py                                      # ayer, todas las cámaras
  py run_live.py --date 2026-09-20 --video-id webcam   # backfill/debug de un día puntual
  ```

  **Scheduling**: pensado para correr **una vez al día**, temprano a la mañana siguiente (nunca el
  día en curso, que está parcial). Esto es un paso de infraestructura, no algo que resuelva el
  script — en la Pi, un cron (`0 6 * * * cd /ruta/baseline && py run_live.py`) o el Task Scheduler
  de Windows en la PC de desarrollo. No hay todavía un daemon ni un hook en `runner/` que lo
  dispare solo.
  - `--max-gap-s` (default 120s): debe ser mayor que `clip_heartbeat_s` configurado en
    `cameras.yaml` para esa cámara, o se subestima sistemáticamente la duración de cada actividad
    (ver comentario en `camera_events.py`).

Probado con eventos sintéticos (store de prueba con historial + un desvío inyectado a mano) — ver
"Pendiente": falta medir con datos reales de cámara una vez que haya historial de producción.

## Setup

```
cd baseline
py -m venv .venv && .venv\Scripts\activate
pip install -r requirements.txt
```

## Cómo correrlo

```bash
cd baseline
# día a día: forma general + cuánto/cuántas veces por clase (incluye ausencia/exceso/deficit)
py -3 inject_synthetic_deviations.py      # genera synthetic_eval/day_vectors_perturbed.parquet + labels.parquet (default: los 80 hogares del parquet)
py -3 evaluate_deviations.py               # barrido de ClassDeviationConfig "en la misma data" (optimista, ver "Cross-validation por hogar")
py -3 evaluate_deviations.py --full-grid   # + barrido de DriftConfig (lento, no hace falta salvo recalibrar el multivariado)
py -3 evaluate_deviations.py --camera      # versión "sin sensor dedicado" (CAMERA_TRACKED_CLASSES, columnas cam_*)
py -3 evaluate_deviations.py --cv 5        # NÚMERO A CITAR: cross-validation por hogar, modo sensor
py -3 evaluate_deviations.py --cv 5 --camera  # ídem, modo cámara
py -3 evaluate_deviations.py --cv 5 --binomial  # experimento del 2026-10-08 (71,0/14,2 sensor), NO es el número a citar

# pruebas del 2026-10-13 (guardan cada barrido en synthetic_eval/cv_grid_*.pkl; borrarlos si se regenera el set)
py -3 evaluate_classifier_noise.py         # modo cámara con el error real del clasificador, 3 semillas
py -3 evaluate_learning_period.py          # min_window_days = 7, 14, 21 y 30, modos sensor y cámara
py -3 verify_cv_grid.py                    # comprueba que cv_grid.py da lo mismo que cross_validate
py -3 verify_cv_grid.py --completo         # además exige que --cv 5 dé 72,8/13,1 y 69,5/13,8 (unos 20 min)
py -3 evaluate_production_vocab.py --con-puerta  # vocabulario de producción (con entradas y salidas): validación cruzada, voto (con días de puerta), sistema, ruido, aprendizaje y pérdida de puerta (~10 min)
py -3 evaluate_production_vocab.py --con-puerta --matriz ../fusion/confusion_holdout_produccion.json  # además, el error del clasificador con la matriz medida el 2026-10-13 (los números de la actualización (8))
py -3 evaluate_production_vocab.py               # variante sin entradas ni salidas, evaluada y no adoptada
py -3 evaluate_production_vocab.py --paso1       # la medición del 2026-10-13 (5): mismos desvíos, defaults anteriores (~12 min)

# pruebas automatizadas (hogares sintéticos, no necesitan CASAS)
pytest baseline/tests                      # desde la raíz del repo

# horario habitual por actividad
py -3 evaluate_timing_deviations.py        # genera e inyecta internamente, no requiere paso previo
py -3 evaluate_timing_cv.py                # 2026-10-13: validación cruzada por hogar, por clase, vocabularios cámara y sensor (~35 min la primera vez)
py -3 evaluate_door_event_loss.py          # 2026-10-13: modo cámara perdiendo 10/20/30 % de salir_casa y entrar_casa

# tendencia gradual (semanas)
py -3 inject_gradual_trend.py              # genera synthetic_eval/day_vectors_trend_perturbed.parquet + labels_trend.parquet
py -3 evaluate_trend_deviations.py         # barrido, guarda grid_results_trend.csv
```
