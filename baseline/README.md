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

# horario habitual por actividad
py -3 evaluate_timing_deviations.py        # genera e inyecta internamente, no requiere paso previo
py -3 evaluate_timing_cv.py                # 2026-10-13: validación cruzada por hogar, por clase, vocabularios cámara y sensor (~35 min la primera vez)
py -3 evaluate_door_event_loss.py          # 2026-10-13: modo cámara perdiendo 10/20/30 % de salir_casa y entrar_casa

# tendencia gradual (semanas)
py -3 inject_gradual_trend.py              # genera synthetic_eval/day_vectors_trend_perturbed.parquet + labels_trend.parquet
py -3 evaluate_trend_deviations.py         # barrido, guarda grid_results_trend.csv
```
