# `dataset_events/` — Dataset de eventos de actividad (CASAS) para la línea base de rutinas

## Por qué existe esta carpeta

El módulo [`baseline/`](../baseline/README.md) necesita **rutinas de actividad de una
persona a lo largo de muchos días** para aprender su patrón habitual y detectar
desvíos (clustering dinámico + *behavioral drift*, Sec. 2.1.7 / 2.2.5 del informe).

No existe ningún dataset **público y de acceso abierto** que combine *video real +
hogar doméstico + registro continuo multi-día* de un adulto mayor: por privacidad y
almacenamiento, nadie graba video crudo día tras día en una casa particular. Lo que sí
existe, y en abundancia, son datasets de **sensores ambientales** (movimiento, puertas)
corridos durante meses. Eso valida el enfoque del proyecto (metadata estructurada, no
video a la nube) y nos sirve directo: `baseline/` trabaja sobre **eventos de actividad
etiquetados**, no sobre frames.

Este dataset cubre esa necesidad para desarrollar y validar `baseline/` en paralelo,
sin esperar a que `vision/` + `vlm/` generen eventos reales.

## Qué es (y qué no es)

- **Es:** CASAS Smart Home — archivo `labeled_data.zip` del *Center for Advanced
  Studies in Adaptive Systems* (CASAS), Washington State University, grupo de Diane
  Cook. Registros de sensores binarios de movimiento/puerta en hogares reales
  instrumentados, con tramos de actividad anotados a mano.
- **No es video ni audio.** Son eventos de sensores ambientales. El pipeline visual
  (YOLO + CLIP) se valida aparte con datos propios que grabamos nosotros.
- **No incluye el testbed "Aruba"** (ese viene en `data.zip`, sin etiquetar en
  detalle). No hace falta: los hogares de este zip tienen etiquetas **mucho más
  finas** y mejor alineadas con las actividades del proyecto (hay `Cook`, `Drink`,
  `Eat`, `Relax` como categorías propias; Aruba no separa "beber").

## Origen y licencia (documentar en el informe)

| Campo | Valor |
|---|---|
| Fuente | CASAS Smart Home Datasets, Washington State University |
| Archivo | `labeled_data.zip` (~236 MB comprimido; ~1.2 GB descomprimido) |
| Repositorio | <http://casas.wsu.edu/datasets/> · espejo Zenodo |
| DOI | `10.5281/zenodo.15708568` |
| Licencia | Creative Commons Attribution 4.0 (CC-BY 4.0) — uso y redistribución libres, solo exige citar |
| Cita obligatoria | Cook, D., Crandall, A., Thomas, B., & Krishnan, N. (2013). *CASAS: A smart home in a box*. IEEE Computer, 46(7), 62–69. DOI: 10.1109/MC.2012.328 |
| Contexto de la colección | Datos ambientales longitudinales de 189 hogares comunitarios, 2007–2024. Perfiles: comunidades de jubilados (`hh`, `rw`, `tm`), adultos mayores independientes (`ihs`, `mn`, `mva`), y hogares internacionales. |

**Nota para el Estado del Arte:** es el benchmark estándar de *Human Activity
Recognition* en smart homes — aparece en decenas de papers 2024–2026. Citarlo tiene
peso; no es un dataset oscuro.

## Contenido de la carpeta

```
dataset_events/
  labeled/           # 81 CSV + 1 .txt — NO versionado (ver .gitignore)
    hh101.csv ...    # 30 hogares "hh"  (retirement community)
    ihs06.csv ...    #  5 hogares "ihs" (adulto mayor independiente)
    mn50.csv  ...    #  8 hogares "mn"
    mv001.csv/.txt   #  1 hogar  "mv"  (.txt es el crudo original, .csv es copia)
    mva001.csv ...   #  4 hogares "mva"
    rw101.csv ...    #  7 hogares "rw"
    tm001.csv ...    # 26 hogares "tm"
```

- **~26,3 millones de eventos de sensor** en total.
- Cobertura por hogar: desde ~2–3 semanas (`rw104`: 24 días) hasta **más de un año**
  (`hh113`: jun-2011 → nov-2012). Muy por encima de los 7–10 días que buscábamos.
- **No están en git** (1.2 GB de datos crudos). Para obtenerlos: descargar
  `labeled_data.zip` del DOI de arriba y descomprimir en `dataset_events/labeled/`.
  El único artefacto versionado será el ETL y el JSON de eventos derivado que consume
  `baseline/`.

## Formato de archivo

Un CSV por hogar, **sin encabezado**, una línea por evento de sensor:

```
fecha,hora,sensor,estado[,etiqueta_actividad]
```

Ejemplo (`hh101.csv`):

```
2012-07-20,10:38:54.512364,OutsideDoor,ON,Step_Out="begin"
2012-07-20,10:38:59.541365,OutsideDoor,OFF
2012-07-20,10:39:02.429671,OutsideDoor,ON
...
2012-07-20,11:05:11.001200,Kitchen,OFF,Cook_Lunch="end"
```

| Columna | Descripción |
|---|---|
| `fecha` | `YYYY-MM-DD` |
| `hora` | `HH:MM:SS.ffffff` (hora local del hogar, sin timezone) |
| `sensor` | Identificador por **ubicación** (`Kitchen`, `Bathroom`, `Bedroom`, `LivingRoom`, `OutsideDoor`, ...). PIR de movimiento, sensores de puerta y de temperatura. Los de temperatura no aportan a reconocimiento de actividad. |
| `estado` | `ON`/`OFF` para PIR y puertas; valor numérico para temperatura |
| `etiqueta_actividad` | *Opcional.* Solo aparece en la línea que **abre** (`="begin"`) o **cierra** (`="end"`) un tramo de actividad anotado. |

**Cómo se leen las actividades:** un tramo de actividad es el intervalo entre una línea
`X="begin"` y la siguiente `X="end"` de la misma etiqueta `X`. Entre medio hay decenas
o cientos de líneas de sensores sin etiqueta (movimiento crudo). La mayor parte del
tiempo del hogar **no** está dentro de ningún tramo etiquetado (transiciones, tiempo
muerto, ruido de sensores) — eso es esperable y se filtra.

## Análisis del dataset (2026-09-01, `scratchpad/analyze.py`)

81 hogares, **26,3M eventos de sensor**, **56 etiquetas de actividad distintas**.

- **~32 actividades distintas por hogar** (mediana). El emparejado `begin`/`end` es
  casi perfecto en la mayoría (0 cierres huérfanos, 0 solapamientos).
- **Cobertura temporal etiquetada: 55–85%** del tiempo de reloj en los hogares `hh`
  buenos (hh101 llega a 84,6%) — más de medio día cubierto con una actividad nombrada.
- **Duración**: `hh113` = 509 días de una sola persona, `hh115` = 320d, `hh117` = 346d.

### Hogares recomendados

Volumen + cobertura + limpieza + medicación anotada:
`hh113`, `hh115`, `hh117`, `hh120`, `hh109`, `hh101`, `mn83`.
Los `hh` son single-resident confirmado y tienen `Morning_Meds`/`Evening_Meds`/
`Take_Medicine`; los `ihs/mn/mva/rw/tm` casi no tienen medicación.

### Exclusiones y reservas (aplicadas en `etl.py`)

| Hogar | Motivo | Acción del ETL |
|---|---|---|
| `mv001` | 0 tramos anotados | excluido (salvo `--include-all`) |
| `hh124` | cobertura ~3% | se procesa, `calidad: "baja"` |
| `hh107`, `tm016`, `tm017` | cobertura ~7–11% | se procesan, marcados en `NOTES` |
| `hh126/127/128` | ruido en emparejado begin/end | se procesan, marcados |
| `ihs06/07/08`, `mn61/77` | muchos sensores + baja cobertura | **verificar si son multi-residente** antes de usarlos como "una persona" |

## Vocabulario y mapeo ([`activity_map.py`](activity_map.py))

Dos niveles, ambos en el `metadata` de cada evento:

- **`actividad`** (canónica, 19 clases): la granularidad que el dataset soporta con
  volumen. Es lo que consume `baseline/` para aprender la rutina.
  `dormir · bano · bano_nocturno · aseo · cocinar · lavar_platos · comer · beber ·
  medicacion · descansar · ver_tv · leer · telefono · trabajo · salir_casa ·
  entrar_casa · tareas_hogar · visitas · otro`
- **`actividad_gruesa`** (9 clases + `otro`): el subconjunto que una **cámara única** puede
  llegar a producir (ver [`../fusion/PLAN_sustitucion_clip.md`](../fusion/PLAN_sustitucion_clip.md)
  y "Vocabulario de cámara" en [`../baseline/README.md`](../baseline/README.md), 2026-09-05).
  Permite medir "desvíos detectables con sensores" vs "solo con cámara".
  `dormir · descanso_sofa · ocio_sentado · cocina · comida_mesa · comida ·
  aseo · salir_casa · entrar_casa · otro`
  — `salir_casa`/`entrar_casa` son eventos puntuales, no el estado `fuera_casa` (versión anterior
  de este mapeo); `comida_mesa` (comer) vs `comida` (beber/medicación) depende de la Fase 1 de
  zonas del hogar, todavía no implementada. **Nota**: los `.jsonl`/`.parquet` ya generados
  (`events/`) tienen `actividad_gruesa` calculada con el mapeo VIEJO (11 clases, con `fuera_casa`)
  — hace falta re-correr `etl.py` para que el campo persistido refleje el mapeo actual. `baseline/`
  no depende de este campo persistido: recalcula la versión de cámara en el momento
  (`features.add_camera_columns()`), así que no se vio afectado por este desfasaje.

Etiquetas que **no** son actividad del residente (`Resperate` = ruido de sensor
médico, `Staff_Enter`/`Staff_Leave`) se mapean a `descartar` y el ETL las omite
(las cuenta en el manifest).

## Contrato de salida ([`events/schema.py`](../events/schema.py) — `MonitoringEvent`)

```json
{
  "video_id": "casas_hh113",
  "event_type": "activity",
  "video_timestamp_s": 1234567.0,
  "metadata": {
    "actividad": "cocinar",
    "actividad_gruesa": "cocina",
    "etiqueta_original": "cook_dinner",
    "inicio_iso": "2011-06-15T19:02:11",
    "fin_iso": "2011-06-15T19:41:55",
    "duracion_s": 2384.0,
    "hora": 19,
    "minuto_del_dia": 1142,
    "dia_semana": 2,
    "fuente": "CASAS",
    "hogar": "hh113"
  }
}
```

- Cada hogar es **proxy de un residente**: `video_id = casas_<hogar>`, consistente con
  el uso de `video_id` como proxy de residente en `baseline/`.
- `video_timestamp_s` = segundos desde el primer evento del hogar (el esquema usa
  timestamp relativo). El reloj real y las features derivadas (`hora`,
  `minuto_del_dia`, `dia_semana`) van en `metadata` — es lo que `baseline/` necesita
  para el clustering por hora del día / día de semana.

## Pipeline por capas

```
labeled/*.csv         BRONZE  crudo CASAS (gitignored, se re-descarga del DOI)
   │  etl.py
   ▼
events/casas_*.jsonl  SILVER  eventos MonitoringEvent, parseados + limpios + validados
events/*.db                   (mismo contenido en SQLite, esquema de events/store.py)
   │  build_table.py
   ▼
events/casas_events.parquet         SILVER tabla tidy tipada (1 fila = 1 evento)
events/casas_timeline_10min.parquet GOLD   flujo de actividad resampleado a bin fijo
events/_quality_report.md           reporte de calidad de datos (versionado)
```

### `etl.py` — bronze → silver

```bash
py -3 dataset_events/etl.py                          # todos los hogares
py -3 dataset_events/etl.py --homes hh113 hh115
py -3 dataset_events/etl.py --limit-days 10          # primeros 10 días de cada hogar
py -3 dataset_events/etl.py --sqlite dataset_events/casas_events.db
py -3 dataset_events/etl.py --include-all            # no excluir hogares marcados
```

Salida en `dataset_events/events/`:
- `casas_<hogar>.jsonl` — un `MonitoringEvent` por línea (gitignored, se regenera).
- `_manifest.json` — resumen por hogar + totales globales (**versionado**).
- SQLite opcional con el esquema de `events/store.py` → `baseline/` apunta
  `EventStore(db_path=...)` ahí sin cambios.

**Solo stdlib** + `pydantic` (valida cada evento contra el contrato compartido).

### `build_table.py` — silver → tabla analítica + gold

```bash
py -3 dataset_events/build_table.py                  # bin de timeline: 10 min
py -3 dataset_events/build_table.py --bin-min 15
```

- `casas_events.parquet` — tabla tidy, columnas tipadas (categóricas para
  `actividad`/`hogar`, `datetime64` para `inicio`/`fin`), + features derivadas
  (`fecha`, `franja`, `fin_de_semana`, `duracion_min`) + flags de calidad
  (`solapa`, `dia_dst`). Es lo que conviene cargar en `baseline/` para análisis
  con pandas/numpy/scikit-learn (10 MB vs. ~120 MB de JSONL).
- `casas_timeline_<bin>.parquet` — **flujo de actividad a intervalo regular** (la
  forma que consume el clustering de rutinas, tipo el paper de Aruba): un bin cada
  N min, actividad gruesa dominante por cobertura, `sin_dato` si no hay ninguna.
- `_quality_report.md` — distribuciones de duración por actividad, duraciones
  sospechosas, cobertura de calendario por residente, solapamientos, DST. Para el
  informe.

Requiere `pandas` + `pyarrow` + `tabulate` (ver `requirements.txt`).

## Decisiones de limpieza y estandarización (para el informe)

| Problema en el crudo | Qué hace el pipeline |
|---|---|
| Líneas con <4 campos / timestamp inválido | se descartan y se cuentan (`_manifest.json`) |
| `X="begin"` sin su `X="end"` (o al revés) | cierre huérfano → descartado; tramo sin cerrar al EOF → descartado |
| `X="begin"` repetido sin cerrar | se cierra el anterior en el nuevo `begin` (`tramo_reabierto`) |
| Tramos con duración ≤ 0 o > 20 h | descartados como par corrupto |
| Etiquetas que no son actividad del residente (`Resperate`, `Staff_*`) | mapeadas a `descartar`, omitidas |
| 56 etiquetas finas heterogéneas entre hogares | estandarizadas a 19 canónicas + 11 gruesas (`activity_map.py`) |
| Timestamps naive sin timezone | **se dejan naive en hora local de pared** — es lo correcto para clustering por hora del día; los días de cambio de hora (DST) se marcan en `dia_dst`, no se corrigen |
| Actividades solapadas/anidadas de CASAS | silver: se conservan como eventos separados (`solapa=True`); gold: el bin toma la actividad con más cobertura |
| Tiempo no anotado (~35–50% del reloj) | no se inventa nada: en el timeline queda como `sin_dato` |

## Resultado de la corrida completa (2026-09-01)

**319.229 eventos, 80 residentes, 19 clases canónicas.** 63.370 h etiquetadas
(2.640 días-persona). Sin etiquetas desconocidas. Solapamientos 0,1%. Solo `hh124`
quedó `calidad: "baja"` (cobertura de calendario 45%).

Top actividades (eventos): `aseo` 55,5k · `cocinar` 40,5k · `trabajo` 32,6k ·
`ver_tv` 32,0k · `bano` 30,1k · `dormir` 24,9k · `lavar_platos` 19,4k ·
`salir_casa` 14,4k · `beber` 13,2k · `comer` 12,6k · `descansar` 11,2k ·
`bano_nocturno` 6,0k · `medicacion` 4,5k.

Timeline 10 min (bins): `sin_dato` 34% · `dormir` 29% · `ocio_sentado` 15% ·
`aseo` 7% · `cocina` 4% · `comida` 3% · `descanso_sofa` 3% · `fuera_casa` 2%.

## Estado

- [x] Dataset descargado y descomprimido en `labeled/` (local, no versionado).
- [x] Análisis del dataset.
- [x] ETL bronze→silver (`etl.py` + `activity_map.py`).
- [x] Capa analítica + gold + reporte de calidad (`build_table.py`).
- [ ] Integración con `baseline/`: clustering de rutinas sobre
      `casas_timeline_10min.parquet` (o `casas_events.parquet`), y validación
      end-to-end inyectando desvíos sintéticos (ver "Validación end-to-end" en
      [`../fusion/PLAN_sustitucion_clip.md`](../fusion/PLAN_sustitucion_clip.md)).
