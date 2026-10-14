# `cloud/` — Backend cloud (API + línea base en la nube)

Implementa la arquitectura cloud que ya define el informe del PFI (capítulos 3 y 4):
FastAPI + Postgres (RDS en producción), recibe eventos sincronizados desde la Pi
(`runner/cloud_sync.py`), corre `baseline/` contra ese historial y expone una API mínima para la
futura app móvil.

**Estado: desplegado en AWS (Elastic Beanstalk + RDS, ver punto 1 de "Cómo seguir") y probado de
punta a punta desde una app real en un celular** (2026-10-07/08, ver "Prueba end-to-end en
celular real" en el README del repo Frontend). Proyecto de Firebase creado y probado con tokens
reales (2026-10-06, ver "Cómo seguir" → punto 2). KIT/vinculación, eventos (resolver,
`ubicacion`, filtros RF13), CRUD de familiares/roles (RF12), resumen y serie de rutina (RF9,
`GET /rutina` y `GET /rutina/serie`, este último sumado 2026-10-08 para el gráfico de evolución
de la app) — todo probado contra Postgres real en producción, con Firebase real. **Sin probar
todavía**: Firebase Cloud Messaging real (push tokens se guardan, `POST /me/push-token`, pero
nada los consume del lado de la app — requiere *development build*, bloqueado en EAS). El
resumen/serie de rutina además se corrió contra `dataset_events/events.zip` (CASAS, un hogar real
con 490 días de historial) como proxy de variabilidad real — ver nota en `app/rutina.py`, no
reemplaza calibrar con cámara real, que sigue sin existir.

## Arquitectura

```
runner/cloud_sync.py (Pi) ──POST /devices/events──► cloud/ (FastAPI)
                                                          │
                                                          ▼
                                                   Evento (Postgres/RDS)
                                                          │
                                                          ▼
                                    app/linea_base_diaria.py (thread de la API, 03:00 ART)
                                       reusa baseline/src/ (ver baseline_runner.py)
                                                          │
                                                          ▼
                                    Evento(event_type=routine_deviation) + push (FCM)
                                                          │
                                                          ▼
                                    app/retencion.py: borra eventos vencidos
                                       (solo si están definidas sus variables de entorno)

app móvil (Firebase Auth) ──GET /hogares/{id}/eventos──► cloud/ (lee Postgres)
```

`baseline/run_live.py` (el que corre en la Pi, de la sesión anterior) **no es el camino de
producción** según el informe (la línea base corre del lado de la nube, RNF6/RNF7) — queda como
herramienta de desarrollo/arnés de prueba local, igual que los `evaluate_*.py` de `baseline/`.

## Modelo de datos

Fiel al diagrama del informe (Figura 3.11, "Diagrama de Base de Datos en la Nube"; detalle campo
por campo en `docs/vinculacion-kit.md` del repo Frontend, sección 4). Ver `app/models.py`.
Alineado el 2026-10-06 (migración `a3f9c2d17b40`):

- `Kit` (la controladora + sus cámaras): existe desde fábrica (`alta_kit.py`), con
  `cantidad_camaras` (plan, 1 a 3), `codigo_vinculacion_hash` (QR) y **`device_api_key_hash`** (la
  credencial de la Pi es del kit, no del hogar: si se cambia la controladora, se vincula un kit
  nuevo al mismo hogar sin perder historial). `hogar_id` null hasta que alguien escanea el QR.
- `Hogar` 1:1 `Residente` (RNF11), 1:N `Kit`; `Camara` N:1 `Kit` (y `hogar_id` null hasta
  vincular, `identificador_local` = el `id` de `cameras.yaml`, único por kit; `ubicacion`
  obligatoria como en la figura: `""` mientras el familiar no la cargó, la API la devuelve `null`).
- `Evento` (FK a `Camara` —nullable: los `routine_deviation` no vienen de una cámara— y a
  `Residente`, desnormalizado), con `resuelta_at`/`resuelta_por` para "Marcar como resuelta".
- `Familiar` N:M `Hogar` vía `HogarFamiliar.rol` (`responsable_principal` | `familiar`);
  `Familiar.push_token` para FCM.
- `LineaBaseComportamiento` 1:1 `Residente`: **creada pero todavía no se escribe** —
  `baseline_runner.py` recalcula la línea base en cada corrida.

Datos previos a la migración: cada hogar recibió un kit `LEGACY-NNNNNN` con su misma API key, así
que una Pi ya configurada sigue autenticando sin cambios.

## Esquema de transporte: reusa `events/schema.py`

El body de `POST /devices/events` es una lista de `MonitoringEvent` (mismo `EventType` que ya usa
todo el pipeline local) — el informe pide explícitamente compartir ese esquema Pydantic entre el
backend local y el de la nube. `occurred_at` (columna de `Evento`) sale directo de
`MonitoringEvent.wall_clock_ts` (agregado a `events/schema.py` en la sesión anterior).

### Ingestión idempotente (2026-10-13)

`POST /devices/events` ya no guarda dos veces el mismo evento. Antes, si la nube guardaba un
lote pero la respuesta no llegaba a la controladora, ésta lo reenviaba y quedaba duplicado
(medido: 10 eventos, 20 filas; ver "Corte y reconexión con la nube, medido" en
`runner/README.md`).

- Cada evento trae `event_id`, un identificador único que la controladora genera al guardarlo
  en su base local y repite en cada reenvío (`events/README.md`). Se guarda en `Evento.event_id`.
- Al recibir un lote, la nube busca cuáles de esos identificadores ya tiene **para las cámaras
  de ese kit** y guarda solo el resto. Lo que mandó un kit no bloquea a otro.
- La respuesta sigue siendo **201**, ahora con `{"insertados": n, "duplicados": m}`. Tiene que
  ser 201 también cuando todo el lote es repetido: así la controladora lo marca como enviado
  y deja de reenviarlo.
- Un evento repetido no vuelve a disparar el aviso de caída o inmovilidad.
- Última barrera: restricción única `(camara_id, event_id)` en la tabla. Si dos pedidos con los
  mismos eventos llegan a la vez, uno de los dos falla al confirmar; la ingestión vuelve a
  consultar y los trata como repetidos.
- Un evento sin `event_id` (controladora con el código anterior) se guarda siempre, como antes:
  compatible, pero sin protección. Los `routine_deviation` que genera la nube tampoco lo
  llevan; su propia protección contra duplicados es la de `baseline_runner.py`.

Migración: `alembic/versions/c41e7d905b12_evento_event_id.py` agrega la columna (null) y la
restricción. Los eventos que ya estaban quedan con `event_id` en NULL, que no cuenta como
repetido. **No está aplicada en RDS ni desplegada**: para desplegar este cambio hay que correr
`alembic upgrade head` contra la base antes de subir el código nuevo (el código nuevo consulta
la columna). Probada en local con SQLite, ida y vuelta, con eventos ya cargados; `alembic check`
confirma que el esquema migrado coincide con los modelos. No se probó contra Postgres.

## Autenticación

Dos esquemas separados (`app/auth.py`):

- **Pi → nube**: header `X-API-Key`, una key por kit (`Kit.device_api_key_hash` — se guarda el
  hash, la key cruda la muestra una sola vez `alta_kit.py`). Cada request autenticado actualiza
  `Kit.ultima_conexion_at`. Mientras el kit no esté vinculado a un hogar con residente,
  `POST /devices/events` responde **409** y la Pi reintenta en el próximo ciclo (no se pierde nada).
  Si el hogar del kit se dio de baja responde **410** y la Pi descarta esos eventos (ver "Baja del
  hogar").
- **App móvil → nube**: `Authorization: Bearer <firebase_id_token>`, verificado con
  `firebase_admin.auth.verify_id_token`. **Si el proceso no tiene `GOOGLE_APPLICATION_CREDENTIALS`
  configurado, estos endpoints devuelven `501` con un mensaje claro** en vez de romper el arranque
  de la app — así se puede levantar y probar la ingestión + `baseline_runner` sin Firebase.
  Con Firebase: `GOOGLE_APPLICATION_CREDENTIALS=~/.secrets/firebase-pfi.json` (clave de cuenta de
  servicio del proyecto `pfi-ellero-ferrario`, **fuera del repo**). `verify_id_token` usa
  `clock_skew_seconds=10` (2026-10-07): un token pedido justo después de crear la cuenta puede
  tener el `iat` un segundo por delante del reloj de este proceso (resolución normal de NTP
  entre dos máquinas, no un reloj mal configurado) y sin tolerancia se rechaza como "usado
  demasiado pronto" — encontrado probando la app en un celular contra un backend local.

## Endpoints de la app móvil

Documentación interactiva (Swagger) en `/docs`, agrupada por función (Controladora, Cuenta,
Vinculación, Hogar, Eventos, Rutina, Familiares). Los grupos, descripciones y candados se definen
en `main.py` y en cada `@router` (`tags`, `summary`); son solo metadatos de OpenAPI.

| Endpoint | Qué hace |
|---|---|
| `GET /me` | Perfil del familiar logueado + sus hogares (`rol`, `residente`). **404** si la cuenta de Firebase todavía no tiene `Familiar` → la app llama a `POST /me`. |
| `POST /me` | Body `{nombre}` (2-60). Crea el `Familiar`; el email sale del token, no del body. Idempotente: si ya existe lo devuelve. |
| `POST /me/push-token` | Body `{token}` (token de FCM del celular, RF8). **204**. Pisa el anterior (un token por familiar). **404** si la cuenta todavía no tiene `Familiar` (llamar `POST /me` primero). |
| `POST /kits/vincular` | Body `{codigo}` (del QR; acepta minúsculas, espacios, guiones). Crea el `Hogar` ("Mi hogar"), le asigna kit y cámaras y deja al familiar como `responsable_principal`. Devuelve `{hogar_id, numero_serie, cantidad_camaras, camaras}`. **404** código inexistente · **409** kit de otro hogar · **429** 5 intentos fallidos en 15 min (por familiar, en memoria del proceso). Reintento del mismo familiar = idempotente. |
| `GET /hogares/{id}/camaras` | Cámaras del hogar. Cualquier familiar vinculado. |
| `PATCH /hogares/{id}/camaras/{camara_id}` | Body `{ubicacion}` (2-40). RF10. **409** si repite la de otra cámara del hogar. Solo `responsable_principal`. |
| `DELETE /hogares/{id}` | Baja del hogar (2026-10-13). **204**. Solo `responsable_principal` (**403** para un `familiar` común o alguien ajeno al hogar). Borra residente, eventos, línea base, vínculos de familiares y el hogar; el kit y sus cámaras quedan sin hogar para volver a vincularse. No se puede deshacer. Ver "Baja del hogar". |
| `PUT /hogares/{id}/residente` | Body `{nombre, telefono}` (E.164 o null). Crea o actualiza; si el hogar se llamaba "Mi hogar" pasa a "Casa de {nombre}". **204**. Solo `responsable_principal`. Es el último paso: con residente, la app pasa a las pestañas y la Pi ya puede sincronizar eventos. |
| `GET /hogares/{id}/eventos` · `/alertas` | Eventos del hogar. **403** si el familiar no está vinculado. `/eventos` acepta filtros opcionales `tipo`, `desde`, `hasta` (RF13, `occurred_at` ISO 8601). Cada evento trae `ubicacion` (de `Camara.ubicacion`, `null` si no vino de una cámara o no se cargó) y `resuelta` (`Evento.resuelta_at is not None`). |
| `POST /hogares/{id}/eventos/{evento_id}/resolver` | "Marcar como resuelta" (mockup 03, RF8). Cualquier familiar vinculado (no solo `responsable_principal`). **204**, idempotente (no pisa quién la marcó primero). **404** si el evento no es de ese hogar. |
| `GET /hogares/{id}/rutina?periodo=` | Resumen de rutina del residente vs. su historial (pantalla Reportes). `periodo` = `hoy`\|`semana`\|`mes` (ventana rolling, no calendario). Ver `app/rutina.py`: solo dos indicadores (`comida`, `sedentarismo`) — los únicos calculables con cámara única hoy, sin inventar `"despertar"` como mostraba el mock de la app. `indicadores: []` si no hay suficiente historial. **No lee `LineaBaseComportamiento`** (sigue sin escribirse, ver modelo de datos) — calcula todo al vuelo sobre `Evento` en cada request; si el volumen de eventos crece mucho, considerar cachear o sí escribir esa tabla. |
| `GET` · `POST /hogares/{id}/familiares` | RF12. Listar (cualquier vinculado) / invitar por email (solo `responsable_principal`; body `{email, rol}`). Invitar es "a quien ya tiene cuenta" — **404** si ese email nunca hizo `POST /me`, sin invitación pendiente. Idempotente: invitar a alguien ya vinculado devuelve su vínculo sin pisarle el rol. |
| `PATCH /hogares/{id}/familiares/{familiar_id}` | Cambiar `rol`. Solo `responsable_principal`. **409** si es el único `responsable_principal` del hogar y se le intenta bajar el rol. |
| `DELETE /hogares/{id}/familiares/{familiar_id}` | Desvincular. Uno mismo siempre puede; a otro familiar, solo `responsable_principal`. **409** si es el único `responsable_principal` del hogar. |

Permisos en `app/permisos.py`. Contrato: `docs/vinculacion-kit.md` (sección 5) y
`src/api/types.ts` del repo Frontend (cambiar los dos juntos). `kit` en `/me` es el kit activo del
hogar (el último vinculado). **Diferencia con la app hoy**: las cámaras salen con
`identificador_local` (nombre del diagrama), la app todavía espera `local_id`.

## Correr localmente (sin Docker, sin Postgres)

```
cd cloud
py -m venv .venv && .venv\Scripts\activate
pip install -r requirements.txt

# crear el esquema (SQLite por default, ver DATABASE_URL en app/db.py)
alembic upgrade head

uvicorn main:app --reload
```

Dar de alta un kit "de fábrica" (muestra UNA vez el código del QR y la API key de la Pi):

```
py alta_kit.py --camaras 2                    # cámaras cam1, cam2
py alta_kit.py --camaras 1 --ids webcam       # ids = los `id` de cameras.yaml
```

El hogar lo crea la vinculación desde la app (`POST /kits/vincular` con el código, después
`PUT /hogares/{id}/residente`). Recién con residente la Pi puede sincronizar eventos (antes: 409).

Después, registrar la cámara (`POST /devices/camaras`, header `X-API-Key`) y postear eventos
(`POST /devices/events`) como ya hace `runner/cloud_sync.py` del lado de la Pi.

## Correr la línea base

```
py run_baseline_daily.py                              # ayer, todos los hogares
py run_baseline_daily.py --date 2026-09-20 --hogar-id 1
```

En producción **la corre la propia API** (`app/linea_base_diaria.py`, 2026-10-09): un thread que
arranca con uvicorn y todos los días a las 03:00 de Argentina (UTC-3 fijo, no la zona del
contenedor) evalúa el día anterior de cada hogar. Antes no la corría nada en AWS (sin EventBridge,
Lambdas ni cron, confirmado en la consola). Con Postgres toma `pg_try_advisory_lock` para que, si
hay más de una instancia, no corra dos veces a la vez. Se desactiva con
`LINEA_BASE_DIARIA_DESACTIVADA=1` (tests, desarrollo local). Después de la línea base, la misma
tarea aplica la retención de eventos (ver "Retención de eventos"). Este CLI queda para correr un día
puntual a mano (backfill/debug). Mismo criterio que `baseline/run_live.py`: nunca evalúa el día en
curso, es idempotente (no duplica un desvío ya emitido).

### Vocabulario y parámetros del proceso diario (2026-10-13)

Dos cambios en lo que evalúa `app/baseline_runner.py`, los dos en `baseline/src/` (acá solo se
importan):

- **Entradas y salidas de casa: encendidas por defecto.** El proceso sigue dos clases más,
  `cam_salir_casa` y `cam_entrar_casa`, además de las cinco del clasificador
  (`camera_events.LIVE_TRACKED_CLASSES`). Salen de los eventos `LeaveHome` / `EnterHome` que la
  controladora emite cuando la cámara tiene configurada la zona `puerta` (`runner/README.md`,
  "Entradas y salidas de casa"). Entran por `POST /devices/events` como cualquier `activity`: no
  hizo falta tocar el esquema ni la ingestión. Un hogar sin esa zona no los manda nunca y no
  recibe alertas de puerta. Se apagan con `ENTRADAS_Y_SALIDAS_ACTIVADAS=0`.
- **De la puerta solo sale un desvío: que falten entradas o salidas.** Nunca una alerta por el
  horario de una salida, ni siquiera con `ALERTAS_DE_HORARIO_ACTIVADAS=1`, ni por exceso,
  déficit o tendencia: el detector de horario y el de tendencia no siguen esas dos clases, y
  el detector por clase las trata aparte (`clases_de_puerta`).
  **Cambio del 2026-10-13 (segunda votación): el primer día sin salir o sin entrar ya genera
  el desvío.** Ese primer día sale como `cam_salir_casa:ausencia` / `cam_entrar_casa:ausencia`;
  del segundo día en adelante, como `...:ausencia_sostenida`. Hasta ese cambio hacían falta 2
  días seguidos y un solo día no generaba nada. Cuántos días hacen falta es un parámetro
  (`dias_consecutivos_ausencia_puerta`) y lo eligió la validación cruzada: las 15 particiones
  de entrenamiento eligieron 1 (ver `baseline/README.md`, "Actualización 2026-10-13 (8)", que
  también muestra lo que cuesta pedir 2, 3 o 4). **La app tiene que mostrar bien el tipo
  `ausencia` para esas dos clases**: no se revisó el frontend, que está en otro repositorio.
- **Parámetros del detector por clase.** Antes corría con los defaults de
  `ClassDeviationConfig()`; ahora con `class_deviations.production_config()`: desde la segunda
  votación del 2026-10-13, `frac_historica_min=0.6`, `excess_k=2.5`,
  `dias_consecutivos_exceso_deficit=2` (un exceso o un déficit tiene que sostenerse 2 días) y
  `dias_consecutivos_ausencia_puerta=1`; el resto igual. Los eligió la validación cruzada
  sobre el vocabulario que producción puebla. Con los defaults, los falsos positivos medidos
  eran 22,5 %. Los números están en `baseline/README.md`, "Actualización 2026-10-13 (8)" (la
  primera votación, con `excess_k=5.0` y la puerta fija en 2 días, está en la (6)).

- **Texto del aviso al celular (2026-10-13).** El push de un desvío de rutina mandaba como
  cuerpo la clave interna (`cam_comida:ausencia`) con el título fijo "Posible desvío de
  rutina". Ahora título y cuerpo salen de `alertas.texto_desvio`, con los mismos textos que
  muestra la app (`presentacion.ts` del Frontend, `desvioDeRutina`): por ejemplo "Cambio en la
  rutina" / "Hoy no se registró comer o beber", o "Hoy no se registró ninguna salida de casa".
  Vale para todos los detectores (por clase, tendencia, multivariado y horario). Los textos
  están copiados en los dos repositorios: si se cambia uno hay que cambiar el otro. El aviso
  de horario (apagado por defecto) lleva dos horas, así que es el único con dos puntos. El
  envío real sigue sin probarse: Firebase no está configurado en las pruebas.

**No cambia ningún número de este README**: las pruebas de `test_linea_base_produccion.py`
(el día sin comidas emite dos desvíos, el mismo día sin tocar no emite ninguno) dan lo mismo con
los parámetros nuevos, sin modificar ninguna; también con los de la segunda votación.

**No está desplegado en AWS.** Requiere además que la controladora esté actualizada y tenga la
zona `puerta` dibujada.

### La línea base por el camino de producción, medida (2026-10-13)

Hasta ahora la detección de desvíos se había medido con los scripts de `baseline/`, no con el
código que corre en la nube. `tests/test_linea_base_produccion.py` recorre el camino real: el
historial entra por `POST /devices/events` con la clave del kit, en lotes de 200 como los manda
`runner/cloud_sync.py`, y lo evalúa `linea_base_diaria.ejecutar_linea_base`, la función que
llama el thread de las 03:00. Todo en local, con SQLite.

**Datos.** 60 días seguidos del hogar `hh113` de CASAS (2011-09-05 a 2011-11-03, 1.514
actividades; `tests/datos/casas_hh113_60dias.csv`, se regenera con
`tests/datos/generar_casas_hh113.py`). Solo las actividades que la cámara puede clasificar,
traducidas a su `activity_label` con el mapeo de `baseline/evaluate_classifier_noise.py`. Cada
actividad se envía como lo haría el runner: un evento al empezar y uno cada 60 segundos mientras
sigue (`ongoing`). Son 14.529 eventos en 73 lotes. Las horas de CASAS (hora local del hogar) se
cargan como hora de Argentina. Son etiquetas limpias: esta prueba mide que el camino funciona,
no cuánto acierta la cámara.

**Desvío inyectado.** El último día (2011-11-03) no se envía ninguna actividad de comida (las
7 `EatingDrinkingMedication` reales de ese día); el resto del día queda igual.

| Qué se verificó | Resultado |
|---|---|
| El proceso diario crea el `routine_deviation` del desvío inyectado | Sí: detector `clase`, `cam_comida:ausencia`, `valor_hoy` 0, mediana histórica 7 |
| Queda marcado como alerta | Sí (`es_alerta`), sin cámara asociada |
| Aparece en `GET /hogares/{id}/alertas` | Sí, sin resolver, con su `fecha` y `clave` |
| Correr el proceso otra vez sobre el mismo día | No duplica: las mismas alertas |
| Mismo día con sus comidas reales (control) | No aparece ningún desvío de comida |
| Día sin ningún evento | No se evalúa y no emite nada |

Ese día el proceso emitió **4 alertas**. Dos se deben al desvío inyectado: `cam_comida:ausencia`
y `forma_del_dia` del multivariado (z = 2,57, con `cam_comida_n` entre los tres factores que
más pesan). Las otras dos son del detector de horario (`cam_cocina:horario`, 177 minutos fuera
de lo habitual, y `cam_ocio_sentado:horario`, 160 minutos) **y salen igual en el control**, con
el día real sin tocar. Sobre un día normal de este hogar, entonces, el camino de producción
emite 2 alertas que nadie inyectó. Es un solo día de un solo hogar, no una tasa; coincide con lo
que `baseline/README.md` ya dice del detector de horario ("estado parcial", 12 a 23 % de falsos
positivos en varias clases).

**Tiempo.** Con 60 días de historial, el proceso diario tardó 42 segundos por hogar en la PC de
desarrollo (8 núcleos, sin otra carga). Casi todo es el detector de horario, que recalcula un
KMeans por cada actividad de todo el historial (1.467 para este hogar) aunque solo importen las
del día evaluado. El costo crece con el historial y con la cantidad de hogares; no se midió en
la instancia `t3.micro` de AWS.

**Segunda ronda, mismo día (2026-10-13).** Dos cosas sobre lo anterior:

- *Tiempo, corregido.* `baseline_runner.py` le pide ahora al detector de horario solo el día
  evaluado (`evaluate_only`; el historial se sigue usando como referencia). Mismo hogar, misma
  PC: de 42 segundos a menos de 2 por hogar (3,6 segundos para dos hogares). Las alertas son
  las mismas seis de antes, valor por valor, y una prueba nueva
  (`test_evaluar_solo_el_dia_pedido_da_lo_mismo_que_evaluar_todo_el_historial`) compara los dos
  caminos. Lo que más tarda ahora es el detector multivariado (1,4 segundos).
- *Las dos alertas de horario sobre el día real, sin corregir.* Se midió con validación cruzada
  por hogar si acotar las clases o calibrar por clase las evitaba sin perder detección: no (ver
  "Actualización 2026-10-13 (3)" en `baseline/README.md`). El detector quedó como estaba. La
  medición muestra que no es una rareza de este hogar: con las clases de cámara, dos de cada
  tres días normales reciben alguna alerta de horario (207 de 315 días de 21 hogares). Dejar de
  emitirlas es una decisión pendiente.

**Tercera ronda, mismo día (2026-10-13): las alertas de horario quedan apagadas por defecto.**
Decisión del usuario. `baseline_runner.py` ya no corre el detector de horario salvo que la
variable de entorno `ALERTAS_DE_HORARIO_ACTIVADAS` valga `1` (también `true`, `si`, `yes`); sin
la variable, o con cualquier otro valor, no emite ningún `routine_deviation` con
`detector: "timing"`. Los detectores multivariado, por clase y de tendencia siguen igual. El
código del detector no se borró.

| Misma prueba, mismo hogar | Antes | Ahora (por defecto) | Con la variable encendida |
|---|---|---|---|
| Día sin comidas | 4 alertas | 2: `cam_comida:ausencia` y `forma_del_dia` | 4 |
| El mismo día sin modificar (control) | 2 alertas, las dos de horario | ninguna | 2 |

- `GET /hogares/{id}/rutina` y `/rutina/serie` (pantalla Reportes) no cambian: el horario
  habitual de la primera comida lo calcula `app/rutina.py`, que no usa este detector.
- Los `routine_deviation` de horario que ya estén guardados en una base no se tocan: siguen
  apareciendo en `/eventos` y `/alertas` hasta que alguien los resuelva o los alcance la
  retención.
- **No está desplegado.** En AWS sigue corriendo el código anterior, que emite alertas de
  horario. Al desplegar, alcanza con no definir la variable.

No se probó: Postgres (la prueba usa SQLite), el envío del push (Firebase no está configurado
en las pruebas) ni el thread que dispara el proceso a las 03:00 (se llama a la función
directamente).

## Retención de eventos (2026-10-13)

El informe (capítulo 3, Viabilidad Legal) declara cuánto tiempo se conservan los eventos en la
nube: **12 meses** los de actividad y **24 meses** los de caída, inmovilidad y desvío de rutina.
`app/retencion.py` lo implementa como un paso más de la tarea diaria de las 03:00
(`app/linea_base_diaria.py`).

> **Antes de definir las variables en producción** (al 2026-10-14 no están definidas en
> `pfi-lares-prod`, así que no se borra nada): la antigüedad se mide con la fecha en que ocurrió
> el evento. Si alguna cuenta tiene cargado el historial de CASAS (`hh113`, fechas de 2011, ver
> "Zona horaria"), la primera tarea de las 03:00 lo borra entero. No se revisó si esa carga
> sigue en la base de producción: hay que mirarlo (o volver a cargarla con fechas recientes)
> antes de poner los plazos.

Los plazos no están escritos en el código. Salen de dos variables de entorno, en meses:

| Variable | Eventos | Valor del informe |
|---|---|---|
| `RETENCION_ACTIVIDAD_MESES` | `activity` | `12` |
| `RETENCION_ALERTAS_MESES` | `fall`, `immobility`, `routine_deviation` | `24` |

- **Sin la variable, ese grupo no se borra.** Es a propósito: desplegar este código no borra
  nada en producción. La retención empieza recién cuando alguien define las variables en el
  environment de Elastic Beanstalk (Configuration → Software). Cada variable activa solo su
  grupo.
- Un valor que no sea un entero mayor que 0 (`0`, `-12`, `doce`, `1.5`) tampoco borra nada y
  deja un error en el log. En particular, `0` no significa "borrar todo".
- **Corre después de calcular la línea base, nunca antes**, y con el mismo candado de Postgres
  tomado: el día recién evaluado se calcula con el historial completo y recién después se
  borra lo vencido. Si la línea base de un hogar falla, la retención corre igual; si falla la
  retención, queda en el log y se reintenta al día siguiente.
- La antigüedad se mide con `occurred_at` (cuándo ocurrió el evento según el reloj de la
  controladora), en meses de calendario. **Ojo:** un evento que llegue con el reloj de la
  controladora muy atrasado nace vencido y se borra en la primera pasada.
- `run_baseline_daily.py` (el CLI) no aplica la retención: solo la tarea diaria de la API.
- Con `LINEA_BASE_DIARIA_DESACTIVADA=1` la tarea diaria no corre y, por lo tanto, tampoco la
  retención.
- Borrar es definitivo. La primera pasada después de activar las variables borra de una vez
  todo lo acumulado más allá del plazo.

Efecto sobre la rutina: `GET /rutina`, `GET /rutina/serie` y la línea base usan los eventos de
actividad, así que con la retención activa trabajan con 12 meses de historial como máximo.

**Estado:** implementado y probado en local con SQLite. No está desplegado ni activado en AWS.

## Baja del hogar (2026-10-13)

`DELETE /hogares/{hogar_id}` (`app/routers/mobile.py`, grupo Hogar de Swagger) es la baja que
describe el informe (capítulo 3, Viabilidad Legal): el responsable del hogar puede pedir que se
borren los datos del residente.

- **Solo `responsable_principal`.** Un `familiar` común recibe 403 y no se borra nada. Si el
  hogar tiene más de un responsable, cualquiera de ellos puede darlo de baja.
- **Qué borra:** el residente, todos sus eventos (incluidos los desvíos de rutina), su línea
  base, los vínculos de todos los familiares con ese hogar y el hogar mismo.
- **Qué no borra:** las cuentas de los familiares (`Familiar` y su usuario de Firebase), que
  pueden estar vinculadas a otros hogares. `GET /me` les devuelve la lista de hogares sin éste.
- **El kit y sus cámaras quedan desvinculados:** `hogar_id` en NULL en los dos, `vinculado_por`
  del kit en NULL y la `ubicacion` de cada cámara vacía. El mismo código QR y la misma API key
  siguen sirviendo: `POST /kits/vincular` crea un hogar nuevo que no hereda nada del anterior.
  `vinculado_at` se conserva a propósito: un kit sin hogar pero con esa fecha es un kit dado
  de baja, y así se lo distingue de uno que nunca se vinculó sin tocar el esquema.
- **No se puede deshacer** y la API no pide confirmación: eso le toca a la app.
- Hasta que el kit se vuelva a vincular, la controladora recibe **410** en
  `POST /devices/events` y descarta esos eventos en vez de guardarlos (ver "Retención local de
  eventos" en `runner/README.md`). Si los acumulara, al vincular el kit de nuevo se subirían
  al hogar siguiente. Un kit que nunca se vinculó sigue recibiendo 409, y ahí sí se guardan.
- **Caso que queda sin cubrir:** si la controladora estaba sin conexión cuando se dio la baja
  y recién vuelve a conectarse después de que el kit se vinculó a otro hogar, nunca ve el 410
  y los eventos que tenía pendientes se suben al hogar nuevo.
- No cambia el esquema de la base: no hizo falta ninguna migración.

**Pendiente del lado de la app:** el endpoint todavía no está en `src/api/types.ts` ni en
`docs/vinculacion-kit.md` del repo Frontend, y ninguna pantalla lo llama.

**Estado:** implementado y probado en local con SQLite. No está desplegado en AWS.

## Pruebas

```
pip install pytest
pytest cloud/tests        # desde la raíz del repo
```

Corren siempre contra un SQLite temporal: `tests/conftest.py` pisa `DATABASE_URL` antes de
importar la aplicación y se niega a seguir si el engine no es ese SQLite, así que no pueden
tocar la base de producción aunque la terminal tenga la variable apuntando a RDS. Firebase se
reemplaza por un token de prueba. SQLite se abre con las claves foráneas activadas, para que un
borrado en el orden equivocado falle igual que fallaría en Postgres. Cubren (2026-10-13):

- `test_retencion_nube.py`: sin las variables de entorno no se borra nada (ni llamando al paso
  directamente ni desde la tarea diaria), cada grupo usa su propio plazo, un valor inválido no
  borra nada y la retención corre después de la línea base.
- `test_baja_hogar.py`: un familiar común no puede dar de baja el hogar, tampoco alguien de
  otro hogar; el responsable sí, y después no quedan residente, eventos ni vínculos; el kit y
  las cámaras quedan desvinculados y se pueden volver a vincular; no se tocan los datos de otro
  hogar; después de la baja la controladora recibe 410 (y 409 un kit que nunca se vinculó); el
  endpoint aparece en el grupo Hogar de Swagger.
- `test_linea_base_produccion.py` (18 pruebas, ver "La línea base por el camino de producción,
  medida"): 60 días de un hogar de CASAS cargados por `POST /devices/events`, un día sin
  comidas genera la alerta `routine_deviation`, se ve en `/alertas` y no se duplica al correr
  el proceso dos veces, y pedirle al detector de horario solo el día evaluado da lo mismo que
  recorrer todo el historial. Tardaba dos minutos; con el cambio del 2026-10-13, unos 20 segundos.
  Desde la tercera ronda del 2026-10-13 comprueba además que el día sin comidas emite
  exactamente dos desvíos y ninguno de horario, que el mismo día sin modificar no emite nada,
  que con `ALERTAS_DE_HORARIO_ACTIVADAS=1` vuelven las dos alertas de horario, y qué valores de
  la variable la encienden (12 casos).
- `test_push_desvio_de_rutina.py` (53 casos, ver "Texto del aviso al celular"): ningún aviso
  de los que el proceso diario puede emitir contiene `cam_`, dos puntos, guiones bajos,
  porcentajes ni números; los textos coinciden con los de la app; una clase o un tipo
  desconocidos salen como "su rutina"; y de punta a punta, el día sin comidas y los días sin
  salir (uno y cuatro) avisan con el texto legible.
- `test_linea_base_puerta.py` (9 pruebas, ver "Vocabulario y parámetros del proceso diario"):
  con una salida y una entrada por día en el historial, cuatro días sin salir generan las
  alertas `cam_salir_casa:ausencia_sostenida` y `cam_entrar_casa:ausencia_sostenida` y se ven
  en `/alertas`; un solo día sin salir genera `cam_salir_casa:ausencia` y
  `cam_entrar_casa:ausencia` (desde la segunda votación del 2026-10-13), y no genera nada si
  el parámetro se sube a 2 días; saliendo todos los días no hay alerta; una
  salida a las 05:00 en vez de las 15:00 no genera un desvío de horario ni con las alertas de
  horario encendidas (y una prueba de control muestra que el detector sí la marcaría si
  siguiera esa clase); un hogar que nunca mandó eventos de puerta no recibe alertas de puerta;
  y con `ENTRADAS_Y_SALIDAS_ACTIVADAS=0` el proceso ignora esos eventos.
- `test_ingestion_idempotente.py` (8 pruebas, ver "Ingestión idempotente"): reenviar el mismo
  lote no duplica, un lote mezclado guarda solo lo nuevo, un evento repetido dentro del lote
  se guarda una vez, una alerta reenviada no vuelve a avisar, los eventos sin identificador se
  guardan siempre, el mismo identificador en otro kit no cuenta como repetido, la base rechaza
  dos filas iguales, y la migración agrega la columna sin tocar los eventos que ya estaban.

## Zona horaria (días y horas del residente)

El contenedor corre en UTC, pero "hoy", la hora de la primera comida y dónde corta cada día son
los del residente. Todo lo que agrupa eventos por día u hora usa `app/horario.py`
(`ZONA_ARGENTINA`, UTC-3 fijo; `hoy_argentina()`), que se le pasa a `camera_events` como `tz`.
**Bug corregido el 2026-10-09** (deploy v6): antes se usaba la zona de la máquina, así que en AWS
una toma de las 22:00 contaba como "primera comida" a la 01:00 del día siguiente y los días
cortaban a las 21:00 de Argentina. Se vio al cargar 60 días de CASAS (`hh113`) en una cuenta de
prueba: "primera comida" habitual 01:02–01:35 → con el arreglo, 08:00–10:34. En la Pi (`tz`
sin pasar) el comportamiento no cambia: usa la hora local de la máquina, que ya es la del hogar.

## Migraciones (Alembic)

`alembic/env.py` lee `DATABASE_URL` del entorno (mismo que usa `app/db.py`) y los modelos de
`app/models.py` para autogenerar. Para un cambio de esquema nuevo:

```
DATABASE_URL=postgresql+psycopg2://... py -m alembic revision --autogenerate -m "descripción"
py -m alembic upgrade head
```

## Cómo seguir (para quien retome esto)

**2 (Firebase), 3 (push tokens), 4 (CRUD de familiares) y 1 (deploy a AWS) ya están hechos.**
Queda pendiente HTTPS (ver "✅ HECHO" del punto 1, al final) — decisión tomada: no hace falta
para esta etapa del PFI.

### 1. Desplegar a AWS (Elastic Beanstalk + RDS) — ✅ HECHO (2026-10-07)

Desplegado y probado de punta a punta con un ID token real contra el endpoint productivo
(`GET`/`POST /me` → 404 → 200, región `us-east-2`). Cuenta: Innovation Sandbox de la facultad
(créditos por alumno), IAM user `lellero@uade.edu.ar`. Recursos reales:
- RDS Postgres `pfi-lares-db`, `db.t3.micro`, sin acceso público (deshabilitado una vez corrida
  la migración inicial — EB le pega por el security group, no por internet).
- Elastic Beanstalk `pfi-lares-prod`, plataforma Docker, instancia única `t3.micro`.
- Credencial de Firebase pasada como `GOOGLE_APPLICATION_CREDENTIALS_JSON` (el JSON entero como
  variable de entorno, no un archivo montado — ver `app/auth.py::_get_firebase_app`; se agregó
  esta alternativa justo por este deploy, ver commit `fc7ed50`).
- **Costo real: $0** — un solo `t3.micro` de EB + un `db.t3.micro` de RDS 24/7 entran dentro del
  Free Tier normal de la cuenta (750 hs/mes cada uno), verificado con
  `aws ce get-cost-and-usage` (toda la semana da ~$0.00 en todos los servicios). Los créditos del
  Innovation Sandbox quedan de colchón si se agrega algo que sí tenga costo (ej. un load
  balancer con HTTPS, ver más abajo).
- **HTTPS: decisión tomada, no se hace por ahora.** Requiere un Application Load Balancer +
  dominio + certificado ACM — el ALB sí tiene costo por hora (no es Free Tier), y esto es una
  demo de PFI, no producción real. Queda documentado como pendiente si en algún momento hace
  falta (pasos originales más abajo, en el punto 5 del runbook de Elastic Beanstalk).

**Deploys posteriores al inicial** (mismo mecanismo: `aws s3 cp` del zip a
`s3://pfi-lares-deploy-115908660276/`, `create-application-version` + `update-environment`, ver
runbook abajo — no se volvió a tocar RDS ni Firebase, solo código de `cloud/app/`):
- **v3 (2026-10-08)**: endpoint `GET /hogares/{id}/rutina/serie` — mismo indicador que
  `GET /rutina` pero un punto por día, para el gráfico de evolución nuevo de la app (`rutina.py`,
  `routers/mobile.py`, `schemas.py`).
- **v4 (2026-10-08)**: indicadores `cocina`/`movilidad` sumados a `app/rutina.py` (antes solo
  `comida`/`sedentarismo`).
- **v8 (2026-10-14)**: `main` en `c8f3a52` (las v5–v7 no quedaron anotadas acá). Trae la línea
  base con entradas y salidas de casa y los parámetros votados (la puerta avisa desde el primer
  día sin eventos), el aviso al celular con el texto de la app, la ingestión idempotente, la
  retención y la baja del hogar. **Este sí tocó RDS**: la migración `c41e7d905b12` (columna
  `eventos.event_id` y restricción única con `camara_id`), corrida antes del deploy con RDS
  abierto temporalmente solo a una IP y cerrado después (sin acceso público, y el security group
  quedó otra vez con una sola regla, la de EB). Antes se tomó el snapshot manual
  `pfi-lares-antes-v8` (20 GB). Verificado: `/health` 200; `GET /me` con un ID token real de un
  usuario de prueba temporal → 404 "Tu cuenta todavía no tiene perfil" (el token se verificó;
  el usuario se borró de Firebase después) y con un token falso → 401; el OpenAPI publicado ya
  tiene `DELETE /hogares/{hogar_id}`. No se probó el envío real de un push. Volver atrás:
  `update-environment --version-label v7`, compatible con la base ya migrada.
- Mientras se armaba el deploy de v3 se encontró que `monitor-backend-env` (el environment viejo
  de una cuenta de prueba previa, ver problema 5 abajo) **seguía corriendo** pese a que este mismo
  README decía que se había borrado — se terminó (`terminate-environment`) el 2026-10-08.

**Problemas encontrados y cómo se resolvieron** (para quien lo repita):
1. **Zip con separadores de Windows**: `Compress-Archive` de PowerShell arma rutas con `\`
   dentro del zip, y el Linux de la instancia EB no las reconoce ("appears to use backslashes as
   path separators"). Armar el zip con el módulo `zipfile` de Python en vez de
   `Compress-Archive`, forzando `/` en los nombres de archivo.
2. **Contraseña de RDS corrompida por un BOM**: generarla con `Out-File -Encoding utf8` de
   PowerShell agrega un BOM invisible al principio del archivo, que se propagaba a mitad de la
   contraseña al armar `DATABASE_URL` por una cadena de `cat`/interpolación en Bash
   ("password authentication failed" con la contraseña aparentemente correcta). Se solucionó
   regenerando contraseña y `DATABASE_URL` con un script de Python usando `encoding='ascii'`, y
   verificando con `repr()` que no quedó ningún byte extra.
3. **RDS sin acceso desde EB** (504 Gateway Timeout): faltaba una regla de entrada en el security
   group de RDS que apunte al security group creado por EB — no alcanza con habilitar el puerto
   5432 desde "mi IP", eso es solo para la migración inicial.
4. **Filtración de la contraseña maestra de RDS en el bundle de deploy**: un `cp -r cloud/
   deploy_bundle/cloud/` incluyó por error archivos de scratch local (`.aws_*`) con la contraseña
   en texto plano, que llegaron a subirse a S3 como parte de una versión de la app de EB. Se
   detectó antes de que quedara expuesto a nadie más, se borró de las dos ubicaciones en S3 y de
   la versión de EB, y se rotó la contraseña dos veces. Desde entonces esos archivos de scratch
   están en `.gitignore` (`cloud/.aws_*`, `cloud/.backend_*.log`, ver commit `c1bf5d9`) y el
   runbook de deploy (paso 2 de Elastic Beanstalk, abajo) advierte sobre esto explícitamente.
5. **Deploy viejo ya corriendo**: la cuenta ya tenía un `monitor-backend-env`/`monitor-db`
   corriendo hace 5 días de una prueba anterior, que contradecía lo que decía este mismo README
   ("nada desplegado todavía"). Se borró antes de desplegar el actual. Moraleja: antes de asumir
   el estado de una cuenta de AWS a partir de este README, listar los recursos reales
   (`aws elasticbeanstalk describe-environments`, `aws rds describe-db-instances`).

Runbook original (para repetirlo desde cero en otra cuenta):

**RDS (Postgres)**:
1. Consola AWS → RDS → "Create database" → PostgreSQL, plantilla "Free tier" (o la que corresponda),
   `db.t3.micro`, storage mínimo (20GB).
2. Usuario/contraseña maestros → anotarlos, van al `DATABASE_URL`.
3. **No marcar "Publicly accessible"** — el backend le va a pegar desde dentro de la misma VPC
   (desde EB), no hace falta exponerlo a internet. Excepción: vas a necesitar acceso público
   *temporal* una sola vez, para correr `alembic upgrade head` la primera vez (ver paso 5) — se
   puede hacer también desde una instancia EC2/bastion en la misma VPC si no querés exponerlo ni
   un minuto; más simple para una sola vez: habilitarlo, correr la migración, deshabilitarlo.
4. Security group de RDS: permitir entrada en el puerto 5432 solo desde el security group de EB
   (se crea en el paso siguiente) — o desde tu IP, temporalmente, para el paso 5.
5. Correr la migración una vez, apuntando `DATABASE_URL` al endpoint de RDS:
   ```
   DATABASE_URL=postgresql+psycopg2://usuario:password@<endpoint-rds>:5432/<nombre_db> \
     py -m alembic upgrade head
   ```

**Elastic Beanstalk (plataforma Docker)**:
1. Consola AWS → Elastic Beanstalk → "Create application" → plataforma **Docker** (no Python — ver
   por qué abajo).
2. Armar el bundle de deploy desde la **raíz del repo** (no desde `cloud/`), porque
   `cloud/Dockerfile` ya está escrito asumiendo ese contexto (copia `baseline/src/` y `events/`
   explícitamente, ver el Dockerfile). EB Docker platform espera un `Dockerfile` en la raíz del
   bundle subido, así que:
   ```
   # desde D:\PFI_Desarrollo
   mkdir deploy_bundle
   cp cloud/Dockerfile deploy_bundle/Dockerfile
   cp -r cloud deploy_bundle/cloud
   cp -r baseline/src deploy_bundle/baseline/src   # (crear baseline/ primero si hace falta)
   cp -r events deploy_bundle/events
   cd deploy_bundle && zip -r ../deploy.zip . && cd ..
   ```
   (Esto es un punto de partida razonable, no se probó — si EB se queja de la estructura, revisar
   que las rutas del `COPY` en `Dockerfile` coincidan con lo que realmente queda en el zip.)
3. Al crear el *environment*, setear las variables de entorno (Configuration → Software):
   - `DATABASE_URL` = la de RDS.
   - `GOOGLE_APPLICATION_CREDENTIALS` = ruta DENTRO del contenedor a la clave de Firebase (ver
     ítem 2 más abajo — hay que resolver cómo llega ese archivo al contenedor, por ejemplo
     subiéndolo como secret de EB o incluyéndolo en el bundle si el repo/pipeline de deploy es
     privado; **no versionar esa clave en git**).
4. Health check path: `/health` (Configuration → Load balancer → health check path) — el endpoint
   ya existe en `main.py` y no requiere auth.
5. HTTPS: el DNS default de EB no trae TLS gratis solo por crear el environment — hace falta
   asociar un dominio propio + certificado de AWS Certificate Manager (ACM) al load balancer del
   environment (Configuration → Load balancer → Listeners → agregar HTTPS:443 con el cert de ACM).
   Sin esto, `runner/cloud_sync.py` seguiría funcionando contra HTTP plano, pero el informe exige
   HTTPS (RNF de la arquitectura de red) — no saltear este paso antes de ir a producción real.
6. Deploy: `eb init` (elegir la app creada) → `eb deploy`, o subir el `deploy.zip` manualmente
   desde la consola.

### 2. Crear el proyecto de Firebase — ✅ HECHO (2026-10-06)

Proyecto `pfi-ellero-ferrario`, Email/Password habilitado, app web `pfi-test` registrada (su
`firebaseConfig` es lo que usa la app en Expo Go, con el SDK web de Firebase — los
`google-services.json`/`GoogleService-Info.plist` recién hacen falta con un development build).
Probado de punta a punta con un ID token real: token inválido → 401, familiar no vinculado → 403,
vinculado → 200 en `/hogares/{id}/eventos`, y `GET`/`POST /me`. Usuario de prueba en Firebase:
`test@pfi-ellero-ferrario.test`. Pasos originales, por si hay que repetirlo:


1. https://console.firebase.google.com → "Add project" → nombre (ej. el mismo del repo).
2. **Authentication** → Sign-in method → habilitar "Email/Password" (es lo que describe el
   informe para el login de la app — Sección "Backend en la nube").
3. **Cloud Messaging** no requiere un paso de habilitación aparte — ya viene disponible al crear
   el proyecto; alcanza con la clave de servicio del paso 4 para poder mandar pushes desde
   `firebase-admin` (API HTTP v1, la que ya usa `app/notifications.py` vía `messaging.send_each_for_multicast`).
4. Project Settings (ícono de engranaje) → **Service accounts** → "Generate new private key" →
   descarga un JSON. Ese archivo es lo que tiene que apuntar `GOOGLE_APPLICATION_CREDENTIALS`
   (variable de entorno con la *ruta* al archivo, no su contenido) — ver `app/auth.py::_get_firebase_app`,
   que ya lee esa variable sola, no hace falta tocar código para esto.
   **No commitear ese JSON al repo** — va como secret en EB (o en `.env` local, ya ignorado por
   `.gitignore`).
5. Para la app móvil (repo aparte, React Native): seguir la guía de `react-native-firebase` para
   sumar `google-services.json` (Android) / `GoogleService-Info.plist` (iOS) del mismo proyecto.
6. Una vez con la clave descargada, probar localmente:
   ```
   GOOGLE_APPLICATION_CREDENTIALS=/ruta/a/la/clave.json uvicorn main:app --reload
   ```
   y pegarle a `GET /hogares/{id}/eventos` con un ID token real (lo saca la app móvil al loguearse,
   o se puede generar uno de prueba con el SDK cliente de Firebase para testear a mano) — debería
   dejar de dar `501` y pasar a `403`/200 según si el `firebase_uid` del token está vinculado a
   ese hogar.

### 3. Push tokens en `Familiar` — ✅ HECHO (2026-10-07)

1. ~~Columna `Familiar.push_token`~~ ✅ (migración `a3f9c2d17b40`, 2026-10-06).
2. ~~Migración~~ ✅ (misma).
3. ~~Nuevo endpoint~~ ✅ `POST /me/push-token` (`app/routers/me.py`), body `{token}`, **204**.
   La app móvil lo llama al loguearse (y cuando el SDK de Firebase avisa con `onTokenRefresh`,
   lo cual aún no está cableado del lado de la app — requiere *development build*, ver
   `README.md` del Frontend).
4. ~~`app/baseline_runner.py::_registrar_desvio`~~ ✅ usa los tokens reales de `hogar.familiares`
   en vez de una lista vacía.
5. ~~`fall`/`immobility` con `es_alerta=True` también disparan push~~ ✅ **decidido y hecho**:
   sí cubren caídas/inmovilidad además de desvíos de rutina (RF8), en `routers/devices.py` justo
   después de insertar los eventos del lote — texto del push en `alertas.py::texto_alerta`.

Probado con un smoke test (SQLite, `enviar_push_alerta` capturado en vez de pegarle a Firebase
real): `fall` confirmado dispara push con el token correcto, `activity` e `immobility` de
severidad baja no disparan nada, el token se pisa correctamente al loguearse en otro equipo. Sin
probar con Firebase Cloud Messaging real todavía (necesita *development build* del lado de la
app para recibir la notificación — Expo Go no la soporta, ver "Próximos pasos" del README del
Frontend).

### 4. CRUD de familiares/roles (RF12) — ✅ HECHO (2026-10-07)

Alta del hogar por vinculación del kit, cámaras y residente (2026-10-06, ver "Endpoints de la
app móvil"). CRUD de familiares completo (2026-10-07, nuevo `app/routers/familiares.py`):

- `GET /hogares/{id}/familiares` — listar vinculados + rol (cualquier familiar del hogar).
- `POST /hogares/{id}/familiares` — invitar por email. Body `{email, rol}`.
  `firebase_admin.auth.get_user_by_email(email)` resuelve el `firebase_uid`; **404** si esa
  persona nunca se registró en la app — decidido: sin invitación pendiente/por mail, consistente
  con "no sobre-ingenierizar antes de medir si hace falta". Crea `Familiar` si no existía (se
  registró en Firebase pero nunca llamó `POST /me`) + `HogarFamiliar`. Solo
  `responsable_principal`. Idempotente: invitar a alguien ya vinculado devuelve su vínculo tal
  cual, sin pisarle el rol.
- `PATCH /hogares/{id}/familiares/{familiar_id}` — cambiar `rol`. Solo `responsable_principal`.
  **409** si es el único `responsable_principal` del hogar y se le intenta bajar el rol.
- `DELETE /hogares/{id}/familiares/{familiar_id}` — desvincular. Uno mismo siempre puede;
  a otro, solo `responsable_principal`. **409** si es el único `responsable_principal` del hogar.

**Modelo de roles** (constantes `ROLES` en `app/models.py`, default `familiar`):
`responsable_principal` (invita/quita familiares y cambia roles, además de lo que ya podía:
configurar residente/cámaras) | `familiar` (solo lectura de eventos/alertas/reportes).

Reusa `permisos.hogar_autorizado(..., solo_responsable=True)`, no duplica el chequeo de rol.
Probado con un smoke test (SQLite, `get_user_by_email`/`_get_firebase_app` mockeados): invitar a
quien no tiene cuenta da 404, invitar dos veces es idempotente, un `familiar` no puede invitar
(403), y las dos reglas de "no te quedes sin responsable" (`PATCH` y `DELETE`) dan 409 en el
caso límite (único `responsable_principal` del hogar).

## Setup

```
cd cloud
py -m venv .venv && .venv\Scripts\activate
pip install -r requirements.txt
```
