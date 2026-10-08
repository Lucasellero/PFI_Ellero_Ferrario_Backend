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
                                    run_baseline_daily.py (cron/EventBridge, 1x/día)
                                       reusa baseline/src/ (ver baseline_runner.py)
                                                          │
                                                          ▼
                                    Evento(event_type=routine_deviation) + push (FCM)

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

## Autenticación

Dos esquemas separados (`app/auth.py`):

- **Pi → nube**: header `X-API-Key`, una key por kit (`Kit.device_api_key_hash` — se guarda el
  hash, la key cruda la muestra una sola vez `alta_kit.py`). Cada request autenticado actualiza
  `Kit.ultima_conexion_at`. Mientras el kit no esté vinculado a un hogar con residente,
  `POST /devices/events` responde **409** y la Pi reintenta en el próximo ciclo (no se pierde nada).
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

| Endpoint | Qué hace |
|---|---|
| `GET /me` | Perfil del familiar logueado + sus hogares (`rol`, `residente`). **404** si la cuenta de Firebase todavía no tiene `Familiar` → la app llama a `POST /me`. |
| `POST /me` | Body `{nombre}` (2-60). Crea el `Familiar`; el email sale del token, no del body. Idempotente: si ya existe lo devuelve. |
| `POST /me/push-token` | Body `{token}` (token de FCM del celular, RF8). **204**. Pisa el anterior (un token por familiar). **404** si la cuenta todavía no tiene `Familiar` (llamar `POST /me` primero). |
| `POST /kits/vincular` | Body `{codigo}` (del QR; acepta minúsculas, espacios, guiones). Crea el `Hogar` ("Mi hogar"), le asigna kit y cámaras y deja al familiar como `responsable_principal`. Devuelve `{hogar_id, numero_serie, cantidad_camaras, camaras}`. **404** código inexistente · **409** kit de otro hogar · **429** 5 intentos fallidos en 15 min (por familiar, en memoria del proceso). Reintento del mismo familiar = idempotente. |
| `GET /hogares/{id}/camaras` | Cámaras del hogar. Cualquier familiar vinculado. |
| `PATCH /hogares/{id}/camaras/{camara_id}` | Body `{ubicacion}` (2-40). RF10. **409** si repite la de otra cámara del hogar. Solo `responsable_principal`. |
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

Pensado para cron/AWS EventBridge (1 vez al día) — mismo criterio que `baseline/run_live.py`:
nunca evalúa el día en curso, es idempotente (no duplica un desvío ya emitido).

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
