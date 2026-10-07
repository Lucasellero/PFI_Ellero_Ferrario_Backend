# `cloud/` — Backend cloud (API + línea base en la nube)

Implementa la arquitectura cloud que ya define el informe del PFI (capítulos 3 y 4):
FastAPI + Postgres (RDS en producción), recibe eventos sincronizados desde la Pi
(`runner/cloud_sync.py`), corre `baseline/` contra ese historial y expone una API mínima para la
futura app móvil.

**Estado: código probado contra Postgres real (Docker local) y SQLite, nada desplegado a AWS
todavía.** Proyecto de Firebase creado y probado con tokens reales (2026-10-06, ver "Cómo
seguir" → punto 2). KIT/vinculación (2026-10-07) y los endpoints chicos de eventos (resolver,
`ubicacion`, filtros RF13; mismo día) probados de punta a punta contra Postgres con un ID token
real. Ver "Cómo seguir" al final, con el runbook de lo que falta (deploy a AWS, push tokens,
CRUD de familiares/roles) para retomarlo sin empezar de cero.

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
  servicio del proyecto `pfi-ellero-ferrario`, **fuera del repo**).

## Endpoints de la app móvil

| Endpoint | Qué hace |
|---|---|
| `GET /me` | Perfil del familiar logueado + sus hogares (`rol`, `residente`). **404** si la cuenta de Firebase todavía no tiene `Familiar` → la app llama a `POST /me`. |
| `POST /me` | Body `{nombre}` (2-60). Crea el `Familiar`; el email sale del token, no del body. Idempotente: si ya existe lo devuelve. |
| `POST /kits/vincular` | Body `{codigo}` (del QR; acepta minúsculas, espacios, guiones). Crea el `Hogar` ("Mi hogar"), le asigna kit y cámaras y deja al familiar como `responsable_principal`. Devuelve `{hogar_id, numero_serie, cantidad_camaras, camaras}`. **404** código inexistente · **409** kit de otro hogar · **429** 5 intentos fallidos en 15 min (por familiar, en memoria del proceso). Reintento del mismo familiar = idempotente. |
| `GET /hogares/{id}/camaras` | Cámaras del hogar. Cualquier familiar vinculado. |
| `PATCH /hogares/{id}/camaras/{camara_id}` | Body `{ubicacion}` (2-40). RF10. **409** si repite la de otra cámara del hogar. Solo `responsable_principal`. |
| `PUT /hogares/{id}/residente` | Body `{nombre, telefono}` (E.164 o null). Crea o actualiza; si el hogar se llamaba "Mi hogar" pasa a "Casa de {nombre}". **204**. Solo `responsable_principal`. Es el último paso: con residente, la app pasa a las pestañas y la Pi ya puede sincronizar eventos. |
| `GET /hogares/{id}/eventos` · `/alertas` | Eventos del hogar. **403** si el familiar no está vinculado. `/eventos` acepta filtros opcionales `tipo`, `desde`, `hasta` (RF13, `occurred_at` ISO 8601). Cada evento trae `ubicacion` (de `Camara.ubicacion`, `null` si no vino de una cámara o no se cargó) y `resuelta` (`Evento.resuelta_at is not None`). |
| `POST /hogares/{id}/eventos/{evento_id}/resolver` | "Marcar como resuelta" (mockup 03, RF8). Cualquier familiar vinculado (no solo `responsable_principal`). **204**, idempotente (no pisa quién la marcó primero). **404** si el evento no es de ese hogar. |

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

Orden recomendado: 2 (Firebase) ya está hecho; 1 (AWS) es independiente. Dentro de 3/4, hace
falta **primero `POST /kits/vincular`** (crea el `Hogar` al escanear el QR, ver punto 4) porque todo
lo demás de RF12 depende de poder crear un `Hogar` sin tocar la base a mano.

### 1. Desplegar a AWS (Elastic Beanstalk + RDS)

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

### 3. Push tokens en `Familiar`

Hoy `notifications.enviar_push_alerta` siempre recibe una lista vacía (ver `_registrar_desvio` en
`app/baseline_runner.py`) — falta dónde guardar el token de FCM de cada familiar y cómo armar esa
lista. Pasos concretos:

1. ~~Columna `Familiar.push_token`~~ ✅ (migración `a3f9c2d17b40`, 2026-10-06).
2. ~~Migración~~ ✅ (misma).
3. Nuevo endpoint (en `app/routers/mobile.py` o un `familiares.py` nuevo, ver ítem 4):
   ```python
   @router.post("/me/push-token")
   def registrar_push_token(token: str, uid: str = Depends(require_familiar_uid), db: Session = Depends(get_db)):
       familiar = db.query(Familiar).filter(Familiar.firebase_uid == uid).first()
       if familiar is None:
           raise HTTPException(404, "Familiar no registrado — ver ítem 4 (alta de familiares)")
       familiar.push_token = token
       db.commit()
   ```
   La app móvil lo llama una vez al loguearse (o cuando el token de FCM rota, que pasa de vez en
   cuando — el SDK de Firebase en el cliente avisa con un callback `onTokenRefresh`).
4. En `app/baseline_runner.py::_registrar_desvio`, reemplazar la lista vacía:
   ```python
   tokens = [
       f.familiar.push_token
       for f in hogar.familiares  # HogarFamiliar
       if f.familiar.push_token
   ]
   enviar_push_alerta(tokens, titulo="Posible desvío de rutina", cuerpo=clave, data=detalle)
   ```
   Mismo cambio aplicaría en `routers/devices.py` si en algún momento se decide notificar también
   por `fall`/`immobility` con `es_alerta=True` en el momento de la ingestión (hoy esos eventos se
   guardan pero no disparan push — evaluar si hace falta, el informe no es explícito sobre si el
   push es exclusivo de desvíos de rutina o también de eventos agudos; RF8 sugiere que sí debería
   cubrir caídas/inmovilidad también, revisar con el resto del equipo antes de implementarlo).

### 4. CRUD de familiares/roles (RF12)

Hecho (2026-10-06): alta del hogar por vinculación del kit, cámaras y residente (ver "Endpoints de
la app móvil"), con permisos por rol en `app/permisos.py`. **Falta el CRUD de familiares**
(invitar/quitar/cambiar rol, abajo).

**~~Prerequisito~~ ✅: crear el `Hogar`.** Decidido
(2026-10-05, `docs/vinculacion-kit.md` del Frontend): no hay `POST /hogares`, el hogar nace al
vincular el kit — `POST /kits/vincular {codigo}` con el token de Firebase del familiar: busca el
`Kit` por `hash_codigo(codigo)` (404 si no existe, 409 si ya tiene hogar, 429 ante muchos
intentos), crea el `Hogar`, le asigna kit y cámaras, y vincula al familiar como
`responsable_principal`. Después `PUT /hogares/{id}/residente` y `PATCH
/hogares/{id}/camaras/{id}` completan la configuración.

**Modelo de roles** (decidido; constantes `ROLES` en `app/models.py`, default `familiar`):
- `responsable_principal`: puede invitar/quitar otros familiares y cambiar roles.
- `familiar`: solo lectura de eventos/alertas/reportes (lo que ya hace `routers/mobile.py`).

Guardar el rol como ya está en `HogarFamiliar.rol` (columna `String`, ya existe en el modelo) — no
hace falta un enum de DB, alcanza con validar los valores permitidos en el endpoint.

**Endpoints a sumar** (nuevo `app/routers/familiares.py`):
- `GET /hogares/{hogar_id}/familiares` — listar vinculados + rol (cualquier familiar del hogar).
- `POST /hogares/{hogar_id}/familiares` — invitar por email. Body `{email, rol}`. Implementación:
  `firebase_admin.auth.get_user_by_email(email)` para resolver el `firebase_uid` (falla si esa
  persona nunca se registró en la app — decidir si eso es un error 404 "primero tiene que crearse
  una cuenta" o si se arma un flujo de invitación pendiente más elaborado; lo primero es
  consistente con "no sobre-ingenierizar antes de medir si hace falta" y alcanza para un primer
  release). Crea `Familiar` si no existía (nombre/email del token decodificado) + `HogarFamiliar`.
  Solo lo puede llamar un `responsable_principal` del hogar (reusar/extender `_hogar_autorizado`
  de `routers/mobile.py` para exigir ese rol, no cualquier vínculo).
- `PATCH /hogares/{hogar_id}/familiares/{familiar_id}` — cambiar `rol`. Mismo chequeo de permiso.
- `DELETE /hogares/{hogar_id}/familiares/{familiar_id}` — desvincular. Un familiar puede
  desvincularse a sí mismo siempre; desvincular a otro requiere `responsable_principal`. Validar
  que no se pueda sacar al último `responsable_principal` de un hogar (si no, nadie puede
  administrar ese hogar nunca más).

**Reusar, no duplicar**: `permisos.hogar_autorizado(..., solo_responsable=True)` ya cubre las dos
variantes ("cualquier rol" vs. "solo responsable_principal").

## Setup

```
cd cloud
py -m venv .venv && .venv\Scripts\activate
pip install -r requirements.txt
```
