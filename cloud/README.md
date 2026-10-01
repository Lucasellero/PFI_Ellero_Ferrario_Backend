# `cloud/` — Backend cloud (API + línea base en la nube)

Implementa la arquitectura cloud que ya define el informe del PFI (capítulos 3 y 4):
FastAPI + Postgres (RDS en producción), recibe eventos sincronizados desde la Pi
(`runner/cloud_sync.py`), corre `baseline/` contra ese historial y expone una API mínima para la
futura app móvil.

**Estado: código probado localmente (SQLite, sin Docker), nada desplegado a AWS/Firebase
todavía** — ver "Cómo seguir" al final, con el runbook concreto de los 4 pendientes (deploy a AWS,
proyecto de Firebase, push tokens, CRUD de familiares/roles) para retomarlo sin empezar de cero.

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

Fiel al diagrama del informe (`fig:diagrama_bd_cloud`): `Hogar` 1:1 `Residente` (RNF11: un único
residente por instalación), `Hogar` 1:N `Camara`, `Evento` (FK a `Camara` —nullable, ver abajo—
y a `Residente`, desnormalizado), `Familiar` N:M `Hogar` vía `HogarFamiliar` (con `rol`). Ver
`app/models.py`.

**Una diferencia con el diagrama original**: `Evento.camara_id` es nullable. Los eventos
`routine_deviation` los genera `baseline_runner.py` a partir del historial del residente, no una
cámara puntual — no tenía sentido forzar una FK a una cámara arbitraria.

## Esquema de transporte: reusa `events/schema.py`

El body de `POST /devices/events` es una lista de `MonitoringEvent` (mismo `EventType` que ya usa
todo el pipeline local) — el informe pide explícitamente compartir ese esquema Pydantic entre el
backend local y el de la nube. `occurred_at` (columna de `Evento`) sale directo de
`MonitoringEvent.wall_clock_ts` (agregado a `events/schema.py` en la sesión anterior).

## Autenticación

Dos esquemas separados (`app/auth.py`):

- **Pi → nube**: header `X-API-Key`, una key por hogar (`Hogar.device_api_key_hash` — se guarda
  el hash, la key cruda se devuelve una sola vez al crear el hogar).
- **App móvil → nube**: `Authorization: Bearer <firebase_id_token>`, verificado con
  `firebase_admin.auth.verify_id_token`. **Si el proceso no tiene `GOOGLE_APPLICATION_CREDENTIALS`
  configurado, estos endpoints devuelven `501` con un mensaje claro** en vez de romper el arranque
  de la app — así se puede levantar y probar la ingestión + `baseline_runner` sin tener un
  proyecto de Firebase todavía (que es el estado actual).

## Correr localmente (sin Docker, sin Postgres)

```
cd cloud
py -m venv .venv && .venv\Scripts\activate
pip install -r requirements.txt

# crear el esquema (SQLite por default, ver DATABASE_URL en app/db.py)
alembic upgrade head

uvicorn main:app --reload
```

Sembrar un hogar de prueba (no hay endpoint público de alta de hogar todavía — es un paso de
administración manual, ver "Qué falta" abajo):

```python
from app.db import SessionLocal, Base, engine
from app.models import Hogar, Residente
Base.metadata.create_all(engine)
db = SessionLocal()
raw_key, hashed = Hogar.generar_api_key()
hogar = Hogar(nombre="...", device_api_key_hash=hashed)
db.add(hogar); db.commit(); db.refresh(hogar)
db.add(Residente(hogar_id=hogar.id, nombre="...")); db.commit()
print(raw_key)  # guardar esta key, no se puede recuperar después
```

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

## Cómo seguir (para quien retome esto — nada de lo de abajo se pudo probar en esta sesión,
## no hay credenciales de AWS ni de Firebase)

Orden recomendado: 2 (Firebase) desbloquea probar 3 y 4 de verdad; 1 (AWS) es independiente y se
puede hacer en paralelo. Dentro de 3/4, hace falta **primero un endpoint de alta de hogar** (no
existe todavía — se sembró a mano con un script en esta sesión, ver "Correr localmente" arriba)
porque todo lo demás de RF12 depende de poder crear un `Hogar` sin tocar la base a mano.

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

### 2. Crear el proyecto de Firebase

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

1. `app/models.py` → sumar a `Familiar`:
   ```python
   push_token: Mapped[str | None] = mapped_column(String(300), nullable=True)
   ```
2. Migración: `py -m alembic revision --autogenerate -m "agregar push_token a familiares"` →
   revisar el archivo generado (debería ser un `ADD COLUMN` nullable, sin pérdida de datos) →
   `py -m alembic upgrade head`.
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

No implementado — es la pieza que falta para que 2 y 3 tengan sentido en una instalación real (sin
esto, un `Hogar`/`Familiar` solo se puede crear a mano, como se hizo en esta sesión). Diseño
concreto para no arrancar de cero:

**Primero, el prerequisito que falta y no es parte de RF12 en sí**: un endpoint para crear el
`Hogar` + su `Residente` (hoy es el script manual de "Correr localmente" arriba). Candidato:
`POST /hogares` (sin auth de hogar porque todavía no existe uno — pensar qué lo protege; podría
ser el mismo `Authorization: Bearer <firebase_id_token>` del familiar que lo está dando de alta,
que de paso queda vinculado como primer familiar con rol `"responsable_principal"`).

**Modelo de roles** (no versionado en código todavía, es una decisión a tomar): al menos dos
niveles —
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

**Reusar, no duplicar**: `_hogar_autorizado` ya existe en `routers/mobile.py` — al sumar
`familiares.py`, factorizarla a un lugar común (ej. `app/auth.py` o un `app/permisos.py` nuevo)
en vez de copiarla, ya que las dos rutas la necesitan con la variante "cualquier rol" vs. "solo
responsable_principal".

## Setup

```
cd cloud
py -m venv .venv && .venv\Scripts\activate
pip install -r requirements.txt
```
