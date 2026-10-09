"""Backend cloud — FastAPI. Ver `cloud/README.md` para setup local y qué falta para desplegar a
Elastic Beanstalk/RDS/Firebase reales.

    uvicorn main:app --reload

La documentación interactiva (Swagger) queda en `/docs`. Los grupos, descripciones y candados de
esa página son solo metadatos de OpenAPI: no cambian cómo se autentica cada endpoint, que sigue
definido en `app/auth.py`.

Al arrancar, la API lanza además la línea base diaria (`app/linea_base_diaria.py`), salvo que
`LINEA_BASE_DIARIA_DESACTIVADA=1`.
"""
from __future__ import annotations

import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.openapi.utils import get_openapi

from app.linea_base_diaria import LineaBaseDiaria, esta_desactivada
from app.routers import devices, familiares, kits, me, mobile

# uvicorn solo configura sus propios loggers; sin esto los logs INFO de `app.*` no se ven.
logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
logger = logging.getLogger(__name__)

TAGS = [
    {"name": "Controladora", "description": "Lo que envía la Raspberry Pi del kit. Se autentica con la clave del kit."},
    {"name": "Cuenta", "description": "Perfil del familiar que usa la aplicación."},
    {"name": "Vinculación", "description": "Vincular un kit nuevo a la cuenta del familiar."},
    {"name": "Hogar", "description": "Configuración del hogar, sus cámaras y el residente."},
    {"name": "Eventos", "description": "Eventos y alertas detectados en el hogar."},
    {"name": "Rutina", "description": "Rutina del residente frente a su propio historial."},
    {"name": "Familiares", "description": "Familiares vinculados al hogar y sus roles."},
    {"name": "Estado", "description": "Estado del servicio."},
]


@asynccontextmanager
async def lifespan(_: FastAPI) -> AsyncIterator[None]:
    tarea = None
    if esta_desactivada():
        logger.info("linea_base_diaria: desactivada por variable de entorno")
    else:
        tarea = LineaBaseDiaria()
        tarea.start()
    yield
    if tarea is not None:
        tarea.detener()


app = FastAPI(
    lifespan=lifespan,
    title="Lares API",
    description="Backend en la nube del sistema Lares. Recibe los eventos de la controladora y los expone a la aplicación móvil.",
    version="1.0",
    openapi_tags=TAGS,
)
app.include_router(devices.router)
app.include_router(me.router)
app.include_router(kits.router)
app.include_router(mobile.router)
app.include_router(familiares.router)


@app.get("/health", tags=["Estado"], summary="Verificar que el servicio está en línea")
def health() -> dict:
    return {"status": "ok"}


def _openapi() -> dict:
    """Agrega los esquemas de seguridad para que Swagger muestre el candado en cada endpoint.
    La controladora usa `X-API-Key`; la aplicación, el token de Firebase como Bearer."""
    if app.openapi_schema:
        return app.openapi_schema
    schema = get_openapi(
        title=app.title, version=app.version, description=app.description,
        routes=app.routes, tags=app.openapi_tags,
    )
    schema.setdefault("components", {})["securitySchemes"] = {
        "ClaveDelKit": {"type": "apiKey", "in": "header", "name": "X-API-Key"},
        "TokenFirebase": {"type": "http", "scheme": "bearer"},
    }
    for operaciones in schema["paths"].values():
        for operacion in operaciones.values():
            tags = operacion.get("tags", [])
            if "Estado" in tags:
                continue
            esquema = "ClaveDelKit" if "Controladora" in tags else "TokenFirebase"
            operacion["security"] = [{esquema: []}]
    app.openapi_schema = schema
    return schema


app.openapi = _openapi
