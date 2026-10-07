"""Backend cloud — FastAPI. Ver `cloud/README.md` para setup local y qué falta para desplegar a
Elastic Beanstalk/RDS/Firebase reales.

    uvicorn main:app --reload
"""
from __future__ import annotations

from fastapi import FastAPI

from app.routers import devices, familiares, kits, me, mobile

app = FastAPI(title="Monitor Cloud API")
app.include_router(devices.router)
app.include_router(me.router)
app.include_router(kits.router)
app.include_router(mobile.router)
app.include_router(familiares.router)


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}
