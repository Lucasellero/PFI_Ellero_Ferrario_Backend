"""Punto de entrada: python -m dashboard"""
import uvicorn

if __name__ == "__main__":
    # timeout_graceful_shutdown: sin esto, Ctrl+C deja a uvicorn esperando para
    # siempre a que cierren las conexiones abiertas, y el stream MJPEG de
    # /api/frame-stream (generador infinito) no cierra nunca mientras haya una
    # pestaña con la vista única abierta.
    uvicorn.run(
        "dashboard.api:app", host="0.0.0.0", port=8000, reload=False,
        timeout_graceful_shutdown=2,
    )
