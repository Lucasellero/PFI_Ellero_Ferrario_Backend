# No probado en esta sesión (no hay Docker instalado en la máquina de desarrollo) — revisar al
# desplegar a Elastic Beanstalk (ver cloud/README.md, "Desplegar a AWS").
#
# IMPORTANTE: el contexto de build es la RAÍZ del repo, no cloud/, porque baseline_runner.py
# importa baseline/src/ y events/ directamente (ver su docstring) y necesita ese código
# disponible en la imagen:
#   docker build -f cloud/Dockerfile -t monitor-cloud .        (desde D:\PFI_Desarrollo)
FROM python:3.12-slim

WORKDIR /app

COPY cloud/requirements.txt cloud/requirements.txt
RUN pip install --no-cache-dir -r cloud/requirements.txt

COPY cloud/ cloud/
COPY baseline/src/ baseline/src/
COPY events/ events/

WORKDIR /app/cloud
ENV PYTHONUNBUFFERED=1

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
