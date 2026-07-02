# `evaluation/` — Validación cualitativa

## Qué hace

Como el dataset no tiene ground truth todavía (ver `../videos/README.md`),
la validación de `vision/` y `vlm/` es visual: dado un video de `../videos/`
y los eventos detectados (`../events/store.py`), generar un video anotado
(esqueleto de pose superpuesto + eventos marcados como texto/overlay en el
frame correspondiente) para inspección manual.

## Entrada

Video original (`../videos/<nombre>.mp4`) + eventos de ese `video_id`
(`EventStore.all(video_id=...)`).

## Salida

Video anotado (`.mp4`) guardado localmente (gitignored) para revisión
manual. No hay métricas cuantitativas (precision/recall) hasta que se
etiquete el dataset — eso es trabajo futuro, ver `../videos/README.md`.

## Setup

```
cd evaluation
py -m venv .venv && .venv\Scripts\activate
pip install -r requirements.txt
```

## Estado

No iniciado — depende de que `vision/` tenga una versión mínima que exponga
las detecciones por frame (no hace falta el pipeline completo ni `vlm/` ni
`baseline/` para arrancar esto).
