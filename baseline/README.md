# `baseline/` — Línea base de comportamiento y detección de desvíos

## Qué hace

Consume eventos ya almacenados (`../events/store.py`) y construye una línea
base de comportamiento mediante clustering dinámico (ver Sec. 2.1.7 y 2.2.5
del informe, enfoque de Snoun et al. 2024 — clustering con ventana móvil y
"behavioral drift"), para luego detectar desvíos significativos respecto al
patrón habitual.

## Entrada

`events.db` vía `EventStore.all()`. **No depende del código interno de
`vision/` ni `vlm/`**, solo del esquema de eventos.

## Cómo desarrollarlo en paralelo sin esperar a vision/vlm

Generar eventos sintéticos que respeten `MonitoringEvent` (por ejemplo,
actividades simuladas distribuidas a lo largo de varios "días" de video
ficticios, con algún día con un patrón anómalo a propósito) para probar el
clustering antes de que el pipeline de visión esté terminado. Insertarlos con
`EventStore.add()` igual que lo haría `vision/`/`vlm/` en producción.

## Salida

Por definir con más detalle al empezar esta etapa. Como mínimo: un score de
desvío/anomalía por ventana temporal, comparando la actividad observada
contra los clusters que representan el comportamiento habitual del "video_id"
(en esta fase local, cada `video_id` hace de proxy de un residente/día).

## Setup

```
cd baseline
py -m venv .venv && .venv\Scripts\activate
pip install -r requirements.txt
```

## Estado

No iniciado — es la etapa siguiente a tener `vision/`/`vlm/` generando
eventos reales, pero se puede arrancar en paralelo con datos sintéticos (ver
arriba).
