# Sistema de monitoreo no invasivo para adultos mayores — PFI Ellero/Ferrario

Repositorio de desarrollo del Proyecto Final de Ingeniería (UADE, 2026). Para
el marco teórico, el estado del arte y la justificación de cada decisión de
diseño, ver el informe del proyecto.

## Fase actual: validación local sobre dataset de video

Esta etapa **no** integra Raspberry Pi, cámaras IP ni servicios en la nube.
Todo el pipeline corre en la PC de desarrollo, procesando un dataset de
videos ya grabados (carpeta `videos/`, ver más abajo). El objetivo es validar
la lógica de detección (pose estimation, caídas, inmovilidad, clasificación
de actividades) de forma reproducible antes de integrar hardware real.

Los módulos están pensados para que la arquitectura final del informe (edge
en Raspberry Pi + cloud) sea una extensión de este código, no una reescritura:
el mismo pipeline que hoy lee un archivo de video de `videos/` va a leer, más
adelante, un stream RTSP en vivo.

## Arquitectura (fase local)

```
videos/ (dataset) ──┐
cameras.yaml (streams en vivo) ──► runner/ (orquesta N cámaras en paralelo)
    │                                  │
    │                     por cámara: ├──► vision/  (pose estimation, caídas e inmovilidad — YOLOv11n-pose)
    │                                  └──► vlm/     (clasificación semántica de actividad — CLIP, consulta periódica)
    │                                            │
    │                                            ▼
    │                                        events/  (esquema de evento común + SQLite)
    │                                            │
    │                                            ▼
    │                                        baseline/ (clustering dinámico → línea base → detección de desvíos)
    │
    ├──► vision/main.py / vlm/main.py (modo batch, sin runner/, para procesar el dataset de una)
    │
    └──► evaluation/ (validación cualitativa: video anotado, inspección manual — no hay ground truth todavía)

dashboard/ (panel web: arranca/para runner/, gestiona cameras.yaml, muestra eventos y preview en vivo)
```

`vision/` y `vlm/` son independientes entre sí (ambos leen frames y escriben
a `events/`). `baseline/` depende únicamente de que existan eventos en el
store, no del código interno de `vision/` o `vlm/`. Esto es intencional: son
las líneas de trabajo pensadas para desarrollarse en paralelo (ver más
abajo).

`runner/` es la forma de correr `vision/` + `vlm/` juntos sobre varias
fuentes en simultáneo (archivos en loop o streams RTSP/HTTP/webcam), pensada
para la fase con cámaras reales. `vision/main.py` y `vlm/main.py` siguen
existiendo como scripts batch independientes para procesar el dataset de
`videos/` de una sola vez, sin pasar por `runner/`. `dashboard/` es la capa
de operación sobre `runner/`: no duplica lógica de detección, solo la
orquesta (arranca el subproceso) y visualiza lo que hay en `events/`.

## Dataset de video

Los archivos van en `videos/` (no versionados en git, ver `.gitignore`):
cualquier `.mp4`/`.avi`/`.mov` copiado ahí es candidato a ser procesado. El
nombre de archivo (sin extensión) se usa como `video_id` en el esquema de
eventos, y en los videos del dataset Toyota Smarthome también codifica la
actividad real como prefijo (`Walk_...`, `Laydown_...`) — ese ground truth
es el que usan `vision/` para calibrar caídas y `vlm/evaluate_activity.py`
para medir la clasificación de actividades. La validación cualitativa por
video anotado (ver `evaluation/`) sigue pendiente.

## Esquema de eventos (contrato entre módulos)

Ver [`events/README.md`](events/README.md) y [`events/schema.py`](events/schema.py).
Este es el contrato que separa `vision/`/`vlm/` (productores) de `baseline/`
(consumidor). **Cualquier cambio a este esquema debe coordinarse**, porque
rompe a todos los módulos que dependen de él.

## Módulos

| Módulo | Qué hace | Spec |
|---|---|---|
| `vision/` | Pose estimation + detección de caídas/inmovilidad prolongada | [vision/README.md](vision/README.md) |
| `vlm/` | Clasificación semántica de actividades vía VLM local | [vlm/README.md](vlm/README.md) |
| `events/` | Esquema común + almacenamiento local (SQLite) | [events/README.md](events/README.md) |
| `baseline/` | Línea base de comportamiento (clustering dinámico) + detección de desvíos | [baseline/README.md](baseline/README.md) |
| `evaluation/` | Validación cualitativa (video anotado, inspección manual) | [evaluation/README.md](evaluation/README.md) |
| `runner/` | Orquesta `vision/` + `vlm/` sobre N cámaras (archivo o stream en vivo) en paralelo | [runner/README.md](runner/README.md) |
| `dashboard/` | Panel web: arranca/para `runner/`, gestiona `cameras.yaml`, eventos y preview en vivo | [dashboard/README.md](dashboard/README.md) |

## Cómo trabajar en paralelo (varias sesiones de Claude Code)

1. `events/schema.py` ya está definido — es el contrato. No cambiarlo sin
   avisar a las demás sesiones/módulos que dependen de él.
2. `vision/` y `vlm/` se pueden desarrollar en paralelo, cada uno en su
   propia sesión: ninguno depende del código del otro, solo del esquema de
   eventos y de `videos/`.
3. `baseline/` puede arrancar en paralelo generando eventos sintéticos que
   respeten `MonitoringEvent` — no necesita esperar a que `vision/`/`vlm/`
   estén terminados, y se integra contra eventos reales después.
4. `evaluation/` puede arrancar apenas `vision/` tenga una versión mínima
   que devuelva detecciones por frame — no necesita el pipeline completo.
5. Cada módulo tiene su propio `requirements.txt` y su propio entorno
   virtual. `vision/`, `vlm/` y `baseline/` **no se importan código entre
   sí** — la única comunicación entre ellos es a través de `events/`, que sí
   está pensado para ser importado directamente (como librería local, sin
   servidor) por cualquiera de los tres. La única excepción es `runner/`,
   que sí importa `vision/` y `vlm/` directamente para correrlos juntos por
   cámara (ver `runner/README.md`) — es orquestación, no lógica de
   detección duplicada. `dashboard/` no importa código de detección de
   ninguno de los tres: solo lanza `runner/` como subproceso y lee
   `events/` para mostrar resultados.

## Setup general

Python 3.12+. Cada módulo documenta su propio setup (dependencias,
comandos) en su README.

## Roadmap (según el Gantt del informe)

- [x] `vision/` — clasificación de postura (parado/sentado/acostado) por
      frame, calibrada empíricamente (proceso completo en
      `vision/README.md`). La calibración usó en su momento una
      clasificación manual por carpeta que resultó no ser ground truth
      confiable de postura (mezclaba actividad con postura) — se descartó
      esa estructura; la referencia de acá en adelante es el modelo y su
      matriz de confusión sobre casos verificados a mano, no una
      categorización previa sin validar.
- [x] `vision/` — detección de caídas como evento (heurística basada en
      YOLOv11n-pose, siguiendo Giri et al. 2025, recalibrada por grid search
      sobre 179 clips de `GMDCSA-24`+`CAUCAFall` — proceso completo en
      `vision/README.md`). **Recall 96.2% (GMDCSA-24) / 68.0% (CAUCAFall),
      18/179 falsos positivos combinado.** *Riesgo encontrado 2026-07-20
      calibrando inmovilidad, cerrado 2026-08-06:* con el setup de cámara de
      Toyota Smarthome, el detector daba falsos positivos frecuentes (hasta
      124 en un solo clip) cuando la persona está sentada en el piso o muy
      abajo. Se confirmó cuantitativamente que **no es un problema de
      calibración de umbrales** (ningún punto del grid search lo resuelve,
      ver `vision/README.md`) — se resolvió con un **filtro de confirmación
      con CLIP zero-shot** como segunda etapa (`vlm/src/fall_confirmation.py`,
      integrado en `runner/`): 82.0% de recall (-3.2pp) y -67% de falsos
      positivos totales (-70% en el setup problemático). `vision/` en sí
      queda sin cambios de umbral, puramente geométrico, para seguir
      calibrándose de forma independiente.
- [x] `vision/` — inmovilidad prolongada como evento, separada en dos ramas
      por postura: **sedentarismo** (`sentado` inmóvil 180s+, severidad
      baja) calibrado y validado end-to-end con datos reales de Toyota
      Smarthome untrimmed (8/8 tramos de validación correctos); **posible
      post-caída** (`acostado` inmóvil, severidad alta) con detección y
      clasificación de postura verificadas contra clips de caída reales,
      pero **sin validar la duración real** (no existe dataset con alguien
      inmóvil en el piso durante minutos) — extrapolación documentada, no
      medida. Proceso completo en `vision/README.md`, sección "Inmovilidad
      prolongada".
- [x] `vlm/`: clasificación de actividades vía CLIP, exportado a ONNX para
      correr sin PyTorch en el Pi (ver `vlm/README.md`). Modo batch
      (`vlm/main.py`) sobre el dataset y modo en vivo (vía `runner/`). El
      modo en vivo, pensado para una cámara fija transmitiendo
      continuamente, recorta cada frame a la persona detectada por pose,
      promedia varias muestras por ventana, descarta ventanas de baja
      confianza y deduplica actividades repetidas (heartbeat con
      `ongoing: true` para reconstruir duraciones). El 65.2% medido
      originalmente sobre 39 videos curados no generalizaba: evaluado con
      participantes disjuntos train/val sobre una muestra grande del dataset
      completo, el zero-shot cae a ~48%. Se entrenó en cambio un **linear
      probe sobre los embeddings de CLIP** (categorías fusionadas donde el
      zero-shot no distinguía, ver `vlm/README.md`), que da **63.3%**
      consistente (±2.2 pts entre folds) — mejor que zero-shot pero todavía
      con una categoría floja (`EatingDrinkingMedication`, gestos de mano
      finos invisibles en un frame estático).
- [x] `events/`: store local funcionando end-to-end (SQLite).
- [x] `runner/` + `dashboard/`: orquestación multi-cámara (archivo o stream
      en vivo) y panel web de control — primera versión de la migración a
      ingesta en vivo (ver más abajo), corriendo todavía sobre PC de
      desarrollo, no sobre Raspberry Pi.
- [ ] `baseline/`: clustering dinámico + detección de desvíos (Snoun et al.
      2024).
- [ ] `evaluation/`: exportar video anotado para inspección visual.
- [ ] *(fase futura, no ahora)* desplegar `runner/` en Raspberry Pi 5 (el
      código ya soporta streams RTSP/webcam vía `cameras.yaml`, falta
      validar en el hardware real).
- [ ] *(fase futura, no ahora)* exponer `events/` + `baseline/` vía API
      (FastAPI) para una futura app móvil — separado de `dashboard/`, que es
      panel de operación local, no la API pública del sistema final.
