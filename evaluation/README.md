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

Video anotado (`.mp4`) guardado en `output/` (gitignored) para revisión
manual. No hay métricas cuantitativas (precision/recall) hasta que se
etiquete el dataset — eso es trabajo futuro, ver `../videos/README.md`.

## Setup

```
cd evaluation
py -m venv .venv && .venv\Scripts\activate
pip install -r requirements.txt
```

El primer uso descarga automáticamente los pesos `yolo11n-pose.pt` (mismo
modelo que usa `vision/`) si no están ya cacheados.

## Cómo probarlo

```
python annotate_video.py --video <nombre>.mp4
```

Por default superpone los eventos que ya estén guardados en
`../events/events.db` para ese `video_id` (de `vision/`, `vlm/`/`fusion/` o
`baseline/`, lo que se haya corrido antes sobre ese video). Si no hay
ninguno, agregar `--rerun` corre el detector de caída/inmovilidad de
`vision/` desde cero sobre el video y guarda esos eventos nuevos en el
store antes de generar el video anotado — no reproduce eventos de
`vlm/`/`fusion/`/`baseline/`, que son pipelines separados.

El esqueleto de pose se dibuja siempre (independiente de si hay eventos
guardados), reutilizando `vision/src/pose_pipeline.py` (`detect_person` +
`draw_pose`) para no duplicar el modelo de pose ni su lógica de
desambiguación de sujeto principal.

## Estado

Implementado (2026-10-01): `annotate_video.py`.

**Primer uso real (2026-10-01):** se corrió `--rerun` sobre los 4 clips de
`../videos/ToyotaSmarthome_FP/` (el caso pendiente documentado en
`../vision/README.md`: inmovilidad no se dispara en 3 de 4 clips largos).
Resultado, contando eventos en `events.db`:

| Clip | `fall` (geométrico puro, sin confirmación CLIP/temporal) | `immobility` |
|---|---|---|
| P07T05C05 | 124 | 0 |
| P09T05C05 | 82 | 0 |
| P09T06C05 | 61 | 0 |
| P07T10C02 | 3 | 1 (180s, sentado, severidad baja) |

Confirma el pendiente ya documentado (solo 1 de 4 dispara inmovilidad). La
inspección visual de los frames donde dispara `fall` en dos de los tres
clips sin inmovilidad (P07T05C05 @88.8s, P09T05C05 @48.5s) muestra en ambos
casos a la persona sentada/acostada normalmente en el sillón — confirma
visualmente, con video real, la ambigüedad de la heurística geométrica sola
que ya estaba documentada (sin la confirmación CLIP/clasificador temporal
que corre en producción vía `vlm/`/`fusion/` y que `--rerun` no reproduce,
estos conteos de `fall` NO son el FP rate de producción, ver nota en
`vision/README.md` sobre no confundir corridas de `vision/` sola con el
pipeline completo).

También se midió por primera vez la tasa de detección de `yolo11n-pose`
sobre el clip COMPLETO (antes solo medida sobre tramos cortos): 50.6-77.8%
en los 3 clips sin inmovilidad vs. 99.2% en el único que sí dispara, con
huecos de hasta 23.1s. Se revisaron a mano los frames antes/durante/después
de los dos huecos más grandes y en ambos casos la persona simplemente no
está en el cuadro — se fue de la habitación, no es el modelo perdiendo a
alguien presente y quieto. Detalle completo y tabla en `vision/README.md`
("Validación visual con `evaluation/annotate_video.py`").
