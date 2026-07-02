# `vlm/` — Clasificación semántica de actividades

## Qué hace

Consulta un VLM local open-source de forma **periódica** (no frame a frame —
ver Sec. 2.1.6 del informe, el costo computacional de un VLM es mucho mayor
al de pose estimation) sobre frames muestreados de un video, y genera una
descripción textual + etiqueta de actividad (ej. "cocinando", "mirando
televisión", "leyendo").

## Entrada

Un archivo de video en `../videos/<nombre>.mp4`. Se muestrea un frame cada
N segundos (configurable) en vez de procesar todos los frames.

## Salida

`MonitoringEvent` con `event_type = "activity"` (ver
`../events/schema.py`), `video_timestamp_s` = momento del frame muestreado,
y `metadata` con al menos `description` (texto libre del VLM) y, si se logra
clasificar en una categoría cerrada, `activity_label`.

Los eventos se persisten vía `../events/store.py`, igual que en `vision/`.

## Decisión pendiente: qué modelo/runtime usar

Evaluar entre modelos locales open-source (ej. Qwen2-VL, LLaVA, Moondream)
corriendo vía Ollama o `transformers`, priorizando velocidad de inferencia en
CPU/GPU local sobre precisión máxima — recordar que en el informe (Sec.
2.2.4) Schönfeldt et al. (2025) muestran que un VLM open-source alcanza
desempeño comparable a modelos especializados (F1 = 0.89) en clasificación de
comportamiento sedentario. Documentar acá la elección final y por qué.

## Setup

Pendiente de definir junto con la elección de modelo (ver arriba). Agregar
la dependencia elegida a `requirements.txt` de este módulo.

## Fuera de alcance por ahora

- Optimización para hardware embebido (Raspberry Pi).
- Usar un VLM vía API cloud (contradice el requisito de privacidad del
  informe — todo el procesamiento debe ser local).
