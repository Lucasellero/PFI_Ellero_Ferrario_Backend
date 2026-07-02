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

## Decisión de modelo: CLIP (OpenAI, ViT-B/32)

Se eligió CLIP en lugar de un VLM generativo (LLaVA, Moondream, etc.) por las
siguientes razones:

- **Velocidad**: corre bien en CPU sin cuantización; ViT-B/32 es el punto de
  entrada más liviano. Para mayor precisión, `--model ViT-L/14`.
- **Zero-shot sin fine-tuning**: los prompts en lenguaje natural
  ("a photo of a person reading a book") permiten clasificar directamente las
  14 actividades del dataset Toyota Smarthome sin ajuste adicional.
- **Sin servidor externo**: se descarga al primer uso y queda cacheado
  localmente (`~/.cache/clip/`), cumpliendo el requisito de privacidad.
- **Trade-off aceptado**: CLIP no genera texto libre — la `description` en
  `metadata` es el prompt textual que ganó la clasificación, no una descripción
  generada. Si en el futuro se necesita texto generativo, se puede cambiar el
  clasificador sin tocar `main.py`.

## Setup

### En la máquina de desarrollo (una sola vez)

```bash
cd vlm/
pip install -r requirements-export.txt
python export_onnx.py
```

Genera `vlm/models/` con:
- `clip_image_encoder_int8.onnx` (~85 MB, INT8 cuantizado para ARM)
- `text_features.npy` (embeddings pre-computados, ~28 KB)
- `labels.json`

Copiar `vlm/models/` al Pi antes de correr.

### En el Pi 5

```bash
cd vlm/
pip install -r requirements.txt   # solo onnxruntime + opencv, sin PyTorch
python main.py --video <archivo>.mp4
```

**Por qué esta separación:** PyTorch pesa ~1.5 GB instalado en ARM y solo se necesita para exportar el modelo. En el Pi solo se usa `onnxruntime`, que pesa ~30 MB y es ~2-4x más rápido en CPU ARM que PyTorch.

### Estimación de rendimiento en Pi 5

| Componente | Tiempo por inferencia | Cores usados |
|---|---|---|
| YOLO11n-pose (vision/) | ~150-300ms/frame | 2-4 |
| CLIP ONNX INT8 (vlm/) | ~300-600ms/frame | 2 (limitado) |

CLIP muestrea cada 10s (default), dejando el CPU libre para YOLO entre clasificaciones.

## Fuera de alcance por ahora

- Optimización para hardware embebido (Raspberry Pi).
- Usar un VLM vía API cloud (contradice el requisito de privacidad del
  informe — todo el procesamiento debe ser local).
