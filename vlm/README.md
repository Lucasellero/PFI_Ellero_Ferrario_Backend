# `vlm/` — Clasificación semántica de actividades

## Qué hace

Consulta un VLM local open-source de forma **periódica** (no frame a frame —
ver Sec. 2.1.6 del informe, el costo computacional de un VLM es mucho mayor
al de pose estimation) sobre frames muestreados de un video, y genera una
descripción textual + etiqueta de actividad (ej. "cocinando", "mirando
televisión", "leyendo").

## Dos modos de uso

- **Batch** (`vlm/main.py`): procesa un archivo o todo `../videos/` de una
  sola vez, muestreando cada `--interval` segundos. Pensado para el dataset
  grabado.
- **En vivo** (vía `../runner/`): `runner/camera_worker.py` importa
  `ClipActivityClassifier` directamente y lo corre sobre cada cámara
  configurada en `cameras.yaml`, en ventanas de muestreo cada
  `clip_interval_s` segundos, en paralelo con la detección de pose. En este
  modo cada frame se recorta a la persona detectada (`crop_to_person`), se
  promedian varias muestras por ventana, y hay umbral de confianza y
  deduplicación de actividades repetidas — ver "Clasificación de
  actividades (CLIP)" en `../runner/README.md`.
  Requiere haber generado los modelos ONNX primero (ver Setup abajo) — si no
  están, `runner/` sigue funcionando pero sin clasificación de actividad.

## Entrada

Un archivo de video en `../videos/<nombre>.mp4` (modo batch) o un frame en
vivo de `../runner/` (modo en vivo). Se muestrea un frame cada N segundos
(configurable) en vez de procesar todos los frames.

## Salida

`MonitoringEvent` con `event_type = "activity"` (ver
`../events/schema.py`), `video_timestamp_s` = momento del frame muestreado,
y `metadata` con al menos `description` (texto libre del VLM) y, si se logra
clasificar en una categoría cerrada, `activity_label`.

En modo en vivo el metadata incluye además `confidence` (promedio de la
ventana), `samples` (cuántos frames se promediaron), `ongoing` (`true` si
es el heartbeat de una actividad que continúa, no un cambio) y `scores`
(distribución completa sobre las 14 clases).

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

## Precisión y evaluación (`evaluate_activity.py`)

Los videos de `../videos/{acostado,parado,sentado}` llevan la actividad
real como prefijo del nombre de archivo (`Walk_...`, `Laydown_...`), lo que
permite medir la clasificación cuantitativamente. Desde la raíz del
proyecto:

```bash
python -m vlm.evaluate_activity
```

Resultado (2026-07-12, 39 videos, 112 ventanas de 3 muestras):

| Enfoque | Ventanas correctas |
|---|---|
| Frame completo, 1 muestra por ventana (enfoque original) | 25.0% |
| Recorte a la persona + promedio de 3 muestras (runner actual) | **40.2%** |

El recorte al bounding box de la persona (`crop_to_person` en
`src/clip_classifier.py`, reutiliza la detección de pose de `vision/`)
elimina el sesgo del fondo fijo — con el frame completo, una TV visible
empuja todo hacia `WatchTV`. Las clases de movimiento y postura gruesa son
las que más mejoran (Walk: 8/24 → 19/24 ventanas; Cook.Stir: 0/3 → 3/3);
las de motricidad fina (`Drink.*`, `Takepills`, `Maketea.*`) siguen flojas
porque desde un frame estático son casi indistinguibles entre sí.

Próximos saltos posibles (en orden de esfuerzo): fusionar clases
visualmente similares en categorías más gruesas, o entrenar un linear probe
sobre los embeddings de CLIP usando estos mismos videos etiquetados
(reemplaza `text_features.npy` por los pesos entrenados, mismo costo en
runtime).

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
