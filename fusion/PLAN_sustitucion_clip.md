# Plan: de 4 a ~10 clases de actividad y sustitución progresiva de CLIP

**Fecha:** 2026-09-01 · **Estado:** plan, nada implementado todavía.
**Fuente de verdad de resultados:** sigue siendo `fusion/README.md` / `vlm/README.md`.
Este documento es el plan; cuando cada fase se ejecute, los números van a los README.

---

## 1. Por qué

El módulo `baseline/` (línea base de rutinas + detección de desvíos) se alimenta de
un stream de eventos de actividad. Su capacidad de detectar desvíos clínicamente
relevantes depende de **cuántas clases distintas** recibe:

- Con 4 clases (`Cooking` / `EatingDrinkingMedication` / `Moving` / `Resting`) no se
  puede detectar "no tomó la medicación matinal", "se saltea el almuerzo", "más
  viajes al baño de noche", "no salió de casa en 3 días", "dejó de leer / ver TV".
- El dataset [`dataset_events/`](../dataset_events/README.md) (CASAS, sensores
  ambientales) demuestra que esas distinciones existen y son anotables: ~15-16
  clases canónicas con volumen, sobre meses de una misma persona.

El cuello de botella **no es la capacidad del modelo de visión** — ya está
documentado en `vlm/README.md` que CLIP ViT-B/16 (4× cómputo) no mejora sobre
ViT-B/32 (52.6% vs 51.6%): el problema es estructural, falta información en el frame
de una cámara única y ancha. Subir a un encoder más grande o a un VLM generativo no
lo resuelve.

**La palanca real es agregar señales que una cámara única sí puede dar**, replicando
lo que en CASAS hace la ubicación del sensor:

1. **Zona del hogar** (dónde está la persona) — hoy no se usa.
2. **Objeto de contexto** (qué mueble/electrodoméstico tiene al lado) — YOLO ya puede
   darlo, hoy no se usa.
3. **Forma del movimiento en el tiempo** — ya lo da `fusion/` (GRU temporal sobre
   esqueleto), hoy fusionado solo con CLIP.

## 2. Qué se sustituye y qué no

| Señal | Rol hoy | Rol objetivo |
|---|---|---|
| Esqueleto temporal (GRU, `fusion/`) | secundaria (fusión con CLIP) | **primaria** de movimiento/postura |
| Zona del hogar | — | **primaria** de ubicación (nueva) |
| Objeto de contexto (YOLO-COCO) | — | **secundaria** de contexto (nueva) |
| CLIP embedding holístico | primaria de contexto | **opcional**: desempate de escena en keyframe + hand-crop para `medicacion` |
| Detector geométrico (`vision/`) | caídas / inmovilidad | sin cambios |

CLIP **no se elimina del repo**: queda como señal opcional (`mode="clip_only"` sigue
siendo el fallback, y el hand-crop para `medicacion` es la única vía que movió esa
categoría — ver `vlm/README.md`). Pero deja de ser el eje de la clasificación de
actividad.

## 3. Techo real de una cámara única (lo que NINGUNA fase resuelve)

Documentar esto en el informe como límite conocido, no como pendiente:

- `comer` / `beber` / `medicacion`: gesto mano-boca de <2s, casi siempre ocluido. Se
  mantienen como **una sola clase** desde visión (`EatingDrinkingMedication`). El
  hand-crop de CLIP sube algo `medicacion` pero no la separa de forma confiable.
- `leer` / `ver_tv` / `telefono`: las tres son "sentado quieto". Sin primer plano no
  se separan → una sola clase `ocio_sentado` desde visión.
- Estas distinciones, cuando importan, las cubre un **sensor ambiental** en la fase Pi
  (contacto en pastillero, en heladera, uso de TV), no la cámara.

Vocabulario objetivo **desde visión**: ~9-10 clases —
`dormir`, `descanso_sofa`, `cocina`, `comida` (incl. beber/meds), `ocio_sentado`
(tv/leer/teléfono), `aseo` (zona baño), `salir_casa` / `entrar_casa`, `moverse`,
`caida`, `inmovilidad`. Converge con el nivel "grueso" del ETL de CASAS a propósito:
el vocabulario canónico debe ser el mismo para el desarrollo con dataset y para el
despliegue con cámara.

## 4. Fases

### Fase 1 — Mapa de zonas del hogar  *(mayor impacto, costo de cómputo ~0)*

- **Qué:** polígonos nombrados dibujados una vez en el setup (`cama`, `sofa`, `mesa`,
  `cocina`, `puerta`, `bano`), en coordenadas normalizadas de frame, por cámara.
- **Dónde vive:** `cameras.yaml` (nueva clave `zones:` por cámara) + un editor en
  `dashboard/` (canvas sobre un frame congelado, ya hay drag & drop de cámaras — commit
  `5011f83`). Sin editor: se puede escribir el YAML a mano para el dataset.
- **Cómo se usa:** en `runner/camera_worker.py`, tras `detector.process_frame(...)`,
  se calcula la zona del centro de cadera (keypoints 11-12, ya disponibles) y se
  agrega como feature one-hot a la ventana de `fusion/` (`frame_features` en
  `fusion/src/keypoint_sequence.py` pasa de 55 a 55+`n_zonas` dims).
- **Entrenamiento:** Toyota Smarthome tiene layout fijo por cámara → se pueden
  anotar zonas una vez por setup del dataset y hornearlas en `extract_dataset.py`.
- **Criterio de aceptación:** +X pts de accuracy holdout por categoría, sin regresión
  en las que ya andan bien (`Cooking`, `Moving`).

### Fase 2 — Contexto de objeto vía YOLO-COCO  *(reusa modelo ya cargado)*

- **Qué:** COCO ya trae `bed`, `couch`, `dining table`, `tv`, `oven`, `sink`,
  `refrigerator`, `microwave`, `toilet`, `chair`, `book`, `cup`, `bottle`, `bowl`.
- **Cómo:** una pasada de `yolo11n` (detección, no pose) cada `clip_interval_s` (no
  por frame), buscar objetos con IoU/cercanía al bbox de la persona → vector de
  presencia (`persona_cerca_de_X`). Se concatena a la ventana de `fusion/` igual que
  la zona.
- **Costo Pi:** ~40-70 ms una vez por tick de clasificación → despreciable amortizado.
  Con Hailo-8L, irrelevante.
- **Nota:** solapa parcialmente con la zona (mesa ≈ zona mesa) — medir si suma sobre
  Fase 1 o es redundante antes de dejarlo en producción (mismo criterio que el
  detector de objeto sostenido descartado en `vlm/README.md`).
- **Medido (2026-09-20, ver "Fase 2 de PLAN_sustitucion_clip.md..." en `vlm/README.md`
  para el detalle):** implementado como feature adicional del linear probe de `vlm/`
  (no integrado a `fusion/` ni al runner todavía — mismo dataset/protocolo de
  `train_category_probe.py`, no la ventana temporal de `fusion/`). Confirma la
  hipótesis solo parcialmente: `Cooking` sube +2.4 pts (80.2%→82.6%) gracias sobre
  todo a `Cook.Cleandishes` (-40% de errores, probablemente por detectar la pileta),
  pero el efecto global es +0.3 pts (dentro de ruido) y una subclase (`Cook.Cut`)
  empeoró. No reemplaza a Fase 1 (zona) — se recomienda combinarlas antes de decidir
  si Fase 2 sola justifica el costo de una segunda pasada de YOLO en vivo.

### Fase 3 — Ampliar el clasificador temporal a ~10 clases, entrenado en serio

- **Qué:** el GRU de `fusion/` hoy sale a 31 clases finas de Toyota Smarthome +
  `Standing`, pero entrenado sobre el dataset local de ~2200 clips (18 clases
  presentes). Entrenar sobre **Toyota Smarthome completo** (16K clips, 31 clases,
  18 sujetos 60-80 años, [project.inria.fr/toyotasmarthome](https://project.inria.fr/toyotasmarthome/))
  + las features nuevas de Fase 1-2.
- **Modelo:** si el GRU no separa clases con gesto sutil (hipótesis de `fusion/`), el
  siguiente paso ya está documentado: ST-GCN (Yan et al. 2018). Presupuesto de
  parámetros actual ~50-90k, límite 1-2M — hay margen.
- **Salida:** GRU clasifica fino, se agrega a las ~10 categorías canónicas
  (`fine_scores_to_category_scores`, ya existe).

### Fase 4 — CLIP opcional / aceleración Hailo

- CLIP degradado a: (a) chequeo de escena en keyframe cuando esqueleto+zona+objeto
  no deciden, (b) hand-crop para `medicacion`. Configurable por `cameras.yaml`
  (`clip_role: off | tiebreak | full`).
- Con **Hailo-8L** (AI Kit, ~USD 70): YOLO pose+detección a 100+ FPS, CLIP a 30-60
  FPS → todo lo anterior corre holgado y CLIP puede volver a ser por-segundo si
  la evaluación muestra que aporta. Documentar como opción de hardware, no requisito.
- **VLM local** (Moondream2 1.8B): ~10-30 s/imagen en Pi 5 CPU → fuera de alcance
  para esta iteración, se revisa en la fase Pi+cloud.

## 5. Cómo medir la mejora

### Baseline a superar (estado actual, de `fusion/README.md`, 2026-08-23)

| Split | CLIP solo | Esqueleto solo | Híbrido |
|---|---|---|---|
| Holdout (273 videos, sin leakage) | 75.8% | 78.0% | **82.8%** |

4 categorías. `EatingDrinkingMedication` es la más floja (56.1% con probe+hand-crop).

### Protocolo (idéntico para cada fase, para que los números sean comparables)

1. **Split leave-subject-out** sobre Toyota Smarthome completo: participantes de
   validación nunca vistos en entrenamiento, promedio de ≥5 splits. (Ya es el
   protocolo de `train_temporal_classifier.py` / `train_category_probe.py`.)
2. **Métricas:**
   - Accuracy global por nivel de categoría (reportar el nº de categorías — no es
     comparable entre vocabularios distintos, ver la nota en `vlm/README.md`).
   - **Accuracy y recall por clase** — es lo que importa, no el global.
   - **Recall de las clases clínicas** (`medicacion`, `bano_nocturno`, `salir_casa`)
     como métrica destacada: son las que justifican el proyecto.
   - Matriz de confusión por categoría (detectar confusores dominantes vs. ruido
     repartido, como ya se hizo en `vlm/README.md`).
3. **Criterio de aceptación por fase:** la fase se integra a producción solo si sube
   el holdout global **y** no regresiona ninguna clase que ya funcionaba (`Cooking`,
   `Moving`) más allá del ruido entre seeds. Si una feature nueva mejora una clase
   pero hunde otra (como pasó con `Standing` vs `Moving`), se documenta el trade-off
   y se decide explícitamente — no se integra a ciegas.
4. **Ablation:** medir cada señal por separado (esqueleto / +zona / +objeto / +CLIP)
   para saber qué aporta cada una y poder sacar la que no mueve la aguja.

### Validación end-to-end (la que realmente cuenta)

El objetivo no es accuracy de clasificación por sí misma, es que `baseline/` detecte
mejor los desvíos. Protocolo:

1. Correr el ETL de CASAS (Fase 0, abajo) → eventos de una persona real, meses.
2. Construir la línea base de rutina de esa persona con `baseline/`.
3. Inyectar desvíos sintéticos (día sin medicación, noche con 5 viajes al baño, semana
   sin salir) y medir si `baseline/` los marca con **N clases canónicas** vs con **4**.
4. Repetir con la salida del clasificador de visión (sobre video, con las clases que
   la cámara alcanza) para ver cuánto se pierde al pasar de las clases de sensor a las
   de cámara.

Ese delta — "cuántos desvíos detectables con sensores se siguen detectando solo con
cámara" — es el número clave para el informe.

## 6. Fase 0 (prerrequisito, en curso): ETL de CASAS

Ver [`dataset_events/README.md`](../dataset_events/README.md). Deja el dataset de
eventos listo para construir la línea base y para la validación end-to-end de arriba.
El vocabulario canónico del ETL y el de este plan **son el mismo** (por diseño).
