"""Clasificación zero-shot con CLIP via ONNX Runtime.

En el Pi no se usa PyTorch: inference via onnxruntime, text features cargados
desde models/text_features.npy (generados por export_onnx.py en dev).

Prompt ensembling: múltiples templates por clase promediados en un vector,
más robusto que un solo prompt frente a fondos ambiguos (TV visible, etc.).
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import onnxruntime as ort
from PIL import Image

MODELS_DIR = Path(__file__).resolve().parents[1] / "models"

# CLIP ViT-B/32 normalization
_MEAN = np.array([0.48145466, 0.4578275, 0.40821073], dtype=np.float32)
_STD = np.array([0.26862954, 0.26130258, 0.27577711], dtype=np.float32)

# Color de relleno para el pad-a-cuadrado: el color medio de entrenamiento
# de CLIP (en BGR uint8), que tras normalizar queda en ~0 (neutro).
_PAD_COLOR_BGR = (_MEAN[::-1] * 255).astype(np.uint8)

# Múltiples prompts por clase — usados por export_onnx.py para generar text_features.npy.
# En el Pi se usan solo para el campo "description" del evento.
ACTIVITY_PROMPTS: dict[str, list[str]] = {
    "Cook.Cleanup": [
        "a photo of a person cleaning up in the kitchen",
        "a person wiping or scrubbing a kitchen counter",
        "someone washing dishes at a kitchen sink",
        "a person tidying up after cooking",
    ],
    "Cook.Stir": [
        "a photo of a person stirring food in a pot on the stove",
        "a person cooking and stirring with a spoon",
        "someone stirring a pot while standing at a stove",
    ],
    "Drink.Fromcan": [
        "a photo of a person drinking from a can",
        "a person raising a soda can to their mouth",
        "someone drinking a beverage from a metal can",
    ],
    "Drink.Fromcup": [
        "a photo of a person drinking from a cup or mug",
        "a person raising a cup to their lips to drink",
        "someone drinking from a glass or mug",
    ],
    "Maketea.Insertteabag": [
        "a photo of a person inserting a tea bag into a cup",
        "a person placing a tea bag into hot water in a cup",
        "someone making tea by putting a tea bag in a mug",
    ],
    "Readbook": [
        "a photo of a person reading a book",
        "a person sitting and holding a book open to read",
        "someone reading printed pages of a book",
    ],
    "Sitdown": [
        "a photo of a person sitting down on a chair or sofa",
        "a person in the act of lowering themselves onto a seat",
        "someone transitioning from standing to sitting",
    ],
    "Takepills": [
        "a photo of a person taking pills or medicine",
        "a person placing tablets or pills into their mouth",
        "someone swallowing medication with water",
    ],
    "Uselaptop": [
        "a photo of a person using a laptop computer",
        "a person typing on or looking at a laptop screen",
        "someone working on a laptop placed on a table",
    ],
    "Usetablet": [
        "a photo of a person using a tablet device",
        "a person holding and tapping on a touchscreen tablet",
        "someone swiping on an iPad or tablet",
    ],
    "WatchTV": [
        "a photo of a person sitting still and watching television",
        "a person seated on a couch staring at a TV screen",
        "someone watching a TV show while sitting or lying on a sofa",
    ],
    "Enter": [
        "a photo of a person entering a room through a doorway",
        "a person walking through a door into a room",
        "someone stepping inside through an open door",
    ],
    "Walk": [
        "a photo of a person actively walking across a room",
        "a person in motion walking through an indoor space",
        "someone walking and moving their legs forward indoors",
        "a person striding across a living room or hallway",
    ],
    "Laydown": [
        "a photo of a person lying down horizontally on a bed",
        "a person lying flat on a sofa or couch",
        "someone resting in a lying-down position",
        "a person with their body horizontal on a surface",
    ],
    "Getup": [
        "a photo of a person getting up from a chair or bed",
        "a person rising to a standing position",
        "someone standing up after sitting or lying down",
    ],
    "Leave": [
        "a photo of a person leaving a room through a doorway",
        "a person walking out through a door",
        "someone exiting a room, walking away from the camera through an exit",
    ],
    "Usetelephone": [
        "a photo of a person talking on the phone",
        "a person holding a phone to their ear",
        "someone having a phone conversation",
    ],
    "Cook.Cleandishes": [
        "a photo of a person washing dishes in the kitchen sink",
        "a person cleaning plates and cutlery at a sink",
        "someone scrubbing dishes with water running",
    ],
    "Cook.Cut": [
        "a photo of a person cutting food on a kitchen counter",
        "a person chopping vegetables with a knife",
        "someone slicing food on a cutting board",
    ],
    "Cook.Usestove": [
        "a photo of a person cooking food on a stove",
        "a person standing at a stove with a pot or pan",
        "someone operating stove burners while cooking",
    ],
    "Cutbread": [
        "a photo of a person cutting a loaf of bread",
        "a person slicing bread on a cutting board",
        "someone cutting bread with a knife in the kitchen",
    ],
    "Drink.Frombottle": [
        "a photo of a person drinking from a bottle",
        "a person raising a bottle to their mouth to drink",
        "someone drinking water or a beverage from a bottle",
    ],
    "Drink.Fromglass": [
        "a photo of a person drinking from a glass",
        "a person raising a drinking glass to their lips",
        "someone drinking a beverage from a clear glass",
    ],
    "Eat.Attable": [
        "a photo of a person eating a meal seated at a table",
        "a person having a meal at a dining table",
        "someone eating food with utensils at a table",
    ],
    "Eat.Snack": [
        "a photo of a person eating a snack by hand",
        "a person eating a small snack while sitting",
        "someone snacking, eating food held in their hand",
    ],
    "Makecoffee.Pourgrains": [
        "a photo of a person pouring coffee grounds into a coffee maker",
        "a person scooping and pouring ground coffee",
        "someone preparing coffee by adding coffee grounds",
    ],
    "Makecoffee.Pourwater": [
        "a photo of a person pouring water into a coffee machine",
        "a person filling a coffee maker with water",
        "someone pouring water while making coffee",
    ],
    "Maketea.Boilwater": [
        "a photo of a person boiling water for tea",
        "a person filling and starting a kettle to boil water",
        "someone standing by a kettle waiting for water to boil",
    ],
    "Pour.Frombottle": [
        "a photo of a person pouring liquid from a bottle into a cup",
        "a person tilting a bottle to pour a drink",
        "someone pouring a beverage from a bottle",
    ],
    "Pour.Fromcan": [
        "a photo of a person pouring liquid from a can into a cup",
        "a person tilting a can to pour a drink",
        "someone pouring a beverage from a can",
    ],
    "Pour.Fromkettle": [
        "a photo of a person pouring hot water from a kettle",
        "a person tilting a kettle to pour water into a cup",
        "someone pouring water from a kettle while making tea or coffee",
    ],
    "Standing": [
        "a photo of a person standing still in a room",
        "a person standing upright without moving",
        "someone standing idle, not walking or sitting",
    ],
}

# Clases negativas: absorben recortes sin persona visible (bbox desfasado,
# persona saliendo de cuadro) en vez de forzar una de las 14 actividades
# con confianza baja. El runner las descarta sin emitir evento.
BACKGROUND_PROMPTS: dict[str, list[str]] = {
    "Background": [
        "a photo of an empty room with furniture and no people",
        "a blurry indoor photo with no person visible",
        "a photo of a wall, floor, or household objects",
        "an empty hallway or doorway with nobody in it",
    ],
}

ALL_PROMPTS: dict[str, list[str]] = {**ACTIVITY_PROMPTS, **BACKGROUND_PROMPTS}
BACKGROUND_LABELS = frozenset(BACKGROUND_PROMPTS)

# Categorías gruesas para el producto: al familiar no le sirve distinguir
# Drink.Fromcan de Drink.Fromcup, y desde un frame estático CLIP tampoco
# puede. Se clasifica con las clases finas (mejor cobertura de prompts) y se
# SUMAN las probabilidades de las subclases — el error entre subclases de la
# misma categoría desaparece.
#
# Takepills se probó como categoría propia ("Medication") por ser clínicamente
# relevante, pero la evaluación contra el dataset grande (2026-07-17, ver
# vlm/README.md) mostró que ni el zero-shot ni un linear probe entrenado
# logran distinguirla de forma consistente: son gestos de mano breves,
# indistinguibles de un frame estático a distancia de cámara de habitación —
# la misma limitación estructural que Drink.* y Maketea.Insertteabag. Se
# fusionan las cuatro en una sola categoría para dejar de reportar una
# distinción que en la práctica es ruido, no señal.
COARSE_CATEGORIES: dict[str, list[str]] = {
    "Cooking": [
        "Cook.Cleanup", "Cook.Stir", "Cook.Cleandishes", "Cook.Cut",
        "Cook.Usestove", "Cutbread",
    ],
    "EatingDrinkingMedication": [
        "Drink.Fromcan", "Drink.Fromcup", "Drink.Frombottle", "Drink.Fromglass",
        "Maketea.Insertteabag", "Maketea.Boilwater",
        "Makecoffee.Pourgrains", "Makecoffee.Pourwater",
        "Pour.Frombottle", "Pour.Fromcan", "Pour.Fromkettle",
        "Eat.Attable", "Eat.Snack", "Takepills",
    ],
    "Resting": ["Laydown", "WatchTV"],
    "ReadingScreen": ["Readbook", "Uselaptop", "Usetablet"],
    "Moving": ["Walk", "Enter", "Sitdown", "Getup", "Leave"],
    # Sola por ahora: no se fusiona a priori con EatingDrinkingMedication
    # como Takepills — hay que medir con evaluate_activity.py/
    # train_category_probe.py si CLIP la distingue antes de decidir.
    "Phone": ["Usetelephone"],
    # Datos de entrenamiento vía extract_omnifall_features.py (segmentos
    # OmniFall sobre videos/CAUCAFall/, ver vlm/README.md) — ninguna clase
    # de Toyota Smarthome cubre "parado sin hacer nada".
    "Standing": ["Standing"],
    "Background": list(BACKGROUND_PROMPTS),
}

LABEL_TO_CATEGORY: dict[str, str] = {
    label: cat for cat, labels in COARSE_CATEGORIES.items() for label in labels
}

CATEGORY_DESCRIPTIONS: dict[str, str] = {
    "Cooking": "a person cooking or working in the kitchen",
    "EatingDrinkingMedication": "a person eating, drinking, or taking medication",
    "Resting": "a person resting, lying down or watching TV",
    "ReadingScreen": "a person reading or using a screen device",
    "Moving": "a person walking or moving around the room",
    "Phone": "a person talking on the phone",
    "Standing": "a person standing still, not doing any particular activity",
    "Background": "no clear person visible in the frame",
}


def aggregate_to_categories(scores: dict[str, float]) -> dict[str, float]:
    """Suma las probabilidades de las clases finas por categoría gruesa."""
    cats: dict[str, float] = {cat: 0.0 for cat in COARSE_CATEGORIES}
    for label, p in scores.items():
        cats[LABEL_TO_CATEGORY[label]] += p
    return cats


def _softmax(x: np.ndarray) -> np.ndarray:
    x = x - x.max()
    e = np.exp(x)
    return e / e.sum()


def crop_to_person(image_bgr: np.ndarray, box, margin: float = 0.25) -> np.ndarray:
    """Recorta el frame al bounding box de la persona, con margen relativo.

    En una cámara fija de habitación el fondo es constante y domina la
    similitud de CLIP (una TV visible empuja todo hacia WatchTV aunque la
    persona esté caminando). Clasificar solo el recorte de la persona
    elimina ese sesgo. `box` es xyxy (como devuelve el detector de pose).
    """
    h, w = image_bgr.shape[:2]
    x1, y1, x2, y2 = (float(v) for v in box)
    mx = (x2 - x1) * margin
    my = (y2 - y1) * margin
    x1 = max(0, int(x1 - mx))
    y1 = max(0, int(y1 - my))
    x2 = min(w, int(x2 + mx))
    y2 = min(h, int(y2 + my))
    if x2 - x1 < 32 or y2 - y1 < 32:  # bbox degenerado: mejor el frame completo
        return image_bgr
    return image_bgr[y1:y2, x1:x2]


# Índices COCO de keypoints de muñeca (mismo formato que vision/src/pose_pipeline.py).
_L_WRIST, _R_WRIST = 9, 10

# Confianza mínima de un keypoint de muñeca individual para confiar en su
# posición (más estricta que el promedio de cuerpo de detect_person, que
# puede pasar aunque una muñeca puntual esté ocluida/adivinada).
_MIN_WRIST_CONFIDENCE = 0.3


def crop_to_hands(
    image_bgr: np.ndarray,
    keypoints: np.ndarray,
    box,
    margin: float = 0.45,
    keypoints_conf: np.ndarray | None = None,
) -> np.ndarray | None:
    """Recorta la zona de las manos a partir de los keypoints de muñeca, sobre
    el frame completo (no el crop de persona — `keypoints` viene en
    coordenadas de píxel del frame completo, igual que las devuelve
    `detect_person`/`FallImmobilityDetector.last_detection` en `vision/`, que
    usa `.xy` de ultralytics, no `.xyn`).

    Categorías como comer/beber/tomar pastillas dependen de un gesto de mano
    breve que el embedding de cuerpo completo de CLIP no resuelve bien (es
    holístico, no hace zoom): este crop se usa como feature adicional para el
    linear probe (ver train_category_probe.py), no para el zero-shot.

    `keypoints_conf`, si se pasa, descarta muñecas cuya confianza individual
    esté debajo de `_MIN_WRIST_CONFIDENCE` — con la persona lejos o de
    espaldas, YOLO-pose puede "adivinar" una posición poco confiable que no
    conviene usar como centro del recorte. Sin este argumento se mantiene el
    comportamiento anterior (solo posición).

    Devuelve `None` si ninguna muñeca tiene una posición válida/confiable
    (YOLO-pose deja keypoints no predichos en (0, 0)) — el caller debe
    decidir el fallback.
    """
    h, w = image_bgr.shape[:2]
    x1, y1, x2, y2 = (float(v) for v in box)
    box_h = y2 - y1
    if box_h <= 0:
        return None
    side = box_h * margin

    wrists = []
    for idx in (_L_WRIST, _R_WRIST):
        kx, ky = float(keypoints[idx, 0]), float(keypoints[idx, 1])
        if kx <= 0.0 or ky <= 0.0:  # no predicho
            continue
        if keypoints_conf is not None and float(keypoints_conf[idx]) < _MIN_WRIST_CONFIDENCE:
            continue
        wrists.append((kx, ky))
    if not wrists:
        return None

    xs = [p[0] for p in wrists]
    ys = [p[1] for p in wrists]
    cx1, cx2 = min(xs) - side / 2, max(xs) + side / 2
    cy1, cy2 = min(ys) - side / 2, max(ys) + side / 2

    cx1, cy1 = max(0, int(cx1)), max(0, int(cy1))
    cx2, cy2 = min(w, int(cx2)), min(h, int(cy2))
    if cx2 - cx1 < 16 or cy2 - cy1 < 16:
        return None
    return image_bgr[cy1:cy2, cx1:cx2]


class ClipActivityClassifier:
    """Clasifica un frame BGR. Sin dependencia de PyTorch en runtime."""

    def __init__(
        self,
        quantized: bool = True,
        pad_square: bool = True,
        models_dir: Path | None = None,
    ) -> None:
        self.pad_square = pad_square
        models_dir = models_dir or MODELS_DIR
        model_file = "clip_image_encoder_int8.onnx" if quantized else "clip_image_encoder.onnx"
        model_path = models_dir / model_file
        if not model_path.exists():
            raise FileNotFoundError(
                f"No se encontró {model_path}. "
                "Corré export_onnx.py en la máquina de desarrollo primero."
            )

        opts = ort.SessionOptions()
        # 2 threads: deja los otros 2 cores del Pi 5 para YOLO
        opts.intra_op_num_threads = 2
        opts.inter_op_num_threads = 1
        self._session = ort.InferenceSession(
            str(model_path), opts, providers=["CPUExecutionProvider"]
        )
        self._input_name: str = self._session.get_inputs()[0].name

        self._text_features: np.ndarray = np.load(models_dir / "text_features.npy")
        self._labels: list[str] = json.loads((models_dir / "labels.json").read_text())

        # Linear probe entrenado sobre categorías gruesas (ver
        # train_category_probe.py) — reemplaza la agregación zero-shot para
        # 'category' cuando está disponible, porque generaliza mejor sobre
        # datos nunca vistos (61-65% vs ~46% zero-shot en la misma evaluación,
        # ver vlm/README.md). Las clases finas ('scores', 'activity_label')
        # siguen siendo zero-shot: el probe solo predice a nivel de categoría.
        probe_path = models_dir / "category_probe.npz"
        self._probe: dict | None = None
        if probe_path.exists():
            probe_data = np.load(probe_path, allow_pickle=True)
            self._probe = {
                "W": probe_data["W"],
                "b": probe_data["b"],
                "classes": [str(c) for c in probe_data["classes"]],
            }

    def _preprocess(self, image_bgr: np.ndarray) -> np.ndarray:
        # Pad a cuadrado antes del resize: un resize directo a 224x224
        # deforma la imagen, y los recortes de persona parada (~1:3) quedan
        # irreconocibles al aplastarse. El relleno con el color medio de
        # CLIP se normaliza a ~0, así que no aporta señal.
        if self.pad_square:
            h, w = image_bgr.shape[:2]
            if h != w:
                side = max(h, w)
                canvas = np.empty((side, side, 3), dtype=image_bgr.dtype)
                canvas[:] = _PAD_COLOR_BGR
                y0, x0 = (side - h) // 2, (side - w) // 2
                canvas[y0 : y0 + h, x0 : x0 + w] = image_bgr
                image_bgr = canvas
        pil = Image.fromarray(image_bgr[..., ::-1]).resize((224, 224), Image.BICUBIC)
        img = np.array(pil, dtype=np.float32) / 255.0
        img = (img - _MEAN) / _STD
        return img.transpose(2, 0, 1)[np.newaxis]  # (1, 3, 224, 224)

    def encode_image(self, image_bgr: np.ndarray) -> np.ndarray:
        """Devuelve el embedding normalizado (512,) de un frame BGR — expuesto
        para reusar el mismo encoder ONNX ya cargado desde otros clasificadores
        zero-shot con su propio set de prompts (ver
        `fall_confirmation.py:FallConfirmationClassifier`), sin cargar un
        segundo modelo en memoria."""
        tensor = self._preprocess(image_bgr)
        image_features: np.ndarray = self._session.run(
            None, {self._input_name: tensor}
        )[0][0]
        return image_features / np.linalg.norm(image_features)

    def classify(self, image_bgr: np.ndarray, hand_crop_bgr: np.ndarray | None = None) -> dict:
        image_features = self.encode_image(image_bgr)

        logits = 100.0 * image_features @ self._text_features.T
        probs = _softmax(logits)

        top_idx = int(probs.argmax())
        top_label = self._labels[top_idx]
        scores = {label: float(s) for label, s in zip(self._labels, probs)}

        zeroshot_category_scores = aggregate_to_categories(scores)
        is_background = max(zeroshot_category_scores, key=zeroshot_category_scores.get) == "Background"

        # El probe no tiene clase Background (no hay datos de entrenamiento
        # para "sin persona visible"): solo se usa cuando el zero-shot no cree
        # que el recorte esté vacío. Si no, se mantiene la agregación
        # zero-shot para no perder la detección de ventanas sin persona.
        if self._probe is not None and not is_background:
            # El probe se entrena sobre [body_embedding, hand_embedding]
            # (ver train_category_probe.py) — si no hay crop de manos
            # disponible (caller sin pose, ej. vlm/main.py batch), se
            # duplica el body embedding, igual que hace
            # extract_clip_features.py en el mismo caso, para no alimentar
            # al probe con una distribución de input distinta a la de
            # entrenamiento.
            hand_features = (
                self.encode_image(hand_crop_bgr) if hand_crop_bgr is not None else image_features
            )
            probe_input = np.concatenate([image_features, hand_features])
            probe_logits = probe_input @ self._probe["W"].T + self._probe["b"]
            probe_probs = _softmax(probe_logits)
            category_scores = dict(zip(self._probe["classes"], (float(p) for p in probe_probs)))
        else:
            category_scores = zeroshot_category_scores
        top_category = max(category_scores, key=category_scores.get)

        return {
            "activity_label": top_label,
            "description": ALL_PROMPTS[top_label][0],
            "confidence": float(probs[top_idx]),
            "scores": scores,
            "category": top_category,
            "category_confidence": category_scores[top_category],
            "category_scores": category_scores,
            # Categoría por agregación zero-shot, calculada siempre (aunque
            # se use el probe para 'category') — permite un doble check
            # barato entre dos señales independientes del mismo embedding,
            # ver "Doble check probe/zero-shot" en runner/README.md.
            "zeroshot_category_scores": zeroshot_category_scores,
        }
