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
}


def _softmax(x: np.ndarray) -> np.ndarray:
    x = x - x.max()
    e = np.exp(x)
    return e / e.sum()


class ClipActivityClassifier:
    """Clasifica un frame BGR. Sin dependencia de PyTorch en runtime."""

    def __init__(self, quantized: bool = True) -> None:
        model_file = "clip_image_encoder_int8.onnx" if quantized else "clip_image_encoder.onnx"
        model_path = MODELS_DIR / model_file
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

        self._text_features: np.ndarray = np.load(MODELS_DIR / "text_features.npy")
        self._labels: list[str] = json.loads((MODELS_DIR / "labels.json").read_text())

    def _preprocess(self, image_bgr: np.ndarray) -> np.ndarray:
        pil = Image.fromarray(image_bgr[..., ::-1]).resize((224, 224), Image.BICUBIC)
        img = np.array(pil, dtype=np.float32) / 255.0
        img = (img - _MEAN) / _STD
        return img.transpose(2, 0, 1)[np.newaxis]  # (1, 3, 224, 224)

    def classify(self, image_bgr: np.ndarray) -> dict:
        tensor = self._preprocess(image_bgr)
        image_features: np.ndarray = self._session.run(
            None, {self._input_name: tensor}
        )[0][0]
        image_features = image_features / np.linalg.norm(image_features)

        logits = 100.0 * image_features @ self._text_features.T
        probs = _softmax(logits)

        top_idx = int(probs.argmax())
        top_label = self._labels[top_idx]
        return {
            "activity_label": top_label,
            "description": ACTIVITY_PROMPTS[top_label][0],
            "confidence": float(probs[top_idx]),
            "scores": {label: float(s) for label, s in zip(self._labels, probs)},
        }
