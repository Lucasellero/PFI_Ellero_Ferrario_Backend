"""Clasificación de actividad a partir de una ventana de keypoints, vía
ONNX Runtime. Sin dependencia de PyTorch en runtime — mismo patrón que
`vlm/src/clip_classifier.py` (ver "Setup" en `vlm/README.md`).

El modelo se entrena en `../train_temporal_classifier.py` y se exporta a
ONNX con `../export_temporal_onnx.py`.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import onnxruntime as ort

MODELS_DIR = Path(__file__).resolve().parents[1] / "models"


def _softmax(x: np.ndarray) -> np.ndarray:
    x = x - x.max()
    e = np.exp(x)
    return e / e.sum()


class SkeletonActivityClassifier:
    """Clasifica una ventana `(n_frames, feature_dim)` de
    `keypoint_sequence.SkeletonSequenceBuffer.build_window()`.

    La normalización de input queda horneada dentro del grafo ONNX (ver
    `src/temporal_model.py::GRUActivityClassifier`) — acá no hace falta
    ningún preprocesado más que armar la ventana.
    """

    def __init__(self, quantized: bool = True, models_dir: Path | None = None) -> None:
        models_dir = models_dir or MODELS_DIR
        model_file = "temporal_classifier_int8.onnx" if quantized else "temporal_classifier.onnx"
        model_path = models_dir / model_file
        if not model_path.exists():
            raise FileNotFoundError(
                f"No se encontró {model_path}. "
                "Corré train_temporal_classifier.py + export_temporal_onnx.py primero."
            )
        opts = ort.SessionOptions()
        # 2 threads: mismo criterio que ClipActivityClassifier, deja cores
        # libres para YOLO-pose y CLIP corriendo en paralelo en el Pi.
        opts.intra_op_num_threads = 2
        opts.inter_op_num_threads = 1
        self._session = ort.InferenceSession(str(model_path), opts, providers=["CPUExecutionProvider"])
        self._input_name: str = self._session.get_inputs()[0].name
        self._labels: list[str] = json.loads((models_dir / "temporal_labels.json").read_text())

    @property
    def labels(self) -> list[str]:
        return list(self._labels)

    def classify(self, window: np.ndarray) -> dict:
        """`window`: `(n_frames, feature_dim)` float32."""
        batch = window.astype(np.float32)[np.newaxis]  # (1, n_frames, feat_dim)
        logits = self._session.run(None, {self._input_name: batch})[0][0]
        probs = _softmax(logits)
        top_idx = int(probs.argmax())
        return {
            "fine_label": self._labels[top_idx],
            "confidence": float(probs[top_idx]),
            "scores": {label: float(p) for label, p in zip(self._labels, probs)},
        }
