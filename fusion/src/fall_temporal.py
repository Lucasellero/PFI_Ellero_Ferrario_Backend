"""Clasificador temporal de caída sobre la ventana de keypoints, vía ONNX
Runtime (sin PyTorch en runtime — mismo patrón que `temporal_classifier.py`
y `vlm/src/clip_classifier.py`).

Segunda etapa del evento de caída, igual que hoy lo es la confirmación CLIP
(`vlm/src/fall_confirmation.py`): la heurística geométrica de
`vision/src/pose_pipeline.py::FallImmobilityDetector` dispara (barato, corre
por frame), y solo entonces se consulta este modelo sobre la ventana de
movimiento de los últimos ~2s. A diferencia de CLIP —que mira la apariencia
de un frame recortado— este modelo mira la *forma del movimiento*, que es la
señal que distingue una caída de sentarse/acostarse a propósito (el modo de
falla estructural documentado en `vision/README.md`).

Se entrena en `../train_fall_temporal.py` (con augmentación de oclusión/
truncamiento, ver ese script) y se exporta con `../export_fall_temporal_onnx.py`.

Clases: `normal` / `caida`. Se probó el esquema de 3 etapas de la literatura
de pre-impacto (`normal` / `caida_en_curso` / `caido`, ConvLSTM 3-stage; LSTM
Normal/Warning/Fall en Raspberry Pi, 99.44%), pero los datasets de caída
disponibles (`GMDCSA-24`, `CAUCAFall`) están recortados al acto de caer y casi
no tienen frames de "ya en el piso" (165 ventanas `caido` vs. 3676
`caida_en_curso` en la extracción) — muy poco para una 3ª clase confiable.
`extract_falls.py` igual emite las etiquetas finas al `.npz` (`caido` se
puede revivir con más `--post-fall-s` o datos post-caída), pero
`train_fall_temporal.py` las colapsa a binario.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import onnxruntime as ort

MODELS_DIR = Path(__file__).resolve().parents[1] / "models"

FALL_LABELS: list[str] = ["normal", "caida"]
# Clases que cuentan como "hay una caída" al confirmar un evento geométrico.
FALL_POSITIVE_LABELS: frozenset[str] = frozenset({"caida"})

# Etiquetas finas que emite extract_falls.py -> etiqueta binaria de entrenamiento.
RAW_TO_LABEL: dict[str, str] = {
    "normal": "normal",
    "caida_en_curso": "caida",
    "caido": "caida",
}


def _softmax(x: np.ndarray) -> np.ndarray:
    x = x - x.max()
    e = np.exp(x)
    return e / e.sum()


class FallTemporalClassifier:
    """Clasifica una ventana `(n_frames, feature_dim)` de
    `keypoint_sequence.SkeletonSequenceBuffer.build_window()`.

    La normalización de input queda horneada en el grafo ONNX (ver
    `src/temporal_model.py::GRUActivityClassifier`) — acá solo hace falta
    armar la ventana con `frame_features`.
    """

    def __init__(self, quantized: bool = True, models_dir: Path | None = None) -> None:
        models_dir = models_dir or MODELS_DIR
        model_file = "fall_temporal_int8.onnx" if quantized else "fall_temporal.onnx"
        model_path = models_dir / model_file
        if not model_path.exists():
            raise FileNotFoundError(
                f"No se encontró {model_path}. "
                "Corré train_fall_temporal.py + export_fall_temporal_onnx.py primero."
            )
        opts = ort.SessionOptions()
        # 2 threads: mismo criterio que ClipActivityClassifier /
        # SkeletonActivityClassifier — deja cores libres para YOLO-pose y CLIP.
        opts.intra_op_num_threads = 2
        opts.inter_op_num_threads = 1
        self._session = ort.InferenceSession(str(model_path), opts, providers=["CPUExecutionProvider"])
        self._input_name: str = self._session.get_inputs()[0].name
        self._labels: list[str] = json.loads((models_dir / "fall_temporal_labels.json").read_text())

    @property
    def labels(self) -> list[str]:
        return list(self._labels)

    def classify(self, window: np.ndarray) -> dict:
        """`window`: `(n_frames, feature_dim)` float32."""
        batch = window.astype(np.float32)[np.newaxis]  # (1, n_frames, feat_dim)
        logits = self._session.run(None, {self._input_name: batch})[0][0]
        probs = _softmax(logits)
        top_idx = int(probs.argmax())
        scores = {label: float(p) for label, p in zip(self._labels, probs)}
        return {
            "label": self._labels[top_idx],
            "confidence": float(probs[top_idx]),
            "is_fall": self._labels[top_idx] in FALL_POSITIVE_LABELS,
            "fall_score": sum(scores.get(lbl, 0.0) for lbl in FALL_POSITIVE_LABELS),
            "scores": scores,
        }
