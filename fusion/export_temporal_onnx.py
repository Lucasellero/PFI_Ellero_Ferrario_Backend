"""Exporta el GRU entrenado (`train_temporal_classifier.py`) a ONNX,
cuantizado a INT8 para la Raspberry Pi — mismo patrón que
`vlm/export_onnx.py` (ver `vlm/README.md`, sección "Setup").

Produce en fusion/models/:
  - temporal_classifier.onnx       (float32, para verificar)
  - temporal_classifier_int8.onnx  (INT8 cuantizado, para el Pi)
  - temporal_labels.json           (lista de labels en el mismo orden que la salida)

Uso:
    python export_temporal_onnx.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch
from onnxruntime.quantization import QuantType, quantize_dynamic

from fusion.src.temporal_model import GRUActivityClassifier

MODELS_DIR = Path(__file__).parent / "models"


def main() -> None:
    ckpt_path = MODELS_DIR / "temporal_classifier.pt"
    if not ckpt_path.exists():
        raise SystemExit(f"No se encontró {ckpt_path}. Corré train_temporal_classifier.py primero.")
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)

    model = GRUActivityClassifier(
        input_size=ckpt["input_size"],
        num_classes=len(ckpt["labels"]),
        hidden_size=ckpt["hidden_size"],
        num_layers=ckpt["num_layers"],
        dropout=ckpt["dropout"],
    )
    model.load_state_dict(ckpt["state_dict"])
    model.eval()

    dummy = torch.zeros(1, 16, ckpt["input_size"])  # (batch, n_frames, feat_dim); n_frames dinámico, ver dynamic_axes
    fp32_path = MODELS_DIR / "temporal_classifier.onnx"
    torch.onnx.export(
        model,
        dummy,
        str(fp32_path),
        input_names=["window"],
        output_names=["logits"],
        dynamic_axes={"window": {0: "batch", 1: "seq_len"}, "logits": {0: "batch"}},
        opset_version=18,
        # dynamo=False: el exportador nuevo (dynamo=True, default desde
        # torch 2.x reciente) genera, para este modelo con GRU +
        # dynamic_axes, un grafo cuya inferencia de shapes falla al
        # cuantizar (InferenceError en quantize_dynamic más abajo). El
        # exportador clásico (TorchScript-based, el mismo que usa
        # vlm/export_onnx.py) no tiene ese problema.
        dynamo=False,
    )
    print(f"FP32 exportado → {fp32_path}")

    int8_path = MODELS_DIR / "temporal_classifier_int8.onnx"
    quantize_dynamic(str(fp32_path), str(int8_path), weight_type=QuantType.QInt8)
    print(f"INT8 cuantizado → {int8_path}")

    labels_path = MODELS_DIR / "temporal_labels.json"
    labels_path.write_text(json.dumps(ckpt["labels"], ensure_ascii=False))
    print(f"Labels → {labels_path}")

    print(f"\nListo. Copiá {MODELS_DIR} al Pi (junto con vlm/models/) antes de correr runner/.")


if __name__ == "__main__":
    main()
