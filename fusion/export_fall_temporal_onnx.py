"""Exporta el GRU de caída entrenado (`train_fall_temporal.py`) a ONNX,
cuantizado a INT8 para la Raspberry Pi — mismo patrón que
`export_temporal_onnx.py` / `vlm/export_onnx.py`.

Produce en fusion/models/:
  - fall_temporal.onnx        (float32, para verificar)
  - fall_temporal_int8.onnx   (INT8 cuantizado, para el Pi)
  - fall_temporal_labels.json (labels en el mismo orden que la salida)

Uso:
    python export_fall_temporal_onnx.py [--name fall_temporal]

`--name`: nombre base del checkpoint (ver `train_fall_temporal.py
--output-name`); los tres archivos de salida llevan ese mismo nombre base.
"""
from __future__ import annotations

import argparse
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
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", default="fall_temporal")
    name = parser.parse_args().name
    ckpt_path = MODELS_DIR / f"{name}.pt"
    if not ckpt_path.exists():
        raise SystemExit(f"No se encontró {ckpt_path}. Corré train_fall_temporal.py primero.")
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

    dummy = torch.zeros(1, 16, ckpt["input_size"])  # (batch, n_frames, feat_dim); n_frames dinámico
    fp32_path = MODELS_DIR / f"{name}.onnx"
    torch.onnx.export(
        model,
        dummy,
        str(fp32_path),
        input_names=["window"],
        output_names=["logits"],
        dynamic_axes={"window": {0: "batch", 1: "seq_len"}, "logits": {0: "batch"}},
        opset_version=18,
        dynamo=False,  # ver nota en export_temporal_onnx.py (el exportador nuevo rompe al cuantizar)
    )
    print(f"FP32 exportado → {fp32_path}")

    int8_path = MODELS_DIR / f"{name}_int8.onnx"
    quantize_dynamic(str(fp32_path), str(int8_path), weight_type=QuantType.QInt8)
    print(f"INT8 cuantizado → {int8_path}")

    labels_path = MODELS_DIR / f"{name}_labels.json"
    labels_path.write_text(json.dumps(ckpt["labels"], ensure_ascii=False))
    print(f"Labels → {labels_path}")

    print(f"\nListo. Copiá {MODELS_DIR} al Pi (junto con vlm/models/) antes de correr runner/.")


if __name__ == "__main__":
    main()
