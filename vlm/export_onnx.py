"""Exporta CLIP a ONNX y pre-computa text features. Corre UNA VEZ en la máquina de desarrollo.

Produce en vlm/models/:
  - clip_image_encoder.onnx       (float32, para verificar)
  - clip_image_encoder_int8.onnx  (INT8 cuantizado, para el Pi)
  - text_features.npy             (float32, shape: num_classes x 512)
  - labels.json                   (lista de labels en el mismo orden)

Uso:
    pip install -r requirements-export.txt
    python export_onnx.py               # exporta todo
    python export_onnx.py --text-only   # solo regenera text_features.npy y
                                        # labels.json (tras cambiar prompts)
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import clip
import numpy as np
import torch

from src.clip_classifier import ALL_PROMPTS

MODELS_DIR = Path(__file__).parent / "models"
MODELS_DIR.mkdir(exist_ok=True)

MODEL_NAME = "ViT-B/32"


class _ImageEncoderWrapper(torch.nn.Module):
    """Wrapper para exportar solo el encoder visual de CLIP a ONNX."""

    def __init__(self, clip_model: torch.nn.Module) -> None:
        super().__init__()
        self.visual = clip_model.visual

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        return self.visual(image.float())


def export_image_encoder(model: torch.nn.Module) -> None:
    from onnxruntime.quantization import QuantType, quantize_dynamic

    wrapper = _ImageEncoderWrapper(model)
    wrapper.eval()
    dummy = torch.zeros(1, 3, 224, 224)
    fp32_path = str(MODELS_DIR / "clip_image_encoder.onnx")
    torch.onnx.export(
        wrapper,
        dummy,
        fp32_path,
        input_names=["image"],
        output_names=["features"],
        opset_version=18,
    )
    print(f"  FP32 exportado → {fp32_path}")

    int8_path = str(MODELS_DIR / "clip_image_encoder_int8.onnx")
    quantize_dynamic(fp32_path, int8_path, weight_type=QuantType.QInt8)
    print(f"  INT8 cuantizado → {int8_path}")


def export_text_features(model: torch.nn.Module) -> None:
    labels = list(ALL_PROMPTS.keys())
    class_features: list[np.ndarray] = []
    with torch.no_grad():
        for prompts in ALL_PROMPTS.values():
            tokens = clip.tokenize(prompts)
            feats = model.encode_text(tokens).float()
            feats /= feats.norm(dim=-1, keepdim=True)
            mean_feat = feats.mean(dim=0)
            mean_feat /= mean_feat.norm()
            class_features.append(mean_feat.numpy())

    text_features = np.stack(class_features).astype(np.float32)
    np.save(MODELS_DIR / "text_features.npy", text_features)
    (MODELS_DIR / "labels.json").write_text(json.dumps(labels, ensure_ascii=False))
    print(f"  Text features → {MODELS_DIR / 'text_features.npy'} {text_features.shape}")
    print(f"  Labels        → {MODELS_DIR / 'labels.json'}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--text-only",
        action="store_true",
        help="Solo regenerar text_features.npy y labels.json (los prompts "
        "cambian seguido; el image encoder no)",
    )
    args = parser.parse_args()

    print(f"Cargando {MODEL_NAME} en CPU...")
    model, _ = clip.load(MODEL_NAME, device="cpu")
    model.eval()

    if not args.text_only:
        print("Exportando image encoder...")
        export_image_encoder(model)

    print("Pre-computando text features con prompt ensembling...")
    export_text_features(model)

    print("Listo. Copiá vlm/models/ al Pi antes de correr main.py.")


if __name__ == "__main__":
    main()
