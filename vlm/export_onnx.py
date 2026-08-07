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
from src.fall_confirmation import FALL_PROMPTS, NOTFALL_PROMPTS


def models_dir_for(model_name: str) -> Path:
    """ViT-B/32 (el modelo de producción) usa vlm/models/, para no romper el
    Pi ni runner/. Cualquier otro modelo (ej. ViT-B/16 en evaluación) usa una
    carpeta separada, para poder comparar sin pisar el modelo de producción."""
    slug = "models" if model_name == "ViT-B/32" else "models_" + model_name.replace("/", "-").lower()
    d = Path(__file__).parent / slug
    d.mkdir(exist_ok=True)
    return d


class _ImageEncoderWrapper(torch.nn.Module):
    """Wrapper para exportar solo el encoder visual de CLIP a ONNX."""

    def __init__(self, clip_model: torch.nn.Module) -> None:
        super().__init__()
        self.visual = clip_model.visual

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        return self.visual(image.float())


def export_image_encoder(model: torch.nn.Module, models_dir: Path) -> None:
    from onnxruntime.quantization import QuantType, quantize_dynamic

    wrapper = _ImageEncoderWrapper(model)
    wrapper.eval()
    dummy = torch.zeros(1, 3, 224, 224)
    fp32_path = str(models_dir / "clip_image_encoder.onnx")
    torch.onnx.export(
        wrapper,
        dummy,
        fp32_path,
        input_names=["image"],
        output_names=["features"],
        opset_version=18,
    )
    print(f"  FP32 exportado → {fp32_path}")

    int8_path = str(models_dir / "clip_image_encoder_int8.onnx")
    quantize_dynamic(fp32_path, int8_path, weight_type=QuantType.QInt8)
    print(f"  INT8 cuantizado → {int8_path}")


def export_text_features(model: torch.nn.Module, models_dir: Path) -> None:
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
    np.save(models_dir / "text_features.npy", text_features)
    (models_dir / "labels.json").write_text(json.dumps(labels, ensure_ascii=False))
    print(f"  Text features → {models_dir / 'text_features.npy'} {text_features.shape}")
    print(f"  Labels        → {models_dir / 'labels.json'}")


def export_fall_confirmation_features(model: torch.nn.Module, models_dir: Path) -> None:
    """Text features para FallConfirmationClassifier (ver src/fall_confirmation.py):
    fila 0 = FALL_PROMPTS promediados, fila 1 = NOTFALL_PROMPTS promediados."""
    with torch.no_grad():
        feats = []
        for prompts in (FALL_PROMPTS, NOTFALL_PROMPTS):
            tokens = clip.tokenize(prompts)
            f = model.encode_text(tokens).float()
            f /= f.norm(dim=-1, keepdim=True)
            mean_feat = f.mean(dim=0)
            mean_feat /= mean_feat.norm()
            feats.append(mean_feat.numpy())
    out = np.stack(feats).astype(np.float32)
    np.save(models_dir / "fall_confirmation_features.npy", out)
    print(f"  Fall confirmation features → {models_dir / 'fall_confirmation_features.npy'} {out.shape}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--text-only",
        action="store_true",
        help="Solo regenerar text_features.npy y labels.json (los prompts "
        "cambian seguido; el image encoder no)",
    )
    parser.add_argument(
        "--model",
        default="ViT-B/32",
        help="Variante de CLIP a exportar (ej. ViT-B/16). ViT-B/32 pisa "
        "vlm/models/ (el que usa el Pi); cualquier otra queda en su propia "
        "carpeta vlm/models_<variante>/ para comparar sin romper producción.",
    )
    args = parser.parse_args()
    models_dir = models_dir_for(args.model)

    print(f"Cargando {args.model} en CPU...")
    model, _ = clip.load(args.model, device="cpu")
    model.eval()

    if not args.text_only:
        print("Exportando image encoder...")
        export_image_encoder(model, models_dir)

    print("Pre-computando text features con prompt ensembling...")
    export_text_features(model, models_dir)

    if args.model == "ViT-B/32":
        print("Pre-computando text features de confirmación de caída...")
        export_fall_confirmation_features(model, models_dir)

    print(f"Listo. Copiá {models_dir} al Pi antes de correr main.py (si es el modelo de producción).")


if __name__ == "__main__":
    main()
