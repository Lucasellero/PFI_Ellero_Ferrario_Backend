"""Compara los tres enfoques de clasificación de actividad — CLIP solo,
esqueleto solo, híbrido— sobre el mismo cache de `extract_dataset.py`:
accuracy por video y matriz de confusión por categoría para cada uno (la
comparación empírica central de esta línea de trabajo de la tesis).

IMPORTANTE: para que la comparación sea honesta, `<features.npz>` tiene que
venir de videos/participantes que el clasificador de esqueleto
(`temporal_classifier.pt`) y el probe de CLIP (`category_probe.npz`) NO
vieron en entrenamiento — mismo cuidado que ya toma
`vlm/evaluate_activity.py` al evaluar contra una carpeta separada de la de
entrenamiento. Este script no vuelve a separar train/val: evalúa el cache
completo que se le pase, tal cual, contra los modelos ya entrenados en
`fusion/models/` y `vlm/models/`.

Uso:
    python evaluate_hybrid.py <features.npz>
                               [--fusion-method weighted_average|mlp]
                               [--skeleton-weight 0.5]
                               [--skeleton-weight-sweep]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import torch

from vlm.src.clip_classifier import COARSE_CATEGORIES as CLIP_COARSE_CATEGORIES
from vlm.src.clip_classifier import LABEL_TO_CATEGORY as CLIP_LABEL_TO_CATEGORY

from fusion.src.fusion import ScoreFuser, fine_scores_to_category_scores
from fusion.src.labels import COARSE_CATEGORIES, LABEL_TO_CATEGORY
from fusion.src.temporal_model import GRUActivityClassifier

parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument("features", help="Cache producido por extract_dataset.py")
parser.add_argument("--fusion-method", choices=["weighted_average", "mlp"], default="weighted_average")
parser.add_argument("--skeleton-weight", type=float, default=0.5, help="Solo para fusion-method=weighted_average")
parser.add_argument(
    "--skeleton-weight-sweep", action="store_true",
    help="Ignora --skeleton-weight y reporta accuracy híbrida para varios pesos (0.0 a 1.0), para elegir el mejor a mano",
)
args = parser.parse_args()

ROOT_VLM_MODELS = ROOT / "vlm" / "models"
FUSION_MODELS = Path(__file__).parent / "models"

data = np.load(args.features, allow_pickle=True)
windows = data["windows"]
body_X = data["body_features"]
hand_X = data["hand_features"]
y_fine = data["labels"]
videos = data["videos"]

# La categoría real de cada muestra usa el mapeo extendido de fusion/ (con
# Standing) — es un superset del de vlm/, así que no pierde nada para las
# clases que también existen ahí.
y_cat = np.array([LABEL_TO_CATEGORY[lbl] for lbl in y_fine])

ALL_CATEGORIES = sorted(set(COARSE_CATEGORIES) | set(CLIP_COARSE_CATEGORIES))


# ---------------------------------------------------------------- esqueleto
ckpt_path = FUSION_MODELS / "temporal_classifier.pt"
if not ckpt_path.exists():
    raise SystemExit(f"No se encontró {ckpt_path}. Corré train_temporal_classifier.py primero.")
ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
skeleton_model = GRUActivityClassifier(
    input_size=ckpt["input_size"], num_classes=len(ckpt["labels"]),
    hidden_size=ckpt["hidden_size"], num_layers=ckpt["num_layers"], dropout=ckpt["dropout"],
)
skeleton_model.load_state_dict(ckpt["state_dict"])
skeleton_model.eval()
with torch.no_grad():
    skeleton_logits = skeleton_model(torch.from_numpy(windows).float()).numpy()
skeleton_probs = np.exp(skeleton_logits - skeleton_logits.max(axis=1, keepdims=True))
skeleton_probs /= skeleton_probs.sum(axis=1, keepdims=True)
skeleton_fine_labels = ckpt["labels"]
skeleton_cat_scores_all = [
    fine_scores_to_category_scores(dict(zip(skeleton_fine_labels, row)), LABEL_TO_CATEGORY)
    for row in skeleton_probs
]

# ---------------------------------------------------------------------- CLIP
text_features = np.load(ROOT_VLM_MODELS / "text_features.npy")
clip_fine_labels = json.loads((ROOT_VLM_MODELS / "labels.json").read_text())
clip_logits = 100.0 * body_X @ text_features.T
clip_probs = np.exp(clip_logits - clip_logits.max(axis=1, keepdims=True))
clip_probs /= clip_probs.sum(axis=1, keepdims=True)

probe_path = ROOT_VLM_MODELS / "category_probe.npz"
use_probe = probe_path.exists()
if use_probe:
    probe = np.load(probe_path, allow_pickle=True)
    probe_input = np.concatenate([body_X, hand_X], axis=1)
    probe_logits = probe_input @ probe["W"].T + probe["b"]
    probe_probs = np.exp(probe_logits - probe_logits.max(axis=1, keepdims=True))
    probe_probs /= probe_probs.sum(axis=1, keepdims=True)
    probe_categories = [str(c) for c in probe["classes"]]
    clip_cat_scores_all = [dict(zip(probe_categories, row)) for row in probe_probs]
    print("CLIP: usando el linear probe (vlm/models/category_probe.npz)")
else:
    clip_cat_scores_all = [
        fine_scores_to_category_scores(dict(zip(clip_fine_labels, row)), CLIP_LABEL_TO_CATEGORY)
        for row in clip_probs
    ]
    print("CLIP: category_probe.npz no encontrado, usando agregación zero-shot")


def evaluate(name: str, per_sample_scores: list[dict[str, float]]) -> None:
    """Agrega por video (promedio de las ventanas de ese video, mismo
    criterio que vlm/evaluate_activity.py) y reporta accuracy + matriz de
    confusión por categoría."""
    unique_videos = np.unique(videos)
    y_true, y_pred = [], []
    for v in unique_videos:
        mask = videos == v
        cats = sorted({c for i in np.where(mask)[0] for c in per_sample_scores[i]})
        avg = {c: float(np.mean([per_sample_scores[i].get(c, 0.0) for i in np.where(mask)[0]])) for c in cats}
        pred_cat = max(avg, key=avg.get)
        y_true.append(y_cat[mask][0])
        y_pred.append(pred_cat)

    y_true, y_pred = np.array(y_true), np.array(y_pred)
    acc = float((y_true == y_pred).mean())
    print(f"\n=== {name}: {acc:.1%} accuracy ({len(unique_videos)} videos) ===")

    cats_present = [c for c in ALL_CATEGORIES if (y_true == c).any()]
    header = "Real \\ Predicho".ljust(26) + "".join(c[:10].ljust(12) for c in cats_present)
    print(header)
    for real_cat in cats_present:
        row_mask = y_true == real_cat
        n = int(row_mask.sum())
        counts = {c: int(((y_pred == c) & row_mask).sum()) for c in cats_present}
        row = real_cat.ljust(26) + "".join(f"{counts[c]}/{n}".ljust(12) for c in cats_present)
        print(row)


evaluate("CLIP solo", clip_cat_scores_all)
evaluate("Esqueleto solo", skeleton_cat_scores_all)

if args.skeleton_weight_sweep:
    print("\n--- Barrido de skeleton_weight (híbrido, weighted_average) ---")
    for w in np.linspace(0.0, 1.0, 11):
        fuser = ScoreFuser(method="weighted_average", skeleton_weight=float(w))
        fused = [fuser.fuse(sk, cl) for sk, cl in zip(skeleton_cat_scores_all, clip_cat_scores_all)]
        unique_videos = np.unique(videos)
        y_true, y_pred = [], []
        for v in unique_videos:
            mask = videos == v
            cats = sorted({c for i in np.where(mask)[0] for c in fused[i]})
            avg = {c: float(np.mean([fused[i].get(c, 0.0) for i in np.where(mask)[0]])) for c in cats}
            y_true.append(y_cat[mask][0])
            y_pred.append(max(avg, key=avg.get))
        acc = float((np.array(y_true) == np.array(y_pred)).mean())
        print(f"skeleton_weight={w:.1f}  accuracy={acc:.1%}")
else:
    fuser = ScoreFuser(method=args.fusion_method, skeleton_weight=args.skeleton_weight)
    fused = [fuser.fuse(sk, cl) for sk, cl in zip(skeleton_cat_scores_all, clip_cat_scores_all)]
    label = f"Híbrido ({args.fusion_method}" + (
        f", skeleton_weight={args.skeleton_weight})" if args.fusion_method == "weighted_average" else ")"
    )
    evaluate(label, fused)
