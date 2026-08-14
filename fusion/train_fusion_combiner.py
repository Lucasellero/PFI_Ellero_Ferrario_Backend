"""Entrena el combinador de fusión "mlp" (`ScoreFuser(method="mlp")` en
`src/fusion.py`): un logistic regression chico (mismo enfoque que el
"linear probe" de `vlm/train_category_probe.py`) sobre
`[scores_esqueleto, scores_clip]` concatenados → categoría real, para poder
aprender por-categoría cuánta confianza darle a cada señal en vez de un
peso fijo global (`ScoreFuser(method="weighted_average")`).

Requiere que ya exista `fusion/models/temporal_classifier.pt`
(`train_temporal_classifier.py`) y `vlm/models/text_features.npy` (y,
opcional, `vlm/models/category_probe.npz` — ver `vlm/README.md`).

Uso:
    python train_fusion_combiner.py [--features dataset_features.npz] [--C 1.0] [--seeds 5]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupShuffleSplit

from vlm.src.clip_classifier import COARSE_CATEGORIES as CLIP_COARSE_CATEGORIES
from vlm.src.clip_classifier import LABEL_TO_CATEGORY as CLIP_LABEL_TO_CATEGORY

from fusion.src.fusion import fine_scores_to_category_scores
from fusion.src.labels import COARSE_CATEGORIES, FINE_LABELS, LABEL_TO_CATEGORY
from fusion.src.temporal_model import GRUActivityClassifier

parser = argparse.ArgumentParser()
parser.add_argument("--features", default=None, help="Default: fusion/dataset_features.npz")
parser.add_argument("--C", type=float, default=1.0, help="Inverso de regularización L2")
parser.add_argument("--seeds", type=int, default=5)
args = parser.parse_args()

MODELS_DIR = Path(__file__).parent / "models"
ROOT_VLM_MODELS = ROOT / "vlm" / "models"

features_path = Path(args.features) if args.features else Path(__file__).parent / "dataset_features.npz"
data = np.load(features_path, allow_pickle=True)
windows = data["windows"]
body_X = data["body_features"]
hand_X = data["hand_features"]
y_fine = data["labels"]
groups = data["participants"]
videos = data["videos"]
y_cat = np.array([LABEL_TO_CATEGORY[lbl] for lbl in y_fine])

# --- Señal de esqueleto: correr el GRU entrenado sobre las ventanas cacheadas ---
ckpt_path = MODELS_DIR / "temporal_classifier.pt"
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

# --- Señal de CLIP: zero-shot (o probe si está entrenado) sobre los embeddings cacheados ---
text_features = np.load(ROOT_VLM_MODELS / "text_features.npy")
import json
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

# Espacio de categorías de entrada al combiner: unión de las de esqueleto
# (incluye Standing) y CLIP (no la incluye) — orden fijo, se guarda en el
# .npz exportado para que ScoreFuser sepa reconstruirlo en inferencia.
input_categories = sorted(set(COARSE_CATEGORIES) | set(CLIP_COARSE_CATEGORIES))

X_rows = []
for i in range(len(y_fine)):
    sk_scores = dict(zip(skeleton_fine_labels, skeleton_probs[i]))
    sk_cat = fine_scores_to_category_scores(sk_scores, LABEL_TO_CATEGORY)

    if use_probe:
        cl_cat = dict(zip(probe_categories, probe_probs[i]))
    else:
        cl_scores = dict(zip(clip_fine_labels, clip_probs[i]))
        cl_cat = fine_scores_to_category_scores(cl_scores, CLIP_LABEL_TO_CATEGORY)

    row = [sk_cat.get(c, 0.0) for c in input_categories] + [cl_cat.get(c, 0.0) for c in input_categories]
    X_rows.append(row)
X = np.array(X_rows, dtype=np.float32)

unique_videos = np.unique(videos)
video_group = np.array([groups[videos == v][0] for v in unique_videos])
video_cat = np.array([y_cat[videos == v][0] for v in unique_videos])

accs = []
for seed in range(args.seeds):
    splitter = GroupShuffleSplit(n_splits=1, test_size=0.25, random_state=seed)
    train_idx, val_idx = next(splitter.split(unique_videos, video_cat, video_group))
    train_videos, val_videos = set(unique_videos[train_idx]), set(unique_videos[val_idx])
    train_mask = np.array([v in train_videos for v in videos])

    if train_mask.sum() == 0 or (~train_mask).sum() == 0:
        continue
    clf = LogisticRegression(max_iter=2000, C=args.C, class_weight="balanced")
    clf.fit(X[train_mask], y_cat[train_mask])

    correct = total = 0
    for v in val_videos:
        mask = videos == v
        pred = clf.predict_proba(X[mask]).mean(axis=0)
        pred_cat = clf.classes_[pred.argmax()]
        correct += pred_cat == y_cat[mask][0]
        total += 1
    if total:
        acc = correct / total
        accs.append(acc)
        print(f"seed={seed}  {total} videos val  fusion-mlp={acc:.1%}")

if accs:
    print(f"\n=== Promedio {len(accs)} seeds, por video ===")
    print(f"fusion-mlp: {np.mean(accs):.1%}  (std {np.std(accs):.1%})")

final_clf = LogisticRegression(max_iter=2000, C=args.C, class_weight="balanced")
final_clf.fit(X, y_cat)

out_path = MODELS_DIR / "fusion_combiner.npz"
np.savez(
    out_path,
    W=final_clf.coef_.astype(np.float32),
    b=final_clf.intercept_.astype(np.float32),
    categories=np.array(final_clf.classes_),
    input_categories=np.array(input_categories),
)
print(f"\nGuardado {out_path} (entrenado sobre {X.shape[0]} ventanas, {len(unique_videos)} videos)")
