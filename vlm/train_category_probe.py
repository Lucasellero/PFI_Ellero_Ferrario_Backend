"""Entrena un linear probe (logistic regression) sobre embeddings CLIP
congelados, directamente a nivel de categoría gruesa (COARSE_CATEGORIES),
usando el cache de extract_clip_features.py. Evalúa por video (promediando
los frames cacheados, comparable a la evaluación por ventana de
evaluate_activity.py) con split por participante para no filtrar datos
entre train/val, y con varios seeds para reportar consistencia.

Al final entrena sobre TODO el cache y exporta vlm/models/category_probe.npz,
que ClipActivityClassifier carga automáticamente si existe (ver
src/clip_classifier.py).

Uso:
    python train_category_probe.py [--C 10.0] [--seeds 5] [--features clip_features.npz]
"""
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupShuffleSplit

from vlm.src.clip_classifier import LABEL_TO_CATEGORY, COARSE_CATEGORIES

parser = argparse.ArgumentParser()
parser.add_argument('--C', type=float, default=10.0, help='Inverso de regularización L2')
parser.add_argument('--seeds', type=int, default=5, help='Splits por participante a promediar')
parser.add_argument('--features', default=None, help='Default: vlm/clip_features.npz')
args = parser.parse_args()

features_path = Path(args.features) if args.features else Path(__file__).parent / "clip_features.npz"
data = np.load(features_path, allow_pickle=True)
X, y_fine, groups, videos = data['features'], data['labels'], data['participants'], data['videos']
y_cat = np.array([LABEL_TO_CATEGORY[label] for label in y_fine])

text_features = np.load(ROOT / 'vlm' / 'models' / 'text_features.npy')
zs_labels = json.loads((ROOT / 'vlm' / 'models' / 'labels.json').read_text())

unique_videos = np.unique(videos)
video_group = np.array([groups[videos == v][0] for v in unique_videos])
video_cat = np.array([y_cat[videos == v][0] for v in unique_videos])

zs_accs, lp_accs = [], []
for seed in range(args.seeds):
    splitter = GroupShuffleSplit(n_splits=1, test_size=0.25, random_state=seed)
    train_idx, val_idx = next(splitter.split(unique_videos, video_cat, video_group))
    train_videos, val_videos = set(unique_videos[train_idx]), set(unique_videos[val_idx])

    frame_train_mask = np.array([v in train_videos for v in videos])
    clf = LogisticRegression(max_iter=2000, C=args.C, class_weight='balanced')
    clf.fit(X[frame_train_mask], y_cat[frame_train_mask])
    categories = clf.classes_

    zs_correct = lp_correct = total = 0
    for v in val_videos:
        mask = videos == v
        feats = X[mask]
        true_cat = y_cat[mask][0]

        logits = 100.0 * feats @ text_features.T
        probs = np.exp(logits - logits.max(axis=1, keepdims=True))
        probs /= probs.sum(axis=1, keepdims=True)
        cat_scores = {c: 0.0 for c in COARSE_CATEGORIES}
        for row in probs:
            for label, p in zip(zs_labels, row):
                cat_scores[LABEL_TO_CATEGORY[label]] += p
        zs_pred = max(cat_scores, key=cat_scores.get)

        lp_probs = clf.predict_proba(feats).mean(axis=0)
        lp_pred = categories[lp_probs.argmax()]

        zs_correct += zs_pred == true_cat
        lp_correct += lp_pred == true_cat
        total += 1

    zs_acc, lp_acc = zs_correct / total, lp_correct / total
    zs_accs.append(zs_acc); lp_accs.append(lp_acc)
    print(f'seed={seed}  {total} videos val  zero-shot={zs_acc:.1%}  linear-probe={lp_acc:.1%}')

print(f'\n=== Promedio {args.seeds} seeds, por video (participantes disjuntos train/val) ===')
print(f'zero-shot:    {np.mean(zs_accs):.1%}  (std {np.std(zs_accs):.1%})')
print(f'linear-probe: {np.mean(lp_accs):.1%}  (std {np.std(lp_accs):.1%})')

# --- Entrenamiento final sobre todo el cache, para producción ---
final_clf = LogisticRegression(max_iter=2000, C=args.C, class_weight='balanced')
final_clf.fit(X, y_cat)

out_path = ROOT / 'vlm' / 'models' / 'category_probe.npz'
np.savez(out_path, W=final_clf.coef_.astype(np.float32), b=final_clf.intercept_.astype(np.float32),
         classes=np.array(final_clf.classes_))
print(f'\nGuardado {out_path} (entrenado sobre {X.shape[0]} frames, {len(unique_videos)} videos)')
