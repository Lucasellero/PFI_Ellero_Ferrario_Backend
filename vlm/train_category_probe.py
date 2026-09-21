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
                                    [--extra-features omnifall_features.npz,...]
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
parser.add_argument(
    '--extra-features', default=None,
    help='Lista separada por comas de .npz adicionales (mismo esquema: '
         'features, hand_features, labels, participants, videos) a sumar '
         'al cache principal antes de entrenar — ej. omnifall_features.npz '
         '(ver extract_omnifall_features.py) para clases que Toyota '
         'Smarthome no cubre.',
)
parser.add_argument(
    '--with-object-context', action='store_true',
    help='Suma el vector de presencia de objetos COCO (object_features, ver '
         'extract_clip_features.py --object-context / fusion/src/object_context.py) '
         'al input del probe, si está presente en TODOS los .npz cargados. '
         'Ablation: mide si esa señal ayuda sobre CLIP solo.',
)
parser.add_argument(
    '-o', '--output', default=None,
    help='Default: vlm/models/category_probe.npz (el que carga producción). '
         'Usar otra ruta para experimentos/ablations que no deben pisar el '
         'modelo desplegado — ver "no integrar a ciegas" en vlm/README.md.',
)
args = parser.parse_args()

features_path = Path(args.features) if args.features else Path(__file__).parent / "clip_features.npz"
datasets = [np.load(features_path, allow_pickle=True)]
if args.extra_features:
    datasets += [np.load(Path(p.strip()), allow_pickle=True) for p in args.extra_features.split(',')]

y_fine = np.concatenate([d['labels'] for d in datasets])
groups = np.concatenate([d['participants'] for d in datasets])
videos = np.concatenate([d['videos'] for d in datasets])
body_X = np.concatenate([d['features'] for d in datasets])
hand_X = np.concatenate([d['hand_features'] for d in datasets])
# El probe usa cuerpo + manos concatenado (ver crop_to_hands en
# src/clip_classifier.py); el zero-shot (comparativo de abajo) solo cuerpo,
# porque text_features.npy es 512-dim.
X = np.concatenate([body_X, hand_X], axis=1)
if args.with_object_context:
    if not all('object_features' in d for d in datasets):
        raise SystemExit(
            '--with-object-context pero algún .npz no tiene object_features '
            '— re-extraer con extract_clip_features.py --object-context.'
        )
    obj_X = np.concatenate([d['object_features'] for d in datasets])
    X = np.concatenate([X, obj_X], axis=1)
    print(f'Usando contexto de objeto: +{obj_X.shape[1]} dims (total {X.shape[1]})')
y_cat = np.array([LABEL_TO_CATEGORY[label] for label in y_fine])

text_features = np.load(ROOT / 'vlm' / 'models' / 'text_features.npy')
zs_labels = json.loads((ROOT / 'vlm' / 'models' / 'labels.json').read_text())

unique_videos = np.unique(videos)
video_group = np.array([groups[videos == v][0] for v in unique_videos])
video_cat = np.array([y_cat[videos == v][0] for v in unique_videos])

zs_accs, lp_accs = [], []
per_cat_correct: dict[str, int] = {}
per_cat_total: dict[str, int] = {}
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

        logits = 100.0 * body_X[mask] @ text_features.T
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
        per_cat_total[true_cat] = per_cat_total.get(true_cat, 0) + 1
        per_cat_correct[true_cat] = per_cat_correct.get(true_cat, 0) + int(lp_pred == true_cat)

    zs_acc, lp_acc = zs_correct / total, lp_correct / total
    zs_accs.append(zs_acc); lp_accs.append(lp_acc)
    print(f'seed={seed}  {total} videos val  zero-shot={zs_acc:.1%}  linear-probe={lp_acc:.1%}')

print(f'\n=== Promedio {args.seeds} seeds, por video (participantes disjuntos train/val) ===')
print(f'zero-shot:    {np.mean(zs_accs):.1%}  (std {np.std(zs_accs):.1%})')
print(f'linear-probe: {np.mean(lp_accs):.1%}  (std {np.std(lp_accs):.1%})')

print('\n=== Linear-probe por categoría (pooled sobre los seeds, participantes disjuntos) ===')
for cat in sorted(per_cat_total):
    n = per_cat_total[cat]
    acc = per_cat_correct[cat] / n
    print(f'{cat:28s} n={n:5d}  acc={acc:.1%}')

# --- Entrenamiento final sobre todo el cache, para producción ---
final_clf = LogisticRegression(max_iter=2000, C=args.C, class_weight='balanced')
final_clf.fit(X, y_cat)

out_path = Path(args.output) if args.output else ROOT / 'vlm' / 'models' / 'category_probe.npz'
np.savez(out_path, W=final_clf.coef_.astype(np.float32), b=final_clf.intercept_.astype(np.float32),
         classes=np.array(final_clf.classes_))
print(f'\nGuardado {out_path} (entrenado sobre {X.shape[0]} frames, {len(unique_videos)} videos)')
