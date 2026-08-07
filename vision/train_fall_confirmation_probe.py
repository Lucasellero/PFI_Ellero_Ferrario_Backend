"""Entrena un linear probe (logistic regression) sobre los embeddings ONNX
de CLIP guardados en fall_confirmation_embeddings.npz, como alternativa al
filtro de confirmación zero-shot actual (`vlm/src/fall_confirmation.py`,
margen contra prompts fijos). Misma técnica que ya mejoró la clasificación
de actividades en `vlm/train_category_probe.py`.

Evaluación con GroupKFold por video (cada evento se predice una única vez
con un modelo que nunca vio ese video en entrenamiento) — evita el leakage
de evaluar sobre el mismo dato usado para entrenar, que sería la forma fácil
de "ganarle" al zero-shot sin que generalice de verdad. El recall/FP
end-to-end se recalcula sobre esas predicciones out-of-fold, comparable a
los 82.0% recall / 96 FP del filtro zero-shot (ver vision/README.md) —
aunque esa cifra zero-shot NO es out-of-fold (no hay nada que entrenar en un
umbral fijo), así que la comparación favorece ligeramente al zero-shot.

Al final entrena sobre TODO el cache y exporta vlm/models/fall_confirmation_probe.npz.

Uso:
    vlm/.venv/Scripts/python.exe train_fall_confirmation_probe.py [--C 1.0]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold

from evaluate_falls import load_ground_truth

parser = argparse.ArgumentParser()
parser.add_argument("--C", type=float, default=1.0, help="Inverso de regularización L2")
parser.add_argument("--folds", type=int, default=5)
args = parser.parse_args()

data = np.load(Path(__file__).parent / "fall_confirmation_embeddings.npz", allow_pickle=True)
X_all = data["features"]
video_all = data["video"]
t_all = data["t"]
is_real_fall_all = data["is_real_fall"]
is_in_nonfall_all = data["is_in_nonfall"]

# Solo eventos no ambiguos entrenan el probe (¿es "real_fall" o "no-caída"?).
labeled_mask = is_real_fall_all | is_in_nonfall_all
X, video, t, y = X_all[labeled_mask], video_all[labeled_mask], t_all[labeled_mask], is_real_fall_all[labeled_mask]
n_ambiguous = (~labeled_mask).sum()
print(f"{len(X)} eventos etiquetados ({y.sum()} caída real, {(~y).sum()} no-caída), "
      f"{n_ambiguous} ambiguos excluidos del entrenamiento\n")

unique_videos = np.unique(video)
n_folds = min(args.folds, len(unique_videos))
gkf = GroupKFold(n_splits=n_folds)

oof_pred = np.zeros(len(X), dtype=bool)
oof_prob = np.zeros(len(X), dtype=float)
for fold, (train_idx, val_idx) in enumerate(gkf.split(X, y, groups=video)):
    clf = LogisticRegression(max_iter=2000, C=args.C, class_weight="balanced")
    clf.fit(X[train_idx], y[train_idx])
    probs = clf.predict_proba(X[val_idx])[:, list(clf.classes_).index(True)]
    oof_prob[val_idx] = probs
    oof_pred[val_idx] = probs >= 0.5
    fold_acc = (oof_pred[val_idx] == y[val_idx]).mean()
    print(f"fold={fold}  {len(val_idx)} eventos val  accuracy={fold_acc:.1%}")

print(f"\naccuracy out-of-fold total: {(oof_pred == y).mean():.1%}")

# --- Recall/FP end-to-end sobre las predicciones out-of-fold ---
cases = load_ground_truth()
fall_cases = [c for c in cases if c["fall_segments"]]

kept_video_t = {(v, tt) for v, tt, keep in zip(video, t, oof_pred) if keep}


def video_has_kept_hit(case_video: str) -> bool:
    return any(v == case_video for v, _tt in kept_video_t)


true_positives = 0
for case in fall_cases:
    hit = any(
        v == case["video"] and any(s <= tt <= en for s, en in case["fall_segments"])
        for v, tt in kept_video_t
    )
    if hit:
        true_positives += 1

total_fp = sum(1 for v, tt, keep, in_nf in zip(video, t, oof_pred, is_in_nonfall_all[labeled_mask]) if keep and in_nf)

recall = true_positives / len(fall_cases)
print(f"\n=== Recall/FP end-to-end, out-of-fold (probe) ===")
print(f"Recall: {true_positives}/{len(fall_cases)} ({recall:.1%})")
print(f"Falsos positivos en tramos no-caída: {total_fp}")
print("(comparar contra 82.0% recall / 96 FP del filtro zero-shot, ver vision/README.md — "
      "esa cifra no es out-of-fold, esta sí)")

# --- Entrenamiento final sobre todo el cache etiquetado, para producción ---
final_clf = LogisticRegression(max_iter=2000, C=args.C, class_weight="balanced")
final_clf.fit(X, y)

out_path = ROOT / "vlm" / "models" / "fall_confirmation_probe.npz"
np.savez(
    out_path,
    W=final_clf.coef_.astype(np.float32),
    b=final_clf.intercept_.astype(np.float32),
)
print(f"\nGuardado {out_path} (entrenado sobre {len(X)} eventos, {len(unique_videos)} videos)")
