"""Compara el filtro zero-shot y el linear probe de confirmación de caídas
sobre exactamente el mismo conjunto de eventos (509 eventos geométricos de
292 clips, `fall_confirmation_embeddings.npz`), para poder decidir si
integrar el probe a producción con un número justo (ver "pendiente" en
vision/README.md, 2026-08-12: el 82.0%/96 FP del zero-shot se midió sobre
183 clips viejos, el 84.8%/30 FP del probe sobre 292 — no comparables tal
cual).

El zero-shot no tiene entrenamiento (umbral fijo), así que no hace falta
out-of-fold para él: se evalúa directo sobre todos los eventos. Para que la
comparación sea justa igual, el número del probe que importa es el
out-of-fold (ya impreso por train_fall_confirmation_probe.py), no el
reentrenado sobre todo el cache.

Uso:
    vlm/.venv/Scripts/python.exe compare_confirmation_filters.py
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold

from evaluate_falls import load_ground_truth

EMB_PATH = Path(__file__).parent / "fall_confirmation_embeddings.npz"
TEXT_FEATURES_PATH = ROOT / "vlm" / "models" / "fall_confirmation_features.npy"
ZEROSHOT_MARGIN_THRESHOLD = -0.028  # vlm/src/fall_confirmation.py

data = np.load(EMB_PATH, allow_pickle=True)
X_all = data["features"]
video_all = data["video"]
t_all = data["t"]
is_real_fall_all = data["is_real_fall"]
is_in_nonfall_all = data["is_in_nonfall"]

labeled_mask = is_real_fall_all | is_in_nonfall_all
X, video, t, y = X_all[labeled_mask], video_all[labeled_mask], t_all[labeled_mask], is_real_fall_all[labeled_mask]
in_nonfall = is_in_nonfall_all[labeled_mask]

cases = load_ground_truth()
fall_cases = [c for c in cases if c["fall_segments"]]


def recall_fp(kept_mask: np.ndarray) -> tuple[float, int]:
    kept_video_t = {(v, tt) for v, tt, keep in zip(video, t, kept_mask) if keep}
    true_positives = 0
    for case in fall_cases:
        hit = any(
            v == case["video"] and any(s <= tt <= en for s, en in case["fall_segments"])
            for v, tt in kept_video_t
        )
        if hit:
            true_positives += 1
    total_fp = sum(1 for v, tt, keep, nf in zip(video, t, kept_mask, in_nonfall) if keep and nf)
    return true_positives / len(fall_cases), total_fp


# --- Zero-shot: umbral fijo, sin entrenamiento, se evalúa directo ---
text_features = np.load(TEXT_FEATURES_PATH)
fall_score = X @ text_features[0]
notfall_score = X @ text_features[1]
margin = fall_score - notfall_score
zeroshot_kept = margin > ZEROSHOT_MARGIN_THRESHOLD
zs_recall, zs_fp = recall_fp(zeroshot_kept)

# --- Probe: out-of-fold, GroupKFold por video (mismo criterio que el training script) ---
unique_videos = np.unique(video)
gkf = GroupKFold(n_splits=min(5, len(unique_videos)))
oof_pred = np.zeros(len(X), dtype=bool)
for train_idx, val_idx in gkf.split(X, y, groups=video):
    clf = LogisticRegression(max_iter=2000, C=1.0, class_weight="balanced")
    clf.fit(X[train_idx], y[train_idx])
    probs = clf.predict_proba(X[val_idx])[:, list(clf.classes_).index(True)]
    oof_pred[val_idx] = probs >= 0.5
probe_recall, probe_fp = recall_fp(oof_pred)

print(f"{len(X)} eventos ({y.sum()} caída real, {(~y).sum()} no-caída) sobre "
      f"{len(unique_videos)} videos, {len(fall_cases)} clips con caída en el ground truth\n")
print("=== Mismo conjunto de evaluación (292 clips) ===")
print(f"Zero-shot (umbral fijo {ZEROSHOT_MARGIN_THRESHOLD}): recall {zs_recall:.1%}, FP {zs_fp}")
print(f"Probe (out-of-fold, GroupKFold):        recall {probe_recall:.1%}, FP {probe_fp}")
