"""Entrena el GRU (`src/temporal_model.py`) como clasificador temporal de
caída de 3 clases (`normal` / `caida_en_curso` / `caido`) sobre las ventanas
de keypoints cacheadas por `extract_falls.py`.

Diferencias con `train_temporal_classifier.py` (actividad):

  1. **Métrica alineada con `vision/evaluate_falls.py`**, no accuracy por
     ventana: se reporta, out-of-fold (GroupKFold por video, nunca el mismo
     video en train y test), el *recall por video de caída* (¿alguna ventana
     `caida_*` del video se predijo como caída?) y los *falsos positivos*
     (ventanas `normal` predichas como caída). Es lo que hay que comparar
     contra el baseline documentado: 84.8% recall / 30 FP.

  2. **Augmentación de oclusión / truncamiento de encuadre** (solo en train),
     siguiendo la literatura de robustez del esqueleto (PointOut / StoneGAT:
     dropout de nodos; VIRA-GCN: viewpoints/oclusiones sintéticas). El
     objetivo concreto: que el modelo aprenda "torso visible + este
     movimiento = caída" en vez de romperse cuando la persona sale de cuadro,
     que es la queja que motivó todo esto (ver `vision/README.md`, evidencia
     parcial).

Al final entrena sobre todo el cache y guarda `fusion/models/fall_temporal.pt`.
Siguiente paso: `python export_fall_temporal_onnx.py`.

Uso:
    python train_fall_temporal.py [--features fall_windows.npz] [--folds 5]
        [--epochs 40] [--aug-prob 0.5] [--hidden-size 64] [--num-layers 2]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import torch
from sklearn.model_selection import GroupKFold
from torch import nn

from fusion.src.fall_temporal import FALL_LABELS, FALL_POSITIVE_LABELS, RAW_TO_LABEL
from fusion.src.keypoint_sequence import NUM_KEYPOINTS
from fusion.src.temporal_model import GRUActivityClassifier

parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument("--features", default=None, help="Default: fusion/fall_windows.npz")
parser.add_argument("--folds", type=int, default=5)
parser.add_argument("--hidden-size", type=int, default=64)
parser.add_argument("--num-layers", type=int, default=2)
parser.add_argument("--dropout", type=float, default=0.3)
parser.add_argument("--epochs", type=int, default=40)
parser.add_argument("--batch-size", type=int, default=64)
parser.add_argument("--lr", type=float, default=1e-3)
parser.add_argument("--aug-prob", type=float, default=0.5, help="Probabilidad de aplicar augmentación de oclusión a una ventana de train")
parser.add_argument("--seed", type=int, default=0, help="Semilla (init del modelo, shuffle, augmentación) — para reproducibilidad")
parser.add_argument("--hard-neg-oversample", type=int, default=4, help="Cuántas veces se repite cada hard negative minado en el set de train (ver extract_falls.py --mine-hard-negatives)")
parser.add_argument("--device", default="cpu")
args = parser.parse_args()

torch.manual_seed(args.seed)
np.random.seed(args.seed)

features_path = Path(args.features) if args.features else Path(__file__).parent / "fall_windows.npz"
data = np.load(features_path, allow_pickle=True)
X = data["windows"].astype(np.float32)       # (N, n_frames, feat_dim)
y_raw = data["labels"]                        # etiquetas finas de extract_falls.py
videos = data["videos"]
datasets = data["datasets"] if "datasets" in data else np.array(["?"] * len(y_raw))
hard_neg = data["is_hard_negative"] if "is_hard_negative" in data else np.zeros(len(y_raw), bool)

# Colapso a binario normal/caida (ver docstring de fall_temporal.py).
unknown = sorted(set(y_raw) - set(RAW_TO_LABEL))
if unknown:
    raise SystemExit(f"Labels crudas no mapeadas: {unknown}")
y_str = np.array([RAW_TO_LABEL[s] for s in y_raw])
label_to_idx = {lbl: i for i, lbl in enumerate(FALL_LABELS)}
y = np.array([label_to_idx[s] for s in y_str])
pos_idx = {label_to_idx[l] for l in FALL_POSITIVE_LABELS}

input_size = X.shape[-1]
n_frames = X.shape[1]
device = torch.device(args.device)

print(f"{len(y)} ventanas, {len(set(videos))} videos, feat_dim={input_size}, n_frames={n_frames}")
print("Por clase:", {lbl: int((y == i).sum()) for lbl, i in label_to_idx.items()})

# --- Layout del vector de features (ver keypoint_sequence.frame_features) ---
#   [0 : 2*K)          posiciones (x, y) centradas/escaladas de K=17 keypoints
#   [2*K : 3*K)        confianza por keypoint
#   [3*K : 3*K+4)      escalares: aspect_ratio, torso_angle/180, hip_y, bbox_height_frac
K = NUM_KEYPOINTS
POS, CONF = slice(0, 2 * K), slice(2 * K, 3 * K)
ASPECT_I, HIP_Y_I, BBOX_H_I = 3 * K + 0, 3 * K + 2, 3 * K + 3
# Grupos de keypoints COCO para el truncamiento de encuadre.
_LEGS = [13, 14, 15, 16]
_HEAD = [0, 1, 2, 3, 4]


def augment(win: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Augmentación de oclusión/truncamiento sobre una ventana `(n_frames, feat_dim)`.

    - *keypoint dropout* (PointOut): pone a 0 posición + confianza de un
      subconjunto aleatorio de keypoints, frame a frame.
    - *truncamiento de encuadre* (VIRA-GCN): simula que parte del cuerpo sale
      de cuadro — elimina piernas o cabeza en TODA la ventana y ajusta los
      escalares que el recorte del bbox distorsiona en la vida real
      (aspect_ratio más ancho, bbox más bajo).
    """
    win = win.copy()
    r = rng.random()

    if r < 0.45:  # truncamiento de encuadre (persistente en la ventana)
        group = _LEGS if rng.random() < 0.6 else _HEAD
        cols_xy = np.array([[2 * k, 2 * k + 1] for k in group]).ravel()
        win[:, cols_xy] = 0.0
        win[:, [2 * K + k for k in group]] = 0.0
        if group is _LEGS:
            win[:, ASPECT_I] *= rng.uniform(1.15, 1.6)     # bbox se ensancha al perder alto
            win[:, BBOX_H_I] *= rng.uniform(0.55, 0.8)
            win[:, HIP_Y_I] = np.clip(win[:, HIP_Y_I] * rng.uniform(1.0, 1.15), 0, 1)
        else:
            win[:, BBOX_H_I] *= rng.uniform(0.7, 0.9)
    else:  # keypoint dropout aleatorio, por frame
        p = rng.uniform(0.1, 0.35)
        mask = rng.random((n_frames, K)) < p
        for f in range(n_frames):
            drop = np.where(mask[f])[0]
            if drop.size:
                cols_xy = np.array([[2 * k, 2 * k + 1] for k in drop]).ravel()
                win[f, cols_xy] = 0.0
                win[f, [2 * K + k for k in drop]] = 0.0

    # jitter suave de posición (siempre) — robustez a ruido de YOLO
    win[:, POS] += rng.normal(0, 0.01, win[:, POS].shape).astype(np.float32)
    return win


class WindowDS(torch.utils.data.Dataset):
    def __init__(self, Xs, ys, train: bool, seed: int = 0):
        self.X, self.y, self.train = Xs, ys, train
        self.rng = np.random.default_rng(seed)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, i):
        x = self.X[i]
        if self.train and self.rng.random() < args.aug_prob:
            x = augment(x, self.rng)
        return torch.from_numpy(x.astype(np.float32)), int(self.y[i])


def make_model() -> GRUActivityClassifier:
    return GRUActivityClassifier(
        input_size=input_size, num_classes=len(FALL_LABELS),
        hidden_size=args.hidden_size, num_layers=args.num_layers, dropout=args.dropout,
    ).to(device)


def class_weights(yt: np.ndarray) -> torch.Tensor:
    counts = np.bincount(yt, minlength=len(FALL_LABELS)).astype(np.float32)
    w = np.zeros(len(FALL_LABELS), dtype=np.float32)
    present = counts > 0
    w[present] = counts[present].sum() / (present.sum() * counts[present])
    return torch.from_numpy(w).to(device)


def train_fold(tr_mask, epochs) -> GRUActivityClassifier:
    model = make_model()
    tr_idx = np.where(tr_mask)[0]
    # Oversample de hard negatives (solo en train): se repiten para que el
    # modelo vea el caso "geometría dispara pero no es caída" muchas veces.
    if args.hard_neg_oversample > 1:
        extra = tr_idx[hard_neg[tr_idx]]
        if len(extra):
            tr_idx = np.concatenate([tr_idx] + [extra] * (args.hard_neg_oversample - 1))
    xt = torch.from_numpy(X[tr_idx]).float()
    model.set_normalization(xt.mean(dim=(0, 1)).to(device), xt.std(dim=(0, 1)).to(device))
    ds = WindowDS(X[tr_idx], y[tr_idx], train=True)
    dl = torch.utils.data.DataLoader(ds, batch_size=args.batch_size, shuffle=True)
    crit = nn.CrossEntropyLoss(weight=class_weights(y[tr_idx]))
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    for _ in range(epochs):
        model.train()
        for xb, yb in dl:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            loss = crit(model(xb), yb)
            loss.backward()
            opt.step()
    model.eval()
    return model


def predict(model, mask) -> np.ndarray:
    with torch.no_grad():
        out = model(torch.from_numpy(X[mask]).float().to(device))
    return out.argmax(1).cpu().numpy()


# --- Evaluación out-of-fold, GroupKFold por video ---
gkf = GroupKFold(n_splits=args.folds)
oof_pred = np.full(len(y), -1)
for k, (tr, te) in enumerate(gkf.split(X, y, groups=videos)):
    tr_mask = np.zeros(len(y), bool); tr_mask[tr] = True
    model = train_fold(tr_mask, args.epochs)
    oof_pred[te] = predict(model, te)
    print(f"fold {k+1}/{args.folds}: train={tr_mask.sum()} test={len(te)} ventanas", flush=True)

# Recall por video de caída: el video tiene al menos una ventana caída_* y
# alguna de ellas se predijo como caída.
true_pos = np.isin(y, list(pos_idx))
pred_pos = np.isin(oof_pred, list(pos_idx))
fall_videos = sorted(set(videos[true_pos]))
recalled = [v for v in fall_videos if pred_pos[(videos == v) & true_pos].any()]
recall = len(recalled) / len(fall_videos) if fall_videos else float("nan")

# Falsos positivos: ventanas 'normal' predichas como caída.
normal_mask = y == label_to_idx["normal"]
fp = int((pred_pos & normal_mask).sum())
fp_by_ds = {d: int((pred_pos & normal_mask & (datasets == d)).sum()) for d in sorted(set(datasets))}

# Matriz de confusión por ventana
cm = np.zeros((len(FALL_LABELS), len(FALL_LABELS)), int)
for t, p in zip(y, oof_pred):
    if p >= 0:
        cm[t, p] += 1

print("\n=== Out-of-fold (GroupKFold por video) ===")
print(f"Recall por video de caída: {len(recalled)}/{len(fall_videos)} ({recall*100:.1f}%)")
print(f"Falsos positivos (ventanas 'normal' -> caída): {fp}  {fp_by_ds}")
print(f"Ventanas 'normal': {int(normal_mask.sum())}  |  FP rate: {fp/max(int(normal_mask.sum()),1)*100:.1f}%")
print("Matriz de confusión (fila=real, col=predicho):")
print("            " + "  ".join(f"{l[:10]:>10}" for l in FALL_LABELS))
for i, l in enumerate(FALL_LABELS):
    print(f"{l[:12]:>12} " + "  ".join(f"{cm[i,j]:>10}" for j in range(len(FALL_LABELS))))
print("\nBaseline documentado (heurística + confirmación CLIP, vision/README.md): "
      "84.8% recall / 30 FP out-of-fold; Benabdennour 2026: 97.7% F1.")

# --- Modelo final sobre todo el cache ---
print("\nEntrenando modelo final sobre todo el cache...")
all_mask = np.ones(len(y), bool)
final = train_fold(all_mask, args.epochs)
print(f"Parámetros: {final.num_parameters():,}")

out_dir = Path(__file__).parent / "models"
out_dir.mkdir(exist_ok=True)
out_path = out_dir / "fall_temporal.pt"
torch.save(
    {
        "state_dict": final.state_dict(),
        "input_size": input_size,
        "hidden_size": args.hidden_size,
        "num_layers": args.num_layers,
        "dropout": args.dropout,
        "labels": FALL_LABELS,
    },
    out_path,
)
print(f"Guardado {out_path}\nSiguiente: python export_fall_temporal_onnx.py")
