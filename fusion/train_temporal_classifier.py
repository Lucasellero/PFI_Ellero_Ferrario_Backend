"""Entrena el GRU (`src/temporal_model.py`) sobre las ventanas de keypoints
cacheadas por `extract_dataset.py`, con split por participante (igual
criterio que `vlm/train_category_probe.py`: nunca mezclar el mismo sujeto
entre train/val, para medir generalización real y no memorización de
persona/cámara) y varios seeds para reportar consistencia.

Al final entrena sobre TODO el cache y guarda `fusion/models/temporal_classifier.pt`
(checkpoint de PyTorch — todavía no el ONNX de producción, ver
`export_temporal_onnx.py` para ese paso).

Uso:
    python train_temporal_classifier.py [--features dataset_features.npz]
                                         [--hidden-size 64] [--num-layers 2]
                                         [--epochs 40] [--seeds 5]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import torch
from sklearn.model_selection import GroupShuffleSplit
from torch import nn

from fusion.src.labels import FINE_LABELS
from fusion.src.temporal_model import GRUActivityClassifier

parser = argparse.ArgumentParser()
parser.add_argument("--features", default=None, help="Default: fusion/dataset_features.npz")
parser.add_argument("--hidden-size", type=int, default=64)
parser.add_argument("--num-layers", type=int, default=2)
parser.add_argument("--dropout", type=float, default=0.3)
parser.add_argument("--epochs", type=int, default=40)
parser.add_argument("--batch-size", type=int, default=32)
parser.add_argument("--lr", type=float, default=1e-3)
parser.add_argument("--seeds", type=int, default=5, help="Splits por participante a promediar")
parser.add_argument("--device", default="cpu")
args = parser.parse_args()

features_path = Path(args.features) if args.features else Path(__file__).parent / "dataset_features.npz"
data = np.load(features_path, allow_pickle=True)

X = data["windows"]          # (N, n_frames, feat_dim)
y_fine = data["labels"]
groups = data["participants"]
videos = data["videos"]

label_to_idx = {label: i for i, label in enumerate(FINE_LABELS)}
unknown = sorted(set(y_fine) - set(label_to_idx))
if unknown:
    raise SystemExit(f"Labels en el cache que no están en FINE_LABELS: {unknown}")
y = np.array([label_to_idx[lbl] for lbl in y_fine])

input_size = X.shape[-1]
device = torch.device(args.device)


def train_one(train_mask: np.ndarray, val_mask: np.ndarray, epochs: int) -> tuple[GRUActivityClassifier, float]:
    model = GRUActivityClassifier(
        input_size=input_size,
        num_classes=len(FINE_LABELS),
        hidden_size=args.hidden_size,
        num_layers=args.num_layers,
        dropout=args.dropout,
    ).to(device)

    x_train = torch.from_numpy(X[train_mask]).float()
    y_train = torch.from_numpy(y[train_mask]).long()
    x_val = torch.from_numpy(X[val_mask]).float().to(device)
    y_val = torch.from_numpy(y[val_mask]).long().to(device)

    mean = x_train.mean(dim=(0, 1))
    std = x_train.std(dim=(0, 1))
    model.set_normalization(mean.to(device), std.to(device))

    # class_weight='balanced' equivalente manual (mismo criterio que
    # LogisticRegression(class_weight='balanced') en vlm/train_category_probe.py):
    # las clases de Toyota Smarthome están lejos de estar balanceadas
    # (ej. Walk/Laydown tienen muchos más clips que Cutbread), sin esto el
    # modelo colapsa a predecir siempre la clase mayoritaria.
    class_counts = np.bincount(y[train_mask], minlength=len(FINE_LABELS)).astype(np.float32)
    class_weight = np.zeros(len(FINE_LABELS), dtype=np.float32)
    present = class_counts > 0
    class_weight[present] = class_counts[present].sum() / (present.sum() * class_counts[present])
    criterion = nn.CrossEntropyLoss(weight=torch.from_numpy(class_weight).to(device))
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    n = x_train.shape[0]
    best_val_acc = 0.0
    best_state = None
    for epoch in range(epochs):
        model.train()
        perm = torch.randperm(n)
        for i in range(0, n, args.batch_size):
            idx = perm[i : i + args.batch_size]
            xb, yb = x_train[idx].to(device), y_train[idx].to(device)
            optimizer.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            optimizer.step()

        model.eval()
        with torch.no_grad():
            val_acc = (model(x_val).argmax(dim=1) == y_val).float().mean().item()
        if val_acc >= best_val_acc:
            best_val_acc = val_acc
            best_state = {k: v.clone() for k, v in model.state_dict().items()}

    model.load_state_dict(best_state)
    return model, best_val_acc


unique_videos = np.unique(videos)
video_group = np.array([groups[videos == v][0] for v in unique_videos])

accs = []
for seed in range(args.seeds):
    splitter = GroupShuffleSplit(n_splits=1, test_size=0.25, random_state=seed)
    train_v_idx, val_v_idx = next(splitter.split(unique_videos, groups=video_group))
    train_videos, val_videos = set(unique_videos[train_v_idx]), set(unique_videos[val_v_idx])
    train_mask = np.array([v in train_videos for v in videos])
    val_mask = np.array([v in val_videos for v in videos])

    if train_mask.sum() == 0 or val_mask.sum() == 0:
        print(f"seed={seed}  split degenerado (dataset chico) — salteado")
        continue

    _, val_acc = train_one(train_mask, val_mask, args.epochs)
    accs.append(val_acc)
    print(f"seed={seed}  train={train_mask.sum()} val={val_mask.sum()} ventanas  val_acc={val_acc:.1%}")

if accs:
    print(f"\n=== Promedio {len(accs)} seeds, por ventana (participantes disjuntos train/val) ===")
    print(f"accuracy: {np.mean(accs):.1%}  (std {np.std(accs):.1%})")
else:
    print("\n(Sin splits válidos para reportar accuracy — dataset demasiado chico. "
          "Igual se entrena el modelo final sobre todo el cache.)")

# --- Entrenamiento final sobre todo el cache, para producción ---
print("\nEntrenando sobre el cache completo...")
final_model = GRUActivityClassifier(
    input_size=input_size,
    num_classes=len(FINE_LABELS),
    hidden_size=args.hidden_size,
    num_layers=args.num_layers,
    dropout=args.dropout,
).to(device)
# Sin split acá: la accuracy honesta ya se reportó arriba con participantes
# disjuntos. train_one usa el set de "val" para quedarse con el mejor
# checkpoint por época — al pasar el mismo set de train como "val" para el
# modelo final, ese criterio pasa a ser simplemente la época de mejor
# accuracy de entrenamiento (no hay fuga real de datos de val, porque acá
# no hay val: es el modelo de producción, entrenado sobre todo lo
# disponible, igual que el paso final de vlm/train_category_probe.py).
all_mask = np.ones(len(y), dtype=bool)
final_model, _ = train_one(all_mask, all_mask, args.epochs)
print(f"Parámetros del modelo: {final_model.num_parameters():,}")

out_dir = Path(__file__).parent / "models"
out_dir.mkdir(exist_ok=True)
out_path = out_dir / "temporal_classifier.pt"
torch.save(
    {
        "state_dict": final_model.state_dict(),
        "input_size": input_size,
        "hidden_size": args.hidden_size,
        "num_layers": args.num_layers,
        "dropout": args.dropout,
        "labels": FINE_LABELS,
    },
    out_path,
)
print(f"Guardado {out_path} (entrenado sobre {X.shape[0]} ventanas, {len(unique_videos)} videos)")
print("Siguiente paso: python export_temporal_onnx.py")
