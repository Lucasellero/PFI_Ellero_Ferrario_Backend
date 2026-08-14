"""GRU liviano para clasificar actividad a partir de una secuencia de
keypoints (ver `keypoint_sequence.py`).

Solo se usa en la máquina de desarrollo (`train_temporal_classifier.py`,
`export_temporal_onnx.py`) — requiere PyTorch. En el Pi corre
`temporal_classifier.SkeletonActivityClassifier` sobre el grafo ONNX ya
exportado, sin PyTorch (mismo patrón de separación dev/Pi que `vlm/`, ver
`vlm/README.md` sección "Setup": PyTorch pesa ~1.5GB en ARM y solo hace
falta para exportar, no para inferir).

Presupuesto de parámetros: <1-2M para poder cuantizar a ONNX INT8 y correr
en la Raspberry Pi. Con `FRAME_FEATURE_DIM` (~55, ver keypoint_sequence.py)
y los defaults de abajo (hidden=64, 2 capas), el modelo pesa del orden de
50-90k parámetros (calculado: GRU 3*(input*hidden+hidden²+hidden) por capa
+ la capa de salida) — dos órdenes de magnitud por debajo del límite, deja
margen grande para escalar si hiciera falta más capacidad antes de
acercarse al presupuesto.

Puerta abierta a ST-GCN (Yan et al., 2018) más adelante si el GRU no
alcanza para separar clases con gestos sutiles (mismo problema que motivó
todo este módulo para CLIP, ver `vlm/README.md`): se arranca con GRU porque
(a) es más simple de exportar/cuantizar a ONNX y de justificar en la tesis,
y (b) la literatura ya citada en `vision/README.md` (Benabdennour et al.
2026: YOLOv11-Pose + Transformer para caídas) confirma que un modelo
temporal chico sobre esta clase de secuencia funciona bien para este
dominio. ST-GCN necesitaría además la topología de grafo del esqueleto COCO
como input estructural, no solo la secuencia plana — más costoso de
implementar/exportar sin evidencia todavía de que el GRU no alcance.
"""
from __future__ import annotations

import torch
from torch import nn


class GRUActivityClassifier(nn.Module):
    """`x`: `(batch, seq_len, input_size)` → logits `(batch, num_classes)`.

    La normalización de input (`input_mean`/`input_std`, z-score por canal
    calculado sobre el set de entrenamiento) se guarda como buffer del
    módulo, no se aplica afuera: así el grafo ONNX exportado queda
    autocontenido — `SkeletonActivityClassifier` en el Pi solo tiene que
    armar la ventana de features crudas (`keypoint_sequence.frame_features`)
    y pasarla tal cual, sin mantener sincronizados unos mean/std por
    separado entre entrenamiento e inferencia.
    """

    def __init__(
        self,
        input_size: int,
        num_classes: int,
        hidden_size: int = 64,
        num_layers: int = 2,
        dropout: float = 0.3,
    ) -> None:
        super().__init__()
        self.register_buffer("input_mean", torch.zeros(input_size))
        self.register_buffer("input_std", torch.ones(input_size))
        self.gru = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden_size, num_classes)

    def set_normalization(self, mean: torch.Tensor, std: torch.Tensor) -> None:
        self.input_mean.copy_(mean)
        self.input_std.copy_(std.clamp_min(1e-6))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = (x - self.input_mean) / self.input_std
        _, h_n = self.gru(x)
        last_hidden = h_n[-1]  # (batch, hidden_size) — última capa del GRU
        return self.classifier(self.dropout(last_hidden))

    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())
