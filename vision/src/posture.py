"""Clasificación de postura (parado / sentado / acostado) a partir de descriptores de pose.

Umbrales calibrados empíricamente (proceso completo documentado en
../README.md, sección "Clasificación de postura"): se probaron combinaciones
con `aspect_ratio` + ángulo de rodilla, y separado por `aspect_ratio` solo
con distintos criterios de optimización, contra 39 videos con clasificación
manual de postura por carpeta. Resultado final: dos cortes sobre
`aspect_ratio` únicamente, optimizando recall promedio entre clases (no
accuracy total, para no favorecer a la clase mayoritaria) — recall
parado=90.0%, sentado=82.1%, acostado=86.6%.

Nota importante: al revisar visualmente los casos que esa evaluación
marcaba como "error", se encontró que la clasificación manual por carpeta
no era ground truth confiable (mezclaba "tipo de actividad" con postura
real — ver ../../videos/README.md). El modelo estaba prediciendo bien en
esos casos. Por eso la carpetización se eliminó y estos números quedan
como referencia histórica de cómo se llegó a los umbrales, no como una
accuracy que haya que seguir persiguiendo contra ese dataset.
"""
from __future__ import annotations

from .pose_pipeline import PoseSample

PARADO_SENTADO_ASPECT_RATIO = 0.58   # aspect ratio por encima de esto => ya no es "parado"
SENTADO_ACOSTADO_ASPECT_RATIO = 1.13  # aspect ratio por encima de esto => "acostado"

POSTURES = ("parado", "sentado", "acostado")


def classify_posture(sample: PoseSample) -> str:
    if sample.aspect_ratio > SENTADO_ACOSTADO_ASPECT_RATIO:
        return "acostado"
    if sample.aspect_ratio > PARADO_SENTADO_ASPECT_RATIO:
        return "sentado"
    return "parado"
