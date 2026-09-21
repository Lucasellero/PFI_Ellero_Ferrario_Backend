"""Fusión de la señal temporal de esqueleto (movimiento) con la señal de
CLIP (contexto/objeto), a nivel de categoría gruesa (ver
`vlm.src.clip_classifier.COARSE_CATEGORIES`, extendido acá en
`fusion.src.labels` con `Standing`).

Configurable para poder comparar empíricamente los tres enfoques —CLIP
solo, esqueleto solo, híbrido— con el mismo código de clasificación en vez
de tres implementaciones separadas (ver `evaluate_hybrid.py`, que barre
`mode` para esa comparación, y `runner/camera_worker.py`, que en producción
usaría `mode="hybrid"`).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from .labels import LABEL_TO_CATEGORY

MODELS_DIR = Path(__file__).resolve().parents[1] / "models"


def fine_scores_to_category_scores(
    fine_scores: dict[str, float],
    label_to_category: dict[str, str] | None = None,
) -> dict[str, float]:
    """Suma las probabilidades de clases finas a categoría gruesa — mismo
    criterio que `aggregate_to_categories` en `vlm/src/clip_classifier.py`,
    generalizado acá a cualquier espacio de labels finos definido en
    `labels.py` (por default, el del esqueleto — 31 clases + Standing)."""
    mapping = label_to_category if label_to_category is not None else LABEL_TO_CATEGORY
    cats: dict[str, float] = {}
    for label, p in fine_scores.items():
        cat = mapping.get(label)
        if cat is None:
            continue
        cats[cat] = cats.get(cat, 0.0) + p
    return cats


def _align(a: dict[str, float], b: dict[str, float]) -> tuple[list[str], np.ndarray, np.ndarray]:
    """Alinea dos distribuciones de categoría por clave, tratando una
    categoría ausente en una de las dos señales como probabilidad 0 para
    esa señal (no como error) — ej. CLIP nunca predice `Standing`, así que
    su score para esa categoría es 0, no indefinido."""
    keys = sorted(set(a) | set(b))
    va = np.array([a.get(k, 0.0) for k in keys], dtype=np.float64)
    vb = np.array([b.get(k, 0.0) for k in keys], dtype=np.float64)
    return keys, va, vb


class ScoreFuser:
    """Combina dos distribuciones de categoría (esqueleto, CLIP) en una
    sola. Dos métodos — arrancar simple, escalar solo si hace falta (mismo
    criterio que ya aplicó `vlm/` con zero-shot → linear probe, ver
    `vlm/README.md`):

    - `"weighted_average"` (default): promedio pesado de las dos
      distribuciones, alineadas por categoría. Sin entrenamiento, un solo
      hiperparámetro (`skeleton_weight`) que se puede barrer a mano en
      `evaluate_hybrid.py` para ver el trade-off.
    - `"mlp"`: clasificador chico (logistic regression, sklearn — mismo
      enfoque que el "linear probe" de `vlm/train_category_probe.py`)
      entrenado sobre `[scores_esqueleto, scores_clip]` concatenados →
      categoría real (ver `train_fusion_combiner.py`). Permite que el
      modelo aprenda, por categoría, cuánto confiar en cada señal (ej. más
      en CLIP para `EatingDrinkingMedication` —depende de qué hay en la
      mano—, más en esqueleto para `Moving`/`Standing` —depende del
      movimiento del cuerpo—) en vez de un peso fijo global.
    """

    def __init__(
        self,
        method: str = "weighted_average",
        skeleton_weight: float = 0.5,
        combiner_path: Path | None = None,
    ) -> None:
        if method not in ("weighted_average", "mlp"):
            raise ValueError(f"método de fusión desconocido: {method!r}")
        if not 0.0 <= skeleton_weight <= 1.0:
            raise ValueError(f"skeleton_weight debe estar en [0, 1], vino {skeleton_weight!r}")
        self.method = method
        self.skeleton_weight = skeleton_weight
        self._combiner: dict | None = None
        if method == "mlp":
            path = combiner_path or (MODELS_DIR / "fusion_combiner.npz")
            if not path.exists():
                raise FileNotFoundError(
                    f"No se encontró {path}. Corré train_fusion_combiner.py primero, "
                    "o usá method='weighted_average'."
                )
            data = np.load(path, allow_pickle=True)
            self._combiner = {
                "W": data["W"],
                "b": data["b"],
                "categories": [str(c) for c in data["categories"]],
                "input_categories": [str(c) for c in data["input_categories"]],
            }

    def fuse(
        self,
        skeleton_category_scores: dict[str, float] | None,
        clip_category_scores: dict[str, float] | None,
    ) -> dict[str, float]:
        """Devuelve la distribución de categorías fusionada.

        Cualquiera de las dos señales puede faltar (todavía no hay ventana
        de esqueleto completa — arranque en frío, ver
        `SkeletonSequenceBuffer.build_window` —, o la ventana de CLIP no
        tiene persona detectada) — en ese caso se devuelve la otra señal
        sin fusionar, degradando con gracia en vez de fallar. Si faltan las
        dos, es un error del caller (no debería invocar classify() sin
        ninguna señal disponible).
        """
        if skeleton_category_scores is None and clip_category_scores is None:
            raise ValueError("Necesito al menos una señal (esqueleto o CLIP) para fusionar")
        if skeleton_category_scores is None:
            return dict(clip_category_scores)  # type: ignore[arg-type]
        if clip_category_scores is None:
            return dict(skeleton_category_scores)

        if self.method == "weighted_average":
            keys, sk, cl = _align(skeleton_category_scores, clip_category_scores)
            fused = self.skeleton_weight * sk + (1 - self.skeleton_weight) * cl
            return dict(zip(keys, (float(v) for v in fused)))

        # method == "mlp": el combiner se entrenó sobre un orden fijo de
        # categorías de entrada (input_categories) — ver train_fusion_combiner.py.
        assert self._combiner is not None
        input_cats = self._combiner["input_categories"]
        x = np.concatenate(
            [
                [skeleton_category_scores.get(c, 0.0) for c in input_cats],
                [clip_category_scores.get(c, 0.0) for c in input_cats],
            ]
        ).astype(np.float32)
        logits = x @ self._combiner["W"].T + self._combiner["b"]
        probs = np.exp(logits - logits.max())
        probs /= probs.sum()
        return dict(zip(self._combiner["categories"], (float(p) for p in probs)))


class HybridActivityClassifier:
    """Clasificador de alto nivel: junta esqueleto (movimiento), CLIP
    (contexto/objeto) y un `ScoreFuser`, con un `mode` para poder aislar
    cualquiera de las dos señales sin dos implementaciones separadas —
    correr `mode="hybrid"` en producción (`runner/`) y barrer
    `"skeleton_only"`/`"clip_only"`/`"hybrid"` al evaluar
    (`evaluate_hybrid.py`) es literalmente el mismo código.
    """

    _MODES = ("hybrid", "skeleton_only", "clip_only")

    def __init__(
        self,
        mode: str = "hybrid",
        skeleton_classifier=None,
        clip_classifier=None,
        fuser: ScoreFuser | None = None,
    ) -> None:
        if mode not in self._MODES:
            raise ValueError(f"mode desconocido: {mode!r} (opciones: {self._MODES})")
        if mode in ("hybrid", "skeleton_only") and skeleton_classifier is None:
            raise ValueError(f"mode={mode!r} necesita skeleton_classifier")
        if mode in ("hybrid", "clip_only") and clip_classifier is None:
            raise ValueError(f"mode={mode!r} necesita clip_classifier")
        self.mode = mode
        self.skeleton_classifier = skeleton_classifier
        self.clip_classifier = clip_classifier
        self.fuser = fuser or ScoreFuser()

    def classify(
        self,
        skeleton_window: np.ndarray | None,
        clip_image_bgr: np.ndarray | None,
        clip_hand_crop_bgr: np.ndarray | None = None,
        clip_zone_vector: np.ndarray | None = None,
    ) -> dict:
        """Cualquiera de las dos entradas puede ser `None` (sin ventana de
        esqueleto lista, o sin frame/persona para CLIP en este tick) — el
        resultado degrada a la señal disponible, ver `ScoreFuser.fuse`.

        `clip_zone_vector`: contexto de zona del hogar (ver
        `fusion/src/zone_context.py`), pasado tal cual a
        `clip_classifier.classify()` — `None` si la cámara no tiene zonas
        configuradas, degrada sola sin romper nada (ver `ClipActivityClassifier.classify`).
        """
        skeleton_result = clip_result = None
        skeleton_cat_scores = clip_cat_scores = None

        if self.mode in ("hybrid", "skeleton_only") and skeleton_window is not None:
            skeleton_result = self.skeleton_classifier.classify(skeleton_window)
            skeleton_cat_scores = fine_scores_to_category_scores(skeleton_result["scores"])

        if self.mode in ("hybrid", "clip_only") and clip_image_bgr is not None:
            clip_result = self.clip_classifier.classify(clip_image_bgr, clip_hand_crop_bgr, clip_zone_vector)
            clip_cat_scores = clip_result["category_scores"]

        if self.mode == "skeleton_only":
            category_scores = skeleton_cat_scores
        elif self.mode == "clip_only":
            category_scores = clip_cat_scores
        else:
            category_scores = (
                self.fuser.fuse(skeleton_cat_scores, clip_cat_scores)
                if (skeleton_cat_scores is not None or clip_cat_scores is not None)
                else None
            )

        if not category_scores:
            return {
                "category": None,
                "category_scores": {},
                "skeleton": skeleton_result,
                "clip": clip_result,
            }

        top_category = max(category_scores, key=category_scores.get)
        return {
            "category": top_category,
            "category_confidence": category_scores[top_category],
            "category_scores": category_scores,
            "skeleton": skeleton_result,
            "clip": clip_result,
        }
