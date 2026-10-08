"""Diagnóstico manual escalado del FP de `class_deviations.py` (ver "Actualización 2026-09-20 (3)"
en README.md): sobre los 48 casos revisados a mano se encontró que 21/48 eran `ausencia` de un
solo día en una clase casi-diaria (ruido esperable) y al menos 1 caso parecía un desvío real
(varias clases disparando el mismo día) mal muestreado como negativo. Esto repite ese diagnóstico
sobre TODOS los días `real_sin_tocar` marcados como desvío (no una muestra de 48), con la hipótesis
concreta: si un día dispara en más de una clase a la vez, es más probable que sea un desvío real de
CASAS que ruido del detector (ruido tiende a ser independiente entre clases).

Uso:
  py -3 diagnose_fp_manual.py                 # modo sensor, config de mayoría de folds
  py -3 diagnose_fp_manual.py --camera        # modo cámara
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from evaluate_deviations import DEFAULT_DIR, load_synthetic
from src.class_deviations import CAMERA_TRACKED_CLASSES, ClassDeviationConfig, TRACKED_CLASSES, score_class_deviations
from src.dynamic_clustering import DriftConfig, score_days

# Config de mayoría de folds del cross-validation (ver README, "Cross-validation por hogar"):
# sensor = frac=0.8/excess_k=3.0; cámara = frac=0.6/excess_k=4.0. El resto quedó en default
# (dias_consecutivos_exceso_deficit=1, ausencia_requiere_racha=False) tras los dos ciclos de
# calibración posteriores (ver "Actualización 2026-09-20" x3) — ningún fold eligió otra cosa.
CONFIG_SENSOR = ClassDeviationConfig(frac_historica_min=0.8, excess_k=3.0, separar_fin_de_semana=True)
CONFIG_CAMARA = ClassDeviationConfig(frac_historica_min=0.6, excess_k=4.0, separar_fin_de_semana=False)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dir", type=Path, default=DEFAULT_DIR)
    ap.add_argument("--camera", action="store_true")
    args = ap.parse_args()

    perturbed, labels = load_synthetic(args.dir)
    tracked = CAMERA_TRACKED_CLASSES if args.camera else TRACKED_CLASSES
    class_config = CONFIG_CAMARA if args.camera else CONFIG_SENSOR

    exclude = {(row.hogar, row.fecha) for row in labels.itertuples() if row.es_positivo}
    class_hits = score_class_deviations(perturbed, tracked_classes=tracked, config=class_config, exclude_from_reference=exclude)

    if args.camera:
        combinado_dias = set(zip(class_hits["hogar"], class_hits["fecha"]))
    else:
        scored_mv = score_days(perturbed, DriftConfig(), exclude_from_reference=exclude)
        mv_dias = set(zip(scored_mv.loc[scored_mv["es_desvio"], "hogar"], scored_mv.loc[scored_mv["es_desvio"], "fecha"]))
        combinado_dias = mv_dias | set(zip(class_hits["hogar"], class_hits["fecha"]))

    negativos = labels[~labels["es_positivo"]]
    fp = negativos[[(h, f) in combinado_dias for h, f in zip(negativos["hogar"], negativos["fecha"])]]

    print(f"Modo {'CÁMARA' if args.camera else 'SENSOR'} — config: {class_config}")
    print(f"Días negativos (real_sin_tocar): {len(negativos)}  ·  marcados como desvío (FP): {len(fp)} ({len(fp) / len(negativos):.1%})")

    # Por cada día FP, cuántas clases distintas dispararon (solo cuenta el detector por clase —
    # el multivariado no desglosa por clase, así que un FP que solo vino del multivariado cuenta 0).
    hits_por_dia: dict[tuple, list[str]] = {}
    for row in class_hits.itertuples():
        hits_por_dia.setdefault((row.hogar, row.fecha), []).append(f"{row.clase}:{row.tipo}")

    detalle = []
    for row in fp.itertuples():
        clases = hits_por_dia.get((row.hogar, row.fecha), [])
        detalle.append({
            "hogar": row.hogar, "fecha": row.fecha, "escenario": row.escenario,
            "n_clases_disparadas": len(clases), "clases": "; ".join(clases),
            "solo_multivariado": len(clases) == 0,
        })
    detalle_df = pd.DataFrame(detalle)

    multi = int((detalle_df["n_clases_disparadas"] >= 2).sum())
    uno = int((detalle_df["n_clases_disparadas"] == 1).sum())
    solo_mv = int(detalle_df["solo_multivariado"].sum())

    print(f"\nDe los {len(fp)} FP:")
    print(f"  {multi:4d} ({multi / len(fp):.1%}) dispararon 2+ clases el mismo día -> hipótesis: más probable que sea un desvío real de CASAS mal muestreado como negativo")
    print(f"  {uno:4d} ({uno / len(fp):.1%}) dispararon 1 sola clase -> consistente con ruido (ausencia de 1 día en clase casi-diaria, etc.)")
    print(f"  {solo_mv:4d} ({solo_mv / len(fp):.1%}) vinieron solo del multivariado (ninguna clase individual disparó)")

    out = args.dir / f"fp_manual_{'camara' if args.camera else 'sensor'}.csv"
    detalle_df.sort_values("n_clases_disparadas", ascending=False).to_csv(out, index=False)
    print(f"\nDetalle completo (uno por fila, ordenado por n_clases_disparadas desc): {out}")
    print("\nTop 10 casos más sospechosos de ser desvíos reales (más clases disparadas a la vez):")
    print(detalle_df.sort_values("n_clases_disparadas", ascending=False).head(10).to_string(index=False))


if __name__ == "__main__":
    main()
