"""Capa *silver/gold*: de los eventos JSONL a tablas analíticas + reporte de calidad.

Toma `events/casas_*.jsonl` (salida de `etl.py`) y produce:

  events/casas_events.parquet      tabla tidy, tipada, un evento por fila (silver)
  events/casas_timeline_<bin>.parquet   timeline resampleado a intervalo fijo (gold)
  events/_quality_report.md        reporte de calidad de datos para el informe

Uso:
  py -3 dataset_events/build_table.py
  py -3 dataset_events/build_table.py --bin-min 15
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

OUT_DIR = Path(__file__).resolve().parent / "events"

_FRANJAS = [
    (0, "madrugada"),   # 00-06
    (6, "mañana"),      # 06-12
    (12, "tarde"),      # 12-19
    (19, "noche"),      # 19-24
]


def _franja(hora: int) -> str:
    out = "madrugada"
    for h, name in _FRANJAS:
        if hora >= h:
            out = name
    return out


def load_events() -> pd.DataFrame:
    rows = []
    for fp in sorted(OUT_DIR.glob("casas_*.jsonl")):
        with fp.open(encoding="utf-8") as fh:
            for line in fh:
                ev = json.loads(line)
                m = ev["metadata"]
                rows.append(
                    {
                        "video_id": ev["video_id"],
                        "hogar": m["hogar"],
                        "actividad": m["actividad"],
                        "actividad_gruesa": m["actividad_gruesa"],
                        "etiqueta_original": m["etiqueta_original"],
                        "inicio": m["inicio_iso"],
                        "fin": m["fin_iso"],
                        "duracion_s": m["duracion_s"],
                        "video_timestamp_s": ev["video_timestamp_s"],
                        "hora": m["hora"],
                        "minuto_del_dia": m["minuto_del_dia"],
                        "dia_semana": m["dia_semana"],
                    }
                )
    df = pd.DataFrame(rows)
    df["inicio"] = pd.to_datetime(df["inicio"], format="ISO8601")
    df["fin"] = pd.to_datetime(df["fin"], format="ISO8601")
    return df


def enrich(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_values(["hogar", "inicio"]).reset_index(drop=True)
    df["fecha"] = df["inicio"].dt.date
    df["fin_de_semana"] = df["dia_semana"] >= 5
    df["franja"] = df["hora"].map(_franja)
    df["duracion_min"] = (df["duracion_s"] / 60).round(2)

    # Solapamiento: el evento empieza antes de que termine el anterior del mismo hogar.
    prev_fin = df.groupby("hogar")["fin"].shift()
    df["solapa"] = df["inicio"] < prev_fin

    # Días con cambio de hora (DST): un día con span < 23.5h o > 24.5h de actividad
    # marca ambigüedad de hora local. Aproximación por hogar+fecha.
    span = df.groupby(["hogar", "fecha"])["inicio"].agg(lambda s: (s.max() - s.min()))
    # marca informativa, no corrige: hour-of-day en hora local de pared es lo correcto
    # para clustering de rutina; solo se documenta cuántos eventos caen en días DST.
    dst_days = set()
    for (h, f), sp in span.items():
        # los cambios de DST en US son la 2da semana de marzo y 1ra de noviembre
        mm = pd.Timestamp(f).month
        if mm in (3, 11) and pd.Timestamp(f).day <= 10:
            dst_days.add((h, f))
    df["dia_dst"] = list(zip(df["hogar"], df["fecha"]))
    df["dia_dst"] = df["dia_dst"].isin(dst_days)

    cats = ["video_id", "hogar", "actividad", "actividad_gruesa", "etiqueta_original", "franja"]
    for c in cats:
        df[c] = df[c].astype("category")
    df["hora"] = df["hora"].astype("int8")
    df["minuto_del_dia"] = df["minuto_del_dia"].astype("int16")
    df["dia_semana"] = df["dia_semana"].astype("int8")
    return df


def build_timeline(df: pd.DataFrame, bin_min: int) -> pd.DataFrame:
    """Resamplea a bins de ancho fijo: actividad dominante (por cobertura) por bin.

    Es el formato que suele consumir el clustering de rutinas (flujo de actividad
    a intervalo regular, tipo el paper de Aruba). Los bins sin ninguna actividad
    quedan como `sin_dato`.
    """
    freq = f"{bin_min}min"
    out = []
    for hogar, g in df.groupby("hogar", observed=True):
        t0 = g["inicio"].min().floor(freq)
        t1 = g["fin"].max().ceil(freq)
        bins = pd.date_range(t0, t1, freq=freq)
        nb = len(bins) - 1
        acts = [a for a in g["actividad_gruesa"].cat.categories if a in set(g["actividad_gruesa"])]
        cover = {a: np.zeros(nb) for a in acts}
        bin_edges = bins.asi8  # ns desde epoch
        for s, e, act in zip(g["inicio"].values.astype("int64"),
                             g["fin"].values.astype("int64"),
                             g["actividad_gruesa"].astype(str).values):
            lo = max(int(np.searchsorted(bin_edges, s, "right")) - 1, 0)
            hi = min(int(np.searchsorted(bin_edges, e, "right")) - 1, nb - 1)
            for b in range(lo, hi + 1):
                ov = min(e, bin_edges[b + 1]) - max(s, bin_edges[b])
                if ov > 0:
                    cover[act][b] += ov
        cov_df = pd.DataFrame(cover, index=bins[:-1])
        dom = cov_df.idxmax(axis=1)
        dom[cov_df.max(axis=1) == 0] = "sin_dato"
        tl = pd.DataFrame(
            {
                "hogar": hogar,
                "video_id": f"casas_{hogar}",
                "bin_inicio": bins[:-1],
                "actividad_gruesa": dom.values,
            }
        )
        tl["hora"] = tl["bin_inicio"].dt.hour.astype("int8")
        tl["dia_semana"] = tl["bin_inicio"].dt.dayofweek.astype("int8")
        tl["minuto_del_dia"] = (tl["bin_inicio"].dt.hour * 60 + tl["bin_inicio"].dt.minute).astype("int16")
        out.append(tl)
    res = pd.concat(out, ignore_index=True)
    for c in ("hogar", "video_id", "actividad_gruesa"):
        res[c] = res[c].astype("category")
    return res


def quality_report(df: pd.DataFrame, tl: pd.DataFrame, bin_min: int) -> str:
    L = []
    L.append("# Reporte de calidad — dataset de eventos CASAS\n")
    L.append(f"_Generado por `build_table.py` · {pd.Timestamp.now():%Y-%m-%d %H:%M}_\n")

    L.append("## Resumen\n")
    L.append(f"- **{len(df):,} eventos** · **{df['hogar'].nunique()} residentes**")
    L.append(f"- Rango de fechas: {df['inicio'].min():%Y-%m-%d} → {df['fin'].max():%Y-%m-%d}")
    total_h = df["duracion_s"].sum() / 3600
    L.append(f"- Tiempo total etiquetado: {total_h:,.0f} h ({total_h/24:,.0f} días-persona)")
    L.append(f"- Solapamientos entre eventos: {df['solapa'].sum():,} ({df['solapa'].mean():.1%})")
    L.append(f"- Eventos en días de cambio de hora (DST, informativo): {df['dia_dst'].sum():,}\n")

    L.append("## Distribución por actividad canónica\n")
    g = df.groupby("actividad", observed=True)["duracion_s"]
    tab = pd.DataFrame(
        {
            "eventos": g.count(),
            "%": (g.count() / len(df) * 100).round(1),
            "dur_mediana_min": (g.median() / 60).round(1),
            "dur_p95_min": (g.quantile(0.95) / 60).round(1),
            "horas_totales": (g.sum() / 3600).round(0),
        }
    ).sort_values("eventos", ascending=False)
    L.append(tab.to_markdown() + "\n")

    L.append("## Duraciones sospechosas (posible ruido de anotación)\n")
    short = df[df["duracion_s"] < 5]
    L.append(f"- Eventos < 5 s: {len(short):,} ({len(short)/len(df):.1%})")
    p99 = df.groupby("actividad", observed=True)["duracion_s"].transform(lambda s: s.quantile(0.99))
    longtail = df[df["duracion_s"] > p99]
    L.append(f"- Eventos por encima del p99 de su actividad: {len(longtail):,}")
    worst = (
        longtail.groupby("actividad", observed=True)["duracion_min"]
        .max()
        .sort_values(ascending=False)
        .head(6)
    )
    L.append(f"- Máximos por actividad (min): {worst.round(0).to_dict()}\n")

    L.append("## Cobertura por residente\n")
    per = []
    for hogar, gh in df.groupby("hogar", observed=True):
        dias_span = (gh["fin"].max() - gh["inicio"].min()).days + 1
        dias_con_datos = gh["fecha"].nunique()
        per.append(
            {
                "hogar": hogar,
                "eventos": len(gh),
                "dias_span": dias_span,
                "dias_con_datos": dias_con_datos,
                "cobertura_calendario_%": round(dias_con_datos / max(dias_span, 1) * 100, 1),
                "actividades_distintas": gh["actividad"].nunique(),
                "eventos_por_dia": round(len(gh) / max(dias_con_datos, 1), 1),
            }
        )
    per_df = pd.DataFrame(per).sort_values("eventos", ascending=False)
    L.append(per_df.to_markdown(index=False) + "\n")

    L.append(f"## Timeline resampleado ({bin_min} min)\n")
    sd = (tl["actividad_gruesa"] == "sin_dato").mean()
    L.append(f"- Bins totales: {len(tl):,}")
    L.append(f"- Bins sin actividad etiquetada (`sin_dato`): {sd:.1%}")
    dist = (tl["actividad_gruesa"].value_counts(normalize=True) * 100).round(1)
    L.append(f"- Distribución de bins por actividad gruesa:\n\n{dist.to_markdown()}\n")

    L.append("## Notas de método\n")
    L.append(
        "- **Timestamps naive en hora local de pared**: es lo correcto para clustering "
        "de rutina (importa *a qué hora del día* la persona hace algo, no el instante "
        "UTC). Los días de cambio de hora (2 por año) quedan marcados en `dia_dst` a "
        "título informativo; no se corrigen.\n"
        "- **Solapamientos**: CASAS permite tramos de actividad anidados/solapados "
        "(p. ej. `Relax` conteniendo `Watch_TV`). En la tabla *silver* se conservan "
        "como eventos separados (marcados en `solapa`). El *timeline gold* los resuelve "
        "por cobertura: la actividad con más segundos dentro del bin gana.\n"
        "- **`sin_dato`**: los bins sin ninguna actividad etiquetada son tiempo real "
        "no anotado por CASAS (transiciones, ruido), no un error del ETL.\n"
    )
    return "\n".join(L)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--bin-min", type=int, default=10, help="ancho del bin del timeline (min)")
    args = ap.parse_args()

    if not list(OUT_DIR.glob("casas_*.jsonl")):
        raise SystemExit("No hay events/casas_*.jsonl — correr primero `etl.py`.")

    print("Cargando eventos ...", flush=True)
    df = enrich(load_events())
    ev_path = OUT_DIR / "casas_events.parquet"
    df.to_parquet(ev_path, index=False)
    print(f"  {ev_path}  ({len(df):,} filas, {ev_path.stat().st_size/1e6:.1f} MB)")

    print(f"Construyendo timeline ({args.bin_min} min) ...", flush=True)
    tl = build_timeline(df, args.bin_min)
    tl_path = OUT_DIR / f"casas_timeline_{args.bin_min}min.parquet"
    tl.to_parquet(tl_path, index=False)
    print(f"  {tl_path}  ({len(tl):,} filas, {tl_path.stat().st_size/1e6:.1f} MB)")

    rep = quality_report(df, tl, args.bin_min)
    rep_path = OUT_DIR / "_quality_report.md"
    rep_path.write_text(rep, encoding="utf-8")
    print(f"  {rep_path}")


if __name__ == "__main__":
    main()
