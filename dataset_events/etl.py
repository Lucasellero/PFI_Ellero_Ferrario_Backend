"""ETL: CSV crudos de CASAS -> eventos `MonitoringEvent` para `baseline/`.

Toma los archivos de `dataset_events/labeled/*.csv` (formato CASAS: una línea
por evento de sensor, con etiquetas de actividad `X="begin"` / `X="end"`),
empareja los tramos begin/end, mapea cada etiqueta al vocabulario del proyecto
(ver `activity_map.py`) y emite un evento por tramo.

Salida:
  - `dataset_events/events/casas_<hogar>.jsonl`  (un JSON por línea)
  - `dataset_events/events/_manifest.json`       (resumen por hogar + global)
  - opcional: SQLite con el mismo esquema que `events/store.py` (--sqlite)

Uso:
  py -3 dataset_events/etl.py                          # todos los hogares
  py -3 dataset_events/etl.py --homes hh113 hh115
  py -3 dataset_events/etl.py --limit-days 10          # solo los primeros 10 días de cada hogar
  py -3 dataset_events/etl.py --sqlite dataset_events/casas_events.db
  py -3 dataset_events/etl.py --include-all            # no excluir hogares marcados

Solo stdlib. El esquema `MonitoringEvent` se importa de `../events/` para
garantizar que la salida respeta el contrato compartido.
"""
from __future__ import annotations

import argparse
import json
import re
import sqlite3
import sys
from collections import Counter, defaultdict
from datetime import datetime, timedelta
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT / "events"))
from schema import EventType, MonitoringEvent  # noqa: E402

from activity_map import (  # noqa: E402
    CANONICAL_CLASSES,
    COARSE_CLASSES,
    normalize_label,
    to_canonical,
    to_coarse,
)

LABELED_DIR = Path(__file__).resolve().parent / "labeled"
OUT_DIR = Path(__file__).resolve().parent / "events"

# Tramo begin/end más largo que esto = par corrupto, se descarta.
MAX_INTERVAL_S = 20 * 3600

# Hogares que no aportan (verificado en el análisis, ver README).
EXCLUDE: dict[str, str] = {
    "mv001": "sin etiquetas de actividad (0 tramos anotados)",
}

# Hogares usables pero con reservas documentadas (se procesan, se marcan).
NOTES: dict[str, str] = {
    "hh124": "cobertura etiquetada ~3% del tiempo de reloj",
    "hh107": "cobertura etiquetada ~10%",
    "tm016": "cobertura etiquetada ~7%",
    "tm017": "cobertura etiquetada ~11%",
    "hh126": "ruido en el emparejado begin/end",
    "hh127": "ruido en el emparejado begin/end",
    "hh128": "ruido en el emparejado begin/end (muchos cierres huérfanos)",
    "ihs06": "muchos sensores + baja cobertura: verificar si es multi-residente",
    "ihs07": "muchos sensores + baja cobertura: verificar si es multi-residente",
    "ihs08": "muchos sensores + baja cobertura: verificar si es multi-residente",
    "mn61": "muchos sensores + baja cobertura: verificar si es multi-residente",
    "mn77": "muchos sensores + baja cobertura: verificar si es multi-residente",
}

_LABEL_RE = re.compile(r'^([A-Za-z_][A-Za-z_0-9]*)\s*=\s*["\']?(begin|end)["\']?$', re.I)


def _parse_ts(date_s: str, time_s: str) -> datetime | None:
    stamp = f"{date_s.strip()} {time_s.strip()}"
    for fmt in ("%Y-%m-%d %H:%M:%S.%f", "%Y-%m-%d %H:%M:%S"):
        try:
            return datetime.strptime(stamp, fmt)
        except ValueError:
            continue
    # a veces la fracción de segundo tiene más de 6 dígitos
    m = re.match(r"(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})\.(\d+)", stamp)
    if m:
        try:
            return datetime.strptime(m.group(1), "%Y-%m-%d %H:%M:%S").replace(
                microsecond=int(m.group(2)[:6].ljust(6, "0"))
            )
        except ValueError:
            return None
    return None


def _find_label(fields: list[str]) -> tuple[str, str] | None:
    """Busca un token `X=begin` / `X=end` entre los campos 5+ de la línea."""
    for f in fields[4:]:
        m = _LABEL_RE.match(f.strip())
        if m:
            return m.group(1), m.group(2).lower()
    return None


class HomeResult:
    def __init__(self, home: str):
        self.home = home
        self.video_id = f"casas_{home}"
        self.events: list[MonitoringEvent] = []
        self.by_canonical: Counter = Counter()
        self.by_coarse: Counter = Counter()
        self.drops: Counter = Counter()
        self.unknown_labels: Counter = Counter()
        self.first_ts: datetime | None = None
        self.last_ts: datetime | None = None


def process_home(path: Path, limit_days: int | None = None) -> HomeResult:
    home = path.stem
    res = HomeResult(home)
    open_tramos: dict[str, datetime] = {}
    home_start: datetime | None = None
    cutoff: datetime | None = None

    with path.open(encoding="utf-8", errors="replace") as fh:
        for line in fh:
            line = line.rstrip("\n")
            if not line:
                continue
            fields = line.split(",")
            if len(fields) < 4:
                res.drops["linea_malformada"] += 1
                continue
            ts = _parse_ts(fields[0], fields[1])
            if ts is None:
                res.drops["timestamp_invalido"] += 1
                continue
            if home_start is None:
                home_start = ts
                if limit_days is not None:
                    cutoff = home_start + timedelta(days=limit_days)
            if cutoff is not None and ts > cutoff:
                break

            lab = _find_label(fields)
            if lab is None:
                continue
            raw, state = lab

            if state == "begin":
                if raw in open_tramos:
                    # tramo reabierto sin cerrar: cerramos el anterior acá
                    _emit(res, raw, open_tramos.pop(raw), ts, home_start)
                    res.drops["tramo_reabierto"] += 1
                open_tramos[raw] = ts
            else:  # end
                start = open_tramos.pop(raw, None)
                if start is None:
                    res.drops["cierre_huerfano"] += 1
                    continue
                _emit(res, raw, start, ts, home_start)

    res.drops["tramo_sin_cerrar"] += len(open_tramos)
    return res


def _emit(
    res: HomeResult,
    raw_label: str,
    start: datetime,
    end: datetime,
    home_start: datetime,
) -> None:
    canonical = to_canonical(raw_label)
    if canonical is None:
        res.unknown_labels[normalize_label(raw_label)] += 1
        res.drops["etiqueta_desconocida"] += 1
        return
    if canonical == "descartar":
        res.drops["etiqueta_descartada"] += 1
        return

    dur = (end - start).total_seconds()
    if dur <= 0:
        res.drops["duracion_no_positiva"] += 1
        return
    if dur > MAX_INTERVAL_S:
        res.drops["tramo_demasiado_largo"] += 1
        return

    coarse = to_coarse(canonical)
    ev = MonitoringEvent(
        video_id=res.video_id,
        event_type=EventType.ACTIVITY,
        video_timestamp_s=round((start - home_start).total_seconds(), 3),
        metadata={
            "actividad": canonical,
            "actividad_gruesa": coarse,
            "etiqueta_original": normalize_label(raw_label),
            "inicio_iso": start.isoformat(sep="T"),
            "fin_iso": end.isoformat(sep="T"),
            "duracion_s": round(dur, 1),
            "hora": start.hour,
            "minuto_del_dia": start.hour * 60 + start.minute,
            "dia_semana": start.weekday(),  # 0 = lunes
            "fuente": "CASAS",
            "hogar": res.home,
        },
    )
    res.events.append(ev)
    res.by_canonical[canonical] += 1
    res.by_coarse[coarse] += 1
    if res.first_ts is None or start < res.first_ts:
        res.first_ts = start
    if res.last_ts is None or end > res.last_ts:
        res.last_ts = end


def _quality(res: HomeResult) -> str:
    n = len(res.events)
    if n < 200:
        return "baja"
    orphan = res.drops["cierre_huerfano"]
    if orphan and orphan / (n + orphan) > 0.1:
        return "baja"
    return "ok"


def write_home(res: HomeResult) -> dict:
    OUT_DIR.mkdir(exist_ok=True)
    res.events.sort(key=lambda e: e.video_timestamp_s)
    out_path = OUT_DIR / f"{res.video_id}.jsonl"
    with out_path.open("w", encoding="utf-8") as fh:
        for ev in res.events:
            fh.write(json.dumps(ev.model_dump(mode="json"), ensure_ascii=False) + "\n")

    days = 0.0
    if res.first_ts and res.last_ts:
        days = round((res.last_ts - res.first_ts).total_seconds() / 86400, 1)
    return {
        "video_id": res.video_id,
        "hogar": res.home,
        "archivo": out_path.name,
        "n_eventos": len(res.events),
        "primer_evento": res.first_ts.isoformat() if res.first_ts else None,
        "ultimo_evento": res.last_ts.isoformat() if res.last_ts else None,
        "dias": days,
        "por_actividad": dict(res.by_canonical.most_common()),
        "por_actividad_gruesa": dict(res.by_coarse.most_common()),
        "descartes": {k: v for k, v in res.drops.items() if v},
        "etiquetas_desconocidas": dict(res.unknown_labels.most_common()),
        "calidad": _quality(res),
        "nota": NOTES.get(res.home),
    }


def write_sqlite(db_path: Path, all_events: list[MonitoringEvent]) -> None:
    conn = sqlite3.connect(db_path)
    conn.execute(
        "CREATE TABLE IF NOT EXISTS events ("
        "id INTEGER PRIMARY KEY AUTOINCREMENT, video_id TEXT NOT NULL, "
        "event_type TEXT NOT NULL, video_timestamp_s REAL NOT NULL, metadata TEXT NOT NULL)"
    )
    conn.executemany(
        "INSERT INTO events (video_id, event_type, video_timestamp_s, metadata) VALUES (?, ?, ?, ?)",
        [
            (e.video_id, e.event_type.value, e.video_timestamp_s, json.dumps(e.metadata, ensure_ascii=False))
            for e in all_events
        ],
    )
    conn.commit()
    conn.close()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--homes", nargs="*", help="hogares a procesar (default: todos)")
    ap.add_argument("--limit-days", type=int, help="solo los primeros N días de cada hogar")
    ap.add_argument("--include-all", action="store_true", help="no excluir hogares marcados")
    ap.add_argument("--sqlite", type=Path, help="además, volcar todos los eventos a este SQLite")
    args = ap.parse_args()

    if not LABELED_DIR.exists():
        sys.exit(f"No existe {LABELED_DIR} — descargar labeled_data.zip (ver README).")

    files = sorted(LABELED_DIR.glob("*.csv"))
    if args.homes:
        wanted = set(args.homes)
        files = [f for f in files if f.stem in wanted]
    if not args.include_all:
        files = [f for f in files if f.stem not in EXCLUDE]

    manifest_homes: dict[str, dict] = {}
    all_events: list[MonitoringEvent] = []
    totals = Counter()
    coarse_totals = Counter()

    for i, path in enumerate(files, 1):
        print(f"[{i}/{len(files)}] {path.stem} ...", flush=True)
        res = process_home(path, limit_days=args.limit_days)
        info = write_home(res)
        manifest_homes[res.home] = info
        all_events.extend(res.events)
        totals.update(res.by_canonical)
        coarse_totals.update(res.by_coarse)
        print(f"    {info['n_eventos']} eventos, {info['dias']}d, calidad={info['calidad']}", flush=True)

    manifest = {
        "generado": datetime.now().isoformat(timespec="seconds"),
        "fuente": "CASAS labeled_data.zip (DOI 10.5281/zenodo.15708568), CC-BY 4.0",
        "cita": "Cook et al. (2013), CASAS: A smart home in a box, IEEE Computer 46(7).",
        "limit_days": args.limit_days,
        "hogares_excluidos": EXCLUDE if not args.include_all else {},
        "vocabulario_canonico": CANONICAL_CLASSES,
        "vocabulario_grueso": COARSE_CLASSES,
        "totales": {
            "hogares": len(manifest_homes),
            "eventos": len(all_events),
            "por_actividad": dict(totals.most_common()),
            "por_actividad_gruesa": dict(coarse_totals.most_common()),
        },
        "hogares": manifest_homes,
    }
    OUT_DIR.mkdir(exist_ok=True)
    (OUT_DIR / "_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"\nManifest -> {OUT_DIR / '_manifest.json'}")

    if args.sqlite:
        write_sqlite(args.sqlite, all_events)
        print(f"SQLite  -> {args.sqlite}  ({len(all_events)} eventos)")

    print(f"\nTotal: {len(all_events):,} eventos / {len(manifest_homes)} hogares")
    print("Top actividades:", ", ".join(f"{k}={v}" for k, v in totals.most_common(8)))


if __name__ == "__main__":
    main()
