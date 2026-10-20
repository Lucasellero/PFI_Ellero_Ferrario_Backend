"""Alta de un kit "en fábrica": crea el `Kit` con sus cámaras (tantas como el plan) y muestra
UNA SOLA VEZ el código de vinculación (va impreso en la etiqueta / QR) y la API key (va en la
sección `cloud:` de `cameras.yaml` de esa controladora). En la base solo quedan los hashes.

    py alta_kit.py --camaras 2
    py alta_kit.py --camaras 1 --ids webcam --serie LR-000200
    py alta_kit.py --camaras 2 --ids cam1 cam2 --kit-yaml ../kit.yaml --url http://...

Con `--kit-yaml` el código y la API key no se muestran: se escriben en ese archivo, que es el
que leen `runner/` (sincronización) y `dashboard/` (QR para vincular desde la app). Ver
"Vincular con la app móvil" en runner/README.md.

Reemplaza al sembrado manual de un `Hogar` (el hogar ahora lo crea la vinculación del kit desde
la app, `POST /kits/vincular`). Ver `cloud/README.md`, "Correr localmente".
"""
from __future__ import annotations

import argparse
from pathlib import Path

from app.db import SessionLocal
from app.models import Camara, Kit


_KIT_YAML = """\
# Kit de esta controladora, generado por cloud/alta_kit.py. NO versionar (está en .gitignore):
# la API key y el código no se pueden recuperar de la nube, que solo guarda sus hashes.
url: {url}
numero_serie: {serie}
codigo_vinculacion: {codigo}   # el del QR: lares://vincular?serie=...&codigo=...
api_key: {api_key}
interval_s: 10
# Cámara del kit a la que se sube la fuente elegida con el botón "App" del dashboard.
camara: {camara}
# true: no sube los eventos viejos ni los de antes de vincular (para demos). false: los sube
# todos cuando el kit se vincula, como una controladora real.
solo_desde_vinculacion: true
"""


def _siguiente_serie(db) -> str:
    series = [s for (s,) in db.query(Kit.numero_serie).all() if s.startswith("LR-") and s[3:].isdigit()]
    ultimo = max((int(s[3:]) for s in series), default=0)
    return f"LR-{ultimo + 1:06d}"


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--camaras", type=int, choices=(1, 2, 3), default=1, help="plan comprado (1 a 3)")
    p.add_argument("--ids", nargs="+", help="identificador_local de cada cámara (default cam1..camN); "
                                             "tiene que coincidir con los `id` de cameras.yaml")
    p.add_argument("--serie", help="número de serie (default: el siguiente LR-NNNNNN libre)")
    p.add_argument("--kit-yaml", type=Path,
                   help="escribir serie, código y API key en este archivo (kit.yaml de la controladora) "
                        "en vez de mostrarlos; no pisa un archivo existente")
    p.add_argument("--url", help="URL base del backend que va en --kit-yaml (sin /devices/events)")
    args = p.parse_args()
    if args.kit_yaml and not args.url:
        p.error("--kit-yaml necesita --url")
    if args.kit_yaml and args.kit_yaml.exists():
        # Tiene la API key del kit anterior, que no se puede recuperar de la base.
        p.error(f"{args.kit_yaml} ya existe: movelo o borralo antes (perderías la API key de ese kit)")

    ids = args.ids or [f"cam{i}" for i in range(1, args.camaras + 1)]
    if len(ids) != args.camaras:
        p.error(f"--ids tiene {len(ids)} cámaras pero el plan es de {args.camaras}")

    db = SessionLocal()
    serie = args.serie or _siguiente_serie(db)
    codigo, codigo_hash = Kit.generar_codigo_vinculacion()
    api_key, api_key_hash = Kit.generar_api_key()
    kit = Kit(numero_serie=serie, cantidad_camaras=args.camaras,
              codigo_vinculacion_hash=codigo_hash, device_api_key_hash=api_key_hash)
    db.add(kit)
    db.flush()
    db.add_all(Camara(kit_id=kit.id, identificador_local=i) for i in ids)
    db.commit()

    print(f"Kit {serie} creado (id {kit.id}), cámaras: {', '.join(ids)}")
    if args.kit_yaml:
        args.kit_yaml.write_text(_KIT_YAML.format(
            url=args.url.rstrip("/"), serie=serie, codigo=codigo, api_key=api_key, camara=ids[0],
        ), encoding="utf-8")
        print(f"  Código y API key guardados en {args.kit_yaml} (no se muestran).")
        return
    print(f"  Código de vinculación: {codigo[:4]}-{codigo[4:]}")
    print(f"  QR:                    lares://vincular?serie={serie}&codigo={codigo}")
    print(f"  API key (cameras.yaml -> cloud.api_key): {api_key}")
    print("Guardalos ahora: en la base solo quedan los hashes, no se pueden recuperar.")


if __name__ == "__main__":
    main()
