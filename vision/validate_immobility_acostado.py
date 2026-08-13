"""Primera validación end-to-end de la rama `acostado` (severidad "alta",
posible caída no detectada) de `FallImmobilityDetector._track_immobility` —
ver vision/README.md, "Inmovilidad prolongada": hasta ahora esta rama no
tenía ninguna validación de duración real, solo un sanity-check de postura
sobre los últimos 2s de clips de caída (muy lejos de los 180s de la
ventana).

Fuente de datos: `smarthome_CS_51.json` (anotación oficial untrimmed de
Toyota Smarthome, recuperada de
https://github.com/dairui01/Toyota_Smarthome/blob/main/pipline/data/smarthome_CS_51.json
— per-video, tuplas [label, start_frame, end_frame], label 32 = Lay_down,
28 = Get_up, ver pipline/data/Action_list del mismo repo). Se buscaron
segmentos `Lay_down -> próximo evento` largos entre los 536 videos ya
descargados en D:\\DESCARGAS\\Videos_mp4 (mismo criterio que
calibrate_immobility.py usó para Sit_down->Get_up). Los 2 más largos
encontrados son la MISMA escena grabada por 2 cámaras sincronizadas
(P25T05C05/C04, ~519s/518s) — se incluyen ambas como chequeo cruzado de
ángulo de cámara, no como 2 muestras independientes.

OJO — matiz importante: son personas acostadas en un SILLÓN, no en el piso
después de una caída (ningún dataset disponible tiene eso, ver README). No
es el escenario clínico exacto del informe (Sec 2.1.2), pero sí valida algo
que hoy no tiene ninguna evidencia: que el mecanismo (clasificación de
postura + desplazamiento suavizado) se sostiene correctamente durante
MINUTOS reales de "acostado quieto", no solo unos segundos.

Corre el pipeline completo (heurística geométrica + confirmación CLIP +
tracking de inmovilidad) frame a frame, a framerate nativo, desde un poco
antes del inicio de `Lay_down` hasta el próximo evento anotado (`Get_up` en
todos los casos elegidos).

Uso:
    vision/.venv/Scripts/python.exe validate_immobility_acostado.py
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import cv2

from src.pose_pipeline import FallImmobilityDetector
from vlm.src.clip_classifier import ClipActivityClassifier, crop_to_person
from vlm.src.fall_confirmation import FallConfirmationClassifier

VIDEOS_DIR = Path(r"D:\DESCARGAS\Videos_mp4")

# (video_id, start_frame del Lay_down, end_frame de la ventana de prueba = próximo evento anotado)
# start se retrasa 3s (75 frames a 25fps) antes del Lay_down para capturar la
# transición completa (sentado -> acostándose), igual que vería el detector en producción.
CASES = [
    ("P25T05C05", 14582 - 75, 27686),   # 519s de Lay_down->Get_up
    ("P25T05C04", 14555 - 75, 27649),   # misma escena, otra cámara (518s)
    ("P18T03C04", 0, 6986),             # 277s, escena distinta (start ya casi en 0)
]


def main() -> None:
    clip_model = ClipActivityClassifier()
    fall_confirm = FallConfirmationClassifier(clip_model)
    print(f"Filtro de confirmación: {'probe' if fall_confirm._probe_W is not None else 'zero-shot'}\n")

    for video_id, start_frame, end_frame in CASES:
        path = VIDEOS_DIR / f"{video_id}.mp4"
        cap = cv2.VideoCapture(str(path))
        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        cap.set(cv2.CAP_PROP_POS_FRAMES, max(start_frame, 0))

        detector = FallImmobilityDetector(model_path="yolo11n-pose.pt")
        raw_falls = []
        confirmed_falls = []
        immobility_events = []

        t0 = time.time()
        frame_i = max(start_frame, 0)
        while frame_i < end_frame:
            ok, image = cap.read()
            if not ok:
                break
            t = (frame_i - max(start_frame, 0)) / fps  # tiempo relativo al inicio de la ventana de prueba
            for raw in detector.process_frame(image, t):
                if raw["event_type"] == "fall":
                    raw_falls.append(raw["video_timestamp_s"])
                    if detector.last_detection is not None:
                        kpts, box = detector.last_detection
                        crop = crop_to_person(image, box)
                        confirmed, margin = fall_confirm.confirm(crop)
                        if confirmed:
                            confirmed_falls.append((raw["video_timestamp_s"], round(margin, 3)))
                elif raw["event_type"] == "immobility":
                    immobility_events.append(raw)
            frame_i += 1
        cap.release()
        elapsed = time.time() - t0
        window_s = (frame_i - max(start_frame, 0)) / fps

        print(f"=== {video_id} (ventana de {window_s:.0f}s desde 3s antes del Lay_down, {elapsed:.0f}s de cómputo) ===")
        print(f"  Caídas geométricas crudas: {len(raw_falls)}"
              + (f"  @{[round(t,1) for t in raw_falls[:5]]}" if raw_falls else ""))
        print(f"  Sobreviven el filtro CLIP: {len(confirmed_falls)}"
              + (f"  {confirmed_falls[:5]}" if confirmed_falls else ""))
        if immobility_events:
            for ev in immobility_events:
                print(f"  Inmovilidad disparada @ {ev['video_timestamp_s']:.1f}s: {ev['metadata']}")
        else:
            print(f"  Inmovilidad: nunca se disparó en la ventana")
        print()


if __name__ == "__main__":
    main()
