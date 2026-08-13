"""Pose estimation + heurística de caída/inmovilidad con YOLOv11n-pose (Ultralytics).

Descriptores según Giri et al. (2025), citado en el informe (Sec. 2.2.7):
velocidad vertical de la cadera y aspect ratio del bounding box.

Heurística de caída calibrada empíricamente: la velocidad vertical y el
aspect ratio "acostado" NO ocurren en el mismo frame — el pico de velocidad
pasa durante la caída (~0.3-1s) y el aspect ratio recién se estabiliza
después, cuando el cuerpo termina de asentarse en el piso. Por eso la
detección es en dos fases (pico de velocidad, después confirmación por
postura), no una condición simultánea. El ángulo de torso, calibrado
inicialmente según Giri et al., se descartó de la condición porque en
caídas laterales no llega a valores altos y hacía perder ~2/16 casos sin
mejorar la precisión de los que sí detectaba.

La velocidad vertical se normaliza por el alto del bbox de la persona (no
por el alto del frame): sin eso, el umbral quedaba atado a la distancia
cámara-sujeto de un único setup. Los umbrales actuales (`FALL_VERTICAL_VELOCITY`,
`FALL_ASPECT_RATIO`) se recalibraron con un grid search sobre 128 clips con
caída + 51 sin caída de dos datasets con cámaras distintas (`videos/GMDCSA-24/`,
mismo encuadre fijo; `videos/CAUCAFall/`, encuadre/distancia variable por
sujeto — ver `recalibrate_fall_thresholds.py`), para generalizar entre
setups de cámara en vez de sobreajustar a uno solo. Recall combinado 85.2%
(vs. 66.4% con los umbrales viejos y la velocidad sin normalizar), FP 18/179
clips evaluados.

Confirmación por postura, segunda condición, EXPERIMENTAL/DESACTIVADA
(2026-08-11): `FALL_ASPECT_RATIO` solo no detecta caídas hacia
adelante/hacia la cámara — la persona termina encogida (bbox angosto), no
"ancha" como en una caída lateral, así que nunca cruza el umbral (9/10
sujetos "Fall forward" de CAUCAFall no generaban NINGÚN evento). Se probó una
condición alternativa (OR): el bbox se derrumba en altura respecto a la
"altura de pie" de referencia (`FALL_HEIGHT_DROP_RATIO`) Y la cadera termina
cerca del piso del frame (`FALL_FLOOR_HIP_Y`, posición absoluta, no relativa
al bbox) — el segundo requisito filtra la mayoría de sentadas/agachadas
normales (que sin él también derrumban el bbox y disparan todo el tiempo).

Calibrado con `recalibrate_fall_thresholds.py` sobre 292 clips (los mismos +
`videos/URFD/` + los 4 clips largos de `ToyotaSmarthome_FP/`): parecía dar
86.1%->87.3% recall por 46->55 FP, PERO ese conteo capea a 1 FP por clip —
ciego a que la condición nueva se puede disparar repetidamente dentro del
mismo video. Validado con `evaluate_falls.py` (conteo real, sin capear): un
solo clip largo (~19min, sedentario) disparó **124 veces** — la persona
cambiando de postura naturalmente re-cumple la condición una y otra vez, cada
~9s en promedio. Total real: 340 FP sobre 292 clips (vs. ~46-146 del
baseline sin esta condición, según el conteo). Inutilizable así en
producción (spam de alertas sobre video continuo). Queda `FALL_HEIGHT_DROP_RATIO
= None` (desactivada) hasta re-tunear con una métrica que penalice disparos
repetidos, no solo si el clip tuvo *algún* FP. Detalle completo en
`vision/README.md`.
"""
from __future__ import annotations

import math
from collections import Counter, deque
from dataclasses import dataclass

import numpy as np
from ultralytics import YOLO

# Índices de keypoints COCO (formato usado por los modelos de pose de YOLO).
L_SHOULDER, R_SHOULDER, L_HIP, R_HIP = 5, 6, 11, 12

# Pares de keypoints COCO que forman el esqueleto, para dibujar overlays de debug.
SKELETON_EDGES = [
    (5, 6), (5, 11), (6, 12), (11, 12),          # torso
    (5, 7), (7, 9), (6, 8), (8, 10),              # brazos
    (11, 13), (13, 15), (12, 14), (14, 16),       # piernas
    (0, 5), (0, 6),                                # cabeza-hombros
]

FALL_VELOCITY_WINDOW_S = 0.4          # ventana para medir la velocidad vertical de la cadera
FALL_VERTICAL_VELOCITY = 0.12         # "alturas de persona" por segundo (hip_y / bbox_height_frac) => posible caída en curso
FALL_ASPECT_RATIO = 1.4               # width/height por encima de esto => postura final acostada
FALL_CONFIRM_WINDOW_S = 1.5           # tiempo máx. entre el pico de velocidad y la postura final
FALL_HEIGHT_DROP_RATIO = None          # bbox_height_frac cae por debajo de esta fracción de la altura "de pie" => posible caída (forward, ver docstring). None = desactivado: la calibración inicial (0.55/0.75) disparaba en loop sobre video largo continuo (124 FP en un solo clip de 19min, ver vision/README.md) porque el barrido de calibración capeaba el conteo de FP a 1 por clip, sin penalizar disparos repetidos. Pendiente re-tunear con una métrica que sí penalice eso.
FALL_FLOOR_HIP_Y = 0.75               # + hip_y (posición absoluta en el frame) por encima de esto => cerca del piso, no sentado a media altura
FALL_BASELINE_WINDOW_S = 2.0          # ventana antes del pico de velocidad para estimar la altura "de pie" de referencia
FALL_COOLDOWN_S = 3.0                 # tras confirmar una caída, no reportar otra hasta pasado esto
MIN_KEYPOINT_CONFIDENCE = 0.2         # confianza promedio mínima de keypoints para no descartar el frame
IMMOBILITY_WINDOW_S = 180.0            # segundos de movimiento casi nulo => evento de inmovilidad (sedentarismo sostenido, no un instante)
IMMOBILITY_MOVEMENT_THRESHOLD = 0.02   # desplazamiento normalizado de keypoints POR SEGUNDO (no por frame, ver _smoothed_displacement_rate) => "se está moviendo"
IMMOBILITY_SAMPLE_WINDOW_S = 2.0       # ventana para promediar el desplazamiento: un solo frame ruidoso (dar vuelta una hoja) no debe resetear la cuenta
IMMOBILITY_POSTURE_VOTE_N = 5          # cuántas clasificaciones de postura recientes se usan (mayoría) para decidir sentado/acostado al reportar
IMMOBILITY_HISTORY_SECONDS = 15.0      # retención propia de _smoothed_displacement_rate, mayor a history_seconds (2.5, pensado para la velocidad de caída) — ver docstring de _smoothed_displacement_rate


@dataclass
class PoseSample:
    timestamp_s: float
    hip_y: float               # normalizado (0 = arriba, 1 = abajo del frame)
    bbox_height_frac: float    # alto del bbox / alto del frame, para normalizar la velocidad por la escala de la persona
    aspect_ratio: float
    torso_angle_deg: float
    keypoints: np.ndarray      # normalizados a [0, 1] por ancho/alto del frame


def detect_person(model: YOLO, image: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray | None] | None:
    """Corre el modelo de pose sobre un frame y devuelve (keypoints, bbox,
    keypoints_conf) de la persona detectada, o None si no hay ninguna
    detección confiable.

    Si hay más de una persona en el frame, se toma la del bounding box más
    grande (más cercana/prominente a cámara) como sujeto principal — el
    informe excluye explícitamente el soporte multi-residente (Sec. 1.2.2),
    así que esto es una heurística de desambiguación, no tracking real.
    Se descartan detecciones con confianza promedio de keypoints por debajo
    de MIN_KEYPOINT_CONFIDENCE (oclusión, movimiento, sujeto lejos de
    cámara) siguiendo el criterio de calidad usado en Benabdennour et al.
    (2026, IEEE Access) para este mismo problema. `keypoints_conf` (confianza
    por keypoint individual, no solo el promedio) se devuelve además para que
    los callers puedan descartar puntos puntuales poco confiables (ej. una
    muñeca ocluida) sin descartar la detección entera.
    """
    result = model(image, verbose=False)[0]
    if result.keypoints is None or len(result.keypoints.xy) == 0:
        return None

    boxes = result.boxes.xyxy.cpu().numpy()
    areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    idx = int(areas.argmax())

    kpts_conf = None
    if result.keypoints.conf is not None:
        kpts_conf = result.keypoints.conf[idx].cpu().numpy()
        mean_confidence = float(kpts_conf.mean())
        if mean_confidence < MIN_KEYPOINT_CONFIDENCE:
            return None

    kpts = result.keypoints.xy[idx].cpu().numpy()
    box = boxes[idx]
    return kpts, box, kpts_conf


def build_pose_sample(kpts: np.ndarray, box: np.ndarray, w: int, h: int, timestamp_s: float) -> PoseSample:
    x1, y1, x2, y2 = box
    bw, bh = max(x2 - x1, 1e-6), max(y2 - y1, 1e-6)
    # float(...) porque box/kpts vienen de numpy (float32), que no es
    # serializable a JSON al guardar el evento en events/store.py.
    aspect_ratio = float(bw / bh)

    hip = np.nanmean([kpts[L_HIP], kpts[R_HIP]], axis=0)
    shoulder = np.nanmean([kpts[L_SHOULDER], kpts[R_SHOULDER]], axis=0)
    dx, dy = shoulder[0] - hip[0], shoulder[1] - hip[1]
    torso_angle_deg = float(math.degrees(math.atan2(abs(dx), abs(dy) + 1e-6)))

    return PoseSample(
        timestamp_s=timestamp_s,
        hip_y=float(hip[1] / h),
        bbox_height_frac=float(bh / h),
        aspect_ratio=aspect_ratio,
        torso_angle_deg=torso_angle_deg,
        keypoints=kpts / np.array([w, h]),
    )


def draw_pose(image: np.ndarray, kpts: np.ndarray, box: np.ndarray) -> np.ndarray:
    """Dibuja el bounding box y el esqueleto de una detección sobre una copia
    del frame. Solo para debug/visualización (ej. preview del dashboard) —
    no se usa en la lógica de detección."""
    import cv2

    out = image.copy()
    x1, y1, x2, y2 = box.astype(int)
    cv2.rectangle(out, (x1, y1), (x2, y2), (0, 255, 0), 2)

    for a, b in SKELETON_EDGES:
        if a >= len(kpts) or b >= len(kpts):
            continue
        xa, ya = kpts[a]
        xb, yb = kpts[b]
        if (xa <= 0 and ya <= 0) or (xb <= 0 and yb <= 0):
            continue
        cv2.line(out, (int(xa), int(ya)), (int(xb), int(yb)), (0, 200, 255), 2)

    for x, y in kpts:
        if x <= 0 and y <= 0:
            continue
        cv2.circle(out, (int(x), int(y)), 3, (0, 0, 255), -1)

    return out


class FallImmobilityDetector:
    """Consume frames de un video y emite eventos 'fall' / 'immobility'.

    Umbrales de caída recalibrados sobre 179 clips de dos datasets con
    cámaras distintas, ver docstring del módulo y
    `recalibrate_fall_thresholds.py`. Umbral de inmovilidad calibrado sobre
    datos reales de sedentarismo (Toyota Smarthome untrimmed, ver
    `calibrate_immobility.py` y `../README.md`, sección "Inmovilidad
    prolongada"); la rama de severidad "alta" (persona acostada) sigue sin
    dataset propio de validación.
    """

    def __init__(self, model_path: str = "yolo11n-pose.pt", history_seconds: float = 2.5, device: str | None = None):
        self._model = YOLO(model_path)
        if device is None:
            import torch
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self._model.to(device)
        self._history: deque[PoseSample] = deque()
        self._history_seconds = history_seconds
        # Historial propio para inmovilidad, separado de self._history (2.5s,
        # dimensionado para la velocidad de caída) — ver _smoothed_displacement_rate.
        self._immobility_history: deque[PoseSample] = deque()
        self._immobile_since: float | None = None
        self._immobility_reported = False
        self._posture_history: deque[str] = deque(maxlen=IMMOBILITY_POSTURE_VOTE_N)
        self._last_velocity_spike_t: float | None = None
        self._baseline_height: float | None = None
        self._heights_history: deque[tuple[float, float]] = deque()  # (t, bbox_height_frac), ventana propia para estimar altura "de pie"
        self._fall_cooldown_until: float = float("-inf")
        self.last_detection: tuple[np.ndarray, np.ndarray] | None = None
        # Confianza por keypoint de la última detección, separada de
        # last_detection para no romper los call sites que hacen
        # `kpts, box = det.last_detection` o `det.last_detection[1]`.
        self.last_detection_conf: np.ndarray | None = None

    def process_frame(self, image: np.ndarray, timestamp_s: float) -> list[dict]:
        events: list[dict] = []
        h, w = image.shape[:2]
        detection = detect_person(self._model, image)
        if detection is None:
            self.last_detection = None
            self.last_detection_conf = None
            return events
        kpts, box, kpts_conf = detection
        self.last_detection = (kpts, box)
        self.last_detection_conf = kpts_conf

        sample = build_pose_sample(kpts, box, w, h, timestamp_s)
        self._history.append(sample)
        self._trim_history(sample.timestamp_s)
        self._immobility_history.append(sample)
        while self._immobility_history and sample.timestamp_s - self._immobility_history[0].timestamp_s > IMMOBILITY_HISTORY_SECONDS:
            self._immobility_history.popleft()
        self._heights_history.append((sample.timestamp_s, sample.bbox_height_frac))
        while self._heights_history and sample.timestamp_s - self._heights_history[0][0] > FALL_BASELINE_WINDOW_S + FALL_CONFIRM_WINDOW_S:
            self._heights_history.popleft()

        if self._detect_fall(sample):
            events.append({
                "event_type": "fall",
                "video_timestamp_s": sample.timestamp_s,
                "metadata": {
                    "aspect_ratio": sample.aspect_ratio,
                    "torso_angle_deg": sample.torso_angle_deg,
                },
            })

        events.extend(self._track_immobility(sample))
        return events

    def _trim_history(self, now: float) -> None:
        while self._history and now - self._history[0].timestamp_s > self._history_seconds:
            self._history.popleft()

    def _detect_fall(self, sample: PoseSample) -> bool:
        """Dos fases: (1) un pico de velocidad vertical marca una posible caída
        en curso, (2) si dentro de FALL_CONFIRM_WINDOW_S la postura confirma
        el aterrizaje, se confirma el evento. No exige que ambas condiciones
        ocurran en el mismo frame (ver docstring del módulo).

        Confirmación por postura, dos condiciones alternativas (OR): el
        aspect ratio se ensancha (caída lateral/de espaldas) O el bbox se
        derrumba en altura Y la cadera termina cerca del piso del frame
        (caída hacia adelante/hacia la cámara, no ensancha el bbox — ver
        docstring del módulo).
        """
        if sample.timestamp_s < self._fall_cooldown_until:
            return False

        velocity = self._vertical_velocity(sample)
        if velocity is not None and velocity > FALL_VERTICAL_VELOCITY:
            self._last_velocity_spike_t = sample.timestamp_s
            pre_spike = [h for t, h in self._heights_history if sample.timestamp_s - t <= FALL_BASELINE_WINDOW_S]
            self._baseline_height = max(pre_spike) if pre_spike else None

        posture_ok = sample.aspect_ratio > FALL_ASPECT_RATIO
        if not posture_ok and FALL_HEIGHT_DROP_RATIO is not None and self._baseline_height:
            posture_ok = (
                sample.bbox_height_frac < self._baseline_height * FALL_HEIGHT_DROP_RATIO
                and sample.hip_y > FALL_FLOOR_HIP_Y
            )

        if (
            self._last_velocity_spike_t is not None
            and posture_ok
            and sample.timestamp_s - self._last_velocity_spike_t <= FALL_CONFIRM_WINDOW_S
        ):
            self._last_velocity_spike_t = None
            self._baseline_height = None
            self._fall_cooldown_until = sample.timestamp_s + FALL_COOLDOWN_S
            return True
        return False

    def _vertical_velocity(self, sample: PoseSample) -> float | None:
        """Velocidad vertical de la cadera vs. la muestra más cercana a
        FALL_VELOCITY_WINDOW_S segundos atrás (no frame-a-frame, no el
        historial completo: ambos extremos diluyen el pico real de la caída).

        Se normaliza por el alto del bbox de la persona (promedio entre la
        muestra actual y la de referencia), no por el alto del frame: sin
        esto, el mismo movimiento real da velocidades distintas según la
        distancia cámara-sujeto (más lejos => bbox más chico => se subestima
        la velocidad y se pierden caídas; más cerca => se sobreestima y
        dispara con movimientos normales). Confirmado empíricamente al
        evaluar sobre CAUCAFall, que varía la distancia de cámara por sujeto
        (recall 28% vs. 91% en los clips propios, todos a la misma distancia).
        """
        if not self._history:
            return None
        target_t = sample.timestamp_s - FALL_VELOCITY_WINDOW_S
        ref = min(self._history, key=lambda s: abs(s.timestamp_s - target_t))
        dt = sample.timestamp_s - ref.timestamp_s
        if dt < FALL_VELOCITY_WINDOW_S / 2:
            return None
        scale = (sample.bbox_height_frac + ref.bbox_height_frac) / 2
        if scale <= 1e-6:
            return None
        return (sample.hip_y - ref.hip_y) / scale / dt

    def _track_immobility(self, sample: PoseSample) -> list[dict]:
        """Calibrado empíricamente sobre 1990 muestras "sentado quieto" (huecos
        Sit_down->Get_up sin eventos etiquetados, dataset untrimmed de Toyota
        Smarthome) vs. 168 muestras "caminando", ver docstring del módulo.
        Mediana de desplazamiento por segundo: 0.0056 (quieto) vs. 0.0606
        (caminando) — separación clara, umbral 0.02 da ~8% de error en cada
        distribución tomado frame a frame.

        Ese ~8% de frames "quieto" que superan el umbral por ruido puntual
        (acomodarse, dar vuelta una hoja) es justamente por lo que la cuenta
        de inmovilidad no se resetea con un solo frame por encima del umbral:
        se usa el desplazamiento *promediado* sobre IMMOBILITY_SAMPLE_WINDOW_S
        (`_smoothed_displacement_rate`), no frame a frame, igual criterio de
        "no exigir un instante perfecto" que ya se usa para caídas.
        """
        from .posture import classify_posture  # import local: posture.py importa PoseSample de este módulo

        events: list[dict] = []
        rate = self._smoothed_displacement_rate(sample)
        moving = rate is None or rate > IMMOBILITY_MOVEMENT_THRESHOLD
        self._posture_history.append(classify_posture(sample))

        if moving:
            self._immobile_since = None
            self._immobility_reported = False
        else:
            if self._immobile_since is None:
                self._immobile_since = sample.timestamp_s
            elapsed = sample.timestamp_s - self._immobile_since
            if elapsed > IMMOBILITY_WINDOW_S and not self._immobility_reported:
                posture = self._majority_posture()
                # "parado" inmóvil no es un patrón contemplado (ni sedentarismo ni
                # post-caída, ver README) — no se reporta. No se marca
                # _immobility_reported=True: si la postura pasa a sentado/acostado
                # más adelante sin que se haya movido, ahí sí se reporta (elapsed
                # sigue corriendo, no se resetea por esto).
                if posture != "parado":
                    self._immobility_reported = True
                    events.append({
                        "event_type": "immobility",
                        "video_timestamp_s": sample.timestamp_s,
                        "metadata": {
                            "duration_s": elapsed,
                            "posture": posture,
                            # "alta": posible caída no detectada, persona inmóvil en el piso (sin
                            # validar con ground truth real, ver README). "baja": sedentarismo
                            # prolongado sentado (sí calibrado con datos reales, ver README).
                            "severity": "alta" if posture == "acostado" else "baja",
                        },
                    })
        return events

    def _smoothed_displacement_rate(self, sample: PoseSample) -> float | None:
        """Desplazamiento de keypoints entre `sample` y la muestra más cercana
        a IMMOBILITY_SAMPLE_WINDOW_S segundos atrás, normalizado por el tiempo
        real transcurrido (mismo criterio que _vertical_velocity: sin esto, el
        frame dropping en tiempo real de `runner/` — ver runner/README.md —
        cambiaría el desplazamiento medido según cuántos frames se saltean,
        no según cuánto se mueve la persona).

        Usa `self._immobility_history` (retención `IMMOBILITY_HISTORY_SECONDS`
        = 15s), NO `self._history` (2.5s, dimensionado para `_vertical_velocity`
        de caída). Bug encontrado el 2026-08-13 validando la rama `acostado`
        con datos reales (ver vision/README.md): con `self._history` (2.5s),
        cualquier hueco de detección más largo que eso —normal en video
        doméstico real: cambios de ángulo, oclusión parcial, blur— vacía el
        historial, `_smoothed_displacement_rate` devuelve None,
        `_track_immobility` lo trata como "moviéndose" y resetea el contador
        de inmovilidad. En un clip real y limpio (sin oclusión) donde la
        persona se quedó quieta 213s medidos offline, el detector end-to-end
        nunca disparó — el motivo no era que la persona se moviera, era que
        un solo hueco de detección de ~4s (nada anómalo) volaba el contador."""
        if not self._immobility_history:
            return None
        target_t = sample.timestamp_s - IMMOBILITY_SAMPLE_WINDOW_S
        ref = min(self._immobility_history, key=lambda s: abs(s.timestamp_s - target_t))
        dt = sample.timestamp_s - ref.timestamp_s
        if dt < IMMOBILITY_SAMPLE_WINDOW_S / 2:
            return None
        return float(np.nanmean(np.abs(sample.keypoints - ref.keypoints))) / dt

    def _majority_posture(self) -> str:
        if not self._posture_history:
            return "sentado"
        return Counter(self._posture_history).most_common(1)[0][0]
