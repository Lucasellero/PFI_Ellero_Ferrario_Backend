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

# Calidad de encuadre (ver "evidence" en el evento de caída y vision/README.md):
# cuando la persona no entra entera en el frame, el bbox se recorta contra el
# borde => aspect_ratio ensanchado artificialmente + hip_y saltando de golpe =>
# la heurística de dos fases lo lee como caída. Un evento así no se descarta,
# pero se marca `evidence="partial"` para que el runner no lo confirme con CLIP
# (recorte truncado => CLIP poco confiable) y en cambio lo escale por
# inmovilidad si la persona queda quieta.
FRAME_EDGE_MARGIN_FRAC = 0.012        # el bbox a menos de esta fracción del borde superior/izq/der => "toca el borde" (el borde inferior no cuenta: los pies tocándolo es encuadre normal de alguien parado)
TORSO_KEYPOINT_MIN_CONF = 0.5         # confianza mínima de hombros+cadera (COCO 5,6,11,12) para considerar el torso visible
IMMOBILITY_WINDOW_POSTFALL_S = 45.0   # tras una caída de evidencia parcial, cuántos segundos de inmovilidad escalan a evento de severidad alta (ventana corta: es una posible emergencia, no sedentarismo)
IMMOBILITY_WINDOW_S = 180.0            # segundos de movimiento casi nulo => evento de inmovilidad (sedentarismo sostenido, no un instante)
IMMOBILITY_MOVEMENT_THRESHOLD = 0.02   # desplazamiento normalizado de keypoints POR SEGUNDO (no por frame, ver _smoothed_displacement_rate) => "se está moviendo"
IMMOBILITY_SAMPLE_WINDOW_S = 2.0       # ventana para promediar el desplazamiento: un solo frame ruidoso (dar vuelta una hoja) no debe resetear la cuenta
IMMOBILITY_POSTURE_VOTE_N = 5          # cuántas clasificaciones de postura recientes se usan (mayoría) para decidir sentado/acostado al reportar
IMMOBILITY_HISTORY_SECONDS = 15.0      # retención propia de _smoothed_displacement_rate, mayor a history_seconds (2.5, pensado para la velocidad de caída) — ver docstring de _smoothed_displacement_rate

# Histéresis de detección (ver FallImmobilityDetector._detect y vision/README.md,
# "Histéresis de detección"): una persona en reposo en un sillón vista desde
# cámara ancha/elevada cae por debajo del umbral de caja de YOLO buena parte
# del tiempo. Bajar el umbral global la recupera pero multiplica las cajas
# espurias (x7 a 0.05). En cambio: umbral ALTO para crear un seguimiento,
# umbral bajo solo para mantenerlo, y solo si la caja débil solapa con la caja
# que lo creó (el "ancla"). Las detecciones débiles alimentan únicamente el
# tracking de inmovilidad — nunca la heurística de caída.
PERSON_BOX_CONFIDENCE = 0.25           # confianza de caja para una detección normal (= default de Ultralytics, el que ya se usaba implícitamente)
TRACK_CREATE_BOX_CONFIDENCE = 0.5      # confianza de caja para que una detección normal fije el ancla. Más alto que PERSON_BOX_CONFIDENCE a propósito: los falsos positivos de YOLO sobre objetos (ej. una planta) caen en 0.25-0.31, y sostenerlos con cajas débiles los convertía en una "persona" permanente
TRACK_KEEP_BOX_CONFIDENCE = 0.05       # confianza de caja mínima para mantener un seguimiento ya creado
TRACK_KEEP_MIN_IOU = 0.5               # solape mínimo de la caja débil con el ancla (no con la débil anterior: encadenar débiles deja que el seguimiento derive hacia otro objeto)
TRACK_KEEP_MAX_GAP_S = 3.0             # sin ninguna caja aceptada por más de esto => el seguimiento se pierde (hace falta otra detección que fije ancla)
TRACK_WEAK_MAX_S = 60.0                # tope desde la última vez que se fijó el ancla: un seguimiento "fantasma" nunca llega a IMMOBILITY_WINDOW_S por sí solo


@dataclass
class PoseSample:
    timestamp_s: float
    hip_y: float               # normalizado (0 = arriba, 1 = abajo del frame)
    bbox_height_frac: float    # alto del bbox / alto del frame, para normalizar la velocidad por la escala de la persona
    aspect_ratio: float
    torso_angle_deg: float
    keypoints: np.ndarray      # normalizados a [0, 1] por ancho/alto del frame
    bbox_touches_edge: bool = False   # el bbox llega al borde superior/izq/der del frame => persona parcialmente fuera de cuadro
    torso_visible: bool = True        # hombros + cadera detectados con confianza suficiente (ver TORSO_KEYPOINT_MIN_CONF)
    bbox: np.ndarray | None = None    # (x1, y1, x2, y2) normalizado a [0, 1] por ancho/alto del frame
    weak: bool = False                # detección débil sostenida por histéresis (ver TRACK_KEEP_BOX_CONFIDENCE): keypoints no confiables, solo sirve la caja

    @property
    def framing_ok(self) -> bool:
        """Encuadre confiable para la heurística geométrica: persona entera en
        cuadro y torso visible. Si es False, un evento de caída se marca
        `evidence="partial"` (ver FallImmobilityDetector._detect_fall)."""
        return not self.bbox_touches_edge and self.torso_visible


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

    kpts, box, kpts_conf = _extract(result, boxes, idx)
    if kpts_conf is not None and float(kpts_conf.mean()) < MIN_KEYPOINT_CONFIDENCE:
        return None
    return kpts, box, kpts_conf


def _extract(result, boxes: np.ndarray, idx: int) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
    kpts_conf = None
    if result.keypoints.conf is not None:
        kpts_conf = result.keypoints.conf[idx].cpu().numpy()
    return result.keypoints.xy[idx].cpu().numpy(), boxes[idx], kpts_conf


def _iou(a: np.ndarray, b: np.ndarray) -> float:
    iw = min(a[2], b[2]) - max(a[0], b[0])
    ih = min(a[3], b[3]) - max(a[1], b[1])
    if iw <= 0 or ih <= 0:
        return 0.0
    inter = iw * ih
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return float(inter / union) if union > 0 else 0.0


def build_pose_sample(
    kpts: np.ndarray,
    box: np.ndarray,
    w: int,
    h: int,
    timestamp_s: float,
    kpts_conf: np.ndarray | None = None,
) -> PoseSample:
    x1, y1, x2, y2 = box
    bw, bh = max(x2 - x1, 1e-6), max(y2 - y1, 1e-6)
    # float(...) porque box/kpts vienen de numpy (float32), que no es
    # serializable a JSON al guardar el evento en events/store.py.
    aspect_ratio = float(bw / bh)

    hip = np.nanmean([kpts[L_HIP], kpts[R_HIP]], axis=0)
    shoulder = np.nanmean([kpts[L_SHOULDER], kpts[R_SHOULDER]], axis=0)
    dx, dy = shoulder[0] - hip[0], shoulder[1] - hip[1]
    torso_angle_deg = float(math.degrees(math.atan2(abs(dx), abs(dy) + 1e-6)))

    # El borde inferior no cuenta: los pies de alguien parado suelen tocarlo y
    # eso es encuadre normal. Recorte por arriba/izquierda/derecha sí trunca el
    # cuerpo y rompe la premisa geométrica de la heurística.
    mx, my = FRAME_EDGE_MARGIN_FRAC * w, FRAME_EDGE_MARGIN_FRAC * h
    bbox_touches_edge = bool(x1 <= mx or y1 <= my or x2 >= w - mx)

    torso_visible = True
    if kpts_conf is not None:
        torso_visible = bool(
            np.all(np.asarray(kpts_conf)[[L_SHOULDER, R_SHOULDER, L_HIP, R_HIP]] >= TORSO_KEYPOINT_MIN_CONF)
        )

    return PoseSample(
        timestamp_s=timestamp_s,
        hip_y=float(hip[1] / h),
        bbox_height_frac=float(bh / h),
        aspect_ratio=aspect_ratio,
        torso_angle_deg=torso_angle_deg,
        keypoints=kpts / np.array([w, h]),
        bbox_touches_edge=bbox_touches_edge,
        torso_visible=torso_visible,
        bbox=np.asarray(box, dtype=float) / np.array([w, h, w, h]),
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

    def __init__(
        self,
        model_path: str = "yolo11n-pose.pt",
        history_seconds: float = 2.5,
        device: str | None = None,
        postfall_window_s: float = IMMOBILITY_WINDOW_POSTFALL_S,
    ):
        self._model = YOLO(model_path)
        if device is None:
            import torch
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self._model.to(device)
        self._history_seconds = history_seconds
        self._postfall_window_s = postfall_window_s
        self.reset()

    def reset(self) -> None:
        """Olvida todo lo visto (historiales, cooldown de caída, cuentas de
        inmovilidad, seguimiento) sin recargar el modelo. Para cuando el video
        arranca de nuevo desde el principio: el contenido no es continuación
        del anterior aunque el timestamp siga creciendo."""
        self._history: deque[PoseSample] = deque()
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
        # Vigilancia post-caída: tras una caída de evidencia parcial (persona
        # parcialmente fuera de cuadro, ver _detect_fall), se arma un contador
        # de inmovilidad con ventana corta (postfall_window_s) que escala a un
        # evento de severidad alta si la persona queda quieta — y se cancela si
        # se levanta. Independiente de _track_immobility (sedentarismo, 180s).
        self._postfall_since: float | None = None
        self._postfall_reported = False
        # Histéresis de detección (ver _detect).
        self._anchor_box: np.ndarray | None = None
        self._anchor_t: float = float("-inf")
        self._track_last_t: float = float("-inf")
        # True si en el último frame hay una persona, por detección normal O
        # por una detección débil sostenida por histéresis. `last_detection`
        # sigue siendo None en el segundo caso (no hay keypoints usables), así
        # que quien solo necesite "¿hay alguien?" (ej. el gate de movimiento
        # de runner/) debe mirar esto y no `last_detection`.
        self.person_present: bool = False
        self.last_detection: tuple[np.ndarray, np.ndarray] | None = None
        # Confianza por keypoint de la última detección, separada de
        # last_detection para no romper los call sites que hacen
        # `kpts, box = det.last_detection` o `det.last_detection[1]`.
        self.last_detection_conf: np.ndarray | None = None

    def process_frame(self, image: np.ndarray, timestamp_s: float) -> list[dict]:
        events: list[dict] = []
        h, w = image.shape[:2]
        detection, weak = self._detect(image, timestamp_s)
        self.person_present = detection is not None or weak is not None
        if detection is None:
            self.last_detection = None
            self.last_detection_conf = None
            if weak is not None:
                # Detección débil: solo cuenta para sedentarismo/inmovilidad.
                # No entra a self._history ni a _detect_fall / _track_postfall,
                # que siguen corriendo sobre los mismos frames que antes.
                kpts, box, kpts_conf = weak
                sample = build_pose_sample(kpts, box, w, h, timestamp_s, kpts_conf=kpts_conf)
                sample.weak = True
                self._push_immobility_sample(sample)
                events.extend(self._track_immobility(sample))
            return events
        kpts, box, kpts_conf = detection
        self.last_detection = (kpts, box)
        self.last_detection_conf = kpts_conf

        sample = build_pose_sample(kpts, box, w, h, timestamp_s, kpts_conf=kpts_conf)
        self._history.append(sample)
        self._trim_history(sample.timestamp_s)
        self._push_immobility_sample(sample)
        self._heights_history.append((sample.timestamp_s, sample.bbox_height_frac))
        while self._heights_history and sample.timestamp_s - self._heights_history[0][0] > FALL_BASELINE_WINDOW_S + FALL_CONFIRM_WINDOW_S:
            self._heights_history.popleft()

        if self._detect_fall(sample):
            # Evidencia del encuadre en el momento del aterrizaje: si la persona
            # está parcialmente fuera de cuadro o sin torso visible, la firma
            # geométrica (aspect_ratio / hip_y) es poco confiable. El evento NO
            # se descarta — se marca para que el runner lo trate como sospecha
            # (sin confirmación CLIP sobre un recorte truncado) y lo escale por
            # inmovilidad si la persona queda quieta.
            evidence = "full" if sample.framing_ok else "partial"
            events.append({
                "event_type": "fall",
                "video_timestamp_s": sample.timestamp_s,
                "metadata": {
                    "aspect_ratio": sample.aspect_ratio,
                    "torso_angle_deg": sample.torso_angle_deg,
                    "evidence": evidence,
                    "bbox_touches_edge": sample.bbox_touches_edge,
                    "torso_visible": sample.torso_visible,
                },
            })
            if evidence == "partial":
                self._postfall_since = sample.timestamp_s
                self._postfall_reported = False

        events.extend(self._track_immobility(sample))
        events.extend(self._track_postfall(sample))
        return events

    def _detect(self, image: np.ndarray, timestamp_s: float):
        """Detección con histéresis. Devuelve `(detection, weak)`, a lo sumo
        uno de los dos distinto de None, ambos con la forma de `detect_person`.

        `detection` es lo mismo que devolvería `detect_person`: la caja más
        grande entre las de confianza >= PERSON_BOX_CONFIDENCE, descartada si
        la confianza promedio de keypoints no llega a MIN_KEYPOINT_CONFIDENCE.
        Si además su confianza de caja llega a TRACK_CREATE_BOX_CONFIDENCE,
        fija el ancla del seguimiento.

        `weak` aparece solo si hay un ancla vigente y no hubo detección normal
        en este frame: la caja de confianza >= TRACK_KEEP_BOX_CONFIDENCE que
        más solapa con el ancla (IoU >= TRACK_KEEP_MIN_IOU). Una caja débil en
        cualquier otro lugar del frame se ignora — por eso esto no "detecta
        cualquier cosa" como sí lo haría bajar el umbral global.
        """
        result = self._model(image, verbose=False, conf=TRACK_KEEP_BOX_CONFIDENCE)[0]
        if result.keypoints is None or len(result.keypoints.xy) == 0:
            return None, None
        boxes = result.boxes.xyxy.cpu().numpy()
        box_conf = result.boxes.conf.cpu().numpy()

        strong = np.flatnonzero(box_conf >= PERSON_BOX_CONFIDENCE)
        if len(strong):
            areas = (boxes[strong, 2] - boxes[strong, 0]) * (boxes[strong, 3] - boxes[strong, 1])
            idx = int(strong[areas.argmax()])
            detection = _extract(result, boxes, idx)
            kpts_conf = detection[2]
            if kpts_conf is None or float(kpts_conf.mean()) >= MIN_KEYPOINT_CONFIDENCE:
                if box_conf[idx] >= TRACK_CREATE_BOX_CONFIDENCE:
                    self._anchor_box = detection[1]
                    self._anchor_t = self._track_last_t = timestamp_s
                elif self._anchor_box is not None and _iou(detection[1], self._anchor_box) >= TRACK_KEEP_MIN_IOU:
                    self._track_last_t = timestamp_s
                return detection, None

        if (
            self._anchor_box is None
            or timestamp_s - self._track_last_t > TRACK_KEEP_MAX_GAP_S
            or timestamp_s - self._anchor_t > TRACK_WEAK_MAX_S
        ):
            return None, None
        overlaps = np.array([_iou(b, self._anchor_box) for b in boxes])
        best = int(overlaps.argmax())
        if overlaps[best] < TRACK_KEEP_MIN_IOU:
            return None, None
        self._track_last_t = timestamp_s
        return None, _extract(result, boxes, best)

    def _push_immobility_sample(self, sample: PoseSample) -> None:
        self._immobility_history.append(sample)
        while self._immobility_history and sample.timestamp_s - self._immobility_history[0].timestamp_s > IMMOBILITY_HISTORY_SECONDS:
            self._immobility_history.popleft()

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
        rate = None if sample.weak else self._smoothed_displacement_rate(sample)
        if rate is None:
            # Sin tasa por keypoints (muestra débil, o muestra normal sin otra
            # normal reciente contra la cual comparar): se cae a la tasa por
            # caja, que solo puede SOSTENER la cuenta de quietud. Si da
            # "se mueve" se ignora en vez de resetear — la caja débil tiembla
            # sola — salvo que no haya ninguna muestra débil en juego, que es
            # el caso de siempre (hueco real de detección => reset).
            weak_rate = self._weak_displacement_rate(sample)
            if weak_rate is None:
                if sample.weak:
                    return events
            elif weak_rate > IMMOBILITY_MOVEMENT_THRESHOLD and (
                sample.weak or any(s.weak for s in self._immobility_history)
            ):
                return events
            rate = weak_rate
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

    def _track_postfall(self, sample: PoseSample) -> list[dict]:
        """Vigilancia tras una caída de evidencia parcial (ver process_frame).

        La heurística geométrica no pudo confirmar la caída con confianza
        (persona parcialmente fuera de cuadro), así que en vez de descartarla
        se observa qué pasa después: si la persona queda inmóvil
        `_postfall_window_s` segundos (ventana corta, ~45s: es una posible
        emergencia, no sedentarismo), se emite un evento de severidad alta. Si
        se mueve antes (se levantó), se cancela sin alerta. Esto resuelve la
        tensión "no descartar una caída donde solo se ve el torso" sin inundar
        de falsos positivos: alguien que pasa caminando por el borde del cuadro
        sigue moviéndose y nunca escala.

        Reusa `_smoothed_displacement_rate` / `self._immobility_history` (mismo
        criterio y tolerancia a huecos de 15s que `_track_immobility`).
        """
        if self._postfall_since is None:
            return []

        rate = self._smoothed_displacement_rate(sample)
        if rate is not None and rate > IMMOBILITY_MOVEMENT_THRESHOLD:
            # Se movió: se levantó (o nunca cayó). Cancelar la vigilancia.
            self._postfall_since = None
            self._postfall_reported = False
            return []

        elapsed = sample.timestamp_s - self._postfall_since
        if elapsed > self._postfall_window_s and not self._postfall_reported:
            self._postfall_reported = True
            return [{
                "event_type": "immobility",
                "video_timestamp_s": sample.timestamp_s,
                "metadata": {
                    "duration_s": elapsed,
                    "posture": self._majority_posture(),
                    "severity": "alta",
                    "trigger": "post_fall_sospecha",
                },
            }]
        return []

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
        # Solo muestras normales: las débiles (ver _detect) comparten el
        # historial pero no tienen keypoints confiables — se miden aparte, por
        # caja, en _weak_displacement_rate.
        ref = self._reference_sample(sample, [s for s in self._immobility_history if not s.weak])
        if ref is None:
            return None
        return float(np.nanmean(np.abs(sample.keypoints - ref.keypoints))) / (sample.timestamp_s - ref.timestamp_s)

    def _weak_displacement_rate(self, sample: PoseSample) -> float | None:
        """Como `_smoothed_displacement_rate` pero midiendo con la caja y
        contra cualquier muestra del historial, normal o débil. En una
        detección débil los keypoints no sirven (Ultralytics además pone en
        (0, 0) los de confianza < 0.5, lo que daría un desplazamiento
        artificialmente bajo). Misma escala que por keypoints (mediana
        caminando: 0.059 por caja vs. 0.059 por keypoints), así que vale el
        mismo umbral — pero la caja débil tiembla más, por eso esta tasa solo
        se usa para sostener la cuenta de quietud, nunca para resetearla (ver
        _track_immobility)."""
        ref = self._reference_sample(sample, self._immobility_history)
        if ref is None or sample.bbox is None or ref.bbox is None:
            return None
        return float(np.mean(np.abs(sample.bbox - ref.bbox))) / (sample.timestamp_s - ref.timestamp_s)

    @staticmethod
    def _reference_sample(sample: PoseSample, history) -> PoseSample | None:
        if not history:
            return None
        target_t = sample.timestamp_s - IMMOBILITY_SAMPLE_WINDOW_S
        ref = min(history, key=lambda s: abs(s.timestamp_s - target_t))
        if sample.timestamp_s - ref.timestamp_s < IMMOBILITY_SAMPLE_WINDOW_S / 2:
            return None
        return ref

    def _majority_posture(self) -> str:
        if not self._posture_history:
            return "sentado"
        return Counter(self._posture_history).most_common(1)[0][0]
