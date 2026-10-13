"""Entradas y salidas de casa a partir de la zona `puerta` de la cámara.

No usa ningún modelo: es geometría sobre la posición de la persona que YOLO ya calcula en cada
frame (`posicion_normalizada`) y el polígono `puerta` de `cameras.yaml` (el mismo vocabulario de
zonas de `fusion/src/zone_context.py`). La cámara nunca ve "afuera": solo puede ver que alguien
desaparece o aparece en la puerta.

- **Salida** (`LeaveHome`): la persona se dejó de ver, lo último que se supo de ella es que
  estaba en la zona `puerta`, y pasan `ausencia_min_s` sin que vuelva a aparecer. El evento
  lleva la hora en que se la vio por última vez, no la hora en que se confirmó.
- **Entrada** (`EnterHome`): después de `ausencia_min_s` o más sin ver a nadie, la primera
  detección cae en la zona `puerta`.

Lo que no produce evento: desaparecer lejos de la puerta (salió de cuadro hacia otro ambiente),
perderse un momento en la puerta y reaparecer antes de `ausencia_min_s`, o reaparecer lejos de
la puerta. Al arrancar no se sabe cuánto hace que no hay nadie: la ausencia se cuenta desde la
primera muestra.

**Validado con video solo en parte** (`runner/evaluate_door_events.py`, ver "Validación con
video" en `runner/README.md`): con Toyota Smarthome, donde se entra y se sale de cuadro, no
por una puerta de calle. Los valores por defecto siguen siendo un punto de partida, no se
barrieron. La lógica está probada con trayectorias sintéticas
(`runner/tests/test_door_events.py`). Supone un solo residente: una visita que entra o sale
cuenta igual.
"""
from __future__ import annotations

from dataclasses import dataclass

ZONA_PUERTA = "puerta"
LABEL_SALIR = "LeaveHome"
LABEL_ENTRAR = "EnterHome"
DOOR_LABELS = (LABEL_SALIR, LABEL_ENTRAR)

DEFAULT_AUSENCIA_MIN_S = 60.0
DEFAULT_MEMORIA_PUERTA_S = 2.0
_MIN_KEYPOINT_CONFIDENCE = 0.3
_CADERA_IZQ, _CADERA_DER = 11, 12


@dataclass(frozen=True)
class DoorEvent:
    label: str
    timestamp_s: float
    wall_clock_ts: float | None


def _point_in_polygon(point: tuple[float, float], polygon: list) -> bool:
    """Ray casting, igual que `fusion/src/zone_context.py` (que no se importa acá para que este
    módulo no dependa de que `fusion/` esté instalado)."""
    x, y = point
    inside = False
    n = len(polygon)
    for i in range(n):
        x1, y1 = polygon[i]
        x2, y2 = polygon[(i + 1) % n]
        if (y1 > y) != (y2 > y) and x < x1 + (y - y1) * (x2 - x1) / (y2 - y1):
            inside = not inside
    return inside


def posicion_normalizada(kpts, box, ancho: int, alto: int, kpts_conf=None) -> tuple[float, float]:
    """Dónde está la persona, en coordenadas 0-1 del frame: el punto medio de las caderas (el
    mismo que usa el contexto de zona del clasificador) o, si YOLO no ve bien las caderas, el
    centro del recuadro."""
    if kpts_conf is None or min(float(kpts_conf[_CADERA_IZQ]), float(kpts_conf[_CADERA_DER])) >= _MIN_KEYPOINT_CONFIDENCE:
        x = (float(kpts[_CADERA_IZQ][0]) + float(kpts[_CADERA_DER][0])) / 2.0
        y = (float(kpts[_CADERA_IZQ][1]) + float(kpts[_CADERA_DER][1])) / 2.0
    else:
        x1, y1, x2, y2 = (float(v) for v in box)
        x, y = (x1 + x2) / 2.0, (y1 + y2) / 2.0
    return x / ancho, y / alto


class DoorEventDetector:
    """Máquina de estados de una cámara. Se le pasa una muestra por frame: la posición de la
    persona, o `None` si no se ve a nadie (también mientras la cámara está dormida)."""

    def __init__(
        self,
        polygon: list,
        ausencia_min_s: float = DEFAULT_AUSENCIA_MIN_S,
        memoria_puerta_s: float = DEFAULT_MEMORIA_PUERTA_S,
    ):
        if len(polygon) < 3:
            raise ValueError("la zona puerta necesita al menos 3 vértices")
        if ausencia_min_s <= 0 or memoria_puerta_s < 0:
            raise ValueError("ausencia_min_s tiene que ser mayor que 0 y memoria_puerta_s no puede ser negativa")
        self._polygon = [(float(x), float(y)) for x, y in polygon]
        self._ausencia_min_s = ausencia_min_s
        self._memoria_puerta_s = memoria_puerta_s
        self.reset()

    def reset(self) -> None:
        self._prev_ts: float | None = None
        self._ausente_desde: float | None = None   # última vez que se vio a alguien, o la primera muestra
        self._ultimo_wall_clock: float | None = None
        self._ultimo_en_puerta_ts: float | None = None
        self._salida_resuelta = True  # no hay ninguna desaparición pendiente de decidir

    def update(self, ts: float, point: tuple[float, float] | None, wall_clock_ts: float | None = None) -> list[DoorEvent]:
        if self._prev_ts is not None and ts < self._prev_ts:
            self.reset()  # video en loop: el reloj volvió atrás
        if self._prev_ts is None:
            self._ausente_desde = ts
        self._prev_ts = ts

        if point is None:
            if self._salida_resuelta or ts - self._ausente_desde < self._ausencia_min_s:
                return []
            self._salida_resuelta = True
            desaparecio_en_la_puerta = (
                self._ultimo_en_puerta_ts is not None
                and self._ausente_desde - self._ultimo_en_puerta_ts <= self._memoria_puerta_s
            )
            if not desaparecio_en_la_puerta:
                return []
            return [DoorEvent(LABEL_SALIR, self._ausente_desde, self._ultimo_wall_clock)]

        eventos: list[DoorEvent] = []
        en_puerta = _point_in_polygon(point, self._polygon)
        ausencia = ts - self._ausente_desde
        if ausencia >= self._ausencia_min_s:
            # Si la salida no llegó a confirmarse (no hubo muestras durante la ausencia), se
            # confirma ahora, antes de la entrada.
            if not self._salida_resuelta and self._ultimo_en_puerta_ts is not None \
                    and self._ausente_desde - self._ultimo_en_puerta_ts <= self._memoria_puerta_s:
                eventos.append(DoorEvent(LABEL_SALIR, self._ausente_desde, self._ultimo_wall_clock))
            if en_puerta:
                eventos.append(DoorEvent(LABEL_ENTRAR, ts, wall_clock_ts))

        self._ausente_desde = ts
        self._ultimo_wall_clock = wall_clock_ts
        self._salida_resuelta = False
        if en_puerta:
            self._ultimo_en_puerta_ts = ts
        return eventos
