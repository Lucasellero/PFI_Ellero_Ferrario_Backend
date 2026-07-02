"""Lectura de frames de un archivo de video local con su timestamp relativo al video."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import cv2
import numpy as np


@dataclass
class Frame:
    image: np.ndarray
    timestamp_s: float  # segundos desde el inicio del video, no timestamp de reloj


class VideoFile:
    def __init__(self, path: Path):
        self.path = Path(path)
        self.video_id = self.path.stem
        self._capture = cv2.VideoCapture(str(self.path))
        if not self._capture.isOpened():
            raise RuntimeError(f"No se pudo abrir el video: {self.path}")
        self._fps = self._capture.get(cv2.CAP_PROP_FPS) or 25.0

    def frames(self) -> Iterator[Frame]:
        frame_idx = 0
        while True:
            ok, image = self._capture.read()
            if not ok:
                break
            yield Frame(image=image, timestamp_s=frame_idx / self._fps)
            frame_idx += 1

    def release(self) -> None:
        self._capture.release()
