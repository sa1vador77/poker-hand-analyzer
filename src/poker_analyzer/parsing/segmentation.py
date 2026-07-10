"""Сегментация скриншота: нарезка области лога на отдельные строки.

:func:`split_rows` режет область лога на строки по разметке Layout. Дальше каждую
строку разбирает конвейер ``recognize_time/keyword/cards/amount`` (см.
:mod:`poker_analyzer.vision.recognition`), отрезая свой элемент слева направо.
"""

from __future__ import annotations

import numpy as np

from poker_analyzer.config import Layout
from poker_analyzer.vision.crop import crop_xywh


def split_rows(screen: np.ndarray, layout: Layout, *, y_shift: int = 0) -> list[np.ndarray]:
    """Режет область лога на отдельные строки сверху вниз по разметке.

    Идём от ``y1`` с шагом ``row_pitch``, вырезая полосу высотой ``row_height``,
    пока строка целиком помещается в область до ``y2``. ``y_shift`` опускает всю
    сетку: лог-«наполнение» (пока строки не коснулись низа) рисует их ниже обычного
    (см. ``Layout.fill_shift`` и :func:`poker_analyzer.pipeline.detect_grid_shift`).
    """
    rows: list[np.ndarray] = []
    frame_height, frame_width = screen.shape[:2]
    # Не вылезаем за реальный кадр: иначе срез даёт пустые строки (скриншот
    # может быть меньше калибровочного — например, из-за масштаба Retina).
    bottom = min(layout.y2 + y_shift, frame_height)
    width = min(layout.x2, frame_width) - layout.x1
    if width <= 0:
        return rows
    top = layout.y1 + y_shift
    while top + layout.row_height <= bottom:
        rows.append(crop_xywh(screen, layout.x1, top, width, layout.row_height))
        top += layout.row_pitch
    return rows
