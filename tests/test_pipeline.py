"""Тесты пайплайна кадра: детект сдвига сетки строк и валидация времени."""

from __future__ import annotations

import numpy as np

from poker_analyzer.pipeline import detect_grid_shift, is_valid_time


def test_detect_grid_shift_blank_frame() -> None:
    # Пустой кадр (в колонке времени нет тёмного текста) → обычная сетка по умолчанию.
    blank = np.zeros((1912, 2940, 3), dtype=np.uint8)
    blank[:] = 255  # белый фон лога без строк
    assert detect_grid_shift(blank) == 0


def test_is_valid_time() -> None:
    # Полное «HH:MM:SS» — валидно; обрезанное/битое чтение (кадр в момент прокрутки) — нет.
    assert is_valid_time("12:03:47")
    assert not is_valid_time("02348")
    assert not is_valid_time("12:03:4x")
    assert not is_valid_time("")
