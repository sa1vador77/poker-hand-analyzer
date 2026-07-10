"""Обрезка изображений и приведение цветовых форматов (numpy / OpenCV).

Изображение OpenCV — это numpy-массив формы ``(height, width)`` (gray) или
``(height, width, channels)``. Обрезка = срез массива, поэтому порядок осей — Y, X.
"""

from __future__ import annotations

import cv2
import numpy as np


def crop_xywh(image: np.ndarray, x: int, y: int, width: int, height: int) -> np.ndarray:
    """Обрезка по левому-верхнему углу ``(x, y)`` и размеру ``(width, height)``.

    Возвращает срез (view), а не копию: правка результата изменит исходник.
    """
    return image[y : y + height, x : x + width]


def to_gray(image: np.ndarray) -> np.ndarray:
    """Приводит изображение к 1-канальному grayscale, какой бы формат ни пришёл."""
    if image.ndim == 2:  # уже серое
        return image
    if image.shape[2] == 4:  # BGRA -> gray
        return cv2.cvtColor(image, cv2.COLOR_BGRA2GRAY)
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)  # BGR -> gray


def to_bgr(image: np.ndarray) -> np.ndarray:
    """Приводит изображение к 3-канальному BGR (нужно для матчинга по цвету)."""
    if image.ndim == 2:  # серое -> BGR
        return cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    if image.shape[2] == 4:  # BGRA -> BGR (альфу отбрасываем)
        return cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)
    return image


def dark_text_mask(image: np.ndarray, *, dark: int = 100, red_delta: int = 60) -> np.ndarray:
    """Маска тёмного НЕ красного текста ника: тёмные пиксели → 255, фон и сумма → 0.

    Ник — самый тёмный текст строки; сумма красная, тире/слово светлее. Берём тёмные
    столбцы и вычитаем красную сумму по цвету (``R − max(G, B) > red_delta``). Единый
    источник «что есть пиксель ника» для :func:`recognition.find_nick_region` (кроп для
    сравнения игроков) и :func:`glyphs.segment_glyphs` (нарезка на глифы для чтения).

    Серое изображение принимается как есть (красную сумму отделить нечем — считаем, что
    в кропе ника её и нет).
    """
    gray = to_gray(image)
    dark_mask = gray < dark
    if image.ndim == 3:
        b = image[:, :, 0].astype(np.int16)
        g = image[:, :, 1].astype(np.int16)
        r = image[:, :, 2].astype(np.int16)
        red_mask = r - np.maximum(b, g) > red_delta
        dark_mask = dark_mask & ~red_mask  # тёмное, но не красная сумма
    return (dark_mask * 255).astype(np.uint8)
