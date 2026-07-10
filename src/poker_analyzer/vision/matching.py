"""Поиск шаблонов на изображении через ``cv2.matchTemplate``.

Два примитива:

- :func:`match_template` — найти ОДНО лучшее совпадение шаблона;
- :func:`match_template_all` — найти ВСЕ вхождения выше порога, с подавлением
  дублей вокруг одного совпадения (NMS по IoU).

Параметры ``method`` / ``use_color`` / ``threshold`` сделаны keyword-only, чтобы
их нельзя было перепутать позиционно.
"""

from __future__ import annotations

import cv2
import numpy as np

from poker_analyzer.config import THRESHOLDS
from poker_analyzer.vision.crop import to_bgr, to_gray
from poker_analyzer.vision.types import Match

# Методы, у которых лучшее совпадение — это МИНИМУМ карты откликов (а не максимум).
_SQDIFF_METHODS = (cv2.TM_SQDIFF, cv2.TM_SQDIFF_NORMED)

Box = tuple[int, int, int, int]  # (x1, y1, x2, y2)


def _iou(box_a: Box, box_b: Box) -> float:
    """Intersection-over-Union двух прямоугольников ``(x1, y1, x2, y2)``."""
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b

    inter_w = max(0, min(ax2, bx2) - max(ax1, bx1))
    inter_h = max(0, min(ay2, by2) - max(ay1, by1))
    intersection = inter_w * inter_h
    if intersection == 0:
        return 0.0

    area_a = (ax2 - ax1) * (ay2 - ay1)
    area_b = (bx2 - bx1) * (by2 - by1)
    return intersection / (area_a + area_b - intersection)


def _prepare(
    image: np.ndarray, template: np.ndarray, *, use_color: bool
) -> tuple[np.ndarray, np.ndarray]:
    """Приводит изображение и шаблон к единому формату и проверяет размеры.

    matchTemplate требует одинаковое число каналов и шаблон не больше изображения.
    """
    convert = to_bgr if use_color else to_gray
    prepared_image = convert(image)
    prepared_template = convert(template)

    template_height, template_width = prepared_template.shape[:2]
    image_height, image_width = prepared_image.shape[:2]
    if template_height > image_height or template_width > image_width:
        raise ValueError(
            f"Шаблон {template_width}x{template_height} больше изображения "
            f"{image_width}x{image_height} — matchTemplate так не умеет."
        )
    return prepared_image, prepared_template


def match_template(
    image: np.ndarray,
    template: np.ndarray,
    *,
    method: int = cv2.TM_CCOEFF_NORMED,
    use_color: bool = False,
) -> Match:
    """Ищет ОДНО лучшее совпадение ``template`` на ``image``.

    :param method: метрика cv2 (по умолчанию ``TM_CCOEFF_NORMED``: score в [-1, 1],
        устойчива к равномерному изменению яркости).
    :param use_color: сравнивать по цвету (BGR) вместо яркости — нужно там, где
        объекты различаются цветом, а не формой (например масти карт).
    :returns: :class:`Match` с лучшим совпадением.
    """
    prepared_image, prepared_template = _prepare(image, template, use_color=use_color)
    template_height, template_width = prepared_template.shape[:2]

    # Карта откликов: в каждой точке — насколько шаблон похож на участок image.
    result = cv2.matchTemplate(prepared_image, prepared_template, method)
    min_value, max_value, min_location, max_location = cv2.minMaxLoc(result)

    # У SQDIFF-методов лучшее совпадение — минимум, у остальных — максимум.
    if method in _SQDIFF_METHODS:
        score, top_left = min_value, min_location
    else:
        score, top_left = max_value, max_location

    x, y = int(top_left[0]), int(top_left[1])
    return Match(
        score=float(score),
        top_left=(x, y),
        bottom_right=(x + template_width, y + template_height),
        size=(template_width, template_height),
    )


def match_template_all(
    image: np.ndarray,
    template: np.ndarray,
    *,
    threshold: float = THRESHOLDS.time_digit,
    method: int = cv2.TM_CCOEFF_NORMED,
    use_color: bool = False,
    overlap_threshold: float = THRESHOLDS.nms_overlap,
) -> list[Match]:
    """Находит ВСЕ вхождения ``template`` на ``image`` с качеством не хуже ``threshold``.

    Дубли откликов вокруг одного совпадения схлопываются жадным NMS по IoU.

    :param threshold: порог совпадения (для ``*_NORMED`` методов; выше — строже).
    :param overlap_threshold: два бокса с IoU выше него считаются одним совпадением.
    :returns: список :class:`Match`, отсортированный слева направо (по X).
    """
    prepared_image, prepared_template = _prepare(image, template, use_color=use_color)
    template_height, template_width = prepared_template.shape[:2]

    result = cv2.matchTemplate(prepared_image, prepared_template, method)

    # Точки-кандидаты, прошедшие порог (полярность зависит от метода).
    is_sqdiff = method in _SQDIFF_METHODS
    if is_sqdiff:
        ys, xs = np.where(result <= threshold)
    else:
        ys, xs = np.where(result >= threshold)

    # rank — всегда «больше = лучше», независимо от метода (для жадного NMS).
    rank = -result[ys, xs] if is_sqdiff else result[ys, xs]
    candidates = sorted(zip(rank, xs, ys, strict=True), key=lambda c: -c[0])

    matches: list[Match] = []
    accepted_boxes: list[Box] = []
    for _, x, y in candidates:
        x, y = int(x), int(y)
        box: Box = (x, y, x + template_width, y + template_height)

        # Берём кандидата, только если он не дублирует уже принятое совпадение.
        if any(_iou(box, kept) >= overlap_threshold for kept in accepted_boxes):
            continue

        accepted_boxes.append(box)
        matches.append(
            Match(
                score=float(result[y, x]),
                top_left=(x, y),
                bottom_right=(x + template_width, y + template_height),
                size=(template_width, template_height),
            )
        )

    matches.sort(key=lambda m: m.top_left[0])  # слева направо
    return matches
