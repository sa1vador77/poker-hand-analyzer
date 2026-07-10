"""Базовые типы слоя зрения."""

from __future__ import annotations

from typing import NamedTuple


class Match(NamedTuple):
    """Результат поиска шаблона на изображении.

    Неизменяемая «запись результата»: распаковывается как кортеж и обращается по
    именам полей. Координаты — в пикселях исходного изображения.
    """

    score: float  # качество совпадения; для *_NORMED 1.0 — идеально
    top_left: tuple[int, int]  # (x, y) левого-верхнего угла найденной области
    bottom_right: tuple[int, int]  # (x, y) правого-нижнего угла
    size: tuple[int, int]  # (width, height) шаблона
