"""Выбор размера ставки по EV (ожидаемой ценности в фишках).

Для каждого размера-кандидата (доля банка) считаем EV из двух частей:

- **fold equity** — доля диапазона оппонента, которой невыгодно коллировать этот размер
  (комбо продолжает, только если его эквити ≥ цены колла ``s/(банк+2s)``); эту долю
  времени забираем банк сразу;
- **когда коллируют** — наше эквити против продолжившей (более сильной) части диапазона.

``EV(s) = f·банк + (1−f)·[e·(банк+2s) − s]``. Берём размер с максимальным EV. Работает по
заранее посчитанным эквити героя против каждого комбо (см. ``equity.hero_equity_vs_each``).
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from poker_analyzer.config import BET_SIZES


@dataclass(frozen=True, slots=True)
class SizingResult:
    """Лучший размер ставки и его показатели."""

    size: float  # размер ставки в фишках
    ev: float  # ожидаемая ценность (фишки) относительно «не ставить»
    fold_equity: float  # доля диапазона оппонента, что сбросит на этот размер


def best_bet_from_eqs(
    hero_eqs: Sequence[float],
    pot: float,
    *,
    sizes: Sequence[float] = BET_SIZES,
) -> SizingResult | None:
    """Размер ставки с макс. EV по эквити героя против каждого комбо оппонента.

    :param hero_eqs: эквити героя против каждого комбо диапазона (доли; ``-1`` —
        заблокированные — отбрасываются вызывающей стороной заранее или здесь).
    :param pot: текущий банк в фишках; ставка считается как ``доля·банк``.
    :returns: лучший :class:`SizingResult` или ``None``, если считать нечего.
    """
    villain_eq = [1.0 - e for e in hero_eqs if e >= 0.0]  # эквити оппонента против героя
    n = len(villain_eq)
    if n == 0 or pot <= 0:
        return None

    best: SizingResult | None = None
    for frac in sizes:
        s = frac * pot
        price = s / (pot + 2 * s)  # pot odds, которые получает оппонент на колле
        cont = [ve for ve in villain_eq if ve >= price]  # продолжающие (коллирующие) комбо
        f = 1.0 - len(cont) / n  # доля фолда (fold equity)
        # Эквити героя, когда коллируют, = среднее (1 − эквити_оппонента) по продолжившим.
        hero_eq_called = sum(1.0 - ve for ve in cont) / len(cont) if cont else 1.0
        ev = f * pot + (1.0 - f) * (hero_eq_called * (pot + 2.0 * s) - s)
        if best is None or ev > best.ev:
            best = SizingResult(s, ev, f)
    return best
