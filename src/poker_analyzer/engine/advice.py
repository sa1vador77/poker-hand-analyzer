"""Рекомендация на ход героя — общий тип слоя советов.

Вынесен отдельно, чтобы и :mod:`advisor`, и :mod:`preflop` использовали его без циклов.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class Advice:
    """Рекомендация на ход героя."""

    action: str  # fold / check / call / bet / raise
    equity: float  # эквити героя (доля) — для показа
    pot_odds: float  # to_call / (pot + to_call); 0, если коллить нечего
    reason: str  # короткое объяснение (для логов/отладки)
    size: float | None = None  # размер ставки/рейза в фишках (постфлоп EV-сайзинг)
    ev: float | None = None  # ожидаемая ценность выбранного действия (фишки)
