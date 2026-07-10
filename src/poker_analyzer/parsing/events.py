"""Модель события лога и сборка событий из распознанных строк.

Это «общий язык» между слоями: parsing превращает :class:`~poker_analyzer.pipeline.RowResult`
(распознанную строку, выход слоя vision) в :class:`LogEvent`, а engine применяет
события к состоянию раздачи. Мост — :func:`event_from_row`: ``keyword → Action`` плюс
перенос полей.
"""

from __future__ import annotations

import logging
from collections.abc import Iterable, Iterator
from dataclasses import dataclass, field
from enum import StrEnum
from typing import TYPE_CHECKING

if TYPE_CHECKING:  # только для типов — рантайм-зависимости на pipeline нет
    from poker_analyzer.pipeline import RowResult

logger = logging.getLogger(__name__)


class Action(StrEnum):
    """Тип события строки лога (по ключевому слову)."""

    DEALER = "Dealer"  # начало новой раздачи (Dealer - ник)
    HAND = "Hand"  # карта героя (видна только рука героя)
    BLIND = "Blind"  # блайнд
    CALL = "Call"  # колл
    CHECK = "Check"  # чек
    BET = "Bet"  # бет
    RAISE = "Raise"  # рейз
    FOLD = "Fold"  # фолд
    ALL_IN = "All-in"  # олл-ин
    DEAL = "Deal"  # банк после очередного круга
    TABLE = "Table"  # карты на столе (борд)
    WIN = "Win"  # конец раздачи, выигрыш
    YOUR_TURN = "Ваш ход"  # дальше — ход героя


@dataclass(frozen=True, slots=True)
class LogEvent:
    """Одно разобранное событие строки лога.

    Часть полей не заполняется для отдельных типов событий: например у ``Deal`` нет
    игрока, а у ``Ваш ход`` нет ни игрока, ни суммы, ни карт.
    """

    time: str  # 'HH:MM:SS'
    action: Action  # тип события
    player_id: int | None = None  # кто (None для Deal/Table/Ваш ход)
    amount: float | None = None  # сумма ставки или банка (float — бывают дробные)
    cards: tuple[str, ...] = field(default_factory=tuple)  # карты строки (Hand/Table/Win)
    is_hero: bool = False  # действие героя (ник совпал с шаблоном героя)
    session_id: int | None = None  # постоянный id игрока за сессию (для профилей и лога)
    nick_text: str | None = None  # ник, прочитанный строкой (этап 2a — проверка перед матчингом)
    nick_confident: bool = False  # чтение ника уверенное


# Имя шаблона ключевого слова (lowercase, как в data/templates/keywords) -> тип события.
KEYWORD_TO_ACTION: dict[str, Action] = {
    "dealer": Action.DEALER,
    "hand": Action.HAND,
    "blind": Action.BLIND,
    "call": Action.CALL,
    "check": Action.CHECK,
    "bet": Action.BET,
    "raise": Action.RAISE,
    "fold": Action.FOLD,
    "allin": Action.ALL_IN,
    "deal": Action.DEAL,
    "table": Action.TABLE,
    "win": Action.WIN,
    "yourturn": Action.YOUR_TURN,
}


def event_from_row(row: RowResult) -> LogEvent | None:
    """Превращает распознанную строку в :class:`LogEvent`.

    Возвращает ``None``, если слово не распознано (``keyword is None``) или неизвестно
    (нет в :data:`KEYWORD_TO_ACTION`) — такие строки в поток событий не попадают.
    """
    if row.keyword is None:
        return None
    action = KEYWORD_TO_ACTION.get(row.keyword)
    if action is None:
        logger.warning("Неизвестное ключевое слово, строка пропущена: %r", row.keyword)
        return None
    return LogEvent(
        time=row.time,
        action=action,
        player_id=row.player_id,
        amount=row.amount,
        cards=row.cards,
        is_hero=row.is_hero,
        session_id=row.session_id,
        nick_text=row.nick_text,
        nick_confident=row.nick_confident,
    )


def events_from_rows(rows: Iterable[RowResult]) -> Iterator[LogEvent]:
    """Поток событий из распознанных строк; нераспознанные строки отбрасываются."""
    for row in rows:
        event = event_from_row(row)
        if event is not None:
            yield event
