"""Тесты сборки событий: RowResult -> LogEvent."""

from __future__ import annotations

from pathlib import Path

from poker_analyzer.config import KEYWORD_TEMPLATES_DIR
from poker_analyzer.parsing.events import (
    KEYWORD_TO_ACTION,
    Action,
    event_from_row,
    events_from_rows,
)
from poker_analyzer.pipeline import RowResult


def _row(
    keyword: str | None = None,
    *,
    player_id: int | None = None,
    amount: int | None = None,
    cards: tuple[str, ...] = (),
) -> RowResult:
    return RowResult(
        index=1,
        time="05:23:01",
        keyword=keyword,
        cards=cards,
        amount=amount,
        player_id=player_id,
    )


def test_mapping_covers_all_keyword_templates() -> None:
    # каждый шаблон ключевого слова имеет отображение в Action (иначе строка теряется)
    template_names = {p.stem for p in Path(KEYWORD_TEMPLATES_DIR).glob("*.png")}
    missing = template_names - set(KEYWORD_TO_ACTION)
    assert not missing, f"нет отображения keyword -> Action для: {missing}"


def test_fields_pass_through() -> None:
    ev = event_from_row(_row("bet", player_id=2, amount=45000))
    assert ev is not None
    assert ev.action is Action.BET
    assert ev.time == "05:23:01"
    assert ev.player_id == 2
    assert ev.amount == 45000


def test_cards_pass_through() -> None:
    ev = event_from_row(_row("table", cards=("7♦", "4♠")))
    assert ev is not None
    assert ev.action is Action.TABLE
    assert ev.cards == ("7♦", "4♠")


def test_unrecognized_keyword_returns_none() -> None:
    assert event_from_row(_row(None)) is None


def test_unknown_keyword_returns_none() -> None:
    assert event_from_row(_row("muck")) is None  # нет в KEYWORD_TO_ACTION


def test_events_from_rows_filters_unrecognized() -> None:
    rows = [
        _row("dealer", player_id=0),
        _row(None),  # нераспознанная строка — выпадает
        _row("call", player_id=1, amount=100),
    ]
    actions = [e.action for e in events_from_rows(rows)]
    assert actions == [Action.DEALER, Action.CALL]
