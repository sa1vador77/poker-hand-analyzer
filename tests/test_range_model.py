"""Тесты линия-осведомлённого сужения (range_model): теги линий + добавка к порогу."""

from __future__ import annotations

from poker_analyzer.engine.range_model import LineTag, classify_lines, line_threshold_delta
from poker_analyzer.parsing.events import Action, LogEvent


def _ev(action: Action, pid: int | None = None, amount: float | None = None) -> LogEvent:
    return LogEvent(time="00:00:00", action=action, player_id=pid, amount=amount)


def test_double_barrel_counts_postflop_streets() -> None:
    # игрок 2 — префлоп-агрессор, бьёт флоп и тёрн (два барреля); игрок 1 только коллирует
    events = [
        _ev(Action.DEALER, 1),
        _ev(Action.BLIND, 1, 1),
        _ev(Action.BLIND, 2, 2),
        _ev(Action.RAISE, 2, 6),
        _ev(Action.CALL, 1, 6),
        _ev(Action.DEAL, amount=12),  # → флоп
        _ev(Action.BET, 2, 8),
        _ev(Action.CALL, 1, 8),
        _ev(Action.DEAL, amount=28),  # → тёрн
        _ev(Action.BET, 2, 20),
        _ev(Action.CALL, 1, 20),
    ]
    tags = classify_lines(events)
    assert tags[2].barrels == 2  # флоп + тёрн (префлоп-рейз не считается)
    assert not tags[2].check_raised and not tags[2].donked
    assert 1 not in tags  # пассивный коллер — линии нет


def test_donk_detected_when_non_aggressor_leads() -> None:
    # игрок 1 — префлоп-агрессор; на флопе первым ставит игрок 2 = донк в агрессора
    events = [
        _ev(Action.DEALER, 1),
        _ev(Action.BLIND, 1, 1),
        _ev(Action.BLIND, 2, 2),
        _ev(Action.RAISE, 1, 6),
        _ev(Action.CALL, 2, 6),
        _ev(Action.DEAL, amount=12),  # → флоп
        _ev(Action.BET, 2, 5),
    ]
    tags = classify_lines(events)
    assert tags[2].donked and tags[2].barrels == 1


def test_check_raise_flagged() -> None:
    # на флопе игрок 1 ставит, игрок 2 рейзит — чек-рейз (рейз поверх ставки)
    events = [
        _ev(Action.DEALER, 1),
        _ev(Action.BLIND, 1, 1),
        _ev(Action.BLIND, 2, 2),
        _ev(Action.RAISE, 1, 6),
        _ev(Action.CALL, 2, 6),
        _ev(Action.DEAL, amount=12),  # → флоп
        _ev(Action.BET, 1, 5),
        _ev(Action.RAISE, 2, 15),
    ]
    tags = classify_lines(events)
    assert tags[2].check_raised


def test_line_threshold_delta_orders() -> None:
    assert line_threshold_delta(None) == 0.0
    single = line_threshold_delta(LineTag(barrels=1, check_raised=False, donked=False))
    double = line_threshold_delta(LineTag(barrels=2, check_raised=False, donked=False))
    cr = line_threshold_delta(LineTag(barrels=1, check_raised=True, donked=False))
    donk = line_threshold_delta(LineTag(barrels=1, check_raised=False, donked=True))
    assert single == 0.0
    assert double > single  # доп. баррель — уже диапазон
    assert cr > single  # чек-рейз — уже диапазон
    assert donk < single  # донк — шире диапазон
