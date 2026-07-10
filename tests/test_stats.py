"""Тесты профилей игроков (этапы B/C): сбор по событиям раздачи и поправка диапазона."""

from __future__ import annotations

from poker_analyzer.config import PROFILE_NARROW_CAP
from poker_analyzer.engine.advisor import _narrowed_ranges, _to_equity_int
from poker_analyzer.engine.stats import PlayerProfile, ProfileBook
from poker_analyzer.parsing.events import Action, LogEvent


def _hand_events() -> list[LogEvent]:
    """Раздача: P3 рейзит префлоп и баррелит, P0 коллирует, блайнды фолдят."""
    e = LogEvent
    return [
        e("00:00:01", Action.DEALER, player_id=0, session_id=10),
        e("00:00:02", Action.BLIND, player_id=1, amount=25, session_id=11),
        e("00:00:03", Action.BLIND, player_id=2, amount=50, session_id=12),
        e("00:00:04", Action.RAISE, player_id=3, amount=150, session_id=13),
        e("00:00:05", Action.CALL, player_id=0, amount=150, session_id=10),
        e("00:00:06", Action.FOLD, player_id=1, session_id=11),
        e("00:00:07", Action.FOLD, player_id=2, session_id=12),
        e("00:00:08", Action.TABLE, cards=("K♥", "7♦", "2♣")),
        e("00:00:09", Action.BET, player_id=3, amount=200, session_id=13),
        e("00:00:10", Action.CALL, player_id=0, amount=200, session_id=10),
        e("00:00:11", Action.WIN, player_id=3, amount=750, session_id=13),
    ]


def test_observe_hand_counts() -> None:
    book = ProfileBook()
    updated = book.observe_hand(_hand_events())
    assert updated == [10, 11, 12, 13]

    p3 = book.get(13)
    assert p3 is not None
    assert (p3.hands, p3.vpip, p3.pfr, p3.wins) == (1, 1, 1, 1)
    assert p3.aggro == 2  # рейз префлоп + бет на флопе

    p0 = book.get(10)
    assert p0 is not None
    assert (p0.hands, p0.vpip, p0.pfr) == (1, 1, 0)  # колл — VPIP, но не PFR
    assert p0.passive == 2

    blind = book.get(11)
    assert blind is not None
    assert (blind.hands, blind.vpip) == (1, 0)  # блайнд+фолд — не добровольное вложение


def test_observe_hand_skips_hero_and_unknown() -> None:
    book = ProfileBook()
    events = [
        LogEvent("00:00:01", Action.RAISE, player_id=0, is_hero=True, session_id=1),
        LogEvent("00:00:02", Action.CALL, player_id=1, session_id=None),  # без S-id
    ]
    assert book.observe_hand(events) == []
    assert book.get(1) is None


def test_threshold_delta_shrinkage_and_caps() -> None:
    book = ProfileBook()
    assert book.threshold_delta(None) == 0.0
    assert book.threshold_delta(99) == 0.0  # игрока ещё не видели

    # Экстремальные VPIP (100% / 0%) клампят дельту в кап жёстко при любом POOL_VPIP
    # (умеренные значения у нового POOL_VPIP=0.55 садятся на самый край капа → float-шум).
    loose = PlayerProfile(hands=40, vpip=40)  # VPIP 100% — маньяк
    tight = PlayerProfile(hands=40, vpip=0)  # VPIP 0% — камень
    fresh = PlayerProfile(hands=1, vpip=1)  # одна рука — шринкедж гасит
    book._profiles.update({1: loose, 2: tight, 3: fresh})

    assert book.threshold_delta(1) == -PROFILE_NARROW_CAP  # лузовый → диапазон ШИРЕ (кап)
    assert book.threshold_delta(2) == PROFILE_NARROW_CAP  # тайтовый → уже (кап)
    assert abs(book.threshold_delta(3)) < 0.03  # мало данных → почти нет поправки


def test_profile_delta_widens_narrowed_range() -> None:
    # Этап C в действии: лузовый профиль оппонента → его суженный диапазон строго шире
    # (тир 1 — широкая база коллера; узкая база агрессора у порога почти не режется).
    board = [_to_equity_int(c) for c in ("K♥", "7♦", "2♣")]
    base = _narrowed_ranges([1], board)[0]
    wide = _narrowed_ranges([1], board, profile_deltas=[-PROFILE_NARROW_CAP])[0]
    tightened = _narrowed_ranges([1], board, profile_deltas=[PROFILE_NARROW_CAP])[0]
    assert len(wide) > len(base) > len(tightened)
