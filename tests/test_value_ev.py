"""EV-регресс вэлью-бета: советник должен совпадать с независимой «истиной» симулятора.

Симулятор (:mod:`engine.sim`) считает EV действий прямо из эквити-движка по явной формуле,
НЕ опираясь на советник, — поэтому тесты проверяют советник независимо, а не его допущениями.
Кодируем ожидаемые EV/исходы для мультивей- и хедз-ап-спотов; харнесс подстройки порогов
(``scripts/tune_value_thresholds.py``) гоняет этот же набор спотов как метрику.
"""

from __future__ import annotations

from poker_analyzer.engine.advisor import _value_bet_threshold, advise
from poker_analyzer.engine.equity import cards
from poker_analyzer.engine.sim import best_value_action, bet_ev, check_ev
from poker_analyzer.engine.state import HandState
from poker_analyzer.parsing.events import Action, LogEvent


def ev(action: Action, *, player_id: int | None = None, amount: int | None = None) -> LogEvent:
    return LogEvent(time="00:00:00", action=action, player_id=player_id, cards=(), amount=amount)


def _mw_initiative(
    hero: tuple[str, ...], board: tuple[str, ...], *, opponents: int, pot: int
) -> HandState:
    """Постфлоп-инициатива в мультивее: ``opponents`` коллеров чекнули, ход героя, ``to_call=0``.

    Префлоп все лимпуют, на флопе оппоненты чекают до героя. Герой — BB (``hero_id=99``).
    Борд кладётся как одна ``Table``-строка (флоп/тёрн/ривер — сколько карт передано).
    """
    s = HandState()
    s.apply(ev(Action.DEALER, player_id=0))
    s.apply(ev(Action.BLIND, player_id=1, amount=10))  # SB
    s.apply(ev(Action.BLIND, player_id=99, amount=20))  # BB — герой
    for pid in range(2, 2 + opponents - 1):  # остальные оппоненты лимпуют
        s.apply(ev(Action.CALL, player_id=pid, amount=20))
    s.apply(ev(Action.CALL, player_id=1, amount=20))  # SB добирает
    s.apply(ev(Action.DEAL, amount=pot))  # банк после префлопа
    flop = LogEvent(time="00:00:00", action=Action.TABLE, cards=board)
    s.apply(flop)
    for pid in [1, *range(2, 2 + opponents - 1)]:  # оппоненты чекают до героя
        s.apply(ev(Action.CHECK, player_id=pid))
    s.apply(ev(Action.YOUR_TURN))
    s.hero_cards = hero
    s.hero_id = 99
    s.hero_to_act = True
    s.table_size = max(6, opponents + 1)
    return s


# --- 1. Сам симулятор: математика порога ------------------------------------


def test_sim_breakeven_is_one_over_n_plus_one() -> None:
    # EV_bet пересекает EV_check ровно в eq = 1/(N+1) при любом размере ставки.
    pot = 100.0
    for n in (1, 2, 3, 4):
        be = 1.0 / (n + 1)
        for size in (0.33, 0.5, 1.0):
            assert abs(bet_ev(be, pot, size, n) - check_ev(be, pot)) < 1e-6  # на пороге — равенство
            assert bet_ev(be + 0.05, pot, size, n) > check_ev(be + 0.05, pot)  # выше → бить
            assert bet_ev(be - 0.05, pot, size, n) < check_ev(be - 0.05, pot)  # ниже → чек


# --- 2. Порог советника принципиален (не косметичен) -------------------------


def test_advisor_threshold_above_breakeven_not_far() -> None:
    # Порог советника ≥ break-even 1/(N+1) (не бьём −EV), но и не намного выше (не недо-бетим).
    for n in (1, 2, 3, 4, 5):
        be = 1.0 / (n + 1)
        thr = _value_bet_threshold(n)
        assert thr >= be - 1e-9  # никогда не бьём ниже break-even
        assert thr <= be + 0.30  # и не зажаты абсурдно высоко (старый линейный давал ~0.46 при N=3)


def test_advisor_threshold_drops_multiway() -> None:
    # Чем больше плательщиков, тем НИЖЕ планка вэлью-бета (раньше падала слишком медленно).
    assert _value_bet_threshold(2) > _value_bet_threshold(3) > _value_bet_threshold(4)


# --- 3. Мультивей-инициатива: советник совпадает с истиной симулятора ---------


def test_strong_made_hand_value_bets_multiway() -> None:
    # Сет на сухом борде, 2 оппонента, инициатива: симулятор однозначно за бет — советник тоже.
    hero = ("9♣", "9♦")
    board = ("9♠", "5♦", "2♣")  # топ-сет
    spot = _mw_initiative(hero, board, opponents=2, pot=60)
    h, b = cards("9c9d"), cards("9s5d2c")
    # независимая истина: против широкого коллящего диапазона станций бет выгоднее чека
    truth = best_value_action(h, b, [_loose(), _loose()], 60.0)
    assert truth.action == "bet"  # симулятор однозначно за вэлью-бет
    a = advise(spot)
    assert a is not None and a.action == "bet"  # советник совпадает с истиной


def test_air_does_not_value_bet_multiway() -> None:
    # Полный воздух (нет пары/дро) мультивей: вэлью-бета нет (класс руки не вэлью) → чек.
    hero = ("7♣", "2♦")
    board = ("A♠", "K♦", "Q♣")
    spot = _mw_initiative(hero, board, opponents=2, pot=60)
    a = advise(spot)
    assert a is not None and a.action == "check"


# --- 4. Хедз-ап-вэлью (через EV-сайзинг _advise_hu) --------------------------


def test_hu_strong_hand_bets() -> None:
    # Хедз-ап сет на сухом борде: EV-сайзинг должен бить (вэлью против одного коллера).
    s = HandState()
    s.apply(ev(Action.DEALER, player_id=0))
    s.apply(ev(Action.BLIND, player_id=1, amount=10))
    s.apply(ev(Action.BLIND, player_id=99, amount=20))
    s.apply(ev(Action.CALL, player_id=0, amount=20))
    s.apply(ev(Action.CALL, player_id=1, amount=20))
    s.apply(ev(Action.FOLD, player_id=1))  # остаётся один оппонент (p0) — хедз-ап
    s.apply(ev(Action.DEAL, amount=60))
    s.apply(LogEvent(time="00:00:00", action=Action.TABLE, cards=("9♠", "5♦", "2♣")))
    s.apply(ev(Action.CHECK, player_id=0))
    s.apply(ev(Action.YOUR_TURN))
    s.hero_cards = ("9♣", "9♦")
    s.hero_id = 99
    s.hero_to_act = True
    s.table_size = 6
    a = advise(s)
    assert a is not None and a.action == "bet"


def _loose() -> list[list[int]]:
    """Широкий коллящий диапазон станции (грубо) — для оценки эквити в тестах."""
    from poker_analyzer.engine.ranges import parse_range

    return [
        list(c) for c in parse_range("22+, A2s+, K5s+, Q8s+, J8s+, T8s+, 97s+, A7o+, KTo+, QJo")
    ]
