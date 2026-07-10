"""Симулятор спота: EV действий героя по эквити-движку — НЕЗАВИСИМАЯ «истина».

Нужен для тест-ориентированной калибровки порогов вэлью-бета (``scripts/tune_value_thresholds.py``,
``tests/test_value_ev.py``). Модель сознательно НЕ опирается на сам советник — иначе регресс-тест
проверял бы советник его же допущениями. Здесь EV выводится прямо из эквити героя против
ДИАПАЗОНОВ оппонентов (нативный ``equity_vs_ranges``) по явной покерной формуле.

Модель пула — ОДНОУЛИЧНАЯ «станции коллят свой диапазон, дальше шоудаун»: против лузово-
ПАССИВНОГО пула (VPIP высок, fold equity ≈ 0) это адекватная нижняя оценка вэлью-бета. Она
НЕ моделирует многоуличный розыгрыш и reverse implied odds — поэтому это «истина» для
ЯСНЫХ спотов (явное вэлью / явный чек), а пограничные оставляют запас (см. порог в
:func:`advisor._value_bet_threshold`, где реализация эквити даёт нужный буфер).

Вывод порога вэлью-бета. Банк ``P``, ставка ``b`` (доля банка), ``N`` коллеров-станций, доля
банка героя на шоудауне ``eq``:

    EV_check = eq · P                         (чек: добираем шоудаун)
    EV_bet   = eq · (P + (N+1)·b) − b         (бьём b, все N коллят b, выигрываем итог с eq)
    EV_bet − EV_check = b · (eq·(N+1) − 1)

Значит вэлью-бет выгоднее чека ⇔ ``eq > 1/(N+1)``, и размер ``b`` на ПОРОГ не влияет (только
на величину профита). Это и есть формула порога советника.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from poker_analyzer.engine.equity import equity_vs_ranges

Combo = Sequence[int]  # одно комбо — две карты-int
Range = Sequence[Combo]  # диапазон оппонента — набор комбо


@dataclass(frozen=True, slots=True)
class ActionEV:
    """EV действия героя в споте (в фишках, относительно текущего стека)."""

    action: str  # 'bet' / 'check' / 'call' / 'fold'
    ev: float
    size: float = 0.0  # размер ставки (фишки), для 'bet'
    equity: float = 0.0  # доля банка героя против коллящих диапазонов


def hero_equity(
    hero: Sequence[int],
    board: Sequence[int],
    ranges: Sequence[Range],
    *,
    iterations: int = 80_000,
    exact_cap: int = 200_000,
) -> float:
    """Доля банка героя против диапазонов оппонентов (фикс-сид → детерминировано)."""
    native = [[list(c) for c in r] for r in ranges]
    return equity_vs_ranges(
        list(hero), list(board), native, iterations=iterations, exact_cap=exact_cap
    ).equity


def check_ev(eq: float, pot: float) -> float:
    """EV чека на инициативе: добираем банк на шоудауне с долей ``eq``."""
    return eq * pot


def bet_ev(eq: float, pot: float, size_frac: float, n_callers: int) -> float:
    """EV вэлью-бета ``size_frac·pot``, когда все ``n_callers`` станций коллят (см. модуль)."""
    b = size_frac * pot
    return eq * (pot + (n_callers + 1) * b) - b


def call_ev(eq: float, pot: float, to_call: float) -> float:
    """EV колла ставки ``to_call`` в банк ``pot`` (до колла) при доле банка героя ``eq``."""
    return eq * (pot + to_call) - to_call


def best_value_action(
    hero: Sequence[int],
    board: Sequence[int],
    ranges: Sequence[Range],
    pot: float,
    *,
    sizes: Sequence[float] = (0.33, 0.5, 0.66, 1.0),
    eq: float | None = None,
) -> ActionEV:
    """Лучшее действие НА ИНИЦИАТИВЕ (вэлью-бет vs чек) по EV — независимая «истина» спота.

    Перебирает размеры ставки, берёт максимум EV; если ни один бет не бьёт чек — ``check``.
    ``eq`` можно передать готовым (если уже посчитан), иначе считается тут.
    """
    n = len(ranges)
    e = hero_equity(hero, board, ranges) if eq is None else eq
    ev_check = check_ev(e, pot)
    best = ActionEV("check", ev_check, 0.0, e)
    for s in sizes:
        ev = bet_ev(e, pot, s, n)
        if ev > best.ev:
            best = ActionEV("bet", ev, s * pot, e)
    return best
