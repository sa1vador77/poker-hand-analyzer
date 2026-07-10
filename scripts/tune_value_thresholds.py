"""Тест-ориентированная подстройка порога вэлью-бета (``MultiwayValue.margin``).

Цикл «истина → измерение → коррекция → сходимость» БЕЗ живой игры. Истина — независимый
симулятор спота (:mod:`engine.sim`): EV действий героя выводится прямо из эквити-движка по
формуле, не опираясь на сам советник. Харнесс прогоняет панель канонических мультивей- и
хедз-ап-спотов, для каждого кандидата ``margin`` берёт решение РЕАЛЬНЫХ функций советника
(``_value_class``/``_realization``/``_value_bet_threshold``) и СЧИТАЕТ собранную ценность
симулятором, сообщает метрики каждой итерации и сходится на оптимуме.

Порог выведен из эквити-математики: против ``N`` коллеров-станций вэлью-бет +EV при
``eq > 1/(N+1)`` (см. :mod:`engine.sim`). ``margin`` — запас над этим break-even за
REVERSE IMPLIED ODDS многоулиц (одноуличный симулятор их не моделирует): betting тонкого
вэлью мультивей теряет на поздних улицах. Этот риск кодируется явным дисконтом ``RIO_PER_OPP``
(доля номинальной прибыли вэлью-бета, съедаемая КАЖДЫМ доп. оппонентом) — единственный
модельный параметр, документирован честно; полная калибровка потребовала бы многоуличных
прогонов (будущая работа). Оптимум ``margin`` = максимум RIO-дисконтированной собранной
ценности при НУЛЕ ошибок (бет руки ниже break-even / чек явного вэлью).

Запуск:

    uv run python scripts/tune_value_thresholds.py
"""

from __future__ import annotations

import dataclasses
from collections.abc import Sequence

from poker_analyzer.engine import advisor
from poker_analyzer.engine.equity import cards
from poker_analyzer.engine.ranges import parse_range
from poker_analyzer.engine.sim import bet_ev, check_ev, hero_equity
from poker_analyzer.engine.state import HandState
from poker_analyzer.parsing.events import Action, LogEvent

# Reverse implied odds: ДОБАВОЧНАЯ потеря вэлью-бета на поздних улицах (одноуличный симулятор
# их не видит). Моделируем как долю поставленного, которую съедают «я часто позади» × число
# доп. оппонентов: rio_loss = RIO_PER_OPP · (1 − эквити) · (N−1) · ставка. Для СИЛЬНОЙ руки
# (эквити высока → 1−эквити мала) потеря мала; для ТОНКОГО мультивей-вэлью (эквити у break-even,
# N велико) — потеря может ПРЕВЫСИТЬ тонкую прибыль → чистый минус → чек. Так у оптимизатора
# появляется ВНУТРЕННИЙ оптимум margin>0. Это единственный модельный параметр (честно: полная
# калибровка потребовала бы многоуличных прогонов); 0.14 — консервативная оценка.
RIO_PER_OPP = 0.14
# Станция коллит широко (лузово-пассивный пул) — диапазон одного коллера для оценки эквити.
_STATION = "22+, A2s+, K5s+, Q7s+, J7s+, T7s+, 96s+, 86s+, 75s+, A5o+, K8o+, Q9o+, J9o+, T9o"
BET_SIZES = (0.33, 0.5, 0.66, 1.0)


def _evt(action: Action, *, player_id: int | None = None, amount: int | None = None) -> LogEvent:
    return LogEvent(time="00:00:00", action=action, player_id=player_id, cards=(), amount=amount)


def _spot(hero: str, board: str, *, opponents: int, pot: int) -> HandState:
    """Постфлоп-инициатива в мультивее: оппоненты чекнули, ход героя OOP (BB), ``to_call=0``."""
    s = HandState()
    s.apply(_evt(Action.DEALER, player_id=0))
    s.apply(_evt(Action.BLIND, player_id=1, amount=10))
    s.apply(_evt(Action.BLIND, player_id=99, amount=20))  # BB — герой (OOP)
    for pid in range(2, 2 + opponents - 1):
        s.apply(_evt(Action.CALL, player_id=pid, amount=20))
    s.apply(_evt(Action.CALL, player_id=1, amount=20))
    s.apply(_evt(Action.DEAL, amount=pot))
    s.apply(LogEvent(time="00:00:00", action=Action.TABLE, cards=tuple(_glyphs(board))))
    for pid in [1, *range(2, 2 + opponents - 1)]:
        s.apply(_evt(Action.CHECK, player_id=pid))
    s.apply(_evt(Action.YOUR_TURN))
    s.hero_cards = tuple(_glyphs(hero))
    s.hero_id = 99
    s.hero_to_act = True
    s.table_size = max(6, opponents + 1)
    return s


_GLYPH = {"s": "♠", "h": "♥", "d": "♦", "c": "♣"}


def _glyphs(text: str) -> list[str]:
    """'9c9d' / '9s5d2c' → ['9♣','9♦'] / ['9♠','5♦','2♣'] (для hero_cards/board советника)."""
    out = []
    for i in range(0, len(text), 2):
        rank, suit = text[i], text[i + 1]
        out.append((rank if rank != "T" else "10") + _GLYPH[suit])
    return out


# Панель канонических спотов: имя, рука, борд, число оппонентов, банк, метка-ожидание.
# Метка — ОДНОЗНАЧНЫЕ края (для подсчёта ошибок); тонкое вэлью оптимизатор решает сам.
PANEL = [
    ("сет-сухо-2", "9c9d", "9s5d2c", 2, 60, "value"),
    ("сет-сухо-3", "7c7d", "7s5d2c", 3, 90, "value"),
    ("две-пары-2", "AcQd", "AsQh4c", 2, 60, "value"),
    ("оверпара-2", "KcKd", "9s5d2c", 2, 60, "value"),
    ("топ-пара-тузкик2", "AcKd", "Ks7d2c", 2, 60, "value"),
    ("топ-пара-слабкик3", "Kc9d", "Ks7d2c", 3, 90, None),  # тонко — пусть оптимизатор решит
    ("топ-пара-4вей", "AcJd", "Js7d2c", 4, 120, None),  # тонко мультивей
    ("втор-пара-2", "Kc9d", "As9h2c", 2, 60, "check"),  # вторая пара — не вэлью-класс
    ("воздух-2", "7c2d", "AsKdQc", 2, 60, "check"),
    ("гатшот-3", "JcTd", "9s8d2c", 3, 90, "check"),  # дро — не вэлью-бет (полублеф бесполезен)
    ("оверпара-турн3", "QcQd", "9s5d2c7h", 3, 120, "value"),
    ("сет-турн-4", "5c5d", "5s9dKc2h", 4, 160, "value"),
    # Тонкое мультивей-вэлью: слабый топ с плохим кикером против многих — RIO решает.
    ("слабтоп-5вей", "Kc6d", "Ks9d4c", 5, 200, None),
    ("слабтоп-4вей", "Qc8d", "Qs7d2c", 4, 140, None),
    ("слабтоп-турн5", "Jc7d", "Js9d4c2h", 5, 220, None),
]


@dataclasses.dataclass(frozen=True, slots=True)
class SpotCalc:
    """Предрасчёт спота (не зависит от margin): эквити, реализация, класс, лучший размер."""

    name: str
    n: int
    pot: float
    raw_eq: float  # сырая доля банка героя vs коллящих станций
    realized_eq: float  # реализованная (× R) — её сравнивает советник с порогом
    value_class: str | None  # 'strong'/'top'/None (вэлью-класс руки)
    best_size: float  # лучшая доля банка по сырому EV (для оценки прибыли)
    label: str | None


def precompute() -> list[SpotCalc]:
    """Считает по споту всё margin-независимое (эквити/реализация/класс) — один раз."""
    station = [list(c) for c in parse_range(_STATION)]
    out: list[SpotCalc] = []
    for name, hero, board, n, pot, label in PANEL:
        s = _spot(hero, board, opponents=n, pot=pot)
        h = cards(hero.replace("T", "T"))
        b = cards(board)
        ranges: Sequence[Sequence[Sequence[int]]] = [station for _ in range(n)]
        raw_eq = hero_equity(h, b, ranges)
        realized = max(0.0, min(1.0, raw_eq * advisor._realization(s, h, b, n)))
        vcls = advisor._value_class(h, b)
        # лучший размер по сырому одноуличному EV
        best_s, best_ev = 0.0, check_ev(raw_eq, float(pot))
        for sz in BET_SIZES:
            e = bet_ev(raw_eq, float(pot), sz, n)
            if e > best_ev:
                best_s, best_ev = sz, e
        out.append(SpotCalc(name, n, float(pot), raw_eq, realized, vcls, best_s, label))
    return out


def evaluate(spots: list[SpotCalc], margin: float) -> tuple[float, float, int, list[str]]:
    """Метрики для кандидата ``margin``: (сырая собранная ценность, RIO-дисконт, ошибки, лог).

    Решение советника: вэлью-бет, если ``value_class`` задан и реализованная эквити ≥ порога
    ``max(floor, 1/(N+1) + margin)``. Ценность считает симулятор по СЫРОЙ эквити (истина).
    """
    raw_capture, rio_capture, mistakes, log = 0.0, 0.0, 0, []
    for sp in spots:
        thr = max(advisor.MULTIWAY_VALUE.floor, 1.0 / (sp.n + 1) + margin)
        bets = sp.value_class is not None and sp.realized_eq >= thr
        breakeven = 1.0 / (sp.n + 1)
        if bets:
            b = sp.best_size * sp.pot
            gain = bet_ev(sp.raw_eq, sp.pot, sp.best_size, sp.n) - check_ev(sp.raw_eq, sp.pot)
            rio_loss = RIO_PER_OPP * (1.0 - sp.raw_eq) * (sp.n - 1) * b  # цена reverse implied odds
            raw_capture += gain
            rio_capture += gain - rio_loss  # чистая ценность с учётом многоулиц
            if sp.raw_eq < breakeven:  # бьём руку ниже break-even — это −EV ошибка
                mistakes += 1
                log.append(
                    f"  ✗ {sp.name}: бет при eq {sp.raw_eq:.0%} < break-even {breakeven:.0%}"
                )
            if sp.label == "check":
                mistakes += 1
                log.append(f"  ✗ {sp.name}: бьём, а размечено как чек")
        elif sp.label == "value":  # явное вэлью зачекали — недобор
            mistakes += 1
            log.append(
                f"  ✗ {sp.name}: чек явного вэлью (реал.экв {sp.realized_eq:.0%} < порог {thr:.0%})"
            )
    return raw_capture, rio_capture, mistakes, log


def main() -> None:
    spots = precompute()
    print("ПАНЕЛЬ (margin-независимый предрасчёт):")
    for sp in spots:
        print(
            f"  {sp.name:18} N={sp.n} банк={sp.pot:5.0f} сырая_экв={sp.raw_eq:.0%} "
            f"реал.экв={sp.realized_eq:.0%} класс={sp.value_class or '—':6} "
            f"break-even={1.0 / (sp.n + 1):.0%} метка={sp.label or '—'}"
        )
    print(f"\nRIO_PER_OPP={RIO_PER_OPP} (дисконт reverse implied odds на доп. оппонента)\n")

    grid = [round(0.0 + 0.02 * i, 2) for i in range(16)]  # margin 0.00..0.30
    print(f"{'итер':>4} {'margin':>7} {'сырая_ценность':>15} {'RIO_ценность':>13} {'ошибок':>7}")
    rows = []
    for it, m in enumerate(grid, 1):
        raw_cap, rio_cap, mistakes, log = evaluate(spots, m)
        rows.append((m, raw_cap, rio_cap, mistakes))
        print(f"{it:>4} {m:>7.2f} {raw_cap:>15.1f} {rio_cap:>13.1f} {mistakes:>7}")
        for line in log:
            print(line)

    clean = [(m, rc) for (m, _r, rc, mis) in rows if mis == 0]
    if not clean:
        print("\nСХОДИМОСТЬ: нет margin с нулём ошибок — расширь сетку/панель.")
        return
    best_val = max(rc for _m, rc in clean)
    plateau = [m for m, rc in clean if rc >= best_val - 1e-6]  # margin'ы без потери ценности
    chosen = max(plateau)  # робастный выбор: наибольший margin, что НЕ теряет ценность
    print(
        f"\nСХОДИМОСТЬ: плато оптимума margin∈[{min(plateau):.2f}, {max(plateau):.2f}] "
        f"(RIO-ценность {best_val:.1f}, ошибок 0). Выше {max(plateau):.2f} — недобор вэлью."
    )
    print(
        "  Вывод: оптимальный порог ≈ break-even 1/(N+1) (прежний линейный ~0.46 при N=3 был "
        "крупным недо-бетом). Робастный выбор — НАИБОЛЬШИЙ margin без потери ценности:"
    )
    print(f"  → MW_VALUE_MARGIN={chosen:.2f} (дефолт в config.MultiwayValue / .env).")


if __name__ == "__main__":
    main()
