"""Эквити героя — Python-фасад над нативным модулем C++ (см. ``native/``).

Тяжёлый расчёт (оценщик 7 карт + Монте-Карло/перебор) живёт в C++
(``poker_analyzer._equity``, nanobind) — в прошлой версии это давало ~10× к Python.
Здесь — удобный API, кодировка карт и типы. Производные метрики (pot odds, EV) считает
слой советов поверх эквити.

Карта — ``int 0..51``: ``rank*4 + suit`` (rank 0..12 = 2..A, suit 0..3). Хелперы
:func:`card` / :func:`cards` парсят строки вида ``'As'`` / ``'AsKs'``.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from poker_analyzer import _equity

_RANKS = "23456789TJQKA"
_SUITS = "cdhs"
DEFAULT_SEED = 0xC0FFEE  # фиксированный сид: одинаковый вход -> одинаковый результат
# Минимальная доля удавшихся MC-сэмплов в equity_vs_ranges: ниже — rejection-collapse
# (плотные/пересекающиеся диапазоны), оценка смещена → откат на vs-random.
_MIN_EFF_FRACTION = 0.2

# Категории готовой руки — как pe::Category в native/evaluator.hpp (больше = сильнее).
HIGH_CARD, PAIR, TWO_PAIR, TRIPS, STRAIGHT, FLUSH, FULL_HOUSE, QUADS, STRAIGHT_FLUSH = range(9)


@dataclass(frozen=True, slots=True)
class EquityResult:
    """Результат расчёта эквити героя (доли в [0, 1])."""

    win: float  # доля раздач, где герой единолично лучший
    tie: float  # доля раздач, где герой делит банк
    equity: float  # доля банка героя (win + доли сплитов) — главная метрика
    reliable: bool = (
        True  # False — оценка деградировала (rejection-collapse) и взята fallback vs-random
    )


@dataclass(frozen=True, slots=True)
class ComboClass:
    """Класс комбо на борде: категория готовой руки + дро.

    Для value/bluff-сплита диапазона оппонента и класса руки героя (made/draw/air) в
    реализации эквити. Заполняется нативным :func:`classify_combos`.
    """

    made: int  # категория 0..8 (см. константы выше, pe::Category); -1 — комбо заблокировано бордом
    flush_draw: bool  # ровно 4 карты одной масти (готового флеша ещё нет)
    oesd: bool  # двусторонний стрит-дро / дабл-гатшот (≥2 достраивающих ранга)
    gutshot: bool  # гатшот (ровно 1 достраивающий ранг)
    outs: int  # грубая оценка аутов дро

    @property
    def blocked(self) -> bool:
        """Комбо невозможно (карта на борде) — в расчётах пропускается."""
        return self.made < 0

    @property
    def is_made(self) -> bool:
        """Готовая рука (пара и выше)."""
        return self.made >= PAIR

    @property
    def is_draw(self) -> bool:
        """Есть дро (флеш-дро / стрит-дро / гатшот)."""
        return self.flush_draw or self.oesd or self.gutshot

    @property
    def is_air(self) -> bool:
        """Воздух: ни готовой руки, ни дро."""
        return self.made == HIGH_CARD and not self.is_draw


def card(label: str) -> int:
    """Карта из строки вида ``'As'``, ``'Td'``, ``'2c'`` в ``int 0..51``."""
    rank = _RANKS.index(label[0].upper())
    suit = _SUITS.index(label[1].lower())
    return rank * 4 + suit


def cards(text: str) -> list[int]:
    """Список карт из строки ``'AsKs'`` или ``'As Ks Qh'``."""
    text = text.replace(" ", "")
    return [card(text[i : i + 2]) for i in range(0, len(text), 2)]


# Масть-глиф из распознавания → буква масти в кодировке эквити.
_SUIT_GLYPH = {"♠": "s", "♥": "h", "♦": "d", "♣": "c"}


def card_from_glyph(label: str) -> int:
    """Карта из распознавания ('7♦', 'A♠', '10♦') → ``int 0..51`` для слоя эквити."""
    rank = label[:-1]
    rank = "T" if rank == "10" else rank  # десятка: '10' → 'T'
    return card(rank + _SUIT_GLYPH[label[-1]])


def _no_duplicates(all_cards: list[int]) -> None:
    if len(set(all_cards)) != len(all_cards):
        raise ValueError("карты не должны повторяться")


def equity(
    hero: Sequence[int],
    board: Sequence[int] = (),
    opponents: int = 1,
    *,
    iterations: int = 100_000,
    seed: int | None = None,
) -> EquityResult:
    """Эквити героя против N оппонентов со случайными руками на текущем борде (Монте-Карло).

    :param hero: 2 карты героя; ``board``: 0–5 общих карт; ``opponents``: число живых.
    :param iterations: число прогонов Монте-Карло (больше — точнее, дольше).
    :param seed: сид RNG; ``None`` -> :data:`DEFAULT_SEED` (воспроизводимо).
    """
    if len(hero) != 2:
        raise ValueError("рука героя — ровно 2 карты")
    if len(board) > 5:
        raise ValueError("борд — не более 5 карт")
    if opponents < 1:
        raise ValueError("оппонентов должно быть ≥ 1")
    _no_duplicates([*hero, *board])
    s = DEFAULT_SEED if seed is None else seed
    win, tie, eq = _equity.equity_random(list(hero), list(board), opponents, iterations, s)
    return EquityResult(win, tie, eq)


def equity_vs(
    hero: Sequence[int],
    board: Sequence[int],
    opp_hands: Sequence[Sequence[int]],
    *,
    iterations: int = 100_000,
    seed: int | None = None,
) -> EquityResult:
    """Эквити героя против ИЗВЕСТНЫХ рук оппонентов (точный перебор борда).

    Неизвестен только борд — его завершения перебираются точно (если их не слишком
    много, иначе Монте-Карло). Удобно для проверок против известных эквити.
    """
    if len(hero) != 2:
        raise ValueError("рука героя — ровно 2 карты")
    if len(board) > 5:
        raise ValueError("борд — не более 5 карт")
    if any(len(h) != 2 for h in opp_hands):
        raise ValueError("рука оппонента — ровно 2 карты")
    _no_duplicates([*hero, *board, *(c for h in opp_hands for c in h)])
    s = DEFAULT_SEED if seed is None else seed
    win, tie, eq = _equity.equity_vs(
        list(hero), list(board), [list(h) for h in opp_hands], iterations, s
    )
    return EquityResult(win, tie, eq)


def equity_vs_ranges(
    hero: Sequence[int],
    board: Sequence[int],
    ranges: Sequence[Sequence[Sequence[int]]],
    *,
    iterations: int = 100_000,
    exact_cap: int = 10_000_000,
    seed: int | None = None,
) -> EquityResult:
    """Эквити героя против ДИАПАЗОНОВ оппонентов (адаптивно: перебор / Монте-Карло).

    У каждого оппонента — набор возможных рук (комбо по 2 карты; собрать из нотации
    помогает :func:`poker_analyzer.engine.ranges.parse_range`). Адаптивно: пока оценка
    работы (произв. размеров диапазонов × завершения борда) ≤ ``exact_cap`` — точный
    перебор; иначе Монте-Карло на ``iterations`` сэмплов.

    :param ranges: по одному диапазону на оппонента; диапазон — последовательность комбо.
    :param exact_cap: порог числа раскладов, выше которого переходим на Монте-Карло.
    :param seed: сид RNG; ``None`` → :data:`DEFAULT_SEED` (воспроизводимо).

    Руки, заблокированные картами героя/борда, выкидываются из диапазонов автоматически.
    Если после этого чей-то диапазон пуст — эквити не определено (вернёт нули).
    """
    if len(hero) != 2:
        raise ValueError("рука героя — ровно 2 карты")
    if len(board) > 5:
        raise ValueError("борд — не более 5 карт")
    if not ranges:
        raise ValueError("нужен хотя бы один диапазон оппонента")
    for r in ranges:
        if not r:
            raise ValueError("диапазон оппонента не должен быть пустым")
        if any(len(h) != 2 for h in r):
            raise ValueError("комбо — ровно 2 карты")
    _no_duplicates([*hero, *board])  # сами диапазоны пересекаться/блокироваться могут
    s = DEFAULT_SEED if seed is None else seed
    native_ranges = [[list(h) for h in r] for r in ranges]
    win, tie, eq, n_eff, was_mc = _equity.equity_vs_ranges(
        list(hero), list(board), native_ranges, iterations, exact_cap, s
    )
    if was_mc and n_eff < iterations * _MIN_EFF_FRACTION:
        # Rejection-collapse: диапазоны слишком плотные/пересекающиеся (флеши, блокеры) — MC
        # расставил мало непротиворечивых рук, оценка смещена (вплоть до абсурдных 0%).
        # Откатываемся на vs-random (грубее, но без катастрофы) и помечаем ненадёжной.
        win, tie, eq = _equity.equity_random(list(hero), list(board), len(ranges), iterations, s)
        return EquityResult(win, tie, eq, reliable=False)
    return EquityResult(win, tie, eq)


def hero_equity_vs_each(
    hero: Sequence[int],
    board: Sequence[int],
    combos: Sequence[Sequence[int]],
    *,
    iterations: int = 20_000,
    exact_cap: int = 2_000_000,
    seed: int | None = None,
) -> list[float]:
    """Эквити героя против КАЖДОГО комбо по отдельности на борде.

    Возвращает список (по комбо) долей банка героя; заблокированное картами героя/борда
    комбо → ``-1.0``. Постфлоп доезд короткий → точный перебор (детерминированно).
    """
    if len(hero) != 2:
        raise ValueError("рука героя — ровно 2 карты")
    if len(board) > 5:
        raise ValueError("борд — не более 5 карт")
    if any(len(c) != 2 for c in combos):
        raise ValueError("комбо — ровно 2 карты")
    _no_duplicates([*hero, *board])
    s = DEFAULT_SEED if seed is None else seed
    return list(
        _equity.hero_equity_vs_each(
            list(hero), list(board), [list(c) for c in combos], iterations, exact_cap, s
        )
    )


def equity_each_vs_random(
    board: Sequence[int],
    combos: Sequence[Sequence[int]],
    *,
    iterations: int = 3_000,
    seed: int | None = None,
) -> list[float]:
    """Эквити КАЖДОГО комбо против одной случайной руки на борде (мерило силы на доске).

    Возвращает список (по комбо) долей банка; заблокированное бордом комбо → ``-1.0``.
    Используется для сужения диапазона (отсев «воздуха»).
    """
    if len(board) > 5:
        raise ValueError("борд — не более 5 карт")
    if any(len(c) != 2 for c in combos):
        raise ValueError("комбо — ровно 2 карты")
    s = DEFAULT_SEED if seed is None else seed
    return list(
        _equity.equity_each_vs_random(list(board), [list(c) for c in combos], iterations, s)
    )


def classify_combos(board: Sequence[int], combos: Sequence[Sequence[int]]) -> list[ComboClass]:
    """Классифицирует каждое комбо на ``board``: категория готовой руки + дро.

    Для каждого комбо — :class:`ComboClass` (made-категория, флеш-дро, OESD/гатшот, ауты).
    Заблокированное бордом (или вырожденное) комбо → ``made == -1`` (``blocked``). Борд
    должен быть постфлоп (3–5 карт) для осмысленной категории; на пустом борде вернёт
    префлоп-силу (пара/старшая) без дро.
    """
    raw = _equity.classify_combos(list(board), [list(c) for c in combos])
    return [ComboClass(m, bool(fd), bool(o), bool(g), int(n)) for m, fd, o, g, n in raw]


def evaluate7(seven_cards: Sequence[int]) -> int:
    """Оценка 7 карт (``int 0..51``) -> сравнимый ранг (больше = сильнее)."""
    return _equity.evaluate7(list(seven_cards))
