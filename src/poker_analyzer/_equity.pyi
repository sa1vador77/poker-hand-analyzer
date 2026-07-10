"""Типовые стабы нативного модуля C++ ``_equity`` (собирается из ``native/equity.cpp``).

Карта — ``int 0..51`` (``rank*4 + suit``). Держать в синхроне с ``NB_MODULE`` в
``native/equity.cpp``: при добавлении/смене сигнатуры нативной функции править и здесь,
иначе ``ty`` ругается «Module has no member …».
"""

def evaluate7(cards: list[int]) -> int:
    """Оценка лучшей 5-карточной руки из 7 карт → сравнимый ранг (больше = сильнее)."""
    ...

def equity_vs(
    hero: list[int],
    board: list[int],
    opp_hands: list[list[int]],
    iterations: int,
    seed: int,
) -> tuple[float, float, float]:
    """``(win, tie, equity)`` — эквити героя против ИЗВЕСТНЫХ рук оппонентов
    (точный перебор оставшегося борда)."""
    ...

def equity_random(
    hero: list[int],
    board: list[int],
    num_opponents: int,
    iterations: int,
    seed: int,
) -> tuple[float, float, float]:
    """``(win, tie, equity)`` — эквити героя против ``num_opponents`` оппонентов со
    случайными руками (Монте-Карло на ``iterations`` сэмплов)."""
    ...

def equity_vs_ranges(
    hero: list[int],
    board: list[int],
    ranges: list[list[list[int]]],
    iterations: int,
    exact_cap: int,
    seed: int,
) -> tuple[float, float, float, int, bool]:
    """``(win, tie, equity, n_eff, was_mc)`` — эквити героя против диапазонов оппонентов;
    ``n_eff``/``was_mc`` для детекта rejection-collapse."""
    ...

def hero_equity_vs_each(
    hero: list[int],
    board: list[int],
    combos: list[list[int]],
    iterations: int,
    exact_cap: int,
    seed: int,
) -> list[float]:
    """Эквити героя против КАЖДОГО комбо по отдельности (-1 для заблокированных картами)."""
    ...

def equity_each_vs_random(
    board: list[int],
    combos: list[list[int]],
    iterations: int,
    seed: int,
) -> list[float]:
    """Эквити каждого комбо против случайной руки на борде (мерило силы для сужения)."""
    ...

def classify_combos(
    board: list[int],
    combos: list[list[int]],
) -> list[tuple[int, int, int, int, int]]:
    """Категория комбо на борде: ``(made_category, flush_draw, oesd, gutshot, outs)``."""
    ...
