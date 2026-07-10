"""Диапазоны рук в покерной нотации → списки комбо (для эквити против диапазонов).

Нотация (через запятую): пары ``AA`` / ``TT+``; одномастные ``AKs`` / ``A5s+``;
разномастные ``AKo`` / ``KQo+``; без суффикса масти ``AK`` = обе; конкретное комбо
``AhKd``. Комбо — пара карт (``int 0..51``, кодировка :func:`~poker_analyzer.engine.equity.card`).

«+» расширяет: ``TT+`` → ``TT, JJ, …, AA``; ``A5s+`` → ``A5s, A6s, …, AKs`` (старшая
карта фиксирована, младшая растёт до старшей − 1).
"""

from __future__ import annotations

from collections.abc import Sequence

from poker_analyzer.engine.equity import card, equity_each_vs_random

_RANKS = "23456789TJQKA"  # индекс = сила ранга (0 = двойка … 12 = туз)
_SUITS = "cdhs"

Combo = tuple[int, int]  # одно комбо — две карты (нормализуем: меньший int первым)


def _rank(ch: str) -> int:
    return _RANKS.index(ch.upper())


def _norm(a: int, b: int) -> Combo:
    """Нормализованное комбо: меньший int карты первым (для сравнения/дедупа)."""
    return (a, b) if a < b else (b, a)


def _pair(r: int) -> list[Combo]:
    """6 комбо пары ранга ``r`` (все пары мастей)."""
    cs = [r * 4 + s for s in range(4)]
    return [(cs[i], cs[j]) for i in range(4) for j in range(i + 1, 4)]


def _suited(hi: int, lo: int) -> list[Combo]:
    """4 одномастных комбо рангов ``hi``/``lo``."""
    return [_norm(hi * 4 + s, lo * 4 + s) for s in range(4)]


def _offsuit(hi: int, lo: int) -> list[Combo]:
    """12 разномастных комбо рангов ``hi``/``lo``."""
    return [_norm(hi * 4 + s1, lo * 4 + s2) for s1 in range(4) for s2 in range(4) if s1 != s2]


def _parse_token(tok: str, out: set[Combo]) -> None:
    plus = tok.endswith("+")
    body = tok[:-1] if plus else tok

    # Конкретное комбо: 4 символа, обе масти из _SUITS (напр. 'AhKd').
    if len(body) == 4 and body[1].lower() in _SUITS and body[3].lower() in _SUITS:
        out.add(_norm(card(body[0:2]), card(body[2:4])))
        return

    # Пара: 'AA' / 'TT+'.
    if len(body) >= 2 and body[0].upper() == body[1].upper():
        r = _rank(body[0])
        top = 12 if plus else r
        for rr in range(r, top + 1):
            out.update(_pair(rr))
        return

    # 'AKs' / 'AKo' / 'AK' (+ опционально '+').
    hi, lo = _rank(body[0]), _rank(body[1])
    if hi < lo:
        hi, lo = lo, hi
    suit = body[2].lower() if len(body) == 3 else None  # 's' | 'o' | None (обе)
    los = range(lo, hi) if plus else range(lo, lo + 1)  # '+' тянет младшую до старшей − 1
    for lr in los:
        if suit in (None, "s"):
            out.update(_suited(hi, lr))
        if suit in (None, "o"):
            out.update(_offsuit(hi, lr))


def parse_range(notation: str) -> list[Combo]:
    """Диапазон из покерной нотации → список уникальных комбо (пар карт ``int 0..51``).

    :param notation: напр. ``"TT+, AQs+, AKo, A5s, KhQh"``. Регистр и пробелы вокруг
        токенов не важны. Дубли комбо схлопываются.
    """
    combos: set[Combo] = set()
    for tok in notation.split(","):
        tok = tok.strip()
        if tok:
            _parse_token(tok, combos)
    return sorted(combos)


def narrow_range(
    combos: Sequence[Sequence[int]],
    board: Sequence[int],
    threshold: float,
    *,
    seed: int | None = None,
) -> list[Combo]:
    """Оставляет комбо с силой на борде ≥ ``threshold`` («продолжающий» диапазон).

    Сила комбо = его эквити против случайной руки на ``board`` (мейд-руки и сильные дро —
    высокое, воздух — низкое). Префлоп (борда нет) и пустой список — без изменений.
    Заблокированные бордом комбо (эквити ``-1``) тоже отсекаются.
    """
    if not board or not combos:
        return [(c[0], c[1]) for c in combos]
    strengths = equity_each_vs_random(board, combos, seed=seed)
    return [(c[0], c[1]) for c, e in zip(combos, strengths, strict=True) if e >= threshold]
