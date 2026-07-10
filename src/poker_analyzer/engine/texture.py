"""Текстура борда: масти, парность, связность — для текстуро-зависимого сужения.

Сужение диапазона оппонента (см. :func:`advisor._narrowed_ranges`) меряет силу комбо
как «эквити vs случайной на борде», но не различает ТИП доски. А пул на разных бордах
продолжает по-разному: на МОКРОЙ (монотон/связка) — широко, с массой дро; на СУХОЙ
(радуга, несвязная) — поляризованно, в основном готовыми руками. :func:`texture_delta`
переводит это в поправку к порогу сужения: мокрый борд → порог ниже (диапазон оппонента
ШИРЕ, готовые руки героя не перефолживаются против «только натсов»), сухой → чуть выше.

Карты — ``int 0..51`` (``rank*4 + suit``; ``rank = c // 4`` 0=2..12=A, ``suit = c % 4``),
как в :mod:`poker_analyzer.engine.equity`. Портировано из board_texture.py прошлого
проекта (полный набор фич там; здесь — то, что нужно сужению).
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass

from poker_analyzer.config import TEXTURE_NARROWING, TextureNarrowing

_BROADWAY_MIN = 8  # ранг T (0=2 → T=8): T/J/Q/K/A — бродвей-карты
_ACE = 12  # туз в кодировке ранга (может играть снизу в A-2-3-4-5)


@dataclass(frozen=True, slots=True)
class Texture:
    """Текстура доски (3–5 карт борда)."""

    board_size: int
    is_paired: bool  # есть пара/трипс на доске
    is_monotone: bool  # все карты одной масти
    is_two_tone: bool  # ровно две масти, максимум 2+ одной (но не монотон)
    is_rainbow: bool  # все масти разные
    max_same_suit: int  # максимум карт одной масти
    is_connected: bool  # связная по рангам (грубо)
    is_very_connected: bool  # 3+ подряд идущих ранга
    max_run: int  # длиннейшая цепочка подряд идущих рангов
    broadway_count: int  # сколько бродвей-карт (T+)
    dynamic: bool  # «мокрая» доска (много дро: связка/3-флеш)
    dry: bool  # «сухая» доска (радуга, несвязная, без пар)


def _max_run(ranks: list[int]) -> int:
    """Длиннейшая цепочка подряд идущих рангов (туз учитывается и снизу для A-2-3-4-5)."""
    uniq = sorted(set(ranks))
    if _ACE in uniq:
        uniq = sorted(set(uniq) | {-1})  # колёсный туз ниже двойки
    best = run = 1
    for i in range(1, len(uniq)):
        if uniq[i] == uniq[i - 1] + 1:
            run += 1
            best = max(best, run)
        else:
            run = 1
    return best


def _connected(ranks: list[int]) -> bool:
    """Грубая связность: 3 уникальных ранга в окне ≤4 или 4+ в окне ≤5."""
    uniq = sorted(set(ranks))
    if len(uniq) < 2:
        return False
    span = uniq[-1] - uniq[0]
    if len(uniq) == 3 and span <= 4:
        return True
    return len(uniq) >= 4 and span <= 5


def analyze_texture(board: list[int]) -> Texture | None:
    """Текстура борда из 3–5 карт; ``None`` для префлопа/некорректного размера."""
    if not 3 <= len(board) <= 5:
        return None
    ranks = [c // 4 for c in board]
    suits = [c % 4 for c in board]
    suit_counter = Counter(suits)
    rank_counter = Counter(ranks)
    max_same_suit = max(suit_counter.values())
    is_monotone = len(suit_counter) == 1
    is_paired = any(v >= 2 for v in rank_counter.values())
    is_connected = _connected(ranks)
    max_run = _max_run(ranks)
    is_very_connected = max_run >= 3
    is_rainbow = max_same_suit == 1
    # «Мокрая» доска: много дро — связка, 3+ к флешу, или связь при двух одной масти.
    dynamic = is_very_connected or max_same_suit >= 3 or (is_connected and max_same_suit >= 2)
    dry = is_rainbow and not is_paired and max_run <= 2
    return Texture(
        board_size=len(board),
        is_paired=is_paired,
        is_monotone=is_monotone,
        is_two_tone=not is_monotone and max_same_suit >= 2,
        is_rainbow=is_rainbow,
        max_same_suit=max_same_suit,
        is_connected=is_connected,
        is_very_connected=is_very_connected,
        max_run=max_run,
        broadway_count=sum(1 for r in ranks if r >= _BROADWAY_MIN),
        dynamic=dynamic,
        dry=dry,
    )


def texture_delta(board: list[int], cfg: TextureNarrowing = TEXTURE_NARROWING) -> float:
    """Поправка к порогу сужения по текстуре доски (см. модуль): отрицательная на мокрой
    (диапазон оппонента ШИРЕ — он продолжает с дро), положительная на сухой. Кап — чтобы
    текстура не доминировала над улицей/линией/поляризацией. Префлоп → 0."""
    tex = analyze_texture(board)
    if tex is None:
        return 0.0
    if tex.is_monotone or tex.max_same_suit >= 3 or tex.is_very_connected:
        delta = cfg.wet  # сильно мокрая: монотон / 3-флеш / явная связка
    elif tex.is_two_tone or tex.is_connected:
        delta = cfg.semi_wet  # полу-мокрая: две масти или связная
    elif tex.dry:
        delta = cfg.dry  # сухая: продолжающий диапазон поляризован в готовые руки
    else:
        delta = 0.0
    return max(-cfg.cap, min(cfg.cap, delta))
