"""Тесты парсера диапазонов: покерная нотация → комбо."""

from __future__ import annotations

from poker_analyzer.engine.equity import card, cards
from poker_analyzer.engine.ranges import narrow_range, parse_range


def test_pair_count() -> None:
    assert len(parse_range("AA")) == 6  # 6 комбо пары


def test_pair_plus() -> None:
    assert len(parse_range("TT+")) == 5 * 6  # TT, JJ, QQ, KK, AA


def test_suited_and_offsuit_counts() -> None:
    assert len(parse_range("AKs")) == 4  # одномастные
    assert len(parse_range("AKo")) == 12  # разномастные
    assert len(parse_range("AK")) == 16  # обе: 4 + 12


def test_suited_plus() -> None:
    assert len(parse_range("A2s+")) == 12 * 4  # A2s … AKs = 12 младших × 4 масти


def test_offsuit_plus() -> None:
    assert len(parse_range("KTo+")) == 3 * 12  # KTo, KJo, KQo


def test_specific_combo() -> None:
    combos = parse_range("AhKd")
    assert combos == [tuple(sorted((card("Ah"), card("Kd"))))]


def test_union_and_dedup() -> None:
    assert set(parse_range("AKs, AKs")) == set(parse_range("AKs"))  # дубли схлопнулись
    combined = parse_range("AA, KK")
    assert len(combined) == 12  # 6 + 6, без пересечений


def test_whitespace_and_case() -> None:
    assert set(parse_range("  aa , kk ")) == set(parse_range("AA,KK"))


def test_narrow_empty_board_is_noop() -> None:
    r = parse_range("22+, AKs")
    assert narrow_range(r, [], 0.9) == r  # префлоп не сужаем


def test_narrow_drops_air_keeps_strong() -> None:
    villain = parse_range("22+, A2s+, K2s+, Q5s+, J7s+")  # широкий, много воздуха на AK7
    kept = narrow_range(villain, cards("Ah Kd 7c"), 0.5)
    assert 0 < len(kept) < len(villain)  # часть отсеялась, но не всё
    assert set(kept) <= set(villain)  # подмножество исходного
