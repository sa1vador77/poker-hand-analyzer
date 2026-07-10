"""Тесты ядра математики: оценщик 7 карт и эквити.

Корректность ловим тремя способами: категории/порядок рук оценщика, известные эквити
(AA vs KK, AKs vs 22) и инвариант нулевой суммы (точный перебор: eq(A)+eq(B)=1).
"""

from __future__ import annotations

import pytest

from poker_analyzer.engine.equity import (
    cards,
    classify_combos,
    equity,
    equity_vs,
    equity_vs_ranges,
    evaluate7,
    hero_equity_vs_each,
)
from poker_analyzer.engine.ranges import parse_range

# Категории руки (старшие биты ранга).
HIGH, PAIR, TWO_PAIR, TRIPS, STRAIGHT, FLUSH, FULL, QUADS, SF = 0, 1, 2, 3, 4, 5, 6, 7, 8


def _cat(seven: str) -> int:
    return evaluate7(cards(seven)) >> 20


def test_category_detection() -> None:
    assert _cat("As Ks Qs Js Ts 2c 7d") == SF
    assert _cat("Ah Ad Ac As Kh 2c 7d") == QUADS
    assert _cat("Ah Ad Ac Kh Kd 2c 7d") == FULL
    assert _cat("As Ks 9s 5s 2s 3d 7h") == FLUSH
    assert _cat("9c 8d 7h 6s 5c Ad Kd") == STRAIGHT
    assert _cat("5c 4d 3h 2s Ac Kd Qh") == STRAIGHT  # колесо A-2-3-4-5
    assert _cat("Ah Ad Ac Kh Qd 2c 7h") == TRIPS
    assert _cat("Ah Ad Kh Kd 9c 2s 7h") == TWO_PAIR
    assert _cat("Ah Ad Kh Qd 9c 2s 7h") == PAIR
    assert _cat("Ah Kd 9h 7d 5c 3s 2h") == HIGH


def test_ordering() -> None:
    # стрит-флеш > каре; фулл-хаус > флеш; старшая пара > младшая
    assert evaluate7(cards("As Ks Qs Js Ts 2c 7d")) > evaluate7(cards("Ah Ad Ac As Kh 2c 7d"))
    assert evaluate7(cards("Ah Ad Ac Kh Kd 2c 7d")) > evaluate7(cards("As Ks 9s 5s 2s 3d 7h"))
    assert evaluate7(cards("Ah Ad 9h 7d 5c 3s 2h")) > evaluate7(cards("Kh Kd 9h 7d 5c 3s 2c"))


def test_aa_vs_kk_known_and_symmetric() -> None:
    # борд пустой -> завершения перебираются точно, результат детерминирован
    a = equity_vs(cards("As Ah"), [], [cards("Ks Kh")])
    b = equity_vs(cards("Ks Kh"), [], [cards("As Ah")])
    assert 0.80 <= a.equity <= 0.84  # каноничные ~82% у тузов
    assert abs(a.equity + b.equity - 1.0) < 1e-9  # нулевая сумма HU (точный перебор)


def test_aks_vs_22_coinflip() -> None:
    r = equity_vs(cards("As Ks"), [], [cards("2c 2d")])
    assert 0.45 <= r.equity <= 0.55  # классический «кофлип»


def test_aa_vs_random_opponent() -> None:
    r = equity(cards("As Ah"), [], 1, iterations=200_000, seed=1)
    assert 0.83 <= r.equity <= 0.87  # AA против случайной руки ~85%


def test_more_opponents_lower_equity() -> None:
    one = equity(cards("As Ah"), [], 1, iterations=100_000, seed=3).equity
    five = equity(cards("As Ah"), [], 5, iterations=100_000, seed=3).equity
    assert five < one  # больше оппонентов — ниже эквити


def test_determinism() -> None:
    r1 = equity(cards("As Ah"), [], 2, iterations=50_000, seed=7)
    r2 = equity(cards("As Ah"), [], 2, iterations=50_000, seed=7)
    assert r1 == r2  # фиксированный сид -> идентичный результат


def test_validation_rejects_duplicates() -> None:
    with pytest.raises(ValueError):
        equity(cards("As Ah"), cards("As 2c"), 1)  # туз пик и в руке, и на борде


def test_range_of_one_hand_matches_equity_vs() -> None:
    # диапазон из единственной руки = точный перебор против неё (оба exact, борд пуст)
    rng = equity_vs_ranges(cards("As Ah"), [], [[cards("Ks Kh")]])
    direct = equity_vs(cards("As Ah"), [], [cards("Ks Kh")])
    assert abs(rng.equity - direct.equity) < 1e-9


def test_aa_dominates_tight_range() -> None:
    villain = parse_range("KK, QQ, AKs")  # AA впереди всего этого
    r = equity_vs_ranges(cards("As Ah"), [], [villain])
    assert 0.75 <= r.equity <= 0.95


def test_blockers_shrink_range_to_zero() -> None:
    # оппонент держит только AhKh — но Ah у героя → расклад невозможен, эквити не определено
    r = equity_vs_ranges(cards("Ah Kd"), [], [[cards("Ah Kh")]])
    assert r.equity == 0.0


def test_range_equity_determinism() -> None:
    wide = parse_range("22+, A2s+, K9o+")  # широкий → уйдёт в Монте-Карло
    r1 = equity_vs_ranges(cards("As Ks"), [], [wide], iterations=40_000, seed=11)
    r2 = equity_vs_ranges(cards("As Ks"), [], [wide], iterations=40_000, seed=11)
    assert r1 == r2  # фиксированный сид → идентичный результат


def test_hero_equity_vs_each_matches_equity_vs() -> None:
    # постфлоп перебор точный → поштучное эквити совпадает с equity_vs по каждому комбо
    board = cards("Ah Kd 7c")
    villain = [cards("Ks Kh"), cards("2c 2d")]
    eachs = hero_equity_vs_each(cards("As Ad"), board, villain)
    for combo, e in zip(villain, eachs, strict=True):
        assert abs(e - equity_vs(cards("As Ad"), board, [combo]).equity) < 1e-9


def test_hero_equity_vs_each_marks_blocked() -> None:
    e = hero_equity_vs_each(cards("As Ad"), cards("Ah Kd 7c"), [cards("As 2c")])
    assert e[0] == -1.0  # туз пик у героя → комбо заблокировано


# --- 1.3 классификатор комбо made/draw/air -----------------------------------


def test_classify_flush_draw_and_made_categories() -> None:
    board = cards("Ah 7h 2c")
    fd = classify_combos(board, [cards("Kh Qh")])[0]
    assert fd.flush_draw and fd.is_draw and not fd.is_made and not fd.is_air  # 4 червы
    pair = classify_combos(board, [cards("Ad Kc")])[0]
    assert pair.is_made and not pair.is_draw  # пара тузов
    two_pair = classify_combos(board, [cards("Ad 7c")])[0]
    assert two_pair.made == TWO_PAIR


def test_classify_straight_draws() -> None:
    oesd = classify_combos(cards("9h 8c 2d"), [cards("Ts 7h")])[0]
    assert oesd.oesd and not oesd.gutshot  # 7-8-9-T: достраивают 6 и J (двусторонний)
    gut = classify_combos(cards("Ah Kd 2c"), [cards("Qh Js")])[0]
    assert gut.gutshot and not gut.oesd  # A-K-Q-J: достраивает только T


def test_classify_marks_blocked() -> None:
    c = classify_combos(cards("Ah 7h 2c"), [cards("Ah Kc")])[0]
    assert c.blocked and c.made == -1  # туз червей на борде → комбо невозможно


def test_classify_empty_board_no_draw() -> None:
    c = classify_combos([], [cards("As Ad")])[0]
    assert c.is_made and not c.is_draw  # на пустом борде — только префлоп-сила (пара)


# --- 1.6 алярм rejection-collapse в equity_vs_ranges -------------------------


def test_equity_vs_ranges_reliable_by_default() -> None:
    r = equity_vs_ranges(cards("As Ks"), [], [parse_range("22+, A2s+")], iterations=40_000, seed=5)
    assert r.reliable  # обычный диапазон — оценка надёжна


def test_equity_vs_ranges_collapse_falls_back() -> None:
    # 5 оппонентов с пересекающимися флеш-комбо на монотонном борде: дефицит червей не даёт
    # расставить непротиворечивые руки → MC схлопывается → откат на vs-random (reliable=False),
    # а не катастрофические 0% (как было в исходном баге). exact_cap=1 форсирует MC-ветку.
    hearts = [cards("Th 9h"), cards("8h 7h"), cards("6h 5h"), cards("4h 3h")]
    r = equity_vs_ranges(
        cards("Qh Jh"), cards("Ah Kh 2h"), [hearts] * 5, iterations=2_000, exact_cap=1, seed=9
    )
    assert not r.reliable
    assert r.equity > 0.0  # герой собрал флеш — fallback даёт осмысленное число, не ноль


def test_low_exact_cap_matches_exact_on_river() -> None:
    # Перф-фикс: низкий exact_cap уводит крупные мультивей-расклады на МК (точный перебор
    # произведения диапазонов на ривере занимает секунды). МК с фикс-сидом детерминирован
    # и близок к точному.
    board = cards("Kh 7d 2c Td 3s")
    hero = cards("As Ks")  # топ-пара тузов-кикер
    rng = parse_range("22+, A2s+, K9s+, QTs+, JTs, ATo+, KJo+")  # ~150 комбо → work ~3.4M
    ranges = [rng, rng, rng]
    exact = equity_vs_ranges(hero, board, ranges, iterations=50_000, exact_cap=10_000_000, seed=7)
    mc = equity_vs_ranges(hero, board, ranges, iterations=50_000, exact_cap=200_000, seed=7)
    assert abs(exact.equity - mc.equity) < 0.03  # МК в пределах 3пп от точного — решение не плывёт
    # детерминизм: тот же вход (фикс-сид) → тот же ответ, без флапа между пересчётами
    mc2 = equity_vs_ranges(hero, board, ranges, iterations=50_000, exact_cap=200_000, seed=7)
    assert mc.equity == mc2.equity
