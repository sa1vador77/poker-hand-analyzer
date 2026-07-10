"""Тесты выбора размера ставки по EV."""

from __future__ import annotations

from poker_analyzer.engine.sizing import best_bet_from_eqs


def test_none_without_combos_or_pot() -> None:
    assert best_bet_from_eqs([], 100) is None  # нет диапазона
    assert best_bet_from_eqs([0.5], 0) is None  # нет банка


def test_blocked_combos_filtered() -> None:
    assert best_bet_from_eqs([-1.0, -1.0], 100) is None  # все заблокированы


def test_nuts_bet_has_positive_ev() -> None:
    # герой выигрывает всегда (эквити 1.0) → ставка в плюс, оппонент весь фолдит
    best = best_bet_from_eqs([1.0] * 20, 100)
    assert best is not None
    assert best.ev > 0
    assert best.fold_equity == 1.0  # против 0%-эквити оппонента все фолдят


def test_bigger_size_folds_at_least_as_much() -> None:
    eqs = [0.3, 0.45, 0.55, 0.7]  # эквити героя против комбо оппонента
    small = best_bet_from_eqs(eqs, 100, sizes=(0.5,))
    big = best_bet_from_eqs(eqs, 100, sizes=(2.0,))
    assert small is not None and big is not None
    assert big.fold_equity >= small.fold_equity  # крупнее ставка → не меньше фолдов
