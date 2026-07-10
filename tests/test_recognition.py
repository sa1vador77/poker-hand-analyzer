"""Тесты разбора суммы (_parse_amount, _drop_detached_multiplier)."""

from __future__ import annotations

from poker_analyzer.vision.recognition import _drop_detached_multiplier, _parse_amount


def test_fractional_amount_not_lost() -> None:
    assert _parse_amount("0.50") == 0.5  # раньше int(round(0.5)) давал 0
    assert _parse_amount("1.50") == 1.5


def test_integer_amounts() -> None:
    assert _parse_amount("25") == 25
    assert _parse_amount("950") == 950


def test_suffix_k_and_m() -> None:
    assert _parse_amount("3.61K") == 3610
    assert _parse_amount("2M") == 2_000_000


def test_empty_or_garbage_returns_none() -> None:
    assert _parse_amount("") is None
    assert _parse_amount("zz") is None


def test_win_row_trailing_junk_ignored() -> None:
    # на строке Win после суммы идут карты комбинации и запятая → берём ведущее число
    assert _parse_amount("313...") == 313  # '313, A♠K♠Q♠J♥10♣ …'
    assert _parse_amount("39.20....") == 39.2  # '39.20, K♦K♠ …'
    assert _parse_amount("1.5K..") == 1500  # множитель сразу за числом


def test_nick_k_after_amount_not_thousands() -> None:
    # ник «K» (S11, живой лог 2026-06-13) стоит за красной суммой с ПРОБЕЛОМ →
    # не множитель: '50' + оторванный 'K' иначе раздувался в 50000. Глифы:
    # (left, right, name). Ширина цифры ~10px; ник отстоит на ~16px (> 0.6×10).
    glyphs = [(0, 10, "5"), (10, 20, "0"), (36, 46, "K")]
    kept = _drop_detached_multiplier(glyphs)
    assert kept == [(0, 10, "5"), (10, 20, "0")]  # «K» отрезан
    text = "".join(name for _l, _r, name in kept)
    assert _parse_amount(text) == 50


def test_real_multiplier_kept_when_snug() -> None:
    # настоящий «K» прижат к цифрам (зазор ~1px ≪ 0.6×ширина) → остаётся множителем
    glyphs = [(0, 10, "1"), (10, 15, "dot"), (15, 25, "5"), (26, 36, "K")]
    kept = _drop_detached_multiplier(glyphs)
    assert kept == glyphs  # ничего не отрезано
    text = "".join({"dot": "."}.get(name, name) for _l, _r, name in kept)
    assert _parse_amount(text) == 1500
