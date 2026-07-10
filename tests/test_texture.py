"""Тесты текстуры борда и её влияния на сужение диапазона (порт board_texture.py)."""

from __future__ import annotations

from poker_analyzer.config import NARROWING, TEXTURE_NARROWING
from poker_analyzer.engine.advisor import _TIER_RANGES
from poker_analyzer.engine.equity import cards
from poker_analyzer.engine.ranges import narrow_range
from poker_analyzer.engine.texture import analyze_texture, texture_delta


def test_texture_features() -> None:
    dry = analyze_texture(cards("Kh 7d 2c"))
    assert dry is not None and dry.is_rainbow and dry.dry and not dry.dynamic

    monotone = analyze_texture(cards("Ah Kh Qh"))
    assert monotone is not None and monotone.is_monotone and monotone.dynamic
    assert monotone.max_same_suit == 3 and monotone.broadway_count == 3

    connected = analyze_texture(cards("9h 8d 7c"))
    assert connected is not None and connected.is_very_connected and connected.max_run == 3

    paired = analyze_texture(cards("Kh Kd 4c"))
    assert paired is not None and paired.is_paired and not paired.dry and not paired.dynamic

    two_tone = analyze_texture(cards("Qh Jh 2c"))
    assert two_tone is not None and two_tone.is_two_tone and two_tone.max_same_suit == 2


def test_texture_wheel_connectivity() -> None:
    # Туз играет снизу: A-2-3 — связка (run учитывает колёсного туза).
    wheel = analyze_texture(cards("Ah 2d 3c"))
    assert wheel is not None and wheel.max_run == 3 and wheel.is_very_connected


def test_texture_invalid_size() -> None:
    assert analyze_texture([]) is None  # префлоп
    assert analyze_texture(cards("Kh 7d")) is None  # 2 карты — не борд


def test_texture_delta_signs() -> None:
    assert texture_delta(cards("Ah Kh Qh")) == TEXTURE_NARROWING.wet  # монотон — мокрый
    assert texture_delta(cards("9h 8d 7c")) == TEXTURE_NARROWING.wet  # связка — мокрый
    assert texture_delta(cards("Qh Jh 2c")) == TEXTURE_NARROWING.semi_wet  # two-tone
    assert texture_delta(cards("Kh 7d 2c")) == TEXTURE_NARROWING.dry  # радуга сухая
    assert texture_delta([]) == 0.0  # префлоп — без поправки


def test_texture_widens_range_on_wet_board() -> None:
    # Мокрый борд: с текстурой суженный диапазон агрессора СТРОГО шире (пул продолжает
    # с дро) — это и снимает перефолд готовых рук против «только натсов».
    board = cards("Jh 9d 8c")
    thr = NARROWING.aggressor[0]  # флоп
    base = len(narrow_range(_TIER_RANGES[2], board, thr))
    wet = len(narrow_range(_TIER_RANGES[2], board, thr + texture_delta(board)))
    assert wet > base

    # Сухой борд: с текстурой чуть УЖЕ (продолжение поляризовано в готовые руки).
    dry_board = cards("Jh 7d 2c")
    dry_base = len(narrow_range(_TIER_RANGES[2], dry_board, thr))
    dry = len(narrow_range(_TIER_RANGES[2], dry_board, thr + texture_delta(dry_board)))
    assert dry < dry_base
