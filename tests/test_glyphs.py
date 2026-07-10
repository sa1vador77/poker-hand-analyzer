"""Тесты чтения ника строкой: сегментация на глифы, матч по атласу, атлас на диске.

Глифы синтетические (рисуем cv2.putText на белом фоне) — тесты герметичны, не зависят от
кропов в ``debug/``. Реальная проверка на живых кропах — в разборе, не в CI.
"""

from __future__ import annotations

import cv2
import numpy as np
import pytest

from poker_analyzer.vision.crop import dark_text_mask
from poker_analyzer.vision.glyphs import (
    NickRead,
    canonical_key,
    key_relation,
    label_glyphs,
    load_glyph_atlas,
    read_nick,
    save_glyph_atlas,
    segment_glyphs,
)

_HEIGHT = 28


def _render_glyph(char: str) -> np.ndarray:
    """Рисует один символ тёмным на белом фоне и обрезает по ширине контента."""
    canvas = np.full((_HEIGHT, 26, 3), 255, np.uint8)
    cv2.putText(canvas, char, (3, _HEIGHT - 7), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
    cols = np.where(dark_text_mask(canvas).any(axis=0))[0]
    return canvas[:, cols[0] : cols[-1] + 1]


def _compose(text: str, *, gap: int = 5, space_gap: int = 14) -> np.ndarray:
    """Собирает полосу ника из символов; пробел в ``text`` → широкий разрыв."""
    parts: list[np.ndarray] = []
    for char in text:
        if char == " ":
            if parts:
                parts[-1] = np.full((_HEIGHT, space_gap, 3), 255, np.uint8)
            continue
        parts.append(_render_glyph(char))
        parts.append(np.full((_HEIGHT, gap, 3), 255, np.uint8))
    return np.hstack(parts[:-1])


def test_segments_count_and_order() -> None:
    spans = segment_glyphs(_compose("ABCDE"))
    assert len(spans) == 5
    lefts = [left for left, _ in spans]
    assert lefts == sorted(lefts)  # слева направо
    assert all(left < right for left, right in spans)


def test_read_roundtrip_recovers_text() -> None:
    strip = _compose("BREAK")
    atlas = dict(label_glyphs(strip, "BREAK"))
    read = read_nick(strip, atlas)
    assert read.text == "BREAK"
    assert read.confident()
    assert read.n_unknown == 0
    assert read.min_score > 0.9


def test_independent_render_of_same_chars_matches() -> None:
    # атлас собран с ОДНОЙ полосы, читаем ДРУГУЮ, независимо собранную из тех же символов
    atlas = dict(label_glyphs(_compose("ABCD"), "ABCD"))
    read = read_nick(_compose("DCBA"), atlas)
    assert read.text == "DCBA"
    assert read.confident()


def test_empty_atlas_is_unreadable() -> None:
    read = read_nick(_compose("ABC"), {})
    assert not read.confident()
    assert read.n_unknown == 3
    assert read.text.count("�") == 3  # все глифы — нечитаемый знак �


def test_space_between_words_detected() -> None:
    strip = _compose("AB CD")
    atlas = dict(label_glyphs(strip, "ABCD"))
    read = read_nick(strip, atlas)
    assert read.text == "AB CD"


def test_label_glyphs_mismatch_raises() -> None:
    with pytest.raises(ValueError, match="не совпало"):
        label_glyphs(_compose("ABC"), "AB")  # сегментов 3, символов 2


def test_atlas_disk_roundtrip_is_case_sensitive(tmp_path) -> None:
    # 'a' и 'A' не должны слипнуться на регистронезависимой ФС macOS (имена u0061/u0041)
    entries = {char: _render_glyph(char) for char in "aA"}
    assert save_glyph_atlas(tmp_path, entries) == 2
    loaded = load_glyph_atlas(tmp_path)
    assert set(loaded) == {"a", "A"}


def test_save_does_not_overwrite_existing_by_default(tmp_path) -> None:
    save_glyph_atlas(tmp_path, {"x": _render_glyph("x")})
    # второй вызов с тем же символом ничего не пишет (первый эталон сохраняется)
    assert save_glyph_atlas(tmp_path, {"x": _render_glyph("x")}) == 0
    assert save_glyph_atlas(tmp_path, {"x": _render_glyph("x")}, overwrite=True) == 1


def test_canonical_key_folds_homoglyphs() -> None:
    # латинская 'a' и кириллическая 'а' визуально одинаковы → один ключ (стабильность)
    assert canonical_key("Mаx") == canonical_key("Max") == "Max"  # 'а' U+0430 → 'a'
    assert canonical_key("злoвeщий") == canonical_key("зловещий")  # 'o','e' свёрнуты к латинским
    assert canonical_key("Šamadzade") == canonical_key("Samadzade")  # гачек Š→S (матч их флапает)


def test_canonical_key_preserves_distinct_letters() -> None:
    # не-гомоглифы трогать нельзя, иначе сольём разные буквы
    assert canonical_key("л") == "л"  # кириллическая 'л' латинского двойника не имеет
    assert canonical_key("зв") == "зв"
    assert canonical_key("heartbreaker") == "heartbreaker"  # чистая латиница не меняется


def test_key_relation_same_tolerates_holes_and_flaps() -> None:
    assert key_relation("Furkatbek lrisov", "Furkatbek lr�sov") == "same"  # � — джокер
    assert key_relation("Arystan Uztemirov", "Arystan Uztemiroy") == "same"  # флап v↔y (1 из 17)
    assert key_relation("Max Ush", "Max Ush") == "same"


def test_canonical_key_strips_edge_noise() -> None:
    # краевой �/пробел — шум сегментации в прокрутке; срезаем, иначе ключ не сматчится с чистым
    assert canonical_key("ch�rka �") == "ch�rka"  # хвост убран, внутренний � (флап) сохранён
    assert canonical_key("Natural �") == "Natural"
    assert key_relation(canonical_key("ch�rka �"), canonical_key("chirka")) == "same"


def test_canonical_key_drops_garbage_trailing_word() -> None:
    # хвостовое «слово» сплошь из � — мусор грязного кадра; краевой strip его не берёт из-за
    # читаемого '-' на конце, но всё слово ≥50% � → срезаем, иначе фантомный id (S5 в логе)
    assert canonical_key("Бишкeк ���� ��-") == "Бишкeк"
    assert key_relation(canonical_key("Бишкeк ���� ��-"), canonical_key("Бишкeк")) == "same"
    # реальное второе слово ника не трогаем (в нём нет �)
    assert canonical_key("Гyлy Cyлeймaнoв") == "Гyлy Cyлeймaнoв"
    assert canonical_key("lHCR KASlANCV") == "lHCR KASlANCV"


def test_key_relation_different_for_distinct_nicks() -> None:
    assert key_relation("CCCEDlTS", "Hикитa") == "different"  # склейка, которую ловит 2b
    assert key_relation("Aze", "Лeв") == "different"


def test_key_relation_unknown_when_too_sparse() -> None:
    assert key_relation("Ab", "Cd") == "unknown"  # < min_solid → судить не по чему
    assert key_relation(None, "Hикитa") == "unknown"
    assert key_relation("���", "Hикитa") == "unknown"  # одни дырки


def test_empty_strip_has_no_glyphs() -> None:
    blank = np.full((_HEIGHT, 40, 3), 255, np.uint8)
    read = NickRead(text="", glyphs=())
    assert segment_glyphs(blank) == []
    assert read.min_score == 1.0  # нет глифов — не «плохо прочитано», а «нечего читать»
    assert not read.confident()  # но и ключом такой ник быть не может
