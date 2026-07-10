"""Тесты примитивов поиска шаблонов — на синтетических картинках, без файлов на диске."""

from __future__ import annotations

import numpy as np

from poker_analyzer.vision.matching import _iou, match_template, match_template_all


def _patch() -> np.ndarray:
    """Шаблон 10x10 с вариативностью (верх чёрный, низ белый).

    Однотонный шаблон не годится: у TM_CCOEFF_NORMED нулевая дисперсия даёт
    вырожденную корреляцию. Поэтому делаем явный контраст.
    """
    patch = np.full((10, 10), 255, dtype=np.uint8)
    patch[:5, :] = 0
    return patch


def test_iou_no_overlap() -> None:
    assert _iou((0, 0, 10, 10), (20, 20, 30, 30)) == 0.0


def test_iou_identical() -> None:
    assert _iou((0, 0, 10, 10), (0, 0, 10, 10)) == 1.0


def test_match_template_finds_single_patch() -> None:
    image = np.full((40, 60), 255, dtype=np.uint8)
    patch = _patch()
    image[10:20, 30:40] = patch

    match = match_template(image, patch)

    assert match.top_left == (30, 10)
    assert match.score > 0.99


def test_match_template_all_finds_two_patches_left_to_right() -> None:
    image = np.full((40, 80), 255, dtype=np.uint8)
    patch = _patch()
    image[10:20, 10:20] = patch
    image[10:20, 50:60] = patch

    matches = match_template_all(image, patch, threshold=0.9)

    assert [m.top_left[0] for m in matches] == [10, 50]
