"""Загрузка наборов шаблонов с диска."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from poker_analyzer.config import (
    HERO_TEMPLATES_DIR,
    KEYWORD_TEMPLATES_DIR,
    NICK_GLYPH_TEMPLATES_DIR,
    RANK_TEMPLATES_DIR,
    SUIT_TEMPLATES_DIR,
    TIME_TEMPLATES_DIR,
)
from poker_analyzer.vision.glyphs import load_glyph_atlas


def load_templates(directory: str | Path) -> dict[str, np.ndarray]:
    """Загружает шаблоны ``*.png`` из папки в grayscale.

    Ключ словаря — имя файла без расширения (например ``'0'``..``'9'`` для цифр
    времени). Поднимает ошибку, если шаблонов в папке нет.
    """
    templates: dict[str, np.ndarray] = {}
    for path in sorted(Path(directory).glob("*.png")):
        image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        if image is None:
            raise OSError(f"Не удалось прочитать шаблон: {path}")
        templates[path.stem] = image

    if not templates:
        raise FileNotFoundError(f"В папке нет шаблонов *.png: {directory}")
    return templates


def load_optional(directory: str | Path) -> dict[str, np.ndarray]:
    """Как :func:`load_templates`, но не падает, если папки/файлов нет — возвращает ``{}``.

    Для необязательных наборов (например, шаблон ника героя может быть ещё не нарезан).
    """
    path = Path(directory)
    if not path.is_dir():
        return {}
    templates: dict[str, np.ndarray] = {}
    for file in sorted(path.glob("*.png")):
        image = cv2.imread(str(file), cv2.IMREAD_GRAYSCALE)
        if image is not None:
            templates[file.stem] = image
    return templates


@dataclass(frozen=True, slots=True)
class Templates:
    """Все наборы шаблонов проекта, загруженные один раз.

    Ключ в каждом словаре — имя файла без расширения.
    """

    digits: dict[str, np.ndarray]  # цифры времени/сумм: '0'..'9'
    keywords: dict[str, np.ndarray]  # ключевые слова: 'call', 'bet', ...
    ranks: dict[str, np.ndarray]  # ранги карт: '2'..'10', 'J', 'Q', 'K', 'A'
    suits: dict[str, np.ndarray]  # масти: 'club', 'diamond', 'heart', 'spade'
    hero: dict[
        str, np.ndarray
    ]  # кропы ника героя для его опознания (опционален: {} если папки нет)
    glyphs: dict[
        str, np.ndarray
    ]  # атлас глифов ника (символ → шаблон); опционален: {} если папки нет


def load_all_templates() -> Templates:
    """Загружает все наборы шаблонов проекта из их папок (см. config)."""
    return Templates(
        digits=load_templates(TIME_TEMPLATES_DIR),
        keywords=load_templates(KEYWORD_TEMPLATES_DIR),
        ranks=load_templates(RANK_TEMPLATES_DIR),
        suits=load_templates(SUIT_TEMPLATES_DIR),
        hero=load_optional(HERO_TEMPLATES_DIR),  # опционален: {} если папки/файлов нет
        glyphs=load_glyph_atlas(NICK_GLYPH_TEMPLATES_DIR),  # опционален: {} (атлас пока не собран)
    )
