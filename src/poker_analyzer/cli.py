"""CLI-вход: прогон пайплайна на статичном скриншоте игрового лога.

Конвейер по строке: время → ключевое слово → карты (для hand/table) → сумма → игрок.
Координаты нарезки и пороги берутся из :mod:`poker_analyzer.config` (правятся там).

Запуск::

    uv run poker-analyzer path/to/screenshot.png
    uv run poker-analyzer path/to/screenshot.png --dump   # выгрузка этапов в debug/
"""

from __future__ import annotations

import argparse
import logging
from collections.abc import Sequence

import cv2

from poker_analyzer.debug import dump_rows
from poker_analyzer.log import setup_logging
from poker_analyzer.pipeline import process_screen
from poker_analyzer.vision.templates import load_all_templates

logger = logging.getLogger(__name__)


def _parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Разбор скриншота покерного лога.")
    parser.add_argument("screen", help="путь к скриншоту с логом раздачи")
    parser.add_argument(
        "--dump",
        action="store_true",
        help="сохранить вырезанные строки в debug/ и выйти (для калибровки)",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    """Точка входа CLI. Возвращает код возврата процесса (0 — успех)."""
    setup_logging()
    args = _parse_args(argv)

    image = cv2.imread(args.screen)
    if image is None:
        logger.error("Не удалось открыть скриншот: %s", args.screen)
        return 1

    templates = load_all_templates()

    if args.dump:
        dump_rows(image, templates.keywords)
        return 0

    logger.info(
        "Шаблоны: цифр %d, слов %d, рангов %d, мастей %d",
        len(templates.digits),
        len(templates.keywords),
        len(templates.ranks),
        len(templates.suits),
    )

    results = process_screen(image, templates)
    logger.info("Обработано строк: %d", len(results))
    for row in results:
        logger.info(
            "Строка %2d: время=%s слово=%s игрок=%s сумма=%s карты=%s",
            row.index,
            row.time,
            row.keyword,
            row.player_id if row.player_id is not None else "—",
            f"{row.amount:g}" if row.amount is not None else "—",
            " ".join(row.cards) or "—",
        )
    return 0
