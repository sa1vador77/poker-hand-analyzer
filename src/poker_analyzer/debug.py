"""Визуальная отладка калибровки: выгрузка строк и остатков после каждого этапа.

Помогает на глаз проверить координаты :class:`~poker_analyzer.config.Layout` и
работу конвейера: по каждой строке кладёт три картинки — целую строку, остаток
после реза времени (фикс-координата) и остаток после реза ключевого слова (поиск
через matchTemplate). По умолчанию пишет в ``debug/`` (он в .gitignore).
"""

from __future__ import annotations

import logging
from pathlib import Path

import cv2
import numpy as np

from poker_analyzer.config import DEBUG_DIR, LAYOUT, THRESHOLDS, Layout, Thresholds
from poker_analyzer.parsing.segmentation import split_rows
from poker_analyzer.pipeline import AMOUNT_KEYWORDS, NICK_KEYWORDS
from poker_analyzer.vision.recognition import (
    find_amount_region,
    find_nick_region,
    recognize_keyword,
    win_card_zone,
)

logger = logging.getLogger(__name__)


def draw_layout_grid(frame: np.ndarray, layout: Layout = LAYOUT) -> np.ndarray:
    """Рисует на КОПИИ кадра разметку Layout — что именно режет пайплайн.

    Зелёная рамка — вся область лога; оранжевые рамки — строки; синяя черта — граница
    зоны времени внутри строки; красные числа слева — номера строк. Накладывается на
    реальный лог игры — сразу видно, совпадают ли координаты.
    """
    canvas = frame.copy()  # не трогаем исходник — его ещё распознавать
    height, _ = canvas.shape[:2]
    cv2.rectangle(canvas, (layout.x1, layout.y1), (layout.x2, layout.y2), (0, 255, 0), 2)

    bottom = min(layout.y2, height)
    time_x = layout.x1 + layout.time_width
    top = layout.y1
    index = 1
    while top + layout.row_height <= bottom:
        cv2.rectangle(
            canvas, (layout.x1, top), (layout.x2, top + layout.row_height), (0, 200, 255), 1
        )
        cv2.line(canvas, (time_x, top), (time_x, top + layout.row_height), (255, 0, 0), 1)
        cv2.putText(
            canvas,
            str(index),
            (max(0, layout.x1 - 30), top + layout.row_height - 8),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 0, 255),
            2,
        )
        top += layout.row_pitch
        index += 1
    return canvas


def dump_rows(
    image: np.ndarray,
    keyword_templates: dict[str, np.ndarray],
    out_dir: str | Path = DEBUG_DIR,
    *,
    layout: Layout = LAYOUT,
    thresholds: Thresholds = THRESHOLDS,
) -> int:
    """Выгружает по каждой строке три картинки: целую и остатки после этапов.

    Имена (сортируются по порядку этапов)::

        row_NN_0_full.png           — вся вырезанная строка
        row_NN_1_after_time.png     — после реза времени (по фикс. time_width)
        row_NN_2_after_keyword.png  — после реза ключевого слова (найдено matchTemplate)
        row_NN_3_amount.png         — красная зона суммы, изолированная по цвету (если есть)
        row_NN_4_nick.png           — тёмная зона ника, изолированная по «тёмности» (если есть)
        row_NN_5_wincards.png       — карты комбинации Win (зона между суммой и ником, бледные
                                      кикеры вытянуты контрастом — как их видит recognize_win_cards)

    :param image: скриншот окна с логом (BGR).
    :param keyword_templates: шаблоны ключевых слов (для поиска границы реза).
    :param out_dir: куда писать (создаётся при необходимости).
    :param layout: координаты нарезки (по умолчанию из config).
    :param thresholds: пороги распознавания (по умолчанию из config).
    :returns: число обработанных строк.
    """
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    rows = split_rows(image, layout)
    for index, row in enumerate(rows, start=1):
        cv2.imwrite(str(out / f"row_{index:02d}_0_full.png"), row)

        # Время режется по фиксированной ширине — это и есть остаток после этапа 1.
        after_time = row[:, layout.time_width :]
        cv2.imwrite(str(out / f"row_{index:02d}_1_after_time.png"), after_time)

        # Ключевое слово ищется matchTemplate; функция вернёт остаток справа от него.
        keyword, after_keyword = recognize_keyword(
            after_time, keyword_templates, threshold=thresholds.keyword
        )
        cv2.imwrite(str(out / f"row_{index:02d}_2_after_keyword.png"), after_keyword)

        # Для слов с суммой — красная зона, как её увидит recognize_amount.
        if keyword in AMOUNT_KEYWORDS:
            region = find_amount_region(after_keyword, red_delta=thresholds.amount_red_delta)
            if region is not None:
                cv2.imwrite(str(out / f"row_{index:02d}_3_amount.png"), region)

        # Для строк с ником — тёмная зона ника, как её увидит идентификация игрока.
        if keyword in NICK_KEYWORDS:
            nick = find_nick_region(
                after_keyword,
                red_delta=thresholds.amount_red_delta,
                dark_threshold=thresholds.nick_dark,
                has_leading_cards=keyword == "win",  # только Win несёт карты слева от ника
            )
            if nick is not None:
                cv2.imwrite(str(out / f"row_{index:02d}_4_nick.png"), nick)

        # Win: карты выигрышной комбинации (зона между суммой и ником, кикеры вытянуты контрастом).
        if keyword == "win":
            cards = win_card_zone(
                after_keyword,
                red_delta=thresholds.amount_red_delta,
                dark_threshold=thresholds.nick_dark,
            )
            if cards is not None:
                cv2.imwrite(str(out / f"row_{index:02d}_5_wincards.png"), cards)

    logger.info("Сохранено строк: %d (этапы конвейера) -> %s", len(rows), out)
    return len(rows)
