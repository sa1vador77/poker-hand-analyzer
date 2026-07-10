"""Докладывает глифы ника в атлас ``data/templates/nick`` из РАЗМЕЧЕННОГО кропа.

Кроп (``row_NN_4_nick.png`` из ``poker-analyzer --dump``) — это ник целиком. Шрифт
фиксированный, поэтому ник режется на глифы по разрывам столбцов
(:func:`vision.glyphs.segment_glyphs`), а МЕТКА — строка-подпись, которую читает глазами
человек/агент, — сопоставляет каждому глифу символ. Новые символы дописываются в атлас,
уже собранные НЕ перетираются (первый эталон обычно чище).

Запуск (по кропу за раз, метку берём, прочитав сам кроп):

    uv run python scripts/add_nick_glyphs.py debug/row_07_4_nick.png heartbreaker
    uv run python scripts/add_nick_glyphs.py debug/row_08_4_nick.png "X Æ A-12"

Пробелы в метке игнорируются (пробел в нике глифа не даёт). Если число сегментов не
совпало с числом символов метки — скрипт НЕ пишет ничего и печатает диагностику
(сегментация разошлась с разметкой — поправить метку или кроп). Точечные/пунктуационные
ники (``.,.``) лучше НЕ скармливать: их глифы крошечные (~4px) и как шаблоны дают ложные
совпадения внутри любой буквы — такие ники остаются на откуп сравнению картинкой.
"""

from __future__ import annotations

import sys

import cv2

from poker_analyzer.config import NICK_GLYPH_TEMPLATES_DIR
from poker_analyzer.vision import glyphs


def main(argv: list[str]) -> int:
    if len(argv) != 2:
        print("использование: add_nick_glyphs.py <кроп.png> <метка>", file=sys.stderr)
        return 2
    path, label = argv

    img = cv2.imread(path, cv2.IMREAD_COLOR)
    if img is None:
        print(f"не удалось прочитать кроп: {path}", file=sys.stderr)
        return 1

    try:
        entries = glyphs.label_glyphs(img, label)
    except ValueError as exc:
        spans = glyphs.segment_glyphs(img)
        print(f"РАЗМЕТКА НЕ СОШЛАСЬ: {exc}", file=sys.stderr)
        print(
            f"  сегментов {len(spans)}, символов в метке "
            f"{len(label.replace(' ', ''))} ({label!r}) — ничего не сохранено",
            file=sys.stderr,
        )
        return 1

    before = set(glyphs.load_glyph_atlas(NICK_GLYPH_TEMPLATES_DIR))
    glyphs.save_glyph_atlas(NICK_GLYPH_TEMPLATES_DIR, dict(entries))
    after = set(glyphs.load_glyph_atlas(NICK_GLYPH_TEMPLATES_DIR))

    added = sorted(after - before)
    skipped = sorted({char for char, _ in entries} & before)
    print(f"{path}: метка {label!r} → {len(entries)} глифов")
    print(f"  добавлено ({len(added)}): {''.join(added) or '—'}")
    print(f"  уже были, пропущено: {''.join(skipped) or '—'}")
    print(f"  атлас теперь: {len(after)} символов: {''.join(sorted(after))}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
