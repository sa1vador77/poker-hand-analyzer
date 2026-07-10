"""Распознавание элементов строки лога по шаблонам.

Закрытый словарь (время, ключевые слова, суммы, ранги/масти карт) распознаётся
через matchTemplate надёжно. Ники текстом НЕ читаются — игроков различает слой
:mod:`poker_analyzer.identity` по картинке.

Распознавание устроено как конвейер: каждая функция распознаёт свой элемент и
возвращает строку лога БЕЗ него, чтобы передать остаток следующему этапу.
"""

from __future__ import annotations

import logging
import re

import cv2
import numpy as np

from poker_analyzer.config import THRESHOLDS
from poker_analyzer.vision.crop import dark_text_mask
from poker_analyzer.vision.matching import match_template, match_template_all
from poker_analyzer.vision.types import Match

logger = logging.getLogger(__name__)


def recognize_time(
    row: np.ndarray,
    digit_templates: dict[str, np.ndarray],
    *,
    time_width: int,
    threshold: float = THRESHOLDS.time_digit,
    quiet: bool = False,
) -> tuple[str, np.ndarray]:
    """Распознаёт время и возвращает ``(время, строка без времени)``.

    Время сидит в фиксированной левой зоне шириной ``time_width`` — её и режем
    (координаты времени стабильны, искать не нужно). Каждый шаблон цифры гоним
    через :func:`match_template_all`, находки сортируем слева направо, двоеточия
    восстанавливаем по структуре ``HH:MM:SS``.

    :returns: ``(время, остаток строки справа от зоны времени)``. Если цифр
        распозналось не 6 — в лог уходит предупреждение, а временем станет сырая
        строка цифр (удобно для отладки порога).
    """
    time_crop = row[:, :time_width]

    found: list[tuple[int, str]] = []  # (x, цифра)
    for digit, template in digit_templates.items():
        if not (len(digit) == 1 and digit.isdigit()):
            continue  # в наборе time лежат ещё K/точка для сумм — времени нужны только 0–9
        for match in match_template_all(time_crop, template, threshold=threshold):
            found.append((match.top_left[0], digit))

    found.sort()  # слева направо по X
    digits = "".join(digit for _, digit in found)

    if len(digits) == 6:
        time = f"{digits[:2]}:{digits[2:4]}:{digits[4:]}"
    else:
        if not quiet:  # quiet — для проб (detect_grid_shift), там промах штатен
            logger.warning("Время распознано неполностью: %r (ожидалось 6 цифр)", digits)
        time = digits

    return time, row[:, time_width:]


def recognize_keyword(
    row: np.ndarray,
    keyword_templates: dict[str, np.ndarray],
    *,
    threshold: float = THRESHOLDS.keyword,
) -> tuple[str | None, np.ndarray]:
    """Находит ключевое слово и возвращает ``(слово, строка после слова)``.

    В отличие от времени, слово стоит на плавающей позиции (зависит от иконки и
    ширины того, что слева), поэтому координаты не фиксированы — ищем слово через
    matchTemplate. Собираем всех кандидатов выше порога и снимаем «префиксную»
    неоднозначность: ``deal`` — это начало ``dealer``, поэтому шаблон ``deal`` хорошо
    матчится в первых буквах ``dealer``. Если короткое совпадение почти целиком лежит
    ВНУТРИ более широкого в той же позиции — это подслово, отбрасываем в пользу длинного.

    :returns: ``(имя слова, остаток строки справа от слова)`` или ``(None, row)``,
        если ни один шаблон не прошёл порог (строка возвращается нетронутой).
    """
    row_height, row_width = row.shape[:2]

    # Все кандидаты выше порога (по одному лучшему месту на шаблон).
    candidates: list[tuple[str, Match]] = []
    for name, template in keyword_templates.items():
        # Шаблон выше/шире строки matchTemplate не примет — пропускаем, чтобы не
        # падать (обычно значит, что row_height мал для этого слова — см. калибровку).
        if template.shape[0] > row_height or template.shape[1] > row_width:
            logger.debug("Шаблон %r не влезает в строку — пропуск", name)
            continue
        match = match_template(row, template)
        if match.score >= threshold:
            candidates.append((name, match))

    if not candidates:
        logger.warning("Ключевое слово не распознано (порог %.2f не пройден)", threshold)
        return None, row

    def _is_subword(name: str, match: Match) -> bool:
        # Подслово: есть более широкий кандидат в той же позиции (короткое внутри него).
        own = (match.top_left[0], match.bottom_right[0])
        for other_name, other in candidates:
            if other_name == name:
                continue
            other_x = (other.top_left[0], other.bottom_right[0])
            if other.size[0] > match.size[0] and _x_overlap(own, other_x) >= 0.7:
                return True
        return False

    survivors = [(name, m) for name, m in candidates if not _is_subword(name, m)]
    # Лучшее по score; при равенстве предпочитаем более широкое (специфичное) слово.
    best_name, best_match = max(survivors, key=lambda c: (c[1].score, c[1].size[0]))

    cut_x = best_match.bottom_right[0]
    return best_name, row[:, cut_x:]


# Глифы мастей для отображения карты строкой ('8' + '♣' = '8♣').
_SUIT_GLYPHS = {"spade": "♠", "heart": "♥", "diamond": "♦", "club": "♣"}


def _x_overlap(a: tuple[int, int], b: tuple[int, int]) -> float:
    """Доля перекрытия по X двух отрезков (относительно меньшего из них)."""
    inter = max(0, min(a[1], b[1]) - max(a[0], b[0]))
    smaller = min(a[1] - a[0], b[1] - b[0])
    return inter / smaller if smaller > 0 else 0.0


def _detect_glyphs(
    strip: np.ndarray,
    templates: dict[str, np.ndarray],
    *,
    threshold: float,
    overlap: float = 0.5,
) -> list[tuple[int, int, str]]:
    """Находит все вхождения набора шаблонов с межшаблонным NMS по X.

    На одном месте оставляет шаблон с лучшим score (чтобы похожие ранги вроде 6/8
    не дублировались). Возвращает ``[(x_left, x_right, name)]`` слева направо.
    """
    strip_height, strip_width = strip.shape[:2]
    candidates: list[tuple[float, int, int, str]] = []  # (score, x_left, x_right, name)
    for name, template in templates.items():
        if template.shape[0] > strip_height or template.shape[1] > strip_width:
            continue
        for match in match_template_all(strip, template, threshold=threshold):
            candidates.append((match.score, match.top_left[0], match.bottom_right[0], name))

    candidates.sort(key=lambda c: -c[0])  # лучшие по score первыми
    kept: list[tuple[int, int, str]] = []
    for _score, x_left, x_right, name in candidates:
        if any(_x_overlap((x_left, x_right), (left, right)) >= overlap for left, right, _ in kept):
            continue
        kept.append((x_left, x_right, name))

    kept.sort(key=lambda item: item[0])  # слева направо
    return kept


def recognize_cards(
    strip: np.ndarray,
    rank_templates: dict[str, np.ndarray],
    suit_templates: dict[str, np.ndarray],
    *,
    rank_threshold: float = THRESHOLDS.rank,
    suit_threshold: float = THRESHOLDS.suit,
) -> list[str]:
    """Распознаёт карты в остатке строки и возвращает список вида ``['7♦', '4♠']``.

    Вызывать только для строк с keyword ∈ {hand, table}. Ранги и масти ищутся по
    форме (grayscale): один цвет шаблона ранга ловит и красный, и чёрный за счёт
    нормировки TM_CCOEFF_NORMED, а 4 масти различаются формой. Карты идут
    «ранг-масть» слева направо, поэтому каждый ранг спаривается с ближайшей мастью
    справа.

    :returns: карты слева направо, например ``['7♦', '4♠', '6♥']``.
    """
    ranks = _detect_glyphs(strip, rank_templates, threshold=rank_threshold)
    suits = _detect_glyphs(strip, suit_templates, threshold=suit_threshold)

    cards: list[str] = []
    suit_pos = 0
    for rank_left, _rank_right, rank_name in ranks:
        # пропускаем масти левее начала ранга, берём ближайшую справа
        while suit_pos < len(suits) and suits[suit_pos][0] < rank_left:
            suit_pos += 1
        if suit_pos >= len(suits):
            break  # рангу не нашлось масти справа — вероятно ложное срабатывание
        suit_name = suits[suit_pos][2]
        cards.append(rank_name + _SUIT_GLYPHS.get(suit_name, suit_name))
        suit_pos += 1
    return cards


# Имена шаблонов, которые в сумме отображаются особым символом ('dot' -> '.').
# Цифры '0'..'9' и буквы 'K'/'M' совпадают с именем файла, мапить не нужно.
_AMOUNT_CHARS = {"dot": "."}

# Разрыв (px) между красными кластерами строки: больше него = сумма кончилась, дальше карты
# комбинации (их красные масти ♦♥ — отдельные кластеры). Внутрисуммовые разрывы (между цифрами,
# у точки/K) заметно меньше. Используется и для зоны суммы, и для изоляции карт на Win-строке.
_AMOUNT_CLUSTER_GAP = 25


def _drop_detached_multiplier(
    glyphs: list[tuple[int, int, str]],
    *,
    gap_ratio: float = 0.6,
) -> list[tuple[int, int, str]]:
    """Убирает оторванный справа суффикс ``K``/``M`` — это НИК, а не множитель тысяч.

    В ЛОГЕ СТОЛА суммы пишутся целиком (``50``, ``3360``), без сокращений — настоящий
    множитель тут не встречается (суммы вроде «3.84K» бывают в других экранах игры, не в логе).
    Одно-буквенный ник «K» (тот же глиф, что суффикс тысяч в «3.84K») стоит сразу за
    красной суммой на строках Blind/Call/Bet и иначе раздувает её ×1000 (25 → 25000,
    живой лог 2026-06-13). Настоящий множитель прижат к цифрам вплотную, ник отделён
    пробелом — режем по разрыву шире доли средней ширины глифа. Цифры не трогаем никогда.
    """
    if len(glyphs) < 2:
        return glyphs
    last_left, _last_right, last_name = glyphs[-1]
    if last_name not in ("K", "k", "M", "m"):
        return glyphs
    prev_right = glyphs[-2][1]
    widths = [right - left for left, right, _name in glyphs]
    median_width = sorted(widths)[len(widths) // 2] or 1
    if last_left - prev_right > gap_ratio * median_width:
        return glyphs[:-1]
    return glyphs


def find_amount_region(
    strip: np.ndarray,
    *,
    red_delta: int = THRESHOLDS.amount_red_delta,
) -> np.ndarray | None:
    """Вырезает столбцы со суммой по красному цвету (сумма красная, остальное серое).

    Возвращает полосу ПОЛНОЙ высоты с колонками суммы (плюс небольшой отступ) или
    ``None``, если красного в строке нет. Высоту не режем — чтобы шаблоны цифр
    гарантированно влезали по вертикали.

    Берём ТОЛЬКО ПЕРВЫЙ красный кластер. Сумма всегда первая и набрана слитно (цифры/точка/
    ``K`` без больших разрывов), а на строке ``Win`` правее идут карты комбинации с КРАСНЫМИ
    мастями (♦♥) — отдельные кластеры за разрывом. Без среза зона суммы расползалась на эти
    карты (``recognize_amount`` всё равно брал ведущее число, но дамп ``row_NN_3_amount`` вводил
    в заблуждение, а ранги-цифры карт могли бы засорять чтение). На прочих строках карт нет —
    разрыва нет, поведение прежнее.
    """
    if strip.ndim != 3:  # нужен цвет, иначе красное не отличить от серого
        return None
    blue = strip[:, :, 0].astype(np.int16)
    green = strip[:, :, 1].astype(np.int16)
    red = strip[:, :, 2].astype(np.int16)
    red_mask = red - np.maximum(blue, green) > red_delta

    cols = np.where(red_mask.any(axis=0))[0]
    if len(cols) == 0:
        return None
    gaps = np.where(np.diff(cols) > _AMOUNT_CLUSTER_GAP)[0]
    if len(gaps):  # обрезаем по первому большому разрыву — дальше начались карты, не сумма
        cols = cols[: int(gaps[0]) + 1]
    pad = 3
    x_left = max(0, int(cols.min()) - pad)
    x_right = min(strip.shape[1], int(cols.max()) + 1 + pad)
    return strip[:, x_left:x_right]


_AMOUNT_RE = re.compile(r"\d+(?:\.\d+)?")  # ведущее число (целое или дробное)


def _parse_amount(text: str) -> float | None:
    """Парсит распознанную сумму вроде '230', '1.5K', '0.50', '2M' в число.

    Берётся ВЕДУЩЕЕ число: сумма стоит первой, а дальше в строке может идти мусор —
    например на ``Win`` после суммы распознаются карты выигрышной комбинации и запятая
    (``'313...'`` → 313, ``'39.20....'`` → 39.2). Множитель ``K``/``M`` — если стоит
    сразу за числом. Возвращает float, а не int: дробные суммы (0.50) терять нельзя.
    """
    match = _AMOUNT_RE.match(text)
    if match is None:
        return None
    rest = text[match.end() :]
    multiplier = 1
    if rest[:1] in ("K", "k"):
        multiplier = 1_000
    elif rest[:1] in ("M", "m"):
        multiplier = 1_000_000
    try:
        value = float(match.group())
    except ValueError:
        return None
    return round(value * multiplier, 2)  # ndigits=2 → float (дробь не теряется)


def recognize_amount(
    strip: np.ndarray,
    digit_templates: dict[str, np.ndarray],
    *,
    threshold: float = THRESHOLDS.amount,
    red_delta: int = THRESHOLDS.amount_red_delta,
) -> float | None:
    """Распознаёт сумму в остатке строки и возвращает целое число (или ``None``).

    Вызывать только для строк с keyword ∈ {deal, bet, call, blind, allin, raise, win}.
    Сумма красная, поэтому изолируется по цвету (тире и ник серые — не мешают), а
    внутри красной зоны цифры / ``K`` / ``M`` / точка распознаются теми же
    шаблонами, что и время (форма; красный ловится grayscale за счёт CCOEFF_NORMED).

    :param digit_templates: шаблоны цифр и символов (папка time: 0–9, K, точка;
        для миллионов нужен ещё M).
    :returns: значение суммы, например 230 для '230' или 1500 для '1.5K'.
    """
    region = find_amount_region(strip, red_delta=red_delta)
    if region is None:
        return None
    glyphs = _detect_glyphs(region, digit_templates, threshold=threshold)
    glyphs = _drop_detached_multiplier(glyphs)  # ник «K» вплотную к сумме ≠ ×1000
    text = "".join(_AMOUNT_CHARS.get(name, name) for _left, _right, name in glyphs)
    return _parse_amount(text)


def find_nick_region(
    strip: np.ndarray,
    *,
    red_delta: int = THRESHOLDS.amount_red_delta,
    dark_threshold: int = THRESHOLDS.nick_dark,
    has_leading_cards: bool = False,
) -> np.ndarray | None:
    """Вырезает зону ника (тёмный серый текст) из остатка строки после слова.

    Ник — самый тёмный текст строки; сумма красная, тире/слово светлее. Берём тёмные
    НЕ красные столбцы и режем по их границам — кроп выходит выровненным по нику, что
    важно и для сравнения игроков корреляцией, и для чтения строкой (см. ``vision.glyphs``).

    :param has_leading_cards: на строке ``Win`` слева от ника идут карты выигрышной комбинации
        (тёмные ранги и ЧЁРНЫЕ масти ♠♣ проходят маску) — их надо отрезать, иначе кроп
        «карты+ник» не совпадёт с чистым ником в других строках. На ВСЕХ прочих строках слева
        от ника только тире (светлее порога, в маску не входит), карт нет — поэтому берём весь
        тёмный кластер ЦЕЛИКОМ. Иначе обрыв по разрыву отрезал бы первое слово двусловного ника
        (``Юлия Файзиева`` → ``Файзиева``): межсловный пробел сравним с порогом-разрывом.

    :returns: полоса со столбцами ника (полной высоты, плюс отступ) или ``None``,
        если тёмного текста нет (строка без ника).
    """
    if strip.ndim != 3:  # нужен цвет, чтобы отделить красную сумму от серого ника
        return None
    nick_mask = dark_text_mask(strip, dark=dark_threshold, red_delta=red_delta)

    cols = np.where(nick_mask.any(axis=0))[0]
    if len(cols) == 0:
        return None
    right = int(cols[-1])
    if has_leading_cards:
        # Win: идём от правого края влево, пока разрыв до следующего тёмного столбца не превысит
        # max_gap — это граница «карты | ник». Порог = высота строки: межсловный пробел ника
        # (≈0.5h, замер 19px при h=37) он перекрывает, а до разрыва «карты→ник» далеко — там
        # светлое тире (в маску не входит) даёт разрыв заметно шире высоты (замер 134px). Иначе
        # узкий порог рубил двусловный ник по пробелу (Max Ush → Ush).
        max_gap = max(16, strip.shape[0])
        left = right
        for col in reversed(cols[:-1].tolist()):
            if left - col <= max_gap:
                left = col
            else:
                break  # большой разрыв — начались карты левее ника
    else:
        left = int(cols[0])  # обычная строка: весь ник целиком, включая все слова
    pad = 3
    x_left = max(0, left - pad)
    x_right = min(strip.shape[1], right + 1 + pad)
    return strip[:, x_left:x_right]


def win_card_zone(
    strip: np.ndarray,
    *,
    red_delta: int = THRESHOLDS.amount_red_delta,
    dark_threshold: int = THRESHOLDS.nick_dark,
) -> np.ndarray | None:
    """Изолированная и КОНТРАСТ-НОРМАЛИЗОВАННАЯ зона карт Win-строки (между суммой и ником).

    Левая граница — правый край ПЕРВОГО красного кластера (сумма красная; красные масти карт
    ♦♥ правее — отдельный кластер за разрывом). Правая — левый край ника (как в
    :func:`find_nick_region`: от правого края тёмной маски влево, разрыв > высоты строки =
    граница «карты | ник»; бледные кикеры и тире в тёмную маску не входят и сами дают разрыв).
    Бледные кикеры вытягиваются растяжкой яркости (серый текст → тёмный), чтобы шаблоны
    рангов/мастей сработали штатным порогом. ``None`` — карт нет (все сфолдили: ник впритык к
    сумме) или строка не Win-картовая. ``strip`` — остаток строки ПОСЛЕ времени и слова.
    Используется и :func:`recognize_win_cards`, и дампом (:func:`debug.dump_rows`).
    """
    if strip.ndim != 3:  # нужен цвет: отделить красную сумму и бледные кикеры
        return None
    blue = strip[:, :, 0].astype(np.int16)
    green = strip[:, :, 1].astype(np.int16)
    red = strip[:, :, 2].astype(np.int16)
    red_cols = np.where((red - np.maximum(blue, green) > red_delta).any(axis=0))[0]
    if len(red_cols) == 0:
        return None  # нет суммы — не Win-строка с картами
    # Правый край СУММЫ = конец первого красного кластера (красные масти карт идут за разрывом).
    gaps = np.where(np.diff(red_cols) > _AMOUNT_CLUSTER_GAP)[0]
    amount_right = int(red_cols[gaps[0]]) if len(gaps) else int(red_cols[-1])

    # Левый край НИКА = логика find_nick_region (has_leading_cards): от правого края тёмной
    # маски влево, пока разрыв ≤ высоты строки; больший разрыв = начались карты левее ника.
    mask = dark_text_mask(strip, dark=dark_threshold, red_delta=red_delta)
    dark_cols = np.where(mask.any(axis=0))[0]
    if len(dark_cols) == 0:
        return None  # нет ника — нечего обрамлять
    nick_left = int(dark_cols[-1])
    max_gap = max(16, strip.shape[0])
    for col in reversed(dark_cols[:-1].tolist()):
        if nick_left - col <= max_gap:
            nick_left = col
        else:
            break

    pad = 4
    if nick_left - pad <= amount_right + pad:
        return None  # ник впритык к сумме — карт между ними нет (все сфолдили)
    zone = strip[:, amount_right + pad : nick_left - pad]
    if zone.shape[1] < 1:
        return None
    # Контраст-нормализация: бледные кикеры → тёмные, чтобы шаблоны рангов/мастей сработали.
    # Растяжка яркости С СОХРАНЕНИЕМ ЦВЕТА: тянем только L-канал (яркость) в LAB, цвет (a,b)
    # не трогаем — масти ♦♥ остаются красными, ♠♣ чёрными. Человеку в дампе видно естественно,
    # а распознаватель матчит ФОРМУ (grayscale внутри) — ему вытянутый контраст и так годится.
    # Инвертируем L (текст → яркий), тянем min→0/max→255, инвертируем назад (текст → тёмный).
    lab = cv2.cvtColor(zone, cv2.COLOR_BGR2LAB)
    light = 255 - lab[:, :, 0].astype(np.int16)  # текст яркий на тёмном фоне
    lo, hi = int(light.min()), int(light.max())
    if hi > lo:
        stretched = ((light - lo) * (255.0 / (hi - lo))).clip(0, 255).astype(np.uint8)
        lab[:, :, 0] = 255 - stretched
    return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)


def recognize_win_cards(
    strip: np.ndarray,
    rank_templates: dict[str, np.ndarray],
    suit_templates: dict[str, np.ndarray],
    *,
    red_delta: int = THRESHOLDS.amount_red_delta,
    dark_threshold: int = THRESHOLDS.nick_dark,
    rank_threshold: float = THRESHOLDS.rank,
    suit_threshold: float = THRESHOLDS.suit,
) -> list[str]:
    """Карты выигрышной КОМБИНАЦИИ на строке ``Win`` (между суммой и ником), или ``[]``.

    ``Win`` показывает 5-карточную руку победителя: собранные карты (нормального цвета, как
    на борде) + КИКЕРЫ в скобках (БЛЕДНО-серые, низкий контраст). Если все сфолдили, карты
    победителя НЕ вскрываются → ``[]``. Зону карт изолирует и вытягивает :func:`win_card_zone`,
    дальше ранги/масти ищутся по форме (grayscale). ``strip`` — остаток ПОСЛЕ времени и слова.
    """
    zone = win_card_zone(strip, red_delta=red_delta, dark_threshold=dark_threshold)
    if zone is None:
        return []
    return recognize_cards(
        zone,
        rank_templates,
        suit_templates,
        rank_threshold=rank_threshold,
        suit_threshold=suit_threshold,
    )
