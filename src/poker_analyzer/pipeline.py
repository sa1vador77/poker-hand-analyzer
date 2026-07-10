"""Пайплайн обработки одного кадра лога: скриншот → результат по каждой строке.

Этапы накапливаются в :class:`RowResult`. Конвейер по строке режет её слева
направо: время → ключевое слово → (для hand/table) карты. Каждый этап отдаёт
строку без своего элемента.

Координаты нарезки и пороги берутся из :mod:`poker_analyzer.config` — правятся там.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from poker_analyzer.config import LAYOUT, THRESHOLDS, Layout, Thresholds
from poker_analyzer.identity.players import PlayerRegistry, SessionPlayers
from poker_analyzer.parsing.segmentation import split_rows
from poker_analyzer.vision.glyphs import canonical_key, read_nick
from poker_analyzer.vision.recognition import (
    find_nick_region,
    recognize_amount,
    recognize_cards,
    recognize_keyword,
    recognize_time,
    recognize_win_cards,
)
from poker_analyzer.vision.templates import Templates

# Ключевые слова, после которых в строке есть карты (рука героя / борд).
CARD_KEYWORDS = frozenset({"hand", "table"})
# Ключевые слова, у которых в строке есть сумма (ставка/банк/выигрыш).
AMOUNT_KEYWORDS = frozenset({"deal", "bet", "call", "blind", "allin", "raise", "win"})
# Ключевые слова, у которых в строке есть ник игрока (различаем игроков картинкой).
NICK_KEYWORDS = frozenset(
    {"dealer", "blind", "call", "check", "bet", "raise", "fold", "allin", "win"}
)


def is_valid_time(time: str) -> bool:
    """Время сложилось в ``HH:MM:SS`` (распознались все 6 цифр).

    Это признак, что строка НЕ поймана в момент прокрутки лога: на нестабильном кадре
    строка режется поперёк двух соседних и время читается криво (например ``02348``).
    """
    return (
        len(time) == 8
        and time[2] == ":"
        and time[5] == ":"
        and time[:2].isdigit()
        and time[3:5].isdigit()
        and time[6:8].isdigit()
    )


# Порог «тёмного» пикселя текста в колонке времени (фон лога белый, цифры тёмно-серые).
_GRID_DARK = 120
# Нижняя граница поиска сдвига сетки (вверх — лёгкое дрожание; вниз — до fill_shift+9).
_GRID_SHIFT_MIN = -4


def detect_grid_shift(frame: np.ndarray, *, layout: Layout = LAYOUT) -> int:
    """Вертикальный сдвиг сетки строк на кадре: 0 — обычный лог, ~``layout.fill_shift``
    — лог-«наполнение».

    После входа в комнату лог растёт СВЕРХУ и, пока строки не коснулись низа, стоит
    ниже обычной сетки — без сдвига цифры времени режутся, и строки такого скрина не
    читаются. Сдвиг меряется
    ПО ФАЗЕ текста, без распознавания: в колонке времени строится профиль тёмных
    пикселей по y и ищется сдвиг, при котором текст максимально попадает ВНУТРЬ
    полос нарезки; из плато равно-хороших сдвигов берётся СЕРЕДИНА — текст по центру
    полосы, обе кромки с запасом. (Жёсткая проба двух кандидатов 0/15 вставала на
    КРАЙ плато: кромка полосы резала цифру, время читалось как «04947», и чтение
    замирало посреди лога.) Пустая колонка времени (нет лога) → 0.
    """
    shifts = range(_GRID_SHIFT_MIN, layout.fill_shift + 10)
    y0 = layout.y1 + shifts.start
    y1 = min(layout.y2 + shifts[-1], frame.shape[0])
    if y1 <= y0 or frame.ndim != 3:
        return 0
    column = frame[y0:y1, layout.x1 : layout.x1 + layout.time_width]
    gray = cv2.cvtColor(column, cv2.COLOR_BGR2GRAY)
    profile = (gray < _GRID_DARK).sum(axis=1).astype(np.int64)
    if not profile.any():
        return 0
    sums = np.concatenate(([0], np.cumsum(profile)))  # сумма профиля на [a, b) = sums[b]-sums[a]

    def covered(shift: int) -> int:
        """Сколько тёмных пикселей колонки попадает внутрь полос нарезки при сдвиге."""
        inside = 0
        top = layout.y1 + shift
        bottom = min(layout.y2 + shift, frame.shape[0])
        while top + layout.row_height <= bottom:
            a = max(0, top - y0)
            b = min(len(profile), top - y0 + layout.row_height)
            if b > a:
                inside += int(sums[b] - sums[a])
            top += layout.row_pitch
        return inside

    scores = [(covered(s), s) for s in shifts]
    best = max(score for score, _ in scores)
    plateau = [s for score, s in scores if score == best]
    return plateau[len(plateau) // 2]


@dataclass(slots=True)
class RowResult:
    """Накопительный результат разбора одной строки лога.

    Поля заполняются поэтапно по мере прохождения пайплайна; не пройденные этапы
    остаются пустыми (``None`` / ``()``).
    """

    index: int  # порядковый номер строки в кадре (сверху вниз)
    time: str  # этап 1: распознанное время 'HH:MM:SS'
    keyword: str | None = None  # этап 2: ключевое слово (call/bet/raise/...)
    cards: tuple[str, ...] = ()  # этап 3: карты строки (hand/table), напр. ('7♦', '4♠')
    amount: float | None = None  # этап: сумма (ставка/банк); float — суммы бывают дробными
    player_id: int | None = None  # этап: игрок по картинке ника (в пределах раздачи)
    is_hero: bool = False  # ник строки совпал с шаблоном героя
    session_id: int | None = None  # постоянный id игрока за сессию (SessionPlayers)
    nick_text: str | None = (
        None  # этап 2a: ник, прочитанный строкой (canonical_key); для лога/проверки
    )
    nick_confident: bool = False  # чтение ника уверенное (все глифы выше порога)


def process_row(
    row: np.ndarray,
    templates: Templates,
    *,
    index: int = 0,
    layout: Layout = LAYOUT,
    thresholds: Thresholds = THRESHOLDS,
    registry: PlayerRegistry | None = None,
) -> RowResult:
    """Распознаёт одну уже вырезанную строку лога (конвейер по строке).

    Этапы режут строку слева направо: время → ключевое слово → карты
    (:data:`CARD_KEYWORDS`) либо сумма (:data:`AMOUNT_KEYWORDS`); для строк с ником
    (:data:`NICK_KEYWORDS`) игрок различается картинкой через ``registry``. Вынесено
    отдельно, чтобы можно было распознавать и одиночные, уже вырезанные строки.

    :param row: вырезанная строка лога (BGR), как её отдаёт ``split_rows``.
    :param index: номер строки в кадре (для :class:`RowResult`).
    :param registry: реестр игроков раздачи; без него ник не определяется (``None``).
    """
    time, row = recognize_time(
        row, templates.digits, time_width=layout.time_width, threshold=thresholds.time_digit
    )
    keyword, row = recognize_keyword(row, templates.keywords, threshold=thresholds.keyword)

    cards: tuple[str, ...] = ()
    amount: float | None = None
    if keyword in CARD_KEYWORDS:
        cards = tuple(
            recognize_cards(
                row,
                templates.ranks,
                templates.suits,
                rank_threshold=thresholds.rank,
                suit_threshold=thresholds.suit,
            )
        )
    elif keyword in AMOUNT_KEYWORDS:
        amount = recognize_amount(
            row,
            templates.digits,
            threshold=thresholds.amount,
            red_delta=thresholds.amount_red_delta,
        )
        if keyword == "win":
            # Win несёт и карты выигрышной комбинации (между суммой и ником); кикеры бледные.
            # Все сфолдили → карт нет → (). Карманку (комбинация − борд) извлечёт слой state.
            cards = tuple(
                recognize_win_cards(
                    row,
                    templates.ranks,
                    templates.suits,
                    red_delta=thresholds.amount_red_delta,
                    dark_threshold=thresholds.nick_dark,
                    rank_threshold=thresholds.rank,
                    suit_threshold=thresholds.suit,
                )
            )

    player_id: int | None = None
    is_hero = False
    nick_text: str | None = None
    nick_confident = False
    if registry is not None and keyword in NICK_KEYWORDS and is_valid_time(time):
        registry.on_keyword(keyword)  # Win → конец раздачи; Dealer после Win → сброс реестра
        nick = find_nick_region(
            row,
            red_delta=thresholds.amount_red_delta,
            dark_threshold=thresholds.nick_dark,
            has_leading_cards=keyword == "win",  # только Win несёт карты слева от ника
        )
        if nick is not None:
            # Шаг 2b: читаем ник строкой ОДИН раз и отдаём ключ в identify — он первичный
            # признак (корреляция тай-брейкером). Без атласа key=None → чистая корреляция.
            if templates.glyphs:
                read = read_nick(nick, templates.glyphs)
                nick_text = canonical_key(read.text)
                nick_confident = read.confident()
            # Win не регистрирует новых игроков: его кроп соседствует с картами комбинации
            # и менее надёжен; победителя-героя ловит matches_hero (Win — последняя строка).
            player_id = registry.identify(nick, key=nick_text, register=keyword != "win")
            is_hero = registry.matches_hero(nick)  # ник совпал с шаблоном героя

    return RowResult(
        index=index,
        time=time,
        keyword=keyword,
        cards=cards,
        amount=amount,
        player_id=player_id,
        is_hero=is_hero,
        session_id=registry.session_of(player_id) if registry is not None else None,
        nick_text=nick_text,
        nick_confident=nick_confident,
    )


def process_screen(
    image: np.ndarray,
    templates: Templates,
    *,
    layout: Layout = LAYOUT,
    thresholds: Thresholds = THRESHOLDS,
) -> list[RowResult]:
    """Прогоняет один кадр через пайплайн и возвращает результат по каждой строке.

    Нарезает кадр на строки и распознаёт каждую через :func:`process_row`. Сетка
    строк сама подстраивается под лог-«наполнение» (:func:`detect_grid_shift`).
    Реестр игроков один на кадр (сбрасывается внутри на каждом ``Dealer``).

    :param image: скриншот окна с логом (BGR).
    :param templates: все наборы шаблонов (см. ``load_all_templates``).
    :param layout: координаты нарезки (по умолчанию из config).
    :param thresholds: пороги распознавания (по умолчанию из config).
    """
    # На статике сессионный реестр живёт в пределах кадра: id игроков стабильны и между
    # раздачами одного скрина (полезно при --dump-разборе).
    registry = PlayerRegistry(
        hero_templates=list(templates.hero.values()), session=SessionPlayers()
    )
    return [
        process_row(
            row, templates, index=index, layout=layout, thresholds=thresholds, registry=registry
        )
        for index, row in enumerate(
            split_rows(image, layout, y_shift=detect_grid_shift(image, layout=layout)), start=1
        )
    ]
