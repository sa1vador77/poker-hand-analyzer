"""Чтение ника СТРОКОЙ через посимвольный matchTemplate (а не OCR).

Шрифт ника фиксированный, но словарь символов ОТКРЫТЫЙ (произвольные имена: латиница,
кириллица, цифры, знаки). Поэтому ник не матчится одним шаблоном целиком — он режется на
ГЛИФЫ по разрывам столбцов, и каждый глиф сравнивается с атласом символов того же шрифта.
Результат — детерминированная строка-ключ: один и тот же кроп всегда даёт одну строку.

Зачем строка вместо сравнения картинок (см. :mod:`poker_analyzer.identity`): корреляция
ников по порогу не может одновременно избежать СКЛЕЕК (разные игроки перепрыгнули порог) и
ДРОБЛЕНИЯ (тот же игрок на ином рендере упал под порог). Строковый ключ — это равенство, а
не «похожесть»: порога нет. Но это не серебряная пуля — два по-настоящему одинаковых ника
(напр. оба `.,.`) дадут одну строку, как и одну картинку; такую неоднозначность из пикселей
не убрать ничем, тут работает только структура раздачи.

Атлас на диске (``data/templates/nick``) — по файлу на символ, имя ``u<hex>.png`` (кодовая
точка), чтобы пережить регистронезависимую ФС macOS (``a.png`` и ``A.png`` иначе слиплись бы).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from poker_analyzer.config import THRESHOLDS
from poker_analyzer.vision.crop import dark_text_mask, to_gray
from poker_analyzer.vision.matching import match_template

# Паддинг глифа перед матчингом — шаблон скользит внутри добавленных полей (а не требует
# точного попадания). По ГОРИЗОНТАЛИ поглощает смещение ника, по ВЕРТИКАЛИ — суб-пиксельное
# дрожание строки (соседние строки режутся с разным сдвигом по Y, иначе ZNCC падает).
_MATCH_PAD = 2
_MATCH_PAD_Y = 2


@dataclass(frozen=True, slots=True)
class GlyphRead:
    """Один прочитанный глиф ника."""

    char: str  # распознанный символ; "" — ниже порога (неуверенно)
    score: float  # ZNCC лучшего совпадения по атласу
    x_left: int  # границы глифа в полосе (для отладки/восстановления)
    x_right: int


@dataclass(frozen=True, slots=True)
class NickRead:
    """Результат чтения ника: строка + по-глифовая уверенность."""

    text: str  # склеенная строка (с пробелами по крупным разрывам)
    glyphs: tuple[GlyphRead, ...]

    @property
    def min_score(self) -> float:
        """Худшая уверенность среди глифов (1.0, если глифов нет — пустой кроп)."""
        return min((g.score for g in self.glyphs), default=1.0)

    @property
    def mean_score(self) -> float:
        """Средняя уверенность по глифам (1.0, если глифов нет)."""
        if not self.glyphs:
            return 1.0
        return sum(g.score for g in self.glyphs) / len(self.glyphs)

    @property
    def n_unknown(self) -> int:
        """Сколько глифов не дотянули до порога (символ ``""``)."""
        return sum(1 for g in self.glyphs if not g.char)

    def confident(self, *, floor: float = THRESHOLDS.glyph_match) -> bool:
        """Чтение надёжно: есть хоть один глиф и ни один не ниже ``floor``.

        Низкая уверенность → один промах глифа создаст новый ключ (дробление), поэтому
        identity должен в этом случае откатиться на сравнение картинкой, а не доверять строке.
        """
        return bool(self.glyphs) and self.n_unknown == 0 and self.min_score >= floor


def segment_glyphs(
    strip: np.ndarray,
    *,
    dark: int = THRESHOLDS.nick_dark,
    min_gap: int = 2,
    min_width: int = 1,
) -> list[tuple[int, int]]:
    """Режет полосу ника на глифы по разрывам тёмных столбцов.

    Берём маску тёмного текста (:func:`crop.dark_text_mask`), сворачиваем по столбцам и
    разбиваем «занятые» столбцы на кластеры: разрыв ≥ ``min_gap`` пустых столбцов — граница
    между глифами (внутри буквы разрывов столбцов в этом шрифте нет). Шрифт фиксированный,
    буквы не слипаются — один кластер ≈ один символ.

    :returns: список ``(x_left, x_right)`` глифов слева направо (полу-открытый интервал).
    """
    mask = dark_text_mask(strip, dark=dark)
    col_has = mask.any(axis=0)
    cols = np.where(col_has)[0]
    if len(cols) == 0:
        return []

    spans: list[tuple[int, int]] = []
    start = prev = int(cols[0])
    for col in cols[1:].tolist():
        if col - prev > min_gap:  # крупный разрыв — закрываем глиф, открываем следующий
            if prev - start + 1 >= min_width:
                spans.append((start, prev + 1))
            start = col
        prev = col
    if prev - start + 1 >= min_width:
        spans.append((start, prev + 1))
    return spans


def _match_glyph(
    glyph_gray: np.ndarray, atlas: dict[str, np.ndarray], *, floor: float
) -> tuple[str, float]:
    """Сравнивает один глиф со всеми шаблонами атласа, возвращает ``(символ, score)``.

    Высоту шаблона подгоняем к высоте глифа (устойчивость к разнице масштаба захвата),
    глиф паддим по горизонтали для поглощения смещения. ZNCC устойчив к яркости фона.

    Выбор — по score, ВЗВЕШЕННОМУ на близость ширины (``score × min/max ширин``): узкий шаблон
    (``r``, ``i``) скользит внутри широкой буквы и даёт ложно высокий ZNCC на её куске — без
    штрафа за ширину ``л`` читается как ``r``, ``в`` как ``i``. Порог ``floor`` сверяем с сырым
    score выбранного символа; ниже — символ неуверенный (``""``), но score отдаём для отладки.
    """
    target_h = glyph_gray.shape[0]
    glyph_w = glyph_gray.shape[1]
    padded = cv2.copyMakeBorder(
        glyph_gray, _MATCH_PAD_Y, _MATCH_PAD_Y, _MATCH_PAD, _MATCH_PAD, cv2.BORDER_REPLICATE
    )
    best_char, best_score, best_weighted = "", -1.0, -1.0
    for char, template in atlas.items():
        tmpl = template
        if tmpl.shape[0] != target_h:  # привести высоту шаблона к высоте глифа
            scale = target_h / tmpl.shape[0]
            new_w = max(1, round(tmpl.shape[1] * scale))
            tmpl = cv2.resize(tmpl, (new_w, target_h), interpolation=cv2.INTER_AREA)
        if tmpl.shape[0] > padded.shape[0] or tmpl.shape[1] > padded.shape[1]:
            continue  # шаблон шире глифа+паддинг — это не он
        score = match_template(padded, tmpl).score
        width_ratio = min(tmpl.shape[1], glyph_w) / max(tmpl.shape[1], glyph_w)
        weighted = score * width_ratio
        if weighted > best_weighted:
            best_char, best_score, best_weighted = char, score, weighted
    if best_score < floor:
        return "", best_score
    return best_char, best_score


def read_nick(
    strip: np.ndarray,
    atlas: dict[str, np.ndarray],
    *,
    floor: float = THRESHOLDS.glyph_match,
    space_gap: int | None = None,
    dark: int = THRESHOLDS.nick_dark,
) -> NickRead:
    """Читает ник строкой: сегментация → матч каждого глифа по атласу → склейка.

    Между глифами с разрывом больше ``space_gap`` вставляется пробел (слова ника, напр.
    ``X Æ A-12``). По умолчанию ``space_gap`` ≈ 0.35 высоты строки — крупнее межбуквенного
    (разрыв внутри ``12`` ≈ 0.27h), но мельче межсловного (≈ 0.43h на живом кропе).

    :param atlas: символ → шаблон глифа (grayscale), см. :func:`load_glyph_atlas`. Пустой
        атлас → строка пустая, все глифы неуверенные (это нормально, пока атлас не собран).
    """
    spans = segment_glyphs(strip, dark=dark)
    if space_gap is None:
        space_gap = max(4, round(strip.shape[0] * 0.35))

    gray = to_gray(strip)
    reads: list[GlyphRead] = []
    parts: list[str] = []
    prev_right: int | None = None
    for x_left, x_right in spans:
        if prev_right is not None and x_left - prev_right > space_gap:
            parts.append(" ")
        char, score = _match_glyph(gray[:, x_left:x_right], atlas, floor=floor)
        reads.append(GlyphRead(char=char, score=score, x_left=x_left, x_right=x_right))
        parts.append(char if char else "�")  # � — нечитаемый глиф (виден в логе)
        prev_right = x_right

    return NickRead(text="".join(parts), glyphs=tuple(reads))


# Гомоглифы: кириллические буквы, ВИЗУАЛЬНО неотличимые от латинских в этом шрифте.
# Матч путает их между строками (``Max`` ↔ ``Mаx``) → один игрок даёт разные строки-ключи.
# Сворачиваем к латинскому представителю: это БЕЗ потерь различения (двух игроков с пиксельно
# одинаковыми никами всё равно не отличить) и делает ключ стабильным. Только заведомо
# идентичные пары — спорные (``к/k``, ``з/3``) НЕ включаем, чтобы не слить разные буквы.
_CONFUSABLES = {
    # кириллица → латиница (строчные)
    "а": "a", "е": "e", "о": "o", "р": "p", "с": "c", "х": "x", "у": "y",
    "і": "i", "ј": "j", "ѕ": "s",
    # кириллица → латиница (прописные)
    "А": "A", "В": "B", "Е": "E", "К": "K", "М": "M", "Н": "H", "О": "O",
    "Р": "P", "С": "C", "Т": "T", "У": "Y", "Х": "X", "І": "I", "Ј": "J", "Ѕ": "S",
    # латиница с гачеком/акутом → база: гачек мелкий и сверху, матч флапает S↔Š (как гомоглиф)
    "Š": "S", "š": "s", "Ž": "Z", "ž": "z", "Č": "C", "č": "c", "Ć": "C", "ć": "c",
}  # fmt: skip


def canonical_key(text: str) -> str:
    """Сворачивает гомоглифы (кирилл↔латиница) к канону — стабильный ключ из строки чтения.

    Для сравнения игроков нужен ключ, одинаковый на всех строках одного ника. Сырое чтение
    флапает гомоглифами (``Max``/``Mаx``); канон убирает этот флап без потери различения.
    Краевой ``�``/пробел срезаем: на краю кропа сегментация ловит лишний глиф-шум (особенно
    в прокрутке до снимка), и хвост ``" �"`` иначе мешал бы ключу сматчиться с чистым.
    Хвостовое «слово» сплошь из ``�`` тоже срезаем (``"Бишкек ���� ��-"`` на грязном кадре →
    ``"Бишкек"``): краевой ``strip`` его не берёт, если оно кончается читаемым символом (``-``),
    а реальное второе слово ника (``Гулу Сулейманов``) не трогаем — режем только ≥50% ``�``.
    """
    folded = "".join(_CONFUSABLES.get(ch, ch) for ch in text)
    words = folded.split(" ")
    while len(words) > 1 and _mostly_unreadable(words[-1]):
        words.pop()
    return " ".join(words).strip("� ")


def _mostly_unreadable(word: str) -> bool:
    """Слово хотя бы наполовину из ``�`` — мусор сегментации, не реальное слово ника."""
    return bool(word) and word.count(_UNREADABLE) * 2 >= len(word)


_UNREADABLE = "�"  # маркер нечитаемого глифа (см. read_nick)


def _wildcard_edit(a: str, b: str) -> int:
    """Расстояние Левенштейна, где ``�`` — джокер (совпадает с любым символом бесплатно).

    Нечитаемый глиф на одной строке не должен считаться отличием от верного на другой.
    """
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            sub = 0 if (ca in (cb, _UNREADABLE) or cb == _UNREADABLE) else 1
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + sub))
        prev = cur
    return prev[-1]


def key_relation(
    a: str | None,
    b: str | None,
    *,
    min_solid: int = 3,
    same_ratio: float = 0.8,
    diff_ratio: float = 0.5,
) -> str:
    """Отношение двух канон-ключей ников: ``"same"`` | ``"different"`` | ``"unknown"``.

    Для identity строка — ПЕРВИЧНЫЙ признак, но не единственный: ``"unknown"`` означает
    «по строкам не решить, спроси корреляцию». Так редкий флап (``v↔y``) и дырки (``�``)
    не дробят игрока, а явно разные ники (``CCCEDlTS`` vs ``Hикитa``) не сливаются.

    Логика: если у любой строки солидных (не ``�``, не пробел) символов меньше ``min_solid``
    — судить не по чему → ``"unknown"``. Иначе нечёткое расстояние (``�`` — джокер): близко
    (``ratio ≥ same_ratio``) → ``"same"``, далеко (``≤ diff_ratio``) → ``"different"``, между
    — ``"unknown"`` (пусть решает корреляция).
    """
    if not a or not b:
        return "unknown"
    solid_a = sum(1 for c in a if c not in (_UNREADABLE, " "))
    solid_b = sum(1 for c in b if c not in (_UNREADABLE, " "))
    if solid_a < min_solid or solid_b < min_solid:
        return "unknown"
    ratio = 1.0 - _wildcard_edit(a, b) / max(len(a), len(b))
    if ratio >= same_ratio:
        return "same"
    if ratio <= diff_ratio:
        return "different"
    return "unknown"


# --- Атлас глифов на диске ----------------------------------------------------


def _char_to_filename(char: str) -> str:
    """Имя файла шаблона символа: ``u<hex кодовой точки>`` (ФС-безопасно, регистрозависимо)."""
    return f"u{ord(char):04x}"


def _filename_to_char(stem: str) -> str | None:
    """Обратно из имени файла в символ; ``None``, если имя не в формате ``u<hex>``."""
    if not stem.startswith("u"):
        return None
    try:
        return chr(int(stem[1:], 16))
    except ValueError:
        return None


def load_glyph_atlas(directory: str | Path) -> dict[str, np.ndarray]:
    """Загружает атлас глифов из папки (``u<hex>.png`` → символ, grayscale).

    Опционален: пустая/несуществующая папка → ``{}`` (пока атлас не собран — читать нечем).
    """
    path = Path(directory)
    if not path.is_dir():
        return {}
    atlas: dict[str, np.ndarray] = {}
    for file in sorted(path.glob("*.png")):
        char = _filename_to_char(file.stem)
        if char is None:
            continue
        image = cv2.imread(str(file), cv2.IMREAD_GRAYSCALE)
        if image is not None:
            atlas[char] = image
    return atlas


def label_glyphs(
    strip: np.ndarray, label: str, *, dark: int = THRESHOLDS.nick_dark
) -> list[tuple[str, np.ndarray]]:
    """Режет полосу и сопоставляет глифы заданным символам — для СБОРКИ атласа.

    ``label`` — известная строка ника БЕЗ пробелов (пробелы в нике не дают глифа). Если
    число сегментов не совпало с числом символов — ``ValueError`` (сегментация разошлась с
    разметкой, шаблоны брать нельзя).

    :returns: список ``(символ, grayscale-кроп глифа)`` в порядке слева направо.
    """
    label = label.replace(" ", "")
    spans = segment_glyphs(strip, dark=dark)
    if len(spans) != len(label):
        raise ValueError(
            f"Сегментов {len(spans)}, символов в метке {len(label)} ({label!r}) — не совпало"
        )
    gray = to_gray(strip)
    return [
        (char, gray[:, x_left:x_right])
        for char, (x_left, x_right) in zip(label, spans, strict=True)
    ]


def save_glyph_atlas(
    directory: str | Path, entries: dict[str, np.ndarray], *, overwrite: bool = False
) -> int:
    """Сохраняет шаблоны глифов в папку атласа (``символ → u<hex>.png``).

    Не перезаписывает уже существующий символ, если ``overwrite=False`` (первый эталон
    обычно чище). :returns: сколько файлов реально записано.
    """
    path = Path(directory)
    path.mkdir(parents=True, exist_ok=True)
    written = 0
    for char, image in entries.items():
        dest = path / f"{_char_to_filename(char)}.png"
        if dest.exists() and not overwrite:
            continue
        cv2.imwrite(str(dest), image)
        written += 1
    return written
