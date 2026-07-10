"""Идентификация игроков по картинке ника — без чтения текста (без OCR).

Ник как текст проекту не нужен: нужно лишь устойчиво ОТЛИЧАТЬ игроков друг от
друга. Внутри одной раздачи игроков немного (2–9), поэтому каждый новый кроп ника
сравнивается корреляцией (matchTemplate) с уже виденными:

- совпало с кем-то из реестра → это тот же игрок (его ``player_id``);
- не совпало ни с кем → новый игрок, заводим новый ``player_id``.

Так снимается старая проблема «один и тот же ник = несколько игроков»: сравниваются
сами картинки, а не их нестабильное OCR-прочтение.

Отдельно: ник героя постоянен между раздачами, поэтому герой опознаётся сверкой ника
с его шаблоном (:meth:`PlayerRegistry.matches_hero`) — это ловит героя на ЛЮБОЙ его
строке (включая блайнд), не дожидаясь «Ваш ход».
"""

from __future__ import annotations

from collections.abc import Sequence

import cv2
import numpy as np

from poker_analyzer.config import THRESHOLDS
from poker_analyzer.vision.crop import to_gray
from poker_analyzer.vision.glyphs import key_relation


def _nick_correlation(
    a: np.ndarray, b: np.ndarray, *, min_width_ratio: float | None = None
) -> float:
    """Сдвигоустойчивая корреляция двух кропов ников через ``matchTemplate``.

    Узкий кроп скользит по широкому (matchTemplate требует шаблон ≤ изображения), плюс
    изображение расширяется по краям для запаса сдвига. Это гасит дрожание границ
    нарезки и небольшую разницу ширины одного ника между кадрами — чего не умеет
    сравнение снимков фиксированного размера.

    ``min_width_ratio`` (если задан) — анти-слияние: при сильно разной ширине ников
    (``min/max < min_width_ratio``) сразу возвращаем ``0.0``. Скользящий max иначе находит
    высокую корреляцию узкого кропа в каком-то окне широкого и склеивает РАЗНЫХ игроков.
    """
    if min_width_ratio is not None:
        wa, wb = a.shape[1], b.shape[1]
        if min(wa, wb) / max(wa, wb) < min_width_ratio:
            return 0.0  # ники сильно разной ширины — заведомо разные игроки
    template, image = (a, b) if a.shape[1] <= b.shape[1] else (b, a)
    image = cv2.copyMakeBorder(image, 0, 0, 4, 4, cv2.BORDER_REPLICATE)  # запас под сдвиг
    if template.shape[0] != image.shape[0]:  # высота строк одинакова, но подстрахуемся
        image = cv2.resize(image, (image.shape[1], template.shape[0]))
    return float(cv2.matchTemplate(image, template, cv2.TM_CCOEFF_NORMED).max())


_EXEMPLAR_ADD = 0.9  # совпадение НИЖЕ — кроп достаточно «другой», чтобы стать доп. эталоном


class SessionPlayers:
    """Сессионный реестр игроков: ПОСТОЯННЫЕ id на всё время работы программы.

    Второй слой над :class:`PlayerRegistry`: тот различает игроков ВНУТРИ раздачи
    (и сбрасывается на новой), а этот узнаёт ник МЕЖДУ раздачами и выдаёт стабильный
    ``session_id`` — основу для профилей игроков (статистика, показанные руки).

    Ошибки двух родов несимметричны: ложное РАЗДЕЛЕНИЕ (один игрок получил два id)
    лишь дробит его историю, ложное СЛИЯНИЕ (два игрока под одним id) отравляет
    профиль и делает советы хуже, чем вовсе без профилей. Поэтому правила
    консервативны:

    - порог совпадения СТРОЖЕ внутрираздачного (``nick_match_session``);
    - «при сомнении — новый игрок»: если второй кандидат тоже выше порога и отстаёт
      от лучшего меньше чем на ``margin`` — не сливаем ни с кем, заводим новый id;
    - на игрока хранится несколько эталонов ника (рендер чуть гуляет между кадрами);
    - размер реестра ограничен: давно не виденные вытесняются (LRU).
    """

    def __init__(
        self,
        *,
        match_threshold: float = THRESHOLDS.nick_match_session,
        margin: float = THRESHOLDS.nick_session_margin,
        min_width_ratio: float = THRESHOLDS.nick_min_width_ratio,
        max_players: int = 64,
        max_exemplars: int = 3,
    ) -> None:
        self._threshold = match_threshold
        self._margin = margin  # отрыв лучшего от второго, ниже которого «сомнение»
        self._min_width_ratio = min_width_ratio  # анти-слияние по ширине ника
        self._max_players = max_players
        self._max_exemplars = max_exemplars
        self._exemplars: dict[int, list[np.ndarray]] = {}  # session_id → эталоны (grayscale)
        self._keys: dict[int, str | None] = {}  # session_id → прочитанный ключ ника (строка, 2b)
        self._last_seen: dict[int, int] = {}  # session_id → такт последнего появления (LRU)
        self._next_id = 0
        self._tick = 0  # монотонный счётчик обращений

    def __len__(self) -> int:
        return len(self._exemplars)

    def identify(self, gray: np.ndarray, *, key: str | None = None) -> int:
        """Стабильный ``session_id`` по кропу ника (+ прочитанный ``key``); при сомнении — новый.

        Шаг 2b: строка-ключ — ПЕРВИЧНЫЙ признак. Явное совпадение ключа с уже виденным →
        тот же игрок (узнаём ник между раздачами даже когда корреляция кропа просела). Иначе
        корреляция эталонов (как раньше), но строка ВЕТИРУЕТ слияние, если ключи явно разные.
        Нет ни того ни другого / сомнение → новый игрок (session любит «при сомнении — новый»).
        """
        self._tick += 1
        if key is not None:  # 1) строка-первичный: явное совпадение ключа = тот же игрок
            for sid, k in self._keys.items():
                if key_relation(key, k) == "same":
                    self._last_seen[sid] = self._tick
                    if k is None:
                        self._keys[sid] = key  # доезжаем ключ, если игрок завёлся без чтения
                    return sid
        best_id, best, second = -1, -1.0, -1.0
        for sid, refs in self._exemplars.items():
            score = max(
                _nick_correlation(gray, ref, min_width_ratio=self._min_width_ratio) for ref in refs
            )
            if score > best:
                best_id, second, best = sid, best, score
            elif score > second:
                second = score
        ambiguous = second >= self._threshold and best - second < self._margin
        if best_id >= 0 and best >= self._threshold and not ambiguous:
            if key is not None and key_relation(key, self._keys.get(best_id)) == "different":
                return self._register(gray, key)  # 2) вето: ключи явно разные — не сливаем
            self._last_seen[best_id] = self._tick
            refs = self._exemplars[best_id]
            # Заметно «другой» рендер того же ника копим как доп. эталон — анти-дробление.
            if best < _EXEMPLAR_ADD and len(refs) < self._max_exemplars:
                refs.append(gray)
            if key is not None and self._keys.get(best_id) is None:
                self._keys[best_id] = key  # доезжаем ключ при опознании по корреляции
            return best_id
        return self._register(gray, key)

    def _register(self, gray: np.ndarray, key: str | None = None) -> int:
        """Заводит нового игрока; при переполнении вытесняет давно не виденного (LRU)."""
        if len(self._exemplars) >= self._max_players:
            oldest = min(self._last_seen, key=self._last_seen.__getitem__)
            del self._exemplars[oldest]
            del self._last_seen[oldest]
            self._keys.pop(oldest, None)
        sid = self._next_id
        self._next_id += 1
        self._exemplars[sid] = [gray]
        self._keys[sid] = key
        self._last_seen[sid] = self._tick
        return sid


class PlayerRegistry:
    """Реестр игроков текущей раздачи: различает их по КАРТИНКЕ ника (без OCR).

    Каждый новый кроп ника сравнивается со всеми виденными сдвигоустойчивой
    корреляцией (matchTemplate): совпал выше порога — тот же ``player_id``, нет —
    регистрируется новый. ``player_id`` — индекс в порядке появления.

    Жизненный цикл раздачи: она кончается на ``Win``; новую начинает следующий
    ``Dealer`` — тогда реестр сбрасывается (см. :meth:`on_keyword`). ``Dealer`` БЕЗ
    предшествующего ``Win`` — это смена дилера внутри той же раздачи (игрок вышел),
    реестр не трогаем.

    Если задан сессионный реестр (``session``), каждому игроку раздачи при регистрации
    сопоставляется ещё и постоянный ``session_id`` (см. :meth:`session_of`) — по ПЕРВОМУ
    кропу его ника в раздаче (одна сверка на игрока за раздачу, маппинг внутри раздачи
    стабилен).
    """

    def __init__(
        self,
        *,
        match_threshold: float = THRESHOLDS.nick_match,
        min_width_ratio: float = THRESHOLDS.nick_min_width_ratio,
        hero_templates: Sequence[np.ndarray] | None = None,
        hero_threshold: float = THRESHOLDS.hero_match,
        session: SessionPlayers | None = None,
    ) -> None:
        self._threshold = match_threshold
        self._min_width_ratio = min_width_ratio  # анти-слияние по ширине ника
        self._nicks: list[np.ndarray] = []  # grayscale кропы ников; индекс = player_id
        self._keys: list[str | None] = []  # прочитанный ключ ника; индекс = player_id (2b)
        self._hand_ended = False  # был ли Win с начала текущей раздачи
        self._hero = [to_gray(t) for t in (hero_templates or ())]  # grayscale шаблоны ника героя
        self._hero_threshold = hero_threshold
        self._session = session  # сессионный реестр (живёт МЕЖДУ раздачами), опционально
        self._session_ids: dict[int, int] = {}  # player_id раздачи → session_id

    def on_keyword(self, keyword: str | None) -> None:
        """Двигает границы раздачи по слову строки: ``Win`` — конец, ``Dealer`` — старт.

        Сброс реестра — только на ``Dealer`` ПОСЛЕ ``Win`` (новая раздача). ``Dealer``
        без предшествующего ``Win`` — смена дилера в той же раздаче, реестр не трогаем.
        """
        if keyword == "win":
            self._hand_ended = True
        elif keyword == "dealer" and self._hand_ended:
            self.reset()

    def identify(
        self, nick_crop: np.ndarray, *, key: str | None = None, register: bool = True
    ) -> int | None:
        """Возвращает ``player_id`` игрока по картинке ника (+ прочитанный строкой ``key``).

        Шаг 2b: строка-ключ — ПЕРВИЧНЫЙ признак. Явное совпадение ключа с уже виденным →
        тот же игрок (ловит сплит корреляции: один ник под двумя id из-за дрожания кропа).
        Иначе скользящая корреляция (как раньше), НО строка ВЕТИРУЕТ слияние, если ключи
        явно разные — так корреляция не склеит двух разных игроков (склейка `CCCEDlTS`+`Hикитa`).

        ``register=False`` для строки ``Win``: её кроп ника соседствует с картами комбинации
        и менее надёжен, а победитель-герой и так ловится :meth:`matches_hero`. Так строка
        ``Win`` не плодит фантомных игроков (``Win`` — последняя строка раздачи).
        """
        gray = to_gray(nick_crop)
        if key is not None:  # 1) строка-первичный: явное совпадение ключа = тот же игрок
            for player_id, stored in enumerate(self._keys):
                if key_relation(key, stored) == "same":
                    if stored is None:
                        self._keys[player_id] = key
                    return player_id
        best_id, best_score = -1, -1.0
        for player_id, reference in enumerate(self._nicks):
            score = _nick_correlation(gray, reference, min_width_ratio=self._min_width_ratio)
            if score > best_score:
                best_id, best_score = player_id, score
        # Корреляция нашла кандидата — принимаем, ЕСЛИ строка не говорит «это разные игроки».
        vetoed = (
            best_id >= 0
            and key is not None
            and key_relation(key, self._keys[best_id]) == "different"
        )
        if best_id >= 0 and best_score >= self._threshold and not vetoed:
            if key is not None and self._keys[best_id] is None:
                self._keys[best_id] = key  # доезжаем ключ при опознании по корреляции
            return best_id
        if not register:
            return None
        self._nicks.append(gray)
        self._keys.append(key)
        player_id = len(self._nicks) - 1
        if self._session is not None:
            # Сессионный id — по кропу РЕГИСТРАЦИИ (первое появление в раздаче): одна
            # сверка на игрока за раздачу, маппинг внутри раздачи не дрожит.
            self._session_ids[player_id] = self._session.identify(gray, key=key)
        return player_id

    def session_of(self, player_id: int | None) -> int | None:
        """Постоянный (сессионный) id игрока раздачи, или ``None``.

        ``None`` — без сессионного реестра, для ``player_id is None`` и для игроков,
        опознанных без регистрации (строка ``Win``).
        """
        if player_id is None:
            return None
        return self._session_ids.get(player_id)

    def matches_hero(self, nick_crop: np.ndarray) -> bool:
        """Совпадает ли ник с шаблоном героя — надёжное опознание героя (даже на блайнде).

        Без шаблона (пусто) всегда ``False`` — тогда героя ловит запасной признак
        «строка после Ваш ход» (см. :mod:`poker_analyzer.engine.state`).
        """
        if not self._hero:
            return False
        gray = to_gray(nick_crop)
        return any(_nick_correlation(gray, t) >= self._hero_threshold for t in self._hero)

    def reset(self) -> None:
        """Очищает реестр и метку конца раздачи (старт новой раздачи).

        Сессионный реестр НЕ трогается — он и есть память между раздачами.
        """
        self._nicks.clear()
        self._keys.clear()
        self._session_ids.clear()
        self._hand_ended = False
