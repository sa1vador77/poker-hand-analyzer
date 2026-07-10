"""Состояние раздачи: накопление потока :class:`LogEvent` в queryable-состояние.

:class:`HandState` применяет события (см. :mod:`poker_analyzer.parsing.events`) и
держит то, что нужно слою советов: рука героя, борд, число живых оппонентов (→ эквити),
банк и «сколько доколлить» (→ pot odds). Раздача кончается на ``Win``; следующий
``Dealer`` начинает новую (сброс) — как в :class:`~poker_analyzer.identity.players.PlayerRegistry`.

Допущения по формату лога (подтверждены под конкретную игру):

- сумма в ``bet/raise/call/blind/allin`` — УРОВЕНЬ ставки на улице (call to / raise to),
  не инкремент → банк растёт на ``(уровень - уже внесённое игроком)``;
- банк берётся из ``Deal`` (авторитетный «банк после круга»); ``Deal`` завершает улицу;
- число игроков выводится из действий (лог не перечисляет стол): живые = кто походил и
  не сфолдил. На ранних решениях возможен недосчёт ещё не ходивших оппонентов;
- герой опознаётся по шаблону ника (``LogEvent.is_hero``) — надёжно, на ЛЮБОЙ его
  строке (включая блайнд); запасной признак — игрок, который ходит сразу после «Ваш ход».
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

from poker_analyzer.engine.stats import ProfileBook
from poker_analyzer.parsing.events import Action, LogEvent

logger = logging.getLogger(__name__)

# Действия, несущие ставку (двигают банк и уровень ставки на улице).
_WAGERS = frozenset({Action.BLIND, Action.BET, Action.CALL, Action.RAISE, Action.ALL_IN})
# Агрессивные действия — показывают силу (→ узкий диапазон оппонента в советах).
_AGGRO = frozenset({Action.BET, Action.RAISE, Action.ALL_IN})
# Действия игрока, задающие порядок мест на префлопе (фолд тоже «появление в очереди»).
_SEAT_ACTIONS = _WAGERS | frozenset({Action.FOLD, Action.CHECK})
# Имя улицы по числу карт борда.
_STREETS = {0: "preflop", 3: "flop", 4: "turn", 5: "river"}


def shown_hole_cards(combo: tuple[str, ...], board: tuple[str, ...]) -> tuple[str, ...]:
    """Карманные карты, ВСКРЫТЫЕ игроком на ``Win`` = карты комбинации МИНУС борд.

    ``Win`` показывает 5-карточную руку победителя; вычитая общие карты борда, получаем его
    карманные карты, попавшие в лучшую руку. Игра предпочитает показывать карты ИГРОКА (а не
    борда) там, где есть выбор, поэтому обычно остаётся 1–2 карты; ``0`` — если играет борд
    (карманка в комбинацию не вошла). Вычитание ПОКАРТНОЕ (ранг+масть): карта комбинации,
    совпавшая с бордовой, — общая; иначе карманная.

    У игрока не больше ДВУХ карманных карт, поэтому результат >2 = рассинхрон распознавания
    (масть бордовой/комбинационной карты прочлась по-разному) → ненадёжно, возвращаем ``()``.
    Пустая комбинация (все сфолдили — победитель не вскрыт) → ``()``.
    """
    board_set = set(board)
    hole = tuple(c for c in combo if c not in board_set)
    return hole if len(hole) <= 2 else ()


@dataclass(slots=True)
class HandState:
    """Состояние текущей раздачи, собранное из потока событий лога."""

    hero_cards: tuple[str, ...] = ()  # карты героя (из Hand)
    board: tuple[str, ...] = ()  # общие карты (из Table, накопительно)
    pot: float = 0.0  # банк (из Deal; между Deal — плюс ставки); float — бывают дробные суммы
    live: set[int] = field(default_factory=set)  # player_id живых (не сфолдивших)
    hero_id: int | None = None  # определяется по ходу после «Ваш ход»
    hero_to_act: bool = False  # True на «Ваш ход» — момент, когда нужен совет
    hero_stack: float | None = None  # стек героя на начало раздачи (бай-ин + P&L) — для SPR/олл-ин
    events: list[LogEvent] = field(default_factory=list)  # история текущей раздачи
    # Профили игроков за сессию (engine/stats.ProfileBook) — ставит вызывающий код один раз;
    # советник берёт поправку диапазона оппонента по его session_id. Сбросом не трогается.
    profiles: ProfileBook | None = None

    _bet_level: float = field(default=0.0, init=False)  # текущий уровень ставки на улице
    _street_in: dict[int, float] = field(default_factory=dict, init=False)  # внесено за улицу
    _last_raise_inc: float = field(
        default=0.0, init=False
    )  # последний ШАГ повышения уровня на улице (NLHE: мин-рейз = повторить его)
    _bb: float | None = field(default=None, init=False)  # большой блайнд (макс. первых двух Blind)
    _blind_count: int = field(default=0, init=False)  # сколько Blind видели (страддл — третий)
    _street_aggression: float = field(
        default=0.0, init=False
    )  # крупнейшая агрессия оппонента на улице (ставка / банк до неё) — для поляризации
    _facing_all_in: bool = field(
        default=False, init=False
    )  # верхний уровень ставки на улице поставлен олл-ином (нет имплайдов под флэт-колл чарта)
    _expect_hero: bool = field(default=False, init=False)  # следующий ход — героя
    _hand_over: bool = field(default=False, init=False)  # был Win — Dealer начнёт новую
    _dealer_seen: bool = field(
        default=False, init=False
    )  # видели Dealer — ждём Hand (старт раздачи)
    _pending_dealer: LogEvent | None = field(
        default=None, init=False
    )  # Dealer текущей раздачи — вернуть его баттон при сбросе на первой Hand
    _hero_invested: float = field(default=0.0, init=False)  # вклад героя в банк за раздачу (P&L)
    last_settled: tuple[str, float] | None = field(
        default=None, init=False
    )  # (время Win, P&L героя) последнего расчёта — вызывающий код копит это в баланс сессии
    _opp_tier: dict[int, int] = field(
        default_factory=dict, init=False
    )  # тир агрессии игрока: 0/1/2
    _session_of: dict[int, int] = field(
        default_factory=dict, init=False
    )  # player_id раздачи → постоянный session_id (из событий; для лога и профилей)
    _shown_hole: dict[int, tuple[str, ...]] = field(
        default_factory=dict, init=False
    )  # session_id → вскрытая на Win карманка (комбинация − борд); для калибровки диапазонов
    _button_id: int | None = field(default=None, init=False)  # баттон раздачи (из Dealer)
    _blind_order: list[int] = field(
        default_factory=list, init=False
    )  # первые два Blind (по порядку)
    _blind_amount: dict[int, float] = field(
        default_factory=dict, init=False
    )  # player_id блайнда → сумма (BB = больший: позицию определяем по сумме, не по порядку)
    _seen_order: list[int] = field(
        default_factory=list, init=False
    )  # порядок первых действий префлопа (seat ring от SB)
    _seen: set[int] = field(default_factory=set, init=False)  # игроки, замеченные в раздаче
    table_size: int = field(
        default=0, init=False
    )  # оценка размера стола (макс. за раздачи; вне сброса)

    # --- запросы для слоя советов -------------------------------------------

    @property
    def street(self) -> str:
        """Улица по числу карт борда: preflop / flop / turn / river."""
        return _STREETS.get(len(self.board), f"board{len(self.board)}")

    @property
    def num_opponents(self) -> int:
        """Число живых оппонентов (живые минус герой).

        Пока герой не сделал ход, его id может быть ещё не определён — тогда счёт
        уточнится, как только герой походит (см. допущения в докстринге модуля).
        """
        if self.hero_id is None:
            return len(self.live)
        return len(self.live - {self.hero_id})

    @property
    def to_call(self) -> float:
        """Сколько герою доколлить до текущего уровня ставки (для pot odds)."""
        contributed = self._street_in.get(self.hero_id, 0.0) if self.hero_id is not None else 0.0
        return max(0.0, self._bet_level - contributed)

    @property
    def facing_all_in(self) -> bool:
        """Верхний уровень ставки на улице поставлен олл-ином оппонента (нет имплайдов).

        Признак для префлоп-советов: флэт-колл чарта (спекулятивные одномастные/мелкие пары)
        калиброван под обычный рейз с деньгами за спиной — за ними playability и имплайд-оддсы.
        Против шова реализуется лишь сырая эквити, поэтому такой флэт переоценивается по прямым
        шансам банка (см. :func:`engine.preflop.decide_preflop`).
        """
        return self._facing_all_in

    @property
    def hero_invested(self) -> float:
        """Сколько герой вложил в банк за ТЕКУЩУЮ раздачу (растёт после его блайнда/колла/бета)."""
        return self._hero_invested

    @property
    def hero_remaining(self) -> float | None:
        """Остаток стека героя СЕЙЧАС (стек на начало раздачи − вложено), или ``None`` без снимка.

        ``hero_stack`` — стек героя на начало раздачи (ставит вызывающий код, если известен).
        Остаток нужен слою советов для SPR: цена колла кап-ится остатком (колл-олл-ин «за
        меньше»), а олл-ин-колл считается по сырой эквити без штрафов будущих улиц.
        """
        if self.hero_stack is None:
            return None
        return max(0.0, self.hero_stack - self._hero_invested)

    @property
    def hero_street_in(self) -> float:
        """Внесённое героем на ТЕКУЩЕЙ улице (его уровень ставки)."""
        return self._street_in.get(self.hero_id, 0.0) if self.hero_id is not None else 0.0

    @property
    def big_blind(self) -> float | None:
        """Размер большого блайнда (макс. из первых двух ``Blind``; страддл не считается).

        Максимум, а не «второй», — потому что распознавание изредка меняет SB/BB местами.
        """
        return self._bb

    @property
    def min_raise_to(self) -> float | None:
        """Минимальный легальный УРОВЕНЬ рейза («рейз до») на улице, или ``None``.

        NLHE: мин-рейз = текущий уровень + последний шаг повышения. Без ставки на улице
        минимальная ставка = большой блайнд. ``None`` — блайнды не прочитались. Это НИЖНИЙ
        предел шкалы допустимого рейза (от мин-рейза до олл-ина).
        """
        if self._bet_level > 0:
            inc = self._last_raise_inc if self._last_raise_inc > 0 else self._bet_level
            return self._bet_level + inc
        return self._bb

    @property
    def all_in_to(self) -> float | None:
        """УРОВЕНЬ олл-ина героя на улице (внесённое на улице + остаток), или ``None``.

        ВЕРХНИЙ предел шкалы рейза (уровни «рейз до»).
        """
        remaining = self.hero_remaining
        if remaining is None:
            return None
        return self.hero_street_in + remaining

    @property
    def street_aggression(self) -> float:
        """Размер крупнейшей агрессии оппонента на улице (ставка/банк до неё); 0 — без агрессии."""
        return self._street_aggression

    def session_of(self, player_id: int | None) -> int | None:
        """Постоянный (сессионный) id игрока раздачи, или ``None`` — для лога и профилей."""
        if player_id is None:
            return None
        return self._session_of.get(player_id)

    @property
    def shown_hole(self) -> dict[int, tuple[str, ...]]:
        """Вскрытые на ``Win`` карманные карты по ``session_id`` (комбинация − борд).

        Заполняется на каждой строке ``Win`` с прочитанными картами (см. :func:`shown_hole_cards`):
        победитель показал руку → его карманка известна. Сплит/сайд-пот = несколько ``Win`` →
        несколько записей (по одной на победителя). Источник для калибровки диапазона игрока
        (этап D профилей). Игроки без ``session_id`` и «играет борд» (0 карт) не попадают.
        """
        return dict(self._shown_hole)

    def opponent_ids(self) -> list[int]:
        """Id живых оппонентов (без героя), отсортированы — порядок как у ``opponent_tiers``.

        Нужен слою советов, чтобы выровнять с тирами параллельные данные по оппоненту
        (например, линии из :func:`~poker_analyzer.engine.range_model.classify_lines`).
        """
        opps = self.live - ({self.hero_id} if self.hero_id is not None else set())
        return sorted(opps)

    def opponent_tiers(self) -> list[int]:
        """Тиры агрессии живых оппонентов (0 пас / 1 коллер / 2 агрессор), без героя.

        По одному на каждого живого оппонента — слой советов берёт по тиру диапазон рук
        (агрессор → узкий, коллер → средний, пас/неизвестно → широкий). Порядок — как у
        :meth:`opponent_ids` (сортировка по id).
        """
        return [self._opp_tier.get(p, 0) for p in self.opponent_ids()]

    def hero_in_position(self) -> bool:
        """Грубая оценка: герой В ПОЗИЦИИ (ходит последним на постфлопе)?

        Прокси по префлоп-кольцу мест (``_seen_order`` идёт SB, BB, …, BTN): последний живой
        в этом порядке действует последним и на постфлопе. Баттон → всегда IP. Консервативно:
        при нехватке данных возвращаем ``False`` (OOP), чтобы не переоценивать реализацию
        эквити слабых рук. Используется для множителя реализации (REq) в слое советов.
        """
        if self.hero_id is None:
            return False
        if self._button_id is not None and self.hero_id == self._button_id:
            return True
        live_in_order = [p for p in self._seen_order if p in self.live]
        return bool(live_in_order) and live_in_order[-1] == self.hero_id

    def hero_position(self) -> str | None:
        """Позиция героя на префлопе: ``'EP'`` / ``'CO'`` / ``'BTN'`` / ``'SB'`` / ``'BB'`` или ``None``.

        Выводится из баттона (``Dealer``), порядка блайндов (SB, BB) и порядка действий
        (seat ring). ``None`` — на постфлопе либо когда данных не хватает (тогда советник
        падает на эвристику, а не угадывает). ``'CO'`` отличается от ``'EP'`` только при
        известном размере стола (``table_size``, копится за прошлые раздачи).

        Известное ограничение (редкое, самоисцеляется за орбиту): если стол ВЫРОС между
        раздачами и новые игроки сели позади ещё не ходившего героя, ``table_size`` пока
        занижен — крайнее среднее место может разово определиться как ``'CO'`` (чуть шире
        диапазон). BTN/SB/BB (главные позиции) определяются точно и этому не подвержены.
        """
        if len(self.board) > 0 or self._button_id is None or len(self._blind_order) < 2:
            return None
        hero = self.hero_id
        if hero is None:
            # Героя ещё не опознали (не блайнд, не баттон, не ходил). Но в момент «Ваш ход»
            # герой — это ровно СЛЕДУЮЩИЙ ходящий: его место выводимо из порядка действий
            # без id (иначе чарты молчат и эвристика системно перефолживает EP/CO-споты,
            # вплоть до фолда карманных пар). Вне своего хода без id позиции нет.
            if not self.hero_to_act:
                return None
        else:
            ordered = self._ordered_blinds()
            if hero == self._button_id:
                return "BTN"
            if hero == ordered[0]:
                return "SB"
            if hero == ordered[1]:
                return "BB"
        # Не блайнд и не баттон → доброволец между UTG и CO (в порядке действий).
        voluntary = [
            p for p in self._seen_order if p not in self._blind_order and p != self._button_id
        ]
        # Если герой ещё не ходил (типичное открытие) — он следующий доброволец после
        # уже сходивших; иначе берём его место в порядке. BTN=0, SB=1, BB=2, UTG=3, …
        offset = 3 + (
            voluntary.index(hero) if hero is not None and hero in voluntary else len(voluntary)
        )
        if self.table_size <= offset:  # размер стола неизвестен/мал — не отличить CO от EP
            return None
        return "CO" if offset == self.table_size - 1 else "EP"

    def position_context(self) -> str:
        """Краткая диагностика вывода позиции для лога СОВЕТ.

        Если позиция «—», по этой строке видно, чего не хватило: баттона (нет строки
        ``Dealer``), двух блайндов или размера стола (CO/EP неразличимы).
        """
        return (
            f"баттон={'+' if self._button_id is not None else '-'}"
            f" блайндов={len(self._blind_order)}"
            f" стол={self.table_size}"
        )

    def player_position(self, pid: int) -> str | None:
        """Позиция любого игрока ``pid`` (EP/CO/BTN/SB/BB) на префлопе или ``None``.

        Та же логика, что :meth:`hero_position`, но для произвольного игрока — для лога
        распознанных строк. ``None`` — на постфлопе, когда данных мало, или игрок ещё не
        замечен в порядке действий. Те же ограничения по ``table_size`` (CO/EP), что и у героя.
        """
        if len(self.board) > 0:  # на постфлопе позицию для лога не показываем
            return None
        return self._seat_of(pid)

    def _seat_of(self, pid: int) -> str | None:
        """Место игрока (EP/CO/BTN/SB/BB) по префлоп-кольцу — БЕЗ гарда постфлопа.

        Кольцо фиксируется на префлопе и не меняется, поэтому для выбора базового
        диапазона (:meth:`opponent_context`) позиция доступна и постфлоп. ``None`` —
        мало данных (нет баттона/блайндов) или игрок не в кольце.
        """
        if self._button_id is None or len(self._blind_order) < 2:
            return None
        ordered = self._ordered_blinds()
        if pid == self._button_id:
            return "BTN"
        if pid == ordered[0]:
            return "SB"
        if pid == ordered[1]:
            return "BB"
        voluntary = [
            p for p in self._seen_order if p not in self._blind_order and p != self._button_id
        ]
        if pid not in voluntary:
            return None
        offset = 3 + voluntary.index(pid)  # BTN=0, SB=1, BB=2, UTG=3, …
        if self.table_size <= offset:  # размер стола неизвестен/мал — не отличить CO от EP
            return None
        return "CO" if offset == self.table_size - 1 else "EP"

    def opponent_context(self, pid: int) -> str:
        """Префлоп-контекст игрока для выбора БАЗОВОГО диапазона (поверх тира).

        Определяется ПЕРВЫМ добровольным префлоп-действием (блайнды вынужденные, не в
        счёт); фиксируется на префлопе и держится постфлоп. Метки:
        ``open_late`` / ``open_early`` (открыл рейзом — по позиции), ``3bet`` (рейз
        поверх рейза), ``call_vs_open`` (доколлил рейз), ``bb_defend`` (BB доколлил
        рейз), ``limp`` (вошёл коллом без рейза), ``unknown`` (только чек / нет данных
        → откат на тир-диапазон).
        """
        n_raises = 0  # рейзов (не блайндов) ДО действия игрока
        for e in self.events:
            if e.action is Action.TABLE:
                break  # префлоп кончился
            if e.action in (Action.RAISE, Action.ALL_IN):
                if e.player_id == pid:
                    if n_raises >= 1:
                        return "3bet"
                    return "open_late" if self._seat_of(pid) in ("BTN", "CO") else "open_early"
                n_raises += 1
            elif e.action is Action.CALL and e.player_id == pid:
                if n_raises >= 1:
                    return "bb_defend" if self._seat_of(pid) == "BB" else "call_vs_open"
                return "limp"
        return "unknown"

    # --- применение событий --------------------------------------------------

    def apply(self, event: LogEvent) -> None:
        """Обновляет состояние одним событием лога."""
        self.events.append(event)
        self.last_settled = None  # заполнится только при расчёте раздачи (Win)
        action = event.action

        # Алярм склейки ников — ДО мутации (по уровню ставки и внесённому ПЕРЕД этой строкой).
        if action in (Action.CALL, Action.FOLD):
            self._suspect_self_action(event)

        if action is Action.WIN:
            self._settle(event)
            self._record_shown_hole(event)  # вскрытая карманка (комбинация − борд) → профили
        elif action is Action.DEALER:
            # Сброс на Win→Dealer (норма; держит раздачи раздельными для профилей даже когда
            # герой сидит аут). Dealer мид-хенд (прошлый дилер вышел/кикнут за неактив) НЕ
            # сбрасывает — `_hand_over` ложно. «Армируем» сброс: НОВАЯ раздача = Dealer, сразу
            # за которым идут ДВЕ строки Hand (карты героя) — реальный сброс делаем на первой
            # Hand (см. ниже), а не на Dealer, иначе мид-хенд Dealer-реассайн рушил бы раздачу.
            if self._hand_over:
                self._reset()
            self._dealer_seen = True
            self._pending_dealer = event  # вернём баттон ЭТОЙ раздачи при сбросе на первой Hand
            self._record_button(event)
        elif action is Action.HAND:
            # Dealer→Hand = старт НОВОЙ раздачи (две Hand-строки сразу за Dealer): сбрасываем
            # прошлую (борд/карты/банк) ЗДЕСЬ, а не на Dealer — это надёжный сигнал «героя
            # раздали заново», и Win прошлой мог не прочитаться (живой баг 2026-06-14: новая
            # рука + старый борд 6♥2♦9♣A♣A♠; первая Dealer-строка после захода без времени).
            if self._dealer_seen:
                pending = self._pending_dealer  # Dealer ЭТОЙ раздачи — его баттон переживёт сброс
                self._reset()  # очистит events (текущая Hand добавлена в начале apply) —
                self.events.append(event)  # ...вернём её; _reset сбросил и _dealer_seen
                if pending is not None:  # ВЕРНУТЬ баттон новой раздачи: _reset стёр _button_id,
                    self._record_button(pending)  # иначе позиция терялась (поз=—, регресс QK-фолд)
            # Дедуп: повторная Hand (рескан / неоднозначный якорь на близнецах) не задваивает.
            fresh = tuple(c for c in event.cards if c not in self.hero_cards)
            self.hero_cards = (self.hero_cards + fresh)[:2]  # у героя ровно две карты
        elif action is Action.TABLE:
            # Флоп = 3 карты одной строкой, тёрн/ривер — по одной. Флоп ЗАМЕНЯЕТ борд, а не
            # дозаписывает: самолечение от ПЕРЕНОСА борда прошлой раздачи (поздний Win не
            # сбросил) и от ДУБЛЯ чтения флопа (рескан / неоднозначный якорь) — оба раздували
            # борд >5, и advise падал в ValueError «борд>5» → совет None — разбор обрывался на
            # флопе (симптом «в игре весь борд, а скрипт видит 3 карты»). Тёрн/ривер дозаписываем
            # ТОЛЬКО к валидному борду (3→4→5); одиночная карта к борду 0/1/2/5 — внеочередная
            # (поздняя строка прошлой раздачи) → игнор. Кап [:5] — последняя страховка от >5.
            cards = tuple(event.cards)
            if len(cards) >= 3:
                self.board = cards[:5]
            elif len(self.board) in (3, 4):
                self.board = (self.board + cards)[:5]
        elif action is Action.DEAL:
            self._dealer_seen = False  # пошли круги торгов — окно «Dealer→Hand» закрыто
            if event.amount is not None:
                self.pot = event.amount  # авторитетный банк после круга
            self._bet_level = 0  # новая улица — обнуляем ставки
            self._street_in.clear()
            self._street_aggression = 0.0  # новая улица — сбрасываем размер агрессии
            self._last_raise_inc = 0.0  # шаг мин-рейза — пер-уличный
            self._facing_all_in = False  # верхняя ставка прошлой улицы больше не актуальна
        elif action is Action.YOUR_TURN:
            self.hero_to_act = True
            self._expect_hero = True
        elif action is Action.FOLD:
            self._note_actor(event.player_id)
            if event.player_id is not None:
                self.live.discard(event.player_id)
        elif action is Action.CHECK:
            self._note_actor(event.player_id)
            self._add_player(event.player_id)
        elif action in _WAGERS:
            pot_before = self.pot
            level_before = self._bet_level
            self._apply_wager(event)
            # Кто поднял верхний уровень — тем он и «помечен»: олл-ин снимает имплайды (флэт-колл
            # чарта невалиден), обычный рейз поверх олл-ина возвращает их (за ним есть стек).
            if self._bet_level > level_before:
                self._facing_all_in = action is Action.ALL_IN
            if action in _AGGRO:
                self._bump_tier(event.player_id, 2)  # бет/рейз/олл-ин — показал силу
                # Размер агрессии оппонента = вложенное им сейчас / банк до ставки (поляризация).
                is_hero_actor = event.is_hero or event.player_id == self.hero_id
                if not is_hero_actor and pot_before > 0:
                    self._street_aggression = max(
                        self._street_aggression, (self.pot - pot_before) / pot_before
                    )
            elif action is Action.CALL:
                self._bump_tier(event.player_id, 1)  # только коллировал — пассивно в игре

        # Надёжное опознание героя: ник совпал с его шаблоном. После обработки события
        # (в т.ч. возможного сброса на новой раздаче), чтобы reset не стёр hero_id.
        if event.is_hero and event.player_id is not None:
            self.hero_id = event.player_id

        # Карта «игрок раздачи → постоянный id»: после обработки события, чтобы сброс
        # на новой раздаче не стёр запись её первой строки (Dealer).
        if event.player_id is not None and event.session_id is not None:
            self._session_of[event.player_id] = event.session_id

        # Порядок мест и блайнды — для позиции героя (только префлоп: до карт борда).
        if len(self.board) == 0 and event.player_id is not None and action in _SEAT_ACTIONS:
            self._observe_seat(event.player_id)
            # Только SB и BB — позиционные блайнды; страддл (3-й+) учитывается как обычное
            # занятое место (в _seen_order, не в _blind_order), иначе сбился бы offset.
            if (
                action is Action.BLIND
                and len(self._blind_order) < 2
                and event.player_id not in self._blind_order
            ):
                self._blind_order.append(event.player_id)
                if event.amount is not None:
                    self._blind_amount[event.player_id] = event.amount

    # --- внутреннее ----------------------------------------------------------

    def _ordered_blinds(self) -> list[int]:
        """Блайнды в позиционном порядке ``[SB, BB]`` — по СУММЕ (BB = больший блайнд).

        Распознавание изредка отдаёт строки блайндов в обратном порядке (BB раньше SB) —
        тогда «второй по порядку = BB» путал SB и BB местами. Позицию определяем по сумме
        блайнда; порядок появления — лишь запасной критерий при равных суммах.
        """
        blinds = self._blind_order
        if len(blinds) < 2:
            return list(blinds)
        sb, bb = blinds[0], blinds[1]
        if self._blind_amount.get(bb, 0.0) < self._blind_amount.get(sb, 0.0):
            sb, bb = bb, sb
        return [sb, bb]

    def _suspect_self_action(self, event: LogEvent) -> None:
        """Алярм СКЛЕЙКИ ников: игрок «отвечает сам себе» (колл/фолд своей же верхней ставки).

        Если игрок уже держит верхнюю ставку улицы (внесённое == уровень), а следующая его
        строка — колл или фолд, под одним ``player_id`` слиплись ДВА разных игрока (картинки
        ников совпали выше порога). Для советов почти безвредно, но для профилей (этап B) —
        яд: чужие действия попали бы в чужую статистику. Только печатаем предупреждение —
        состояние не трогаем (вдруг это мис-рид реальной строки).
        """
        pid = event.player_id
        if pid is None or self._bet_level <= 0:
            return
        if self._street_in.get(pid, 0.0) < self._bet_level:
            return  # игрок НЕ на верхней ставке — его колл/фолд законны
        if event.action is Action.CALL:
            logger.warning("СКЛЕЙКА? P%s коллирует собственную ставку %s", pid, self._bet_level)
        elif event.action is Action.FOLD:
            logger.warning(
                "СКЛЕЙКА? P%s фолдит собственную неотвеченную ставку %s", pid, self._bet_level
            )

    def _apply_wager(self, event: LogEvent) -> None:
        self._note_actor(event.player_id)
        self._add_player(event.player_id)
        if event.amount is None:
            return
        level = event.amount  # сумма = уровень ставки на улице
        pid = event.player_id
        prev = self._street_in.get(pid, 0.0) if pid is not None else 0.0
        increment = max(0.0, level - prev)  # в банк идут только новые деньги
        self.pot += increment
        if event.is_hero or (pid is not None and pid == self.hero_id):
            self._hero_invested += increment  # копим вклад героя за раздачу (для P&L)
        if pid is not None:
            self._street_in[pid] = max(prev, level)
        self._track_min_raise(event, level)
        self._bet_level = max(self._bet_level, level)

    def _track_min_raise(self, event: LogEvent, level: float) -> None:
        """Поддерживает шаг мин-рейза (``_last_raise_inc``) и размер ББ — для уровней рейза.

        NLHE: следующий мин-рейз = уровень + последний шаг повышения. Блайнд считается
        шагом ЦЕЛИКОМ (мин-рейз префлопа = 2×ББ; страддл — 2×страддл). Олл-ин «за меньше»
        полного рейза шаг НЕ уменьшает (``max``). Полный бет/рейз задаёт новый шаг.
        """
        if event.action is Action.BLIND:
            self._blind_count += 1
            if self._blind_count <= 2:  # страддл (3-й+) не большой блайнд
                self._bb = max(self._bb or 0.0, level)
            self._last_raise_inc = max(self._last_raise_inc, level)
            return
        jump = level - self._bet_level  # насколько поднят уровень улицы
        if jump <= 0:
            return  # колл/недобор — шаг не меняется
        if event.action is Action.ALL_IN:
            self._last_raise_inc = max(self._last_raise_inc, jump)  # короткий олл-ин не уменьшает
        else:  # полноценный бет/рейз
            self._last_raise_inc = jump

    def _settle(self, event: LogEvent) -> None:
        """Расчёт раздачи на ``Win``: P&L героя → ``last_settled`` (вызывающий код копит в баланс).

        Раздача кончается на ``Win``. Вложенное героем (``_hero_invested``) списывается
        ОДИН раз — на первой ``Win`` раздачи; выигрыш прибавляется на КАЖДОЙ ``Win``, где
        победил герой (на случай сплита банка). Сумма выигрыша — из строки ``Win`` (если
        прочиталась), иначе текущий банк.
        """
        won = event.amount if event.amount is not None else self.pot
        hero_won = event.is_hero or (
            event.player_id is not None and event.player_id == self.hero_id
        )
        invested = 0.0 if self._hand_over else self._hero_invested  # вложенное — единожды
        self.last_settled = (event.time, (won if hero_won else 0.0) - invested)
        self._hand_over = True

    def _record_shown_hole(self, event: LogEvent) -> None:
        """Запоминает вскрытую на ``Win`` карманку победителя: комбинация (``event.cards``) − борд.

        Привязка к ``session_id`` (постоянный id для профилей). Записываем только при известном
        ``session_id`` и непустой карманке (см. :func:`shown_hole_cards`: «играет борд» / >2 карт
        от рассинхрона распознавания → пусто). ``board`` берём ТЕКУЩИЙ — на ``Win`` он уже полный
        (river-``Table`` пришла до ``Win``), а сброс прошлой раздачи делается лишь на следующей
        ``Hand``. Несколько ``Win`` (сплит/сайд-пот) пишут по записи на победителя.
        """
        if event.session_id is None or not event.cards:
            return
        hole = shown_hole_cards(event.cards, self.board)
        if hole:
            self._shown_hole[event.session_id] = hole

    def _add_player(self, player_id: int | None) -> None:
        if player_id is not None:
            self.live.add(player_id)

    def _record_button(self, event: LogEvent) -> None:
        """Эффекты строки ``Dealer``: игрок в живые + баттон/размер стола.

        Баттон и счёт стола — только по ПЕРВОМУ ``Dealer`` раздачи (``_button_id is None``):
        ложный/мис-ридный ``Dealer`` mid-hand не перезаписывает баттон и не раздувает
        ``table_size``. Зовётся и из обработки ``Dealer``, и при сбросе на первой ``Hand``
        (вернуть баттон новой раздачи, который ``_reset`` стёр).
        """
        self._add_player(event.player_id)
        if event.player_id is not None and self._button_id is None:
            self._button_id = event.player_id  # баттон = первый Dealer раздачи
            self._seen.add(event.player_id)  # баттон — игрок за столом (для размера)
            self.table_size = max(self.table_size, len(self._seen))

    def _bump_tier(self, player_id: int | None, tier: int) -> None:
        if player_id is not None:
            self._opp_tier[player_id] = max(self._opp_tier.get(player_id, 0), tier)

    def _observe_seat(self, pid: int) -> None:
        """Запоминает первое появление игрока на префлопе (порядок мест) и размер стола."""
        if pid not in self._seen:
            self._seen.add(pid)
            self._seen_order.append(pid)
            if len(self._seen) > self.table_size:
                self.table_size = len(self._seen)

    def _note_actor(self, player_id: int | None) -> None:
        if player_id is None:
            return
        # Строка сразу после «Ваш ход» — всегда 100% герой. Переопределяем id каждый
        # раз (самокоррекция, если раньше определили ошибочно), а не только при None.
        if self._expect_hero:
            self.hero_id = player_id
            self._expect_hero = False
        if player_id == self.hero_id:  # ход героя завершает его «очередь»
            self.hero_to_act = False

    def _reset(self) -> None:
        self.hero_cards = ()
        self.board = ()
        self.pot = 0.0
        self.live = set()
        self.hero_id = None
        self.hero_to_act = False
        self.hero_stack = None
        self.events = []
        self._bet_level = 0.0
        self._street_in = {}
        self._last_raise_inc = 0.0
        self._bb = None  # блайнды новой раздачи перечитаются из её строк Blind
        self._blind_count = 0
        self._street_aggression = 0.0
        self._expect_hero = False
        self._hand_over = False
        self._dealer_seen = False
        self._pending_dealer = None
        self._hero_invested = 0.0  # вклад героя — пер-раздачный; баланс сессии живёт вне state
        self._opp_tier = {}
        self._session_of = {}
        self._shown_hole = {}
        self._button_id = None
        self._blind_order = []
        self._blind_amount = {}
        self._seen_order = []
        self._seen = set()
        # table_size НЕ сбрасываем — оценка размера стола живёт между раздачами
