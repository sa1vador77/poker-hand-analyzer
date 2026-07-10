"""Выдача подсказки по состоянию раздачи.

:func:`advise` срабатывает на ход героя (``hero_to_act``). Диапазон каждого живого
оппонента берётся по тиру его действий (см. :meth:`HandState.opponent_tiers`) и на
постфлопе сужается по доске (оставляем «продолжающие» руки — отсев воздуха, **B**).
Дальше:

- **хедз-ап и постфлоп** — EV-решение с выбором РАЗМЕРА ставки/рейза (**A**): для набора
  размеров считаем fold equity + эквити-при-колле и берём размер с макс. EV;
- **мультивей или префлоп** — эвристика «эквити героя vs pot odds» (без размера).

Постфлоп-реализм: диапазон агрессора дополнительно сужается по РАЗМЕРУ его ставки
(крупная ставка → поляризация во value), а готовая, но ДОМИНИРОВАННАЯ рука против тяжёлой
агрессии получает штраф reverse implied odds (:func:`_apply_domination`) — колл может стать
фолдом (учёт будущих потерь). Это не солвер на одну улицу, но уже не наивная эвристика.
"""

from __future__ import annotations

from dataclasses import replace
from functools import lru_cache

from poker_analyzer.config import (
    BET_POLARIZATION,
    COMMIT_GATE,
    DOMINATION_CRUSH,
    DOMINATION_PENALTY,
    JAM_BB_THRESHOLD,
    MULTIWAY_VALUE,
    NARROWING,
    OPPONENT_RANGES,
    POSITION_PRESETS,
    REALIZATION,
    VALUE_CATEGORY,
)
from poker_analyzer.engine.advice import Advice
from poker_analyzer.engine.equity import (
    ComboClass,
    card,
    classify_combos,
    equity_vs_ranges,
    hero_equity_vs_each,
)
from poker_analyzer.engine.preflop import decide_preflop
from poker_analyzer.engine.range_model import LineTag, classify_lines, line_threshold_delta
from poker_analyzer.engine.ranges import narrow_range, parse_range
from poker_analyzer.engine.sizing import best_bet_from_eqs
from poker_analyzer.engine.state import HandState
from poker_analyzer.engine.texture import texture_delta

# Масть-глиф из распознавания → буква масти в кодировке эквити.
_SUIT_LETTER = {"♠": "s", "♥": "h", "♦": "d", "♣": "c"}
RAISE_EQUITY = 0.66  # порог агрессии для прежней (мультивей/префлоп) эвристики
EQUITY_ITERATIONS = 50_000  # бюджет Монте-Карло на эквити vs диапазоны (мультивей/префлоп)
# Порог точного перебора. Дефолт движка (10M) на РИВЕРЕ в мультивее даёт точный перебор
# произведения диапазонов (3 лузовых диапазона ≈ 200³ ≈ 8M раскладов) — секунды на ответ.
# Низкий кап уводит крупные расклады на Монте-Карло (50k сэмплов, фикс-сид → детерминирован,
# не флапает): ~0.02 с и тот же ответ (≤3пп от точного, см. тест). Точный перебор остаётся
# лишь для мелких раскладов, где он дешевле выборки.
EQUITY_EXACT_CAP = 200_000

# Диапазон оппонента по тиру агрессии (индекс = тир: 0 пас, 1 коллер, 2 агрессор).
_TIER_RANGES = [
    parse_range(OPPONENT_RANGES.default),
    parse_range(OPPONENT_RANGES.caller),
    parse_range(OPPONENT_RANGES.aggressor),
]
# Диапазон префлоп-ПУШЕРА (рейз ≥ JAM_BB_THRESHOLD ББ): шире рейз-диапазона.
_JAMMER_RANGE = parse_range(OPPONENT_RANGES.jammer)
# Узкий ВЭЛЬЮ-диапазон добровольного захода всем стеком (лузово-пассивный пул налегке стек
# не коммитит). Применяется (1) гейтом коммита в decide_preflop через stackoff_equity и
# (2) к МУЛЬТИВЕЙ-олл-ину ВМЕСТО широкого джеммера: несколько добровольных стеков = пересечение
# узких вэлью, а не один широкий пуш (фикс A — иначе доминированные руки шли в шов по сырой
# эквити против завышенно-широких диапазонов).
_STACKOFF_RANGE = parse_range(OPPONENT_RANGES.stackoff)


def _committed_count(tiers: list[int]) -> int:
    """Сколько оппонентов ДОБРОВОЛЬНО вложились в банк (tier ≥ 1: коллер/агрессор).

    Для произведения вэлью-диапазонов в стек-офф-спотах считать нужно ИМЕННО их: пассивный
    ещё-не-ходивший блайнд / баттон (tier 0) почти всегда фолдит на шов и стек не коммитит —
    моделировать его узким вэлью раздувало число диапазонов и резко занижало эквити героя
    (QQ vs 2 шова при живых SB/BB = 0.30 → фолд вместо 0.45 → колл). Минимум 1 (сам агрессор)."""
    return max(1, sum(1 for t in tiers if t >= 1))


# Индекс улицы по числу карт борда (для порогов сужения): флоп / тёрн / ривер.
_STREET_IDX = {3: 0, 4: 1, 5: 2}

# Базовый диапазон по ПРЕФЛОП-КОНТЕКСТУ (метка из HandState.opponent_context). Точнее
# тира: «агрессор» = и опен-EP, и стил-BTN, и 3-бет — у них РАЗНЫЕ диапазоны. unknown
# (только чек / нет данных) в словаре нет → откат на тир.
_CONTEXT_RANGES = {
    "open_early": parse_range(POSITION_PRESETS.open_early),
    "open_late": parse_range(POSITION_PRESETS.open_late),
    "3bet": parse_range(POSITION_PRESETS.threebet),
    "call_vs_open": parse_range(POSITION_PRESETS.call_vs_open),
    "limp": parse_range(POSITION_PRESETS.limp),
    "bb_defend": parse_range(POSITION_PRESETS.bb_defend),
}


def _to_equity_int(label: str) -> int:
    """Карта из распознавания ('7♦', 'A♠', '10♦') → int 0..51 для слоя эквити."""
    rank = label[:-1]
    rank = "T" if rank == "10" else rank  # десятка: '10' → 'T'
    return card(rank + _SUIT_LETTER[label[-1]])


def _narrowed_ranges(
    tiers: list[int],
    board: list[int],
    aggression: float = 0.0,
    line_tags: list[LineTag | None] | None = None,
    jam: bool = False,
    profile_deltas: list[float] | None = None,
    contexts: list[str] | None = None,
) -> list[list[tuple[int, int]]]:
    """Диапазоны оппонентов по тиру; на постфлопе активные сужаются по доске.

    Крупная агрессия (``aggression`` — ставка/банк) дополнительно поднимает порог сужения
    агрессора (он поляризован во value) — чем больше ставка, тем уже и сильнее диапазон.
    Линия оппонента (``line_tags`` параллельно ``tiers``: баррели / чек-рейз / донк) добавляет
    к порогу: «тяжёлая» линия → уже диапазон — чек-колл и чек-рейз перестают быть равны.
    ``jam`` (шов: явный олл-ин или уровень ≥ ``JAM_BB_THRESHOLD`` ББ): пуш-диапазоны
    ШИРОКИЕ — префлоп агрессоры получают ПУШ-диапазон (шире рейз-диапазона), а постфлоп
    сужаются мягко, «как коллер», БЕЗ поляризации. Иначе шов сужал агрессора до натсов,
    и ассистент перефолживал всё вплоть до сильных готовых рук — пул это эксплойтил
    тупым олл-ином (живой лог 2026-06-12: топ-пара «экв 6%», фолд при цене банка 9%).
    """
    street = _STREET_IDX.get(len(board))
    # Текстура доски — одна на всех оппонентов (от борда, не от игрока): мокрый борд
    # ослабляет порог (диапазон шире — пул продолжает с дро), сухой чуть усиливает.
    tex_delta = texture_delta(board)
    out: list[list[tuple[int, int]]] = []
    for i, t in enumerate(tiers):
        # База: префлоп-шов → джеммер; иначе диапазон по КОНТЕКСТУ (опен/3бет/лимп/защита),
        # а при unknown/без контекста — откат на тир.
        if jam and t == 2 and not board:
            base = _JAMMER_RANGE
        else:
            ctx = contexts[i] if contexts is not None and i < len(contexts) else "unknown"
            base = _CONTEXT_RANGES.get(ctx, _TIER_RANGES[t])
        if street is not None and t in (1, 2):  # пас (тир 0) не сужаем
            if t == 2 and jam:
                # Постфлоп-шов: широкий «коллерский» порог, без поляризации — пуш в этом
                # пуле не значит «натсы», модель натс-диапазона делала фолд-эксплойт.
                thr = NARROWING.caller[street]
            else:
                thr = (NARROWING.aggressor if t == 2 else NARROWING.caller)[street]
                if t == 2:  # размер ставки агрессора → поляризация диапазона
                    # Прирост порога ограничен: иначе мультивей-диапазон схлопывается до
                    # пары флешей и нативное MC не может расставить их (дефицит карт
                    # масти) → 0% эквити.
                    thr += min(0.12, BET_POLARIZATION * max(0.0, aggression - 0.5))
            tag = line_tags[i] if line_tags is not None and i < len(line_tags) else None
            thr += line_threshold_delta(tag)  # линия: баррели/чек-рейз сужают, донк расширяет
            thr += tex_delta  # текстура доски: мокрый борд шире, сухой уже
            if profile_deltas is not None and i < len(profile_deltas):
                # Этап C: лузовость ПРОФИЛЯ оппонента (сглаженный VPIP со шринкеджем) —
                # лузовый шире (его агрессия стоит меньше), тайтовый уже.
                thr += profile_deltas[i]
            narrowed = list(_narrow_cached(tuple(base), tuple(board), round(thr * 1000)))
            base = narrowed or base  # не опустошаем диапазон, если порог срезал всё
        out.append(base)
    return out


@lru_cache(maxsize=512)
def _narrow_cached(
    base: tuple[tuple[int, int], ...], board: tuple[int, ...], thr_millis: int
) -> tuple[tuple[int, int], ...]:
    """Кэш сужения: ``narrow_range`` дорогой (эквити каждого комбо vs случайной на борде),
    а мультивей зовёт его с ОДИНАКОВЫМИ (база, борд, порог) для оппонентов одного контекста
    и повторяет на каждом пересчёте совета той же улицы — без кэша совет на 4 оппонентах
    занимал секунды. Ключ — сам базовый диапазон (кортеж), борд и порог (квантован до 0.001)."""
    return tuple(narrow_range(list(base), list(board), thr_millis / 1000))


def _top_pair_or_overpair(hero: list[int], board: list[int]) -> bool:
    """Топ-пара (старшая карта борда спарена) или оверпара (карманка выше борда).

    Зовётся ТОЛЬКО для руки «одна пара» (``made == PAIR``): карманная пара тогда НЕ на борде
    (иначе сет), а непарная рука даёт ровно одну пару с бордом. Топ-пара/оверпара реализуют
    заметно лучше второй/нижней пары → отдельный бакет реализации. Ранг карты = ``int // 4``.
    """
    if not board:
        return False
    top_board = max(c // 4 for c in board)
    if hero[0] // 4 == hero[1] // 4:  # карманная пара
        return hero[0] // 4 > top_board  # оверпара — выше старшей карты борда
    board_ranks = {c // 4 for c in board}
    for c in hero:  # непарная рука: ровно один ранг спарил борд — это и есть наша пара
        if c // 4 in board_ranks:
            return c // 4 == top_board  # топ-пара, если спарили СТАРШУЮ карту борда
    return False


def _realization(state: HandState, hero: list[int], board: list[int], n_opp: int) -> float:
    """Множитель реализации эквити R для руки героя на борде (REq; мультивей-постфлоп).

    Сырая эквити ≠ реализованной: позиция, класс руки (made/draw/air, топ-пара vs слабая) и
    УЛИЦА определяют, какую долю своей доли рука реально заберёт (реализация растёт к риверу —
    там шоудаун, выбить нечем). Таблица — в :data:`config.REALIZATION`. Возвращает R в
    ``[0.6, 1.15]``; при сбое классификации — нейтральный ``1.0``.
    """
    try:
        cls = classify_combos(board, [(hero[0], hero[1])])[0]
    except (KeyError, ValueError):
        return 1.0
    if cls.made >= VALUE_CATEGORY:
        bucket, pair = "strong", REALIZATION.made_strong
    elif cls.is_made:  # одна пара: топ-пара/оверпара реализуют лучше второй/нижней
        if _top_pair_or_overpair(hero, board):
            bucket, pair = "top", REALIZATION.made_top
        else:
            bucket, pair = "weak", REALIZATION.made_weak
    elif cls.is_draw:  # флеш-дро/OESD реализуют лучше голого гатшота (имплайды, больше аутов)
        if cls.flush_draw or cls.oesd:
            bucket, pair = "draw", REALIZATION.draw_strong
        else:
            bucket, pair = "draw", REALIZATION.draw_weak
    else:
        bucket, pair = "air", REALIZATION.air
    r = pair[0] if state.hero_in_position() else pair[1]
    if bucket in ("weak", "air") and n_opp > 1:  # слабые руки тяжело реализовать мультивей
        r *= REALIZATION.multiway_penalty ** (n_opp - 1)
    street = _STREET_IDX.get(len(board))  # реализация растёт к риверу (шоудаун — забирает всё)
    if street is not None:
        r *= REALIZATION.street_baseline[street]
    return max(0.6, min(1.15, r))


def _apply_domination(
    advice: Advice,
    hero: list[int],
    board: list[int],
    ranges: list[list[tuple[int, int]]],
    tiers: list[int],
    aggression: float,
    to_call: float,
    pot: float,
) -> Advice:
    """Reverse implied odds: фолдим колл/чек И снимаем рейз, если диапазон агрессора КРУШИТ героя.

    Готовая, но доминированная рука (напр. слабый флеш против старшего) против тяжёлой
    агрессии — это будущие потери: требуем эквити выше сырой цены банка. Дро не штрафуем —
    у них эквити против комбо выше ``DOMINATION_CRUSH`` (они не «мертвы»).

    Рейз обрабатываем НАРАВНЕ с коллом: иначе рука, которую штраф сфолдил бы при колле, на
    ещё большей агрессии (рейз) проходила без проверки и ре-рейзила в крушащий диапазон
    (T-флеш ре-рейзил олл-ин против старших флешей). Логика: не хватает дисконтированной
    эквити даже на колл → фолд; хватает на колл, но БОЛЬШИНСТВО вэлью агрессора всё равно
    бьёт героя → рейз снимаем, оставляем колл (рейз ловил бы только лучшее).
    """
    if advice.action not in ("call", "check", "raise") or aggression <= 0:
        return advice
    aggr = next((r for t, r in zip(tiers, ranges, strict=True) if t == 2), None)
    if aggr is None:
        return advice  # нет агрессора — некого бояться
    try:
        each = hero_equity_vs_each(hero, board, aggr)
        classes = classify_combos(board, aggr)
    except (KeyError, ValueError):
        return advice
    # Только реальные расклады: each == -1 — комбо заблокировано картами героя/борда. Это и
    # есть учёт блокеров: заблокированные героем вэлью-комбо агрессора выпадают из счёта →
    # его диапазон становится более блефовым → грань колла смещается в нашу пользу.
    items = [(e, c) for e, c in zip(each, classes, strict=True) if e >= 0.0 and not c.blocked]
    if not items:
        return advice
    # Поляризация по размеру ставки: крупный бет → диапазон тяжелее во value (вэлью весит больше).
    pol = min(0.5, BET_POLARIZATION * max(0.0, aggression))

    def _weight(c: ComboClass) -> float:
        return (1.0 + pol) if c.made >= VALUE_CATEGORY else (1.0 - 0.5 * pol)

    total = sum(_weight(c) for _, c in items)
    crushed = sum(_weight(c) for e, c in items if e < DOMINATION_CRUSH) / total  # вес «мёртвых»
    pot_odds = to_call / (pot + to_call) if to_call > 0 else 0.0
    required = pot_odds + DOMINATION_PENALTY * crushed * min(1.5, aggression)
    if advice.equity < required:
        # Дисконтированной эквити не хватает даже на колл — фолд (и для колла, и для рейза).
        reason = f"доминация: {crushed:.0%} вэлью агрессора крушит (нужно {required:.0%})"
        return Advice("fold", advice.equity, advice.pot_odds, reason)
    if advice.action == "raise" and crushed >= 0.5:
        # Эквити хватает на колл, но большинство вэлью агрессора бьёт героя — рейз снимаем
        # (ре-рейз ловит только лучшее), оставляем колл без размера.
        reason = f"доминация: {crushed:.0%} вэлью крушит — рейз снят, колл"
        return Advice("call", advice.equity, advice.pot_odds, reason)
    return advice


def _stackoff_equity(
    state: HandState,
    hero: list[int],
    tiers: list[int],
    level: float,
    remaining: float | None,
) -> float | None:
    """Эквити героя против узкого ВЭЛЬЮ-диапазона захода всем стеком — для гейта коммита (B).

    Считаем ЛИШЬ когда есть снимок стека и потенциальный чарт-рейз (≈ ``open_raise_mult ×
    уровня``) коммитил бы большую долю стека: в глубоких спотах гейт всё равно пропускает
    рейз, лишний расчёт эквити не делаем (префлоп-эквити — дорогой полный перебор борда).
    Число вэлью-диапазонов = ДОБРОВОЛЬНО вложившиеся (``_committed_count``), не все живые:
    пассивный непоходивший блайнд на 3-бет-шов фолдит, моделировать его вэлью занижало бы
    эквити героя. ``None`` — стека нет / коммит маловероятен / расчёт не удался → гейт молчит.
    """
    if remaining is None:
        return None
    eff = state.hero_street_in + remaining
    if eff <= 0 or COMMIT_GATE.open_raise_mult * level < COMMIT_GATE.raise_commit_frac * eff:
        return None  # потенциальный рейз стек не коммитит — гейт не сработает, эквити не нужна
    try:
        return equity_vs_ranges(
            hero,
            [],
            [list(_STACKOFF_RANGE) for _ in range(_committed_count(tiers))],
            iterations=EQUITY_ITERATIONS,
            exact_cap=EQUITY_EXACT_CAP,
        ).equity
    except (KeyError, ValueError):
        return None


def advise(state: HandState) -> Advice | None:
    """Рекомендация на ход героя, или ``None`` если совет сейчас не нужен/невозможен.

    ``None`` — если сейчас не ход героя (``hero_to_act`` ложен), рука героя ещё не
    распознана (нет ровно 2 карт / карта не парсится) или расчёт невозможен.
    """
    if not state.hero_to_act or len(state.hero_cards) != 2:
        return None
    tiers = state.opponent_tiers()[:9] or [0]  # по тиру на живого оппонента; пусто → один дефолтный
    lines = classify_lines(state.events)  # линии оппонентов: баррели / чек-рейз / донк
    line_tags = [lines.get(pid) for pid in state.opponent_ids()[:9]] or [None]
    aggression = state.street_aggression  # размер агрессии на улице (для поляризации/доминации)
    # Шов: явный олл-ин (facing_all_in) или уровень ставки ≥ порога в ББ. Пуш-диапазоны
    # ШИРОКИЕ: префлоп агрессорам даётся пуш-диапазон, постфлоп — мягкое сужение без
    # поляризации (см. _narrowed_ranges) — иначе «олл-ин = натсы» и перефолд всего.
    bb = state.big_blind
    level = state.to_call + state.hero_street_in
    jam = state.facing_all_in or (bb is not None and bb > 0 and level >= JAM_BB_THRESHOLD * bb)
    # Этап C: поправки порога сужения по ПРОФИЛЯМ оппонентов (лузовость, шринкедж).
    profile_deltas = None
    if state.profiles is not None:
        profile_deltas = [
            state.profiles.threshold_delta(state.session_of(pid))
            for pid in state.opponent_ids()[:9]
        ] or [0.0]
    # Префлоп-КОНТЕКСТ оппонента (опен/3бет/лимп/защита BB) → точнее базовый диапазон,
    # чем грубый тир; unknown → откат на тир внутри _narrowed_ranges.
    contexts = [state.opponent_context(pid) for pid in state.opponent_ids()[:9]] or ["unknown"]
    try:
        hero = [_to_equity_int(c) for c in state.hero_cards]
        board = [_to_equity_int(c) for c in state.board]
    except (KeyError, ValueError):
        return None  # нечистое распознавание (дубли карт / мусор / борд>5) — без совета

    # SPR: цена колла кап-ится остатком стека (колл-олл-ин «за меньше»). Если колл забирает
    # ВЕСЬ стек — это олл-ин: будущих улиц нет, значит решаем по СЫРОЙ эквити vs честные pot
    # odds, без REq и без штрафа за доминацию (нечего терять на несуществующих улицах). Банк
    # урезаем на неоплатный избыток оверщова (его героя не выиграть). Без снимка стека —
    # remaining=None → всё как раньше (полная цена, штрафы включены).
    remaining = state.hero_remaining
    to_call_eff = state.to_call
    pot_eff = state.pot
    all_in = False
    if remaining is not None and remaining > 0 and state.to_call > 0:
        to_call_eff = min(state.to_call, remaining)
        # state.pot УЖЕ включает ставку оппонента (_apply_wager: pot += increment), а формула
        # pot_odds = to_call/(pot+to_call) трактует pot как банк ДО колла героя. Поэтому неоплатный
        # избыток оверщова (to_call − to_call_eff) — деньги виллана, которые ему вернут и герой их
        # не выиграет — вычитаем из банка РОВНО один раз. Эфф. цена = to_call_eff/(pot_eff+to_call_eff).
        pot_eff = state.pot - (state.to_call - to_call_eff)
        all_in = state.to_call >= remaining

    # Фикс A: ПРЕФЛОП-МУЛЬТИВЕЙ-олл-ин (колл забирает весь стек И ≥2 ДОБРОВОЛЬНО вложившихся) —
    # стеки лузово-пассивного пула = узкое ВЭЛЬЮ, а не широкий джеммер. Каждому КОММИТНУВШЕМУ
    # даём вэлью-диапазон (доминация моделируется самим диапазоном); пассивных непоходивших
    # блайндов (tier 0) в произведение НЕ берём — иначе эквити героя завышенно занижалась и
    # премиум перефолживался (QQ vs 2 шова при живых SB/BB). Лечит KJ-колл-в-три-олл-ина.
    # ТОЛЬКО префлоп: постфлоп-шов = широкий пуш («как коллер», см. _narrowed_ranges jam) —
    # узкий стек-офф-диапазон (он префлопный) там был бы кривой моделью и вернул бы перефолд.
    committed = _committed_count(tiers)
    multiway_allin = all_in and committed >= 2 and len(board) == 0
    try:
        if multiway_allin:
            ranges = [list(_STACKOFF_RANGE) for _ in range(committed)]
        else:
            ranges = _narrowed_ranges(
                tiers, board, aggression, line_tags, jam, profile_deltas, contexts
            )
    except (KeyError, ValueError):
        return None  # нечистое распознавание (дубли карт / мусор / борд>5) — без совета

    # A: размер по EV — только хедз-ап и постфлоп. Иначе эвристика / префлоп-чарты.
    if len(ranges) == 1 and len(board) >= 3:
        advice = _advise_hu(hero, board, ranges[0], pot_eff, to_call_eff, all_in=all_in)
    else:
        try:
            eq = equity_vs_ranges(
                hero, board, ranges, iterations=EQUITY_ITERATIONS, exact_cap=EQUITY_EXACT_CAP
            ).equity
        except (KeyError, ValueError):
            return None
        advice = None
        if len(board) == 0:  # префлоп: позиционные чарты первичны
            advice = decide_preflop(
                state,
                eq,
                to_call=to_call_eff,
                pot=pot_eff,
                committed=all_in or state.facing_all_in,
                stackoff_equity=_stackoff_equity(state, hero, tiers, level, remaining),
            )
        if advice is None:
            value_cls = None
            if len(board) >= 3 and not all_in:  # REq — только когда впереди ещё есть улицы
                eq = max(0.0, min(1.0, eq * _realization(state, hero, board, len(ranges))))
                if to_call_eff <= 0:  # инициатива (чек-чек до героя) — оценим вэлью-бет
                    value_cls = _value_class(hero, board)
            advice = _decide_simple(eq, pot_eff, to_call_eff, len(ranges), value_cls)

    # Постфлоп: штраф за доминацию (reverse implied odds) — только если впереди есть улицы.
    if advice is not None and len(board) >= 3 and to_call_eff > 0 and not all_in:
        advice = _apply_domination(
            advice, hero, board, ranges, tiers, aggression, to_call_eff, pot_eff
        )
    # Колл забирает ВЕСЬ стек → РЕЙЗА НЕ СУЩЕСТВУЕТ: кнопка Raise в игре неактивна
    # (разбор раздачи: AK на BTN, доколл = остатку — кнопки Raise в этой точке не существует).
    # «Рейз» чарта/эвристики здесь на деле выбор «колл-олл-ин vs фолд» — решаем по
    # эквити против эффективной цены банка.
    if advice is not None and all_in and advice.action in ("bet", "raise"):
        price = to_call_eff / (pot_eff + to_call_eff)
        action = "call" if advice.equity >= price else "fold"
        advice = replace(
            advice,
            action=action,
            size=None,
            pot_odds=price,
            reason=f"{advice.reason}; рейза нет — колл и есть олл-ин",
        )
    # Инвариант: бесплатный чек НЕ фолдим (страховка от мисклассификации позиции/чарта) —
    # важно: фолд при to_call=0 отдал бы руку даром.
    if advice is not None and advice.action == "fold" and to_call_eff <= 0:
        advice = replace(advice, action="check", reason=f"{advice.reason}; чек бесплатно")
    return _cap_size(advice, remaining)


def _cap_size(advice: Advice | None, remaining: float | None) -> Advice | None:
    """Кап размера ставки/рейза остатком стека: больше стека не поставить → это олл-ин."""
    if advice is None or advice.size is None or remaining is None or remaining <= 0:
        return advice
    if advice.size >= remaining:
        return replace(advice, size=remaining, reason=f"{advice.reason}; олл-ин (весь стек)")
    return advice


def _value_class(hero: list[int], board: list[int]) -> str | None:
    """Класс готовой руки для ВЭЛЬЮ-БЕТА на инициативе в мультивее.

    ``'strong'`` — две пары+; ``'top'`` — топ-пара/оверпара; ``None`` — слабая пара / дро /
    воздух (их НЕ вэлью-бетим: против станций дро-полублеф бесполезен — они не фолдят, а
    слабая пара хочет контроль банка). Зеркалит классификацию :func:`_realization`.
    """
    try:
        cls = classify_combos(board, [(hero[0], hero[1])])[0]
    except (KeyError, ValueError):
        return None
    if cls.made >= VALUE_CATEGORY:
        return "strong"
    if cls.is_made and _top_pair_or_overpair(hero, board):
        return "top"
    return None


def _value_bet_threshold(opponents: int) -> float:
    """Порог РЕАЛИЗОВАННОЙ эквити для вэлью-бета на инициативе — из математики эквити.

    Против ``N`` коллеров-станций (каждый доплачивает ``b`` в банк) вэлью-бет выгоднее чека,
    когда доля банка героя ``eq > 1/(N+1)``. Вывод (одноуличная модель «станции коллят,
    шоудаун», см. :mod:`engine.sim`): ``EV_bet − EV_check = b·(eq·(N+1) − 1)`` > 0 ⇔
    ``eq > 1/(N+1)``; размер ставки ``b`` на ПОРОГ не влияет (классический результат против
    станции), только на величину профита. Сравнивается с реализованной эквити (она уже учла
    reverse implied odds), плюс калибруемый ``margin`` и абсолютный ``floor``."""
    return max(MULTIWAY_VALUE.floor, 1.0 / (opponents + 1) + MULTIWAY_VALUE.margin)


def _decide_simple(
    eq: float, pot: float, to_call: float, opponents: int, value_class: str | None = None
) -> Advice:
    """Мультивей/префлоп: эквити героя vs pot odds → действие.

    ``value_class`` (``'strong'``/``'top'``/``None``) — класс готовой руки для ВЭЛЬЮ-БЕТА на
    инициативе в мультивее; считается в :func:`advise` только постфлоп (``board ≥ 3``) на
    инициативе. ``None`` → прежнее поведение (чек / рейз монстром при ``eq ≥ RAISE_EQUITY``).
    """
    if to_call <= 0:  # коллить нечего — вэлью-бет готовой рукой / чек / (рейз монстром)
        if value_class is not None:
            threshold = _value_bet_threshold(opponents)
            if eq >= threshold:
                frac = (
                    MULTIWAY_VALUE.size_strong
                    if value_class == "strong"
                    else MULTIWAY_VALUE.size_top
                )
                reason = f"вэлью-бет {value_class} экв {eq:.0%} ≥ {threshold:.0%} ({opponents} опп)"
                return Advice("bet", eq, 0.0, reason, size=frac * pot)
        action = "raise" if eq >= RAISE_EQUITY else "check"
        return Advice(action, eq, 0.0, f"эквити {eq:.0%} против {opponents}")
    pot_odds = to_call / (pot + to_call)
    if eq < pot_odds:
        action = "fold"
    elif eq >= RAISE_EQUITY:
        action = "raise"
    else:
        action = "call"
    return Advice(action, eq, pot_odds, f"эквити {eq:.0%} vs шанс банка {pot_odds:.0%}")


def _advise_hu(
    hero: list[int],
    board: list[int],
    villain: list[tuple[int, int]],
    pot: float,
    to_call: float,
    *,
    all_in: bool = False,
) -> Advice | None:
    """Хедз-ап постфлоп: EV-решение с выбором размера ставки/рейза (A).

    ``all_in`` — колл забирает весь стек героя: рейз невозможен, а ``pot``/``to_call`` уже
    «эффективные» (банк без избытка оверщова) → ``ev_call`` становится EV олл-ин-колла.
    """
    try:
        eqs = [e for e in hero_equity_vs_each(hero, board, villain) if e >= 0.0]
    except (KeyError, ValueError):
        return None
    if not eqs:
        return None
    eq = sum(eqs) / len(eqs)  # эквити героя против (суженного) диапазона оппонента
    best = best_bet_from_eqs(eqs, pot)  # лучший размер по EV

    if to_call <= 0:  # первым ходить: чек или ставка
        ev_check = eq * pot
        if best is not None and best.ev > ev_check and best.ev > 0.0:
            reason = f"EV {best.ev:+.0f} > чек {ev_check:+.0f}; фолдов {best.fold_equity:.0%}"
            return Advice("bet", eq, 0.0, reason, size=best.size, ev=best.ev)
        return Advice("check", eq, 0.0, f"эквити {eq:.0%}; ставка невыгодна", ev=ev_check)

    # Перед ставкой соперника: фолд / колл / рейз (EV). Под олл-ин рейза нет — только колл/фолд.
    pot_odds = to_call / (pot + to_call)
    ev_call = eq * (pot + to_call) - to_call
    if not all_in and best is not None and best.ev > max(0.0, ev_call):
        reason = f"рейз EV {best.ev:+.0f}; фолдов {best.fold_equity:.0%}"
        return Advice("raise", eq, pot_odds, reason, size=best.size, ev=best.ev)
    if ev_call > 0.0:
        note = "олл-ин колл" if all_in else "EV колла"
        return Advice("call", eq, pot_odds, f"{note} {ev_call:+.0f}", ev=ev_call)
    return Advice("fold", eq, pot_odds, f"эквити {eq:.0%} < шанс банка {pot_odds:.0%}")
