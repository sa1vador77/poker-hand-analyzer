"""Префлоп-решение по позиционным чартам (см. :class:`config.PreflopCharts`).

Решение берётся ЛУКАПОМ: позиция героя (:meth:`HandState.hero_position`) + ситуация
(нет агрессии / один рейз / 3-бет — по тирам оппонентов) → действие из чарта. Чарт — это
готовая компиляция ценности руки через позицию, инициативу и постфлоп-плей; именно её
одноуличное «эквити vs pot odds» на префлопе систематически недооценивает (перефолд).
Эквити считается отдельно (в advisor) и сюда передаётся ТОЛЬКО для показа в окне.
"""

from __future__ import annotations

from poker_analyzer.config import (
    COMMIT_GATE,
    PREFLOP_CHARTS,
    PREFLOP_ODDS_MARGIN,
    PREMIUM_FLOOR,
    SET_MINE_MAX_FRAC,
)
from poker_analyzer.engine.advice import Advice
from poker_analyzer.engine.equity import card_from_glyph
from poker_analyzer.engine.ranges import Combo, parse_range
from poker_analyzer.engine.state import HandState

# Предпарсенные чарты: позиция → множество комбо (для быстрой проверки принадлежности).
_OPEN: dict[str, frozenset[Combo]] = {
    pos: frozenset(parse_range(s)) for pos, s in PREFLOP_CHARTS.opens.items()
}
# Премиум-руки для случая «позиция не определилась» (поз=—): их играем рейзом-для-вэлью,
# а не отдаём грубой эквити-эвристике (она фолдила AK против лузового поля по сырой цене).
_PREMIUM_FLOOR: frozenset[Combo] = frozenset(parse_range(PREMIUM_FLOOR))
_VS_RAISE: dict[str, tuple[frozenset[Combo], frozenset[Combo]]] = {
    pos: (frozenset(parse_range(call)), frozenset(parse_range(raise_)))
    for pos, (call, raise_) in PREFLOP_CHARTS.vs_raise.items()
}
_VS_3BET: dict[str, tuple[frozenset[Combo], frozenset[Combo]]] = {
    pos: (frozenset(parse_range(call)), frozenset(parse_range(raise_)))
    for pos, (call, raise_) in PREFLOP_CHARTS.vs_3bet.items()
}


def _hero_combo(cards: tuple[str, ...]) -> Combo | None:
    """Две карты героя (глифы) → нормализованная пара ``int`` или ``None``."""
    if len(cards) != 2:
        return None
    try:
        a, b = card_from_glyph(cards[0]), card_from_glyph(cards[1])
    except (KeyError, ValueError, IndexError):
        return None
    return (a, b) if a < b else (b, a)


def _set_mine(
    state: HandState, combo: Combo, equity: float, pot_odds: float, aggressors: int, to_call: float
) -> Advice | None:
    """Колл на сет-майнинг карманной парой против ОДНОГО рейза при глубоком стеке, иначе ``None``.

    Карманная пара флопает сет ~7.5:1; против лузово-пассивного пула (платит широко) имплайды
    велики, поэтому дешёвый колл (цена ≤ :data:`config.SET_MINE_MAX_FRAC` стека) профитен. Берёт
    только ОДИН обычный рейз (не 3бет/шов — там имплайдов нет, премиум-пары 4бетят чартом раньше)
    и требует известного стека. ПОЗИЦИОННО-АГНОСТИЧНО: срабатывает и когда позиция не определилась
    (``поз=—`` — частый случай против рейза из поздней позиции, где иначе чарт молчит → перефолд
    карманной пары грубой эквити-эвристикой; живой лог: 77 против рейза, экв 21% < банк 41% → фолд).
    """
    if combo[0] // 4 != combo[1] // 4:  # не карманная пара (ранги карт различны)
        return None
    if aggressors != 1 or to_call <= 0 or state.facing_all_in or SET_MINE_MAX_FRAC <= 0:
        return None  # только один обычный рейз; против шова имплайдов нет
    eff = state.hero_remaining if state.hero_remaining is not None else state.hero_stack
    if eff is None or eff <= 0 or to_call > SET_MINE_MAX_FRAC * eff:
        return None  # стек неизвестен / колл дорог относительно стека (имплайдов не хватит)
    return Advice("call", equity, pot_odds, "сет-майнинг (карманная пара, глубоко)")


def _commit_ok(state: HandState, pot: float, stackoff_equity: float | None) -> bool:
    """Можно ли РЕЙЗИТЬ, не превращаясь в проигрышный стек-офф? (гейт коммита, фикс B).

    Оценка размера чарт-рейза — ``open_raise_mult × уровень``. Если
    рейз коммитит ≥ ``raise_commit_frac`` эффективного стека, против НЕ-фолдящего пула это
    заход всем стеком: оправдан, лишь когда эквити героя против узкого ВЭЛЬЮ-диапазона,
    который коллит (``stackoff_equity``), ≥ break-even (риск/банк, учитывает мёртвые деньги).
    Без снимка стека (``hero_remaining is None``) или без эквити vs вэлью — НЕ вмешиваемся
    (возвращаем True): гейт работает только когда есть чем судить.
    """
    rem = state.hero_remaining
    if rem is None or rem <= 0:
        return True
    eff = state.hero_street_in + rem
    raise_to = COMMIT_GATE.open_raise_mult * (state.to_call + state.hero_street_in)
    if eff <= 0 or raise_to < COMMIT_GATE.raise_commit_frac * eff:
        return True  # рейз не коммитит стек — обычный 3-бет с деньгами за спиной, не трогаем
    if stackoff_equity is None:
        return True  # нет эквити vs вэлью-диапазон — судить нечем
    denom = (
        pot + 2.0 * rem
    )  # break-even олл-ина: рискуем rem, виллан матчит rem, банк = мёртвые+2·rem
    breakeven = rem / denom if denom > 0 else 1.0
    return stackoff_equity >= breakeven + COMMIT_GATE.stackoff_margin


def _flat_commits(state: HandState, to_call: float) -> bool:
    """Колл забирает ≥ ``flat_commit_frac`` эффективного стека → низкий SPR, имплайдов нет.

    Флэт-колл чартов откалиброван под обычный опен с деньгами за спиной (playability,
    имплайд-оддсы). Когда колл сам по себе коммитит большую долю стека, спекулятивный флэт
    переоценён (реализация падает) — трактуем как ``committed`` (флэт переоценивается по
    прямым шансам банка, как против шова).
    """
    rem = state.hero_remaining
    if rem is None:
        return False
    eff = state.hero_street_in + rem
    return eff > 0 and to_call >= COMMIT_GATE.flat_commit_frac * eff


def decide_preflop(
    state: HandState,
    equity: float,
    *,
    to_call: float | None = None,
    pot: float | None = None,
    committed: bool = False,
    stackoff_equity: float | None = None,
) -> Advice | None:
    """Префлоп-совет по чартам, или ``None`` если позиция/рука не определены (→ эвристика).

    :param equity: эквити героя для ПОКАЗА (решение берётся из чарта, не из эквити).
    :param to_call: цена колла; ``None`` → ``state.to_call``. Передаётся «эффективной» (кап по
        остатку стека), чтобы оверрайд по pot odds против шорт-стек-шова считался честно.
    :param pot: банк для pot odds; ``None`` → ``state.pot`` (эффективный — без избытка оверщова).
    :param committed: имплайдов нет — оппонент в олл-ине (``state.facing_all_in``) или колл
        забирает весь стек героя. Флэт-колл чарта откалиброван под обычный опен (2.5–3бб) с
        деньгами за спиной; против шова он переоценивается по прямым шансам банка (`8To` на BTN
        иначе доколливал 20бб-шов при экв 18% < 47%).
    :param stackoff_equity: эквити героя против узкого ВЭЛЬЮ-диапазона (
        :data:`config.OpponentRanges.stackoff`) — для гейта коммита (фикс B): чарт-рейз,
        коммитящий стек, оправдан только при перевесе над этим вэлью. ``None`` → гейт не
        вмешивается (нет снимка стека). Считается в :func:`advisor.advise`.
    """
    combo = _hero_combo(state.hero_cards)
    if combo is None:
        return None

    aggressors = sum(1 for t in state.opponent_tiers() if t == 2)  # сколько оппонентов рейзило
    to_call = state.to_call if to_call is None else to_call
    pot = state.pot if pot is None else pot
    pot_odds = to_call / (pot + to_call) if to_call > 0 else 0.0
    # Низкий SPR: спекулятивный флэт переоценён (имплайдов за спиной нет) — как committed.
    committed = committed or _flat_commits(state, to_call)

    def gated_raise(reason: str) -> Advice | None:
        """Рейз, если он не превращается в проигрышный стек-офф (гейт коммита), иначе None."""
        if _commit_ok(state, pot, stackoff_equity):
            return Advice("raise", equity, pot_odds, reason)
        return None  # коммитящий рейз без перевеса над вэлью-каллом → провал в флэт/фолд

    pos = state.hero_position()
    if pos is None:
        # Позиция не определилась (поз=—) → чарт молчит. Премиум фолдить ВСЛЕПУЮ нельзя:
        # AK на 27% против лузового поля = рейз-для-вэлью (через гейт коммита), не фолд.
        if combo in _PREMIUM_FLOOR:
            r = gated_raise("премиум (поз=—)")
            if r is not None:
                return r
            if to_call > 0 and equity >= pot_odds + PREFLOP_ODDS_MARGIN:
                return Advice("call", equity, pot_odds, "премиум колл по шансам (поз=—)")
        # Иначе спасаем хотя бы сет-майнинг карманной пары против рейза (иначе фолд, как 77 в логе).
        return _set_mine(state, combo, equity, pot_odds, aggressors, to_call)

    if aggressors == 0:  # нет рейза впереди — спот открытия
        if combo in _OPEN.get(pos, frozenset()):
            r = gated_raise(f"открытие {pos}")
            if r is not None:
                return (
                    r  # иначе опен коммитит стек супершорта без перевеса → провал в фолд/чек ниже
                )
        if pos == "BB" and to_call <= 0:
            return Advice("check", equity, 0.0, "BB добор бесплатно")
        # В блайндах (мало игроков позади → эквити надёжна) при хорошей цене банка —
        # колл/комплит по прямым шансам, а не фолд (напр. SB добирает лимп-пот). Из ранних
        # позиций НЕ лимпуем по оддсам: за спиной много игроков и эквити завышена.
        if pos in ("SB", "BB") and to_call > 0 and equity >= pot_odds + PREFLOP_ODDS_MARGIN:
            return Advice("call", equity, pot_odds, f"колл по шансам банка {pos}")
        return Advice("fold", equity, pot_odds, f"вне открытия {pos}")

    # Есть агрессия: один рейз → vs_raise, рейз+3бет → vs_3bet. 3bet/4bet проверяем ПЕРВЫМ.
    chart = _VS_3BET if aggressors >= 2 else _VS_RAISE
    call_set, raise_set = chart.get(pos, (frozenset(), frozenset()))
    tag = "4-бет" if aggressors >= 2 else "3-бет"
    if combo in raise_set:
        r = gated_raise(f"{tag} {pos}")
        if r is not None:
            return (
                r  # иначе лёгкий 3-бет коммитит стек против не-фолдящего пула → провал в флэт/фолд
            )
    # Флэт-колл чарта — только против обычного рейза. Без имплайдов (шов или низкий SPR: будущих
    # улиц/денег за спиной нет) флэт-коллируемую руку оставляем шансам банка ниже: 8To на BTN
    # иначе доколливал 20бб-шов; A5s в BB иначе флэтил коммитящий 3x при SPR 1.1.
    if combo in call_set and not (committed and equity < pot_odds + PREFLOP_ODDS_MARGIN):
        return Advice("call", equity, pot_odds, f"колл {pos}")
    # Чарт за фолд, но если эквити кроет цену банка (+буфер реализации) — коллим по шансам
    # банка, а не слепо фолдим руку вне диапазона (учёт прямых pot odds против рейза).
    if to_call > 0 and equity >= pot_odds + PREFLOP_ODDS_MARGIN:
        return Advice("call", equity, pot_odds, f"колл по шансам банка {pos}")
    sm = _set_mine(state, combo, equity, pot_odds, aggressors, to_call)  # карманная пара → сет-майн
    if sm is not None:
        return sm
    return Advice("fold", equity, pot_odds, f"фолд {pos}")
