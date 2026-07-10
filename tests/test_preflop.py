"""Тесты префлоп-советов по позиционным чартам (engine.preflop.decide_preflop)."""

from __future__ import annotations

from poker_analyzer.engine.advisor import advise
from poker_analyzer.engine.preflop import decide_preflop
from poker_analyzer.engine.state import HandState
from poker_analyzer.parsing.events import Action, LogEvent


def ev(
    action: Action,
    *,
    player_id: int | None = None,
    amount: int | None = None,
) -> LogEvent:
    return LogEvent(time="00:00:00", action=action, player_id=player_id, amount=amount)


def _spot(
    hero: tuple[str, ...],
    *,
    hero_id: int,
    button: int = 0,
    blinds: tuple[int, int] = (1, 2),
    before: tuple[tuple[Action, int], ...] = (),
    table_size: int = 6,
) -> HandState:
    """Префлоп-состояние: Dealer, блайнды, действия ДО героя, затем рука/позиция героя."""
    s = HandState()
    s.apply(ev(Action.DEALER, player_id=button))
    for pid in blinds:
        s.apply(ev(Action.BLIND, player_id=pid, amount=10))
    for action, pid in before:
        amount = 30 if action in (Action.RAISE, Action.BET) else 10
        s.apply(ev(action, player_id=pid, amount=amount))
    s.hero_cards = hero
    s.hero_id = hero_id
    s.hero_to_act = True
    s.table_size = table_size
    return s


_FOLDS_AROUND = (
    (Action.FOLD, 3),
    (Action.FOLD, 4),
    (Action.FOLD, 5),
    (Action.FOLD, 0),
    (Action.FOLD, 1),
)


def test_bb_defends_wide_vs_raise() -> None:
    # ГЛАВНЫЙ фикс перефолда: BB защищается широко против одного рейза
    s = _spot(("K♠", "9♠"), hero_id=2, before=((Action.RAISE, 3), *_FOLDS_AROUND[1:]))
    a = decide_preflop(s, 0.5)
    assert a is not None and a.action == "call"  # K9s в диапазоне защиты BB


def test_bb_folds_trash_vs_raise() -> None:
    s = _spot(("7♦", "2♣"), hero_id=2, before=((Action.RAISE, 3), *_FOLDS_AROUND[1:]))
    a = decide_preflop(s, 0.18)  # вне защиты И эквити ниже цены банка → фолд
    assert a is not None and a.action == "fold"


def test_pot_odds_override_calls_when_priced_in() -> None:
    # рука вне диапазона CO против рейза, но эквити кроет цену банка → колл по шансам
    s = _spot(("9♦", "8♦"), hero_id=5, before=((Action.RAISE, 3), (Action.FOLD, 4)))
    a = decide_preflop(s, 0.55)
    assert a is not None and a.action == "call"


def test_pot_odds_override_folds_low_equity() -> None:
    s = _spot(("9♦", "8♦"), hero_id=5, before=((Action.RAISE, 3), (Action.FOLD, 4)))
    a = decide_preflop(s, 0.20)  # эквити ниже цены банка → остаётся фолд
    assert a is not None and a.action == "fold"


def test_sb_completes_limped_pot_by_odds() -> None:
    # SB вне диапазона открытия, но добирает лимп-пот по хорошей цене → колл, не фолд
    s = HandState()
    s.apply(ev(Action.DEALER, player_id=0))
    s.apply(ev(Action.BLIND, player_id=1, amount=5))  # SB (герой)
    s.apply(ev(Action.BLIND, player_id=2, amount=10))  # BB
    s.apply(ev(Action.CALL, player_id=3, amount=10))  # лимпер
    s.apply(ev(Action.FOLD, player_id=4))
    s.apply(ev(Action.FOLD, player_id=5))
    s.hero_cards = ("6♣", "8♣")
    s.hero_id = 1
    s.hero_to_act = True
    s.table_size = 6
    a = decide_preflop(s, 0.30)  # SB доколл 5, банк 25, цена ~17% < эквити 30%
    assert a is not None and a.action == "call"


def test_ep_open_does_not_limp_by_odds() -> None:
    # из ранней позиции по оддсам НЕ лимпуем (за спиной много игроков) → фолд вне диапазона
    s = _spot(("A♣", "9♦"), hero_id=3)  # EP, открытие
    a = decide_preflop(s, 0.55)  # даже высокое эквити не делает лимп
    assert a is not None and a.action == "fold"


def test_co_opens_in_range() -> None:
    # никто не вкладывался добровольно (фолды) → открытие; AJs в диапазоне открытия CO
    s = _spot(("A♥", "J♥"), hero_id=5, before=((Action.FOLD, 3), (Action.FOLD, 4)))
    a = decide_preflop(s, 0.5)
    assert a is not None and a.action == "raise"


def test_ep_folds_weak_open() -> None:
    s = _spot(("9♣", "7♦"), hero_id=3)  # UTG, первый доброволец
    a = decide_preflop(s, 0.5)
    assert a is not None and a.action == "fold"  # 97o вне открытия EP


def test_premium_3bets_vs_raise() -> None:
    s = _spot(("A♠", "K♠"), hero_id=5, before=((Action.RAISE, 3), (Action.FOLD, 4)))
    a = decide_preflop(s, 0.5)
    assert a is not None and a.action == "raise"  # AKs 3-бетит из CO


def test_kk_4bets_vs_3bet() -> None:
    # два агрессора (рейз + 3-бет) → vs_3bet; KK 4-бетит
    s = _spot(
        ("K♥", "K♦"),
        hero_id=2,
        before=(
            (Action.RAISE, 3),
            (Action.FOLD, 4),
            (Action.RAISE, 5),
            (Action.FOLD, 0),
            (Action.FOLD, 1),
        ),
    )
    a = decide_preflop(s, 0.5)
    assert a is not None and a.action == "raise"


def test_no_position_nonpremium_returns_none() -> None:
    # Не-премиум без позиции → None (откат на эквити-эвристику в advise).
    s = HandState(hero_cards=("9♣", "7♦"), hero_to_act=True)
    s.hero_id = 0  # нет Dealer/блайндов → позиция не определяется
    assert decide_preflop(s, 0.5) is None


def test_no_position_premium_raises() -> None:
    # Фикс D: премиум (AA) без позиции НЕ фолдим вслепую — рейз-для-вэлью, не None/фолд.
    # Регресс из живого лога: AK на 27% против лузового поля фолдился, т.к. поз=—.
    s = HandState(hero_cards=("A♠", "A♥"), hero_to_act=True)
    s.hero_id = 0  # нет Dealer/блайндов → позиция не определяется
    a = decide_preflop(s, 0.5)
    assert a is not None and a.action == "raise"


def test_advise_routes_preflop_to_chart() -> None:
    # интеграция: advise на префлопе уходит в чарт (AKs открывает CO → raise)
    s = _spot(("A♠", "K♠"), hero_id=5, before=((Action.FOLD, 3), (Action.FOLD, 4)))
    a = advise(s)
    assert a is not None and a.action == "raise"


def test_set_mine_pocket_pair_not_folded_vs_raise_when_deep() -> None:
    # Живой лог 2026-06-14: 77 против ОДНОГО рейза, экв 21% < банк 41% → фолд (грубой эвристикой).
    # Карманная пара при глубоком стеке должна сет-майнить (флоп-сет ~7.5:1, имплайды велики).
    s = _spot(("7♣", "7♠"), hero_id=5, before=((Action.RAISE, 3),))
    s.hero_stack = 3000.0  # цена рейза мала относительно стека → имплайды на сет
    a = decide_preflop(s, 0.21, to_call=30, pot=50)
    assert a is not None
    assert a.action == "call"  # НЕ фолд


def test_commit_gate_blocks_light_3bet_stackoff() -> None:
    # Фикс B (ГЛАВНЫЙ лик живого слива): A5s в BB против опена; короткий стек → 3-бет
    # коммитит ~весь стек, а эквити против вэлью-калла низкая → НЕ заходим лёгким 3-бет-шовом
    # (против не-фолдящего пула fold equity нет). A5s 3-бет-шов при SPR 1.1 стоил ~половину банка.
    s = _spot(
        ("A♥", "5♥"),
        hero_id=2,
        before=((Action.RAISE, 3), (Action.FOLD, 4), (Action.FOLD, 5), (Action.FOLD, 0)),
    )
    s.hero_stack = 100.0  # SPR ~1: 3-бет (≈90) коммитит почти весь стек
    a = decide_preflop(s, 0.35, to_call=20, pot=50, stackoff_equity=0.22)
    assert a is not None and a.action != "raise"  # лёгкий 3-бет-шов снят (флэт/фолд)


def test_commit_gate_allows_3bet_when_deep() -> None:
    # Тот же A5s, но ГЛУБОКО: 3-бет — малая доля стека → гейт не вмешивается, играем 3-бет
    # (имплайды/playability есть). Гейт SPR-ОСОЗНАН, а не «никогда не 3-бетить A5s».
    s = _spot(
        ("A♥", "5♥"),
        hero_id=2,
        before=((Action.RAISE, 3), (Action.FOLD, 4), (Action.FOLD, 5), (Action.FOLD, 0)),
    )
    s.hero_stack = 5000.0  # глубоко
    a = decide_preflop(s, 0.35, to_call=20, pot=50, stackoff_equity=0.22)
    assert a is not None and a.action == "raise"  # A5s 3-бетит из BB глубоко


def test_commit_gate_allows_value_stackoff() -> None:
    # Премиум коммитит стек, НО впереди вэлью-калла (stackoff_equity ≥ break-even) → заходим:
    # это вэлью-стек-офф (AA/KK/AK), а не лёгкий 3-бет. Гейт режет лишь блефы, не вэлью.
    s = _spot(
        ("A♠", "K♠"),
        hero_id=2,
        before=((Action.RAISE, 3), (Action.FOLD, 4), (Action.FOLD, 5), (Action.FOLD, 0)),
    )
    s.hero_stack = 100.0
    a = decide_preflop(s, 0.6, to_call=20, pot=50, stackoff_equity=0.55)
    assert a is not None and a.action == "raise"  # AKs впереди вэлью → стек-офф ОК


def test_multiway_allin_folds_dominated_hand() -> None:
    # Фикс A: KJ против ДВУХ олл-инов. Добровольные стеки лузово-пассивного пула = узкое
    # ВЭЛЬЮ (доминирует KJ), а не широкий джеммер. Раньше шло по сырой эквити vs широкие
    # пуш-диапазоны (KJ «27% > банк» → колл-в-доминацию). Теперь vs вэлью → фолд.
    s = HandState()
    s.apply(ev(Action.DEALER, player_id=0))
    s.apply(ev(Action.BLIND, player_id=1, amount=10))  # SB
    s.apply(ev(Action.BLIND, player_id=2, amount=20))  # BB (герой)
    s.apply(ev(Action.ALL_IN, player_id=3, amount=500))
    s.apply(ev(Action.ALL_IN, player_id=4, amount=500))
    s.apply(ev(Action.YOUR_TURN))
    s.hero_cards = ("K♣", "J♦")
    s.hero_id = 2
    s.hero_to_act = True
    s.hero_stack = 500.0  # колл забирает весь стек → олл-ин, ≥2 коммитнувших → мультивей-олл-ин
    s.table_size = 6
    a = advise(s)
    assert a is not None and a.action == "fold"


def test_multiway_allin_keeps_premium_vs_two_shoves_with_live_blinds() -> None:
    # Фикс блокера (committed-count): QQ против ДВУХ олл-инов при ещё живых пассивных SB/BB.
    # Считать ВСЕХ живых (4) узким вэлью занижало эквити QQ до ~0.30 → фолд (обратный лик).
    # Стек-офф даём только ДВУМ коммитнувшим → QQ ≈ 0.45 → НЕ фолд (колл/рейз). AK тоже.
    def _qq_spot(hero: tuple[str, ...]) -> HandState:
        s = HandState()
        s.apply(ev(Action.DEALER, player_id=0))
        s.apply(ev(Action.BLIND, player_id=1, amount=10))  # SB живой, не ходил (tier 0)
        s.apply(ev(Action.BLIND, player_id=2, amount=20))  # BB живой, не ходил (tier 0)
        s.apply(ev(Action.ALL_IN, player_id=3, amount=500))
        s.apply(ev(Action.ALL_IN, player_id=4, amount=500))
        s.apply(ev(Action.YOUR_TURN))
        s.hero_cards = hero
        s.hero_id = 5  # герой — отдельный игрок (за столом 0,1,2,3,4 + герой 5)
        s.hero_to_act = True
        s.hero_stack = 500.0
        s.table_size = 6
        return s

    qq = advise(_qq_spot(("Q♣", "Q♦")))
    assert qq is not None and qq.action != "fold"  # QQ vs 2 реальных шова — это колл/рейз


def test_set_mine_helper_edges() -> None:
    from poker_analyzer.engine.preflop import _hero_combo, _set_mine

    s = HandState()
    s.hero_cards = ("7♣", "7♠")
    s.hero_stack = 3000.0
    combo = _hero_combo(s.hero_cards)
    assert combo is not None  # рука читаема — сужаем тип для _set_mine
    # глубоко (30 из 3000 = 1%) → сет-майн-колл
    deep = _set_mine(s, combo, 0.21, 0.41, 1, 30)
    assert deep is not None and deep.action == "call" and "сет-майнинг" in deep.reason
    # дорого (1500 из 3000 = 50% > 7%) → None
    assert _set_mine(s, combo, 0.21, 0.41, 1, 1500) is None
    # 3-бет/4-бет (два агрессора) → None (имплайдов нет, премиум-пары 4бетят чартом)
    assert _set_mine(s, combo, 0.21, 0.41, 2, 30) is None
    # шов (facing_all_in) → None
    s._facing_all_in = True
    assert _set_mine(s, combo, 0.21, 0.41, 1, 30) is None
    s._facing_all_in = False
    # не карманная пара → None
    s.hero_cards = ("A♣", "K♠")
    ak = _hero_combo(s.hero_cards)
    assert ak is not None
    assert _set_mine(s, ak, 0.3, 0.4, 1, 30) is None
