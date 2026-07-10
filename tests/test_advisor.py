"""Тесты слоя советов (advisor): эквити + pot odds → рекомендация."""

from __future__ import annotations

from poker_analyzer.config import MULTIWAY_VALUE
from poker_analyzer.engine.advice import Advice
from poker_analyzer.engine.advisor import (
    _apply_domination,
    _cap_size,
    _decide_simple,
    _narrowed_ranges,
    _realization,
    _to_equity_int,
    _value_bet_threshold,
    _value_class,
    advise,
)
from poker_analyzer.engine.equity import card, cards
from poker_analyzer.engine.state import HandState


def _state(
    hero: tuple[str, ...],
    *,
    board: tuple[str, ...] = (),
    opponents: int = 1,
    to_call: int = 0,
    pot: int = 0,
) -> HandState:
    s = HandState(hero_cards=hero, board=board, pot=pot, hero_to_act=True)
    s.hero_id = 0
    s.live = set(range(opponents + 1))  # оппоненты + герой(0) → num_opponents == opponents
    s._bet_level = to_call  # герой ещё не вносил на улице → to_call == уровень
    return s


def _state_tier(hero: tuple[str, ...], tiers: list[int]) -> HandState:
    """Состояние на ход героя с заданными тирами агрессии оппонентов (1..N)."""
    s = HandState(hero_cards=hero, hero_to_act=True)
    s.hero_id = 0
    s.live = set(range(1, len(tiers) + 1))
    s._opp_tier = {i + 1: t for i, t in enumerate(tiers)}
    return s


def _state_hu(
    hero: tuple[str, ...],
    board: tuple[str, ...],
    *,
    tier: int = 0,
    to_call: int = 0,
    pot: int = 100,
) -> HandState:
    """Хедз-ап постфлоп: один оппонент заданного тира, борд, банк."""
    s = HandState(hero_cards=hero, board=board, pot=pot, hero_to_act=True)
    s.hero_id = 0
    s.live = {0, 1}  # герой(0) + один оппонент(1)
    s._opp_tier = {1: tier}
    s._bet_level = to_call
    return s


def test_strong_hand_raises_when_free() -> None:
    a = advise(_state(("A♠", "A♥"), opponents=1, to_call=0))
    assert a is not None
    assert a.action == "raise"  # AA ~85% > порога агрессии


def test_weak_hand_folds_to_big_bet() -> None:
    a = advise(_state(("2♣", "7♦"), opponents=1, to_call=100, pot=100))  # pot odds 0.5
    assert a is not None
    assert a.action == "fold"  # 72o ~35% < 50%


def test_not_hero_turn_returns_none() -> None:
    assert advise(HandState(hero_cards=("A♠", "A♥"))) is None  # hero_to_act ложен


def test_missing_hero_cards_returns_none() -> None:
    assert advise(HandState(hero_to_act=True)) is None


def test_duplicate_cards_returns_none_not_crash() -> None:
    # глюк распознавания дал две одинаковые карты — equity бросает ValueError,
    # advise должен вернуть None, а не упасть (иначе падал вызывающий поток)
    assert advise(_state(("A♠", "A♠"))) is None


def test_card_labels_convert_to_equity_encoding() -> None:
    assert _to_equity_int("10♦") == card("Td")  # десятка-глиф → 'Td'
    assert _to_equity_int("A♠") == card("As")
    assert _to_equity_int("7♣") == card("7c")


def test_pot_odds_reported() -> None:
    a = advise(_state(("A♠", "A♥"), opponents=1, to_call=50, pot=150))
    assert a is not None
    assert abs(a.pot_odds - 50 / 200) < 1e-9


def test_tighter_opponent_range_lowers_equity() -> None:
    # одна рука героя: против агрессора (узкий сильный диапазон) эквити ниже,
    # чем против пассивного (широкий диапазон)
    vs_loose = advise(_state_tier(("K♠", "Q♠"), [0]))  # пас → широкий диапазон
    vs_tight = advise(_state_tier(("K♠", "Q♠"), [2]))  # агрессор → узкий диапазон
    assert vs_loose is not None and vs_tight is not None
    assert vs_tight.equity < vs_loose.equity


def test_hu_postflop_size_only_on_bet() -> None:
    # хедз-ап постфлоп, первым ходить: совет bet (с размером) или check (без)
    a = advise(_state_hu(("A♠", "A♥"), ("A♦", "7♣", "2♠"), to_call=0, pot=100))
    assert a is not None
    assert a.action in ("bet", "check")
    assert (a.size is not None) == (a.action == "bet")  # размер только у ставки
    if a.action == "bet":
        assert a.size and a.size > 0 and a.ev is not None


def test_hu_postflop_trash_facing_bet_folds() -> None:
    # мусор без дро на AKQ против ставки агрессора → фолд
    a = advise(_state_hu(("2♣", "7♦"), ("A♦", "K♣", "Q♠"), tier=2, to_call=80, pot=100))
    assert a is not None
    assert a.action == "fold"


def _multiway_postflop(
    hero: tuple[str, ...],
    board: tuple[str, ...],
    *,
    tiers: list[int],
    pot: int,
    to_call: int,
    aggression: float,
) -> HandState:
    """Мультивей постфлоп: оппоненты с тирами, банк/доколл и размер агрессии на улице."""
    s = HandState(hero_cards=hero, board=board, pot=pot, hero_to_act=True)
    s.hero_id = 0
    s.live = {0, *(i + 1 for i in range(len(tiers)))}
    s._opp_tier = {i + 1: t for i, t in enumerate(tiers)}
    s._bet_level = to_call  # герой ещё не вносил на улице → to_call == уровень
    s._street_aggression = aggression
    return s


def test_top_pair_calls_postflop_all_in() -> None:
    # регресс из живого лога (2026-06-12): «чит против ассистента — тупо олл-инить».
    # Шов сужал агрессора до натсов (порог + поляризация), топ-пара получала экв ~6%
    # и фолдила при отличной цене. Постфлоп-шов = ШИРОКИЙ пуш-диапазон («как коллер»,
    # без поляризации) — топ-пара обязана коллировать.
    s = _multiway_postflop(
        ("K♠", "10♣"),
        ("K♥", "7♦", "2♣"),
        tiers=[2],
        pot=6_000,
        to_call=3_000,
        aggression=2.0,  # овербет-шов
    )
    s._facing_all_in = True  # верхний уровень ставки поставлен олл-ином
    s.hero_stack = 50_000.0  # герой глубокий: его колл НЕ олл-ин (доминация активна)
    a = advise(s)
    assert a is not None
    assert a.action in ("call", "raise"), a.reason  # не фолд: пуш-диапазон широкий


def test_jam_widens_postflop_aggressor_range() -> None:
    # Шов (jam=True) даёт агрессору заметно ШИРЕ диапазон, чем обычная агрессия с
    # поляризацией: модель «олл-ин = натсы» делала фолд-эксплойт.
    board = [_to_equity_int(c) for c in ("K♥", "7♦", "2♣")]
    narrow = _narrowed_ranges([2], board, aggression=2.0, jam=False)[0]
    wide = _narrowed_ranges([2], board, aggression=2.0, jam=True)[0]
    assert len(wide) > len(narrow)


def test_context_selects_base_range() -> None:
    # Префлоп-контекст оппонента выбирает базовый диапазон точнее тира: лимпер (широкий)
    # → суженный диапазон СТРОГО шире, чем у опен-рейзера/3-беттора на той же доске.
    board = cards("Kh 7d 2c")
    limp = len(_narrowed_ranges([1], board, contexts=["limp"])[0])
    call_vs = len(_narrowed_ranges([1], board, contexts=["call_vs_open"])[0])
    open_early = len(_narrowed_ranges([2], board, contexts=["open_early"])[0])
    threebet = len(_narrowed_ranges([2], board, contexts=["3bet"])[0])
    assert limp > call_vs > open_early
    assert open_early > threebet  # 3-бет — самый узкий
    # unknown / без контекста → откат на тир-диапазон (как раньше).
    tier = len(_narrowed_ranges([1], board)[0])
    assert len(_narrowed_ranges([1], board, contexts=["unknown"])[0]) == tier


def test_dominated_flush_folds_to_big_aggression() -> None:
    # 10-high флеш на монотонной доске против ТЯЖЁЛОЙ агрессии (овербет) → фолд по доминации
    # (reverse implied odds). Под лузовый пул (калибровка 2026-06-13) полубанковый бет флеш
    # уже коллит — он бьёт широкий диапазон; фича доминации срабатывает на крупной ставке.
    s = _multiway_postflop(
        ("9♥", "4♥"),
        ("3♥", "6♥", "10♥", "4♠"),
        tiers=[1, 2, 2, 1],
        pot=700,
        to_call=700,
        aggression=1.8,
    )
    a = advise(s)
    assert a is not None
    assert a.action == "fold"
    assert "доминаци" in a.reason


def test_nut_flush_not_folded_by_domination() -> None:
    # натсовый флеш против той же агрессии — не доминирован → не фолдим
    s = _multiway_postflop(
        ("A♥", "K♥"),
        ("3♥", "6♥", "10♥", "4♠"),
        tiers=[1, 2, 2, 1],
        pot=700,
        to_call=350,
        aggression=1.5,
    )
    a = advise(s)
    assert a is not None
    assert a.action != "fold"


def test_domination_downgrades_raise_to_call_or_fold() -> None:
    # 10-high флеш «крушится» старшими флешами агрессора. Раньше рейз проходил мимо
    # доминации (проверялись только call/check) и ре-рейзил в крушащий диапазон.
    board = cards("3h 6h Th 4s")
    hero = cards("9h 4h")  # 10-high флеш
    # диапазон агрессора — только СТАРШИЕ флеши (две черви выше десятки): бьют героя
    aggr = [
        (card("Ah"), card("Kh")),
        (card("Ah"), card("Qh")),
        (card("Kh"), card("Qh")),
        (card("Ah"), card("Jh")),
    ]
    base = Advice("raise", 0.9, 0.2, "исходный рейз", size=2000.0)
    out = _apply_domination(base, hero, board, [aggr], [2], 1.5, 712, 2841)
    assert out.action != "raise"  # рейз в крушащий диапазон снят
    assert out.size is None  # размер рейза сброшен


def test_domination_keeps_value_raise() -> None:
    # натсовый флеш бьёт весь диапазон агрессора → рейз остаётся рейзом
    board = cards("3h 6h Th 4s")
    hero = cards("Ah Kh")  # натсовый флеш
    aggr = [(card("Qh"), card("Jh")), (card("Qh"), card("9h")), (card("Jh"), card("9h"))]
    base = Advice("raise", 0.95, 0.2, "вэлью-рейз", size=2000.0)
    out = _apply_domination(base, hero, board, [aggr], [2], 1.5, 712, 2841)
    assert out.action == "raise" and out.size == 2000.0


def test_bet_size_narrows_aggressor_range() -> None:
    board = cards("3h 6h Th 4s")
    small = _narrowed_ranges([2], board, 0.0)[0]
    big = _narrowed_ranges([2], board, 1.5)[0]
    assert len(big) < len(small)  # крупная ставка → уже (поляризованный) диапазон агрессора


# --- 1.1 реализуемая эквити (REq) --------------------------------------------


def _pos_state(*, ip: bool) -> HandState:
    """Минимальное состояние с детерминированной позицией героя (IP/OOP) для REq."""
    s = HandState(hero_to_act=True)
    s.hero_id = 0
    if ip:
        s._button_id = 0  # герой на баттоне → всегда в позиции
    else:
        s._button_id = 1
        s.live = {0, 1}
        s._seen_order = [0, 1]  # герой ходит раньше последнего живого → вне позиции
    return s


def test_realization_strong_beats_air_and_oop_not_above_ip() -> None:
    board = cards("Ah 7h 2c")
    trips = cards("7c 7d")  # тройка семёрок (7h на борде) — сильная готовая рука
    air = cards("Kd Qc")  # старшая карта без пары и дро — воздух
    strong_ip = _realization(_pos_state(ip=True), trips, board, 1)
    strong_oop = _realization(_pos_state(ip=False), trips, board, 1)
    air_ip = _realization(_pos_state(ip=True), air, board, 1)
    assert strong_oop <= strong_ip  # вне позиции реализуем не больше
    assert air_ip < strong_ip  # воздух реализует хуже готовой руки


def test_realization_multiway_penalizes_weak_made() -> None:
    board = cards("Ah 7h 2c")
    pair = cards("8c 7d")  # пара семёрок — ВТОРАЯ пара (туз старше), слабый бакет
    heads_up = _realization(_pos_state(ip=True), pair, board, 1)
    multiway = _realization(_pos_state(ip=True), pair, board, 3)
    assert multiway < heads_up  # слабую готовую руку тяжело реализовать против многих


def test_realization_top_pair_and_overpair_beat_weak_pair() -> None:
    board = cards("Jh 7d 2c")
    top_pair = cards("As Jd")  # валет спарил СТАРШУЮ карту борда — топ-пара
    overpair = cards("Qs Qh")  # карманные дамы выше борда (J) — оверпара
    weak_pair = cards("Ad 7s")  # семёрки — вторая пара
    r_top = _realization(_pos_state(ip=True), top_pair, board, 1)
    r_over = _realization(_pos_state(ip=True), overpair, board, 1)
    r_weak = _realization(_pos_state(ip=True), weak_pair, board, 1)
    assert r_top > r_weak  # топ-пара реализует больше второй пары
    assert r_over > r_weak  # оверпара — тоже


def test_realization_underpair_is_weak_not_top() -> None:
    board = cards("Kh 7d 2c")
    underpair = cards("9s 9h")  # девятки НИЖЕ короля — андерпара, не оверпара
    weak_pair = cards("Ad 7s")  # вторая пара (семёрки)
    r_under = _realization(_pos_state(ip=True), underpair, board, 1)
    r_weak = _realization(_pos_state(ip=True), weak_pair, board, 1)
    assert r_under == r_weak  # андерпара = тот же слабый бакет, что вторая пара


def test_realization_top_pair_no_multiway_penalty() -> None:
    board = cards("Kh 7d 2c")
    top_pair = cards("As Kd")  # топ-пара королей
    heads_up = _realization(_pos_state(ip=True), top_pair, board, 1)
    multiway = _realization(_pos_state(ip=True), top_pair, board, 4)
    assert multiway == heads_up  # топ-пара БЕЗ мультивей-штрафа (в отличие от слабой пары)


def test_realization_strong_draw_beats_bare_gutshot() -> None:
    oesd = _realization(_pos_state(ip=True), cards("9c 8d"), cards("7h 6s 2c"), 1)  # OESD (8 аутов)
    flush = _realization(_pos_state(ip=True), cards("Ah 5h"), cards("Kh 8h 2c"), 1)  # флеш-дро (9)
    gutshot = _realization(_pos_state(ip=True), cards("Jc 9d"), cards("Qs 8h 2c"), 1)  # гатшот (4)
    assert oesd > gutshot  # OESD реализует лучше голого гатшота
    assert flush > gutshot  # флеш-дро — тоже
    assert oesd == flush  # флеш-дро и OESD — один бакет «сильное дро»


def test_realization_grows_toward_river_for_made_hand() -> None:
    # реализация готовой руки РАСТЁТ к риверу: на ривере шоудаун — забирает всю эквити,
    # на флопе её ещё могут выбить. Множитель улицы флоп не трогает (1.0), тёрн/ривер ↑.
    pair = cards("Ad Kc")  # пара тузов — одна и та же категория на всех улицах
    flop = cards("Ah 7h 2c")
    turn = cards("Ah 7h 2c 3d")
    river = cards("Ah 7h 2c 3d 9s")
    r_flop = _realization(_pos_state(ip=True), pair, flop, 1)
    r_turn = _realization(_pos_state(ip=True), pair, turn, 1)
    r_river = _realization(_pos_state(ip=True), pair, river, 1)
    assert r_flop <= r_turn <= r_river  # не убывает по улицам
    assert r_river > r_flop  # ривер реализует строго больше флопа (меньше перефолда «впереди»)


# --- 1.5 блокеры героя в доминации -------------------------------------------


def test_domination_all_value_blocked_returns_unchanged() -> None:
    # весь вэлью-диапазон агрессора использует карты героя → расклады невозможны,
    # доминация не применяется (совет остаётся прежним объектом)
    board = cards("Ah Kd 7c")
    base = Advice("call", 0.2, 0.4, "исходный")
    hero = cards("Qs Qd")
    aggr = [(card("Qs"), card("Qh")), (card("Qd"), card("Qc"))]  # обе дамы заняты героем
    out = _apply_domination(base, hero, board, [aggr], [2], 1.0, 50, 50)
    assert out is base


# --- SPR / стек: эффективная цена колла и олл-ин без штрафов улиц --------------


def test_cap_size_limits_bet_to_stack() -> None:
    a = Advice("bet", 0.6, 0.0, "ставка", size=500.0, ev=100.0)
    capped = _cap_size(a, 300.0)  # стек 300 < 500 → кап до стека (олл-ин)
    assert capped is not None and capped.size == 300.0 and "олл-ин" in capped.reason
    big_enough = _cap_size(a, 800.0)  # стек больше размера — без изменений
    assert big_enough is not None and big_enough.size == 500.0
    assert _cap_size(a, None) is a  # нет стека — совет не трогаем


def test_short_stack_lowers_effective_pot_odds() -> None:
    # оверщов 1900 при банке 2000, но у героя остаток 100 → колл-олл-ин «за меньше»:
    # эффективная цена банка НИЖЕ полной (рискуем 100, а не 1900)
    board = ("A♦", "K♣", "7♠", "2♥", "3♣")
    full = advise(_state_hu(("9♥", "9♦"), board, tier=2, to_call=1900, pot=2000))
    short_state = _state_hu(("9♥", "9♦"), board, tier=2, to_call=1900, pot=2000)
    short_state.hero_stack = 100.0  # остаток 100 ≪ доколл 1900
    short = advise(short_state)
    assert full is not None and short is not None
    # state.pot (2000) включает шов 1900; герой коллит олл-ин 100, избыток 1800 возвращается
    # виллану. Оспоримый банк = 2000−1800=200, эфф. цена = 100/(200+100) = 0.333 (НЕ полная 0.487
    # и НЕ 0.048 — герой не выигрывает возвращённый избыток). Фиксируем точное значение.
    assert abs(short.pot_odds - 100 / 300) < 1e-6  # эффективная цена банка
    assert abs(full.pot_odds - 1900 / 3900) < 1e-6  # полная цена (без учёта стека)


def test_jam_widens_preflop_aggressor_range() -> None:
    # префлоп-шов: пуш-диапазон агрессора шире рейз-диапазона (иначе перефолд средних пар)
    normal = _narrowed_ranges([2], [], jam=False)[0]
    jam = _narrowed_ranges([2], [], jam=True)[0]
    assert len(jam) > len(normal)
    # Постфлоп шов тоже РАСШИРЯЕТ диапазон агрессора (фикс «тупо олл-инить» 2026-06-12:
    # коллерский порог без поляризации) — детальнее в test_jam_widens_postflop_aggressor_range.
    board = cards("3h 6h Th")
    assert len(_narrowed_ranges([2], board, jam=True)[0]) >= len(
        _narrowed_ranges([2], board, jam=False)[0]
    )


def test_77_better_equity_vs_jammer_than_vs_tight_raiser() -> None:
    from poker_analyzer.engine.advisor import _JAMMER_RANGE, _TIER_RANGES
    from poker_analyzer.engine.equity import equity_vs_ranges

    hero = cards("7h 7d")
    vs_jam = equity_vs_ranges(hero, [], [_JAMMER_RANGE], iterations=30_000).equity
    vs_tight = equity_vs_ranges(hero, [], [_TIER_RANGES[2]], iterations=30_000).equity
    assert vs_jam > vs_tight  # против пуш-диапазона 77 заметно сильнее


def test_ato_cutoff_open_with_table_size_from_view() -> None:
    # регресс ATo из лога: герой не опознан, но размер стола задан селектором вида (5-max)
    # → позиция CO выводится на «Ваш ход» → чарт CO открывает ATo рейзом (а не фолд эвристикой)
    from poker_analyzer.parsing.events import Action, LogEvent

    s = HandState(hero_cards=("10♠", "A♥"))
    for e in (
        LogEvent("00:00:01", Action.DEALER, player_id=0),
        LogEvent("00:00:02", Action.BLIND, player_id=1, amount=5),
        LogEvent("00:00:03", Action.BLIND, player_id=2, amount=10),
        LogEvent("00:00:04", Action.CALL, player_id=3, amount=10),
        LogEvent("00:00:05", Action.YOUR_TURN),
    ):
        s.apply(e)
    s.table_size = 5  # вид стола 5-max
    assert s.hero_position() == "CO"
    a = advise(s)
    assert a is not None and a.action == "raise"


def test_pocket_pair_open_spot_not_folded_without_hero_id() -> None:
    # регресс «сюра» из лога: 8♠8♥ на префлопе фолдились, потому что герой ещё не опознан
    # (не блайнд/не баттон) → поз=None → чарт молчал → эвристика 22%<29%. Теперь позиция
    # выводится на «Ваш ход» из порядка действий → чарт EP открывает 88 рейзом.
    from poker_analyzer.parsing.events import Action, LogEvent

    s = HandState(hero_cards=("8♠", "8♥"))
    for e in (
        LogEvent("00:00:01", Action.DEALER, player_id=0),
        LogEvent("00:00:02", Action.BLIND, player_id=1, amount=25),
        LogEvent("00:00:03", Action.BLIND, player_id=2, amount=50),
        LogEvent("00:00:04", Action.CALL, player_id=3, amount=50),
        LogEvent("00:00:05", Action.YOUR_TURN),
    ):
        s.apply(e)
    s.table_size = 9
    a = advise(s)
    assert a is not None
    assert a.action == "raise"  # 88 в EP-чарте открытия — не фолд


def test_free_check_never_folded() -> None:
    # инвариант: при to_call=0 совет «fold» невозможен (страховка от мисклассификации) —
    # фолд при to_call=0 отдал бы руку даром
    a = advise(_state(("2♣", "7♦"), opponents=3, to_call=0, pot=100))
    assert a is not None
    assert a.action != "fold"


def test_all_in_call_skips_domination() -> None:
    # тот же доминированный флеш, что фолдится по доминации, но колл = весь стек (олл-ин):
    # будущих улиц нет → штраф reverse implied odds НЕ применяется
    s = _multiway_postflop(
        ("9♥", "4♥"),
        ("3♥", "6♥", "10♥", "4♠"),
        tiers=[1, 2, 2, 1],
        pot=700,
        to_call=350,
        aggression=1.0,
    )
    s.hero_stack = 300.0  # остаток 300 < доколл 350 → олл-ин
    a = advise(s)
    assert a is not None
    assert "доминаци" not in a.reason  # на олл-ине доминацию не считаем


def test_raise_advice_becomes_call_when_call_is_all_in() -> None:
    # регресс из живого лога (2026-06-12): A♥K♦ на BTN, доколл = остатку стека героя.
    # Чарт говорил «3-бет BTN», но рейза в игре НЕ СУЩЕСТВУЕТ — кнопка Raise неактивна,
    # Совет должен переоцениться как колл-олл-ин.
    from poker_analyzer.parsing.events import Action, LogEvent

    s = HandState(hero_cards=("A♥", "K♦"))
    for e in (
        LogEvent("00:00:01", Action.DEALER, player_id=0, is_hero=True),  # герой на баттоне
        LogEvent("00:00:02", Action.BLIND, player_id=1, amount=25),  # SB
        LogEvent("00:00:03", Action.BLIND, player_id=2, amount=50),  # BB
        LogEvent("00:00:04", Action.RAISE, player_id=3, amount=1000),  # пуш-рейз 20бб
        LogEvent("00:00:05", Action.FOLD, player_id=1),  # блайнды ушли — хедз-ап
        LogEvent("00:00:06", Action.FOLD, player_id=2),
        LogEvent("00:00:07", Action.YOUR_TURN),
    ):
        s.apply(e)
    s.table_size = 5
    s.hero_stack = 800.0  # остаток 800 < доколл 1000 → колл сам по себе олл-ин
    a = advise(s)
    assert a is not None
    assert a.action == "call"  # не «raise»: AK легко оправдывает цену колл-олл-ина
    assert a.size is None
    assert "олл-ин" in a.reason


def test_flat_call_chart_folds_to_all_in_overbet() -> None:
    # регресс из живого лога: 8♠10♠ на BTN против олл-ина CO на 1000 (20бб). Флэт-колл чарта
    # выдавал «колл BTN» (экв 18% < банк 47%) и герой доколливал шов. Шов снимает имплайды →
    # флэт переоценивается по прямым шансам банка → фолд.
    from poker_analyzer.parsing.events import Action, LogEvent

    s = HandState(hero_cards=("8♠", "10♠"))
    for e in (
        LogEvent("00:00:01", Action.DEALER, player_id=0, is_hero=True),  # герой на баттоне
        LogEvent("00:00:02", Action.BLIND, player_id=1, amount=25),  # SB
        LogEvent("00:00:03", Action.BLIND, player_id=2, amount=50),  # BB
        LogEvent("00:00:04", Action.CALL, player_id=3, amount=50),  # EP лимп
        LogEvent("00:00:05", Action.ALL_IN, player_id=4, amount=1000),  # CO шов 20бб
        LogEvent("00:00:06", Action.YOUR_TURN),
    ):
        s.apply(e)
    s.table_size = 5
    assert s.hero_position() == "BTN"
    assert s.facing_all_in is True  # верхний уровень поставлен олл-ином
    a = advise(s)
    assert a is not None
    assert a.action == "fold"  # 8To не доколливает шов вне шансов банка


def test_flat_call_chart_still_calls_normal_raise() -> None:
    # контраст к шову: тот же 8♠10♠ на BTN против ОБЫЧНОГО рейза (за ним есть стек → имплайды)
    # остаётся флэт-коллом чарта — гейт против олл-ина не должен ломать нормальный флэт.
    from poker_analyzer.parsing.events import Action, LogEvent

    s = HandState(hero_cards=("8♠", "10♠"))
    for e in (
        LogEvent("00:00:01", Action.DEALER, player_id=0, is_hero=True),
        LogEvent("00:00:02", Action.BLIND, player_id=1, amount=25),
        LogEvent("00:00:03", Action.BLIND, player_id=2, amount=50),
        LogEvent("00:00:04", Action.RAISE, player_id=4, amount=150),  # CO рейз 3бб (не шов)
        LogEvent("00:00:05", Action.YOUR_TURN),
    ):
        s.apply(e)
    s.table_size = 5
    assert s.hero_position() == "BTN"
    assert s.facing_all_in is False
    a = advise(s)
    assert a is not None
    assert a.action == "call"  # обычный рейз → флэт-колл чарта в силе


# --- мультивей-вэлью-бет на инициативе (главный вэлью-лик) --------------------


def test_value_class_top_pair_overpair_strong_and_none() -> None:
    assert _value_class(cards("As Kd"), cards("Kh 7d 2c")) == "top"  # топ-пара королей
    assert _value_class(cards("Qs Qh"), cards("Jh 7d 2c")) == "top"  # оверпара
    assert _value_class(cards("Jc 7c"), cards("Jh 7d 2c")) == "strong"  # две пары
    assert _value_class(cards("Ad 7s"), cards("Kh 7d 2c")) is None  # вторая пара — не вэлью
    assert _value_class(cards("9c 8d"), cards("7h 6s 2c")) is None  # дро — не вэлью


def test_value_bet_threshold_drops_with_opponents_but_not_below_floor() -> None:
    assert _value_bet_threshold(2) > _value_bet_threshold(4)  # больше плательщиков — ниже планка
    assert _value_bet_threshold(9) == MULTIWAY_VALUE.floor  # но не ниже флора


def test_decide_simple_value_bets_made_hand_on_initiative() -> None:
    # топ-пара на инициативе в мультивее: вэлью-бет (раньше чек при eq<0.66)
    adv = _decide_simple(0.55, 100.0, 0.0, 3, "top")
    assert adv.action == "bet"
    assert adv.size == MULTIWAY_VALUE.size_top * 100.0  # доля банка по силе руки
    strong = _decide_simple(0.60, 100.0, 0.0, 3, "strong")
    assert strong.action == "bet"
    assert strong.size == MULTIWAY_VALUE.size_strong * 100.0  # две пары+ бьют крупнее


def test_decide_simple_checks_weak_or_behind_on_initiative() -> None:
    assert _decide_simple(0.55, 100.0, 0.0, 3, None).action == "check"  # не готовая → чек
    # Топ-пара ЯВНО ниже break-even 1/(N+1) (порог 4-вей = max(floor, 1/5+margin)) → чек.
    # NB: при экв ~0.30 4-вей теперь ВЭЛЬЮ-БЕТ (тонко, но +EV против липкого пула — фикс недо-бета).
    assert _decide_simple(0.18, 100.0, 0.0, 4, "top").action == "check"  # явно позади → чек
    # без value_class и ниже RAISE_EQUITY — прежнее поведение (чек), не фолд бесплатно
    assert _decide_simple(0.40, 100.0, 0.0, 2, None).action == "check"


def test_decide_simple_value_bet_only_on_initiative_not_facing_bet() -> None:
    # есть ставка перед героем (to_call>0): вэлью-бет не предлагается — обычная логика
    adv = _decide_simple(0.55, 100.0, 40.0, 3, "top")
    assert adv.action in ("call", "fold", "raise")  # НЕ bet на инициативе
