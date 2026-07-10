"""Тесты накопления состояния раздачи (HandState)."""

from __future__ import annotations

from poker_analyzer.engine.state import HandState, shown_hole_cards
from poker_analyzer.parsing.events import Action, LogEvent


def ev(
    action: Action,
    *,
    player_id: int | None = None,
    amount: int | None = None,
    cards: tuple[str, ...] = (),
    is_hero: bool = False,
    session_id: int | None = None,
) -> LogEvent:
    return LogEvent(
        time="00:00:00",
        action=action,
        player_id=player_id,
        amount=amount,
        cards=cards,
        is_hero=is_hero,
        session_id=session_id,
    )


def test_session_of_recorded_from_events() -> None:
    """Карта player_id → session_id копится из событий (для лога СОВЕТ и профилей)."""
    s = HandState()
    s.apply(ev(Action.DEALER, player_id=0, session_id=17))
    s.apply(ev(Action.BLIND, player_id=1, amount=50, session_id=4))
    s.apply(ev(Action.CALL, player_id=2, amount=100))  # session_id не определился — None
    assert s.session_of(0) == 17  # запись первой строки раздачи (Dealer) не теряется
    assert s.session_of(1) == 4
    assert s.session_of(2) is None
    assert s.session_of(None) is None


def test_hero_from_template_fixes_bb_check() -> None:
    s = HandState()
    # Герой (P3) ставит BB 50 и опознан по шаблону (is_hero) — id известен сразу,
    # не дожидаясь «Ваш ход».
    s.apply(ev(Action.BLIND, player_id=3, amount=50, is_hero=True))
    assert s.hero_id == 3
    s.apply(ev(Action.CALL, player_id=1, amount=50))  # оппонент уравнивает блайнд
    s.apply(ev(Action.YOUR_TURN))
    assert s.to_call == 0  # герой уже внёс 50 = уровню → колл не нужен (можно чек)


def test_preflop_pot_players_and_hero() -> None:
    s = HandState()
    s.apply(ev(Action.DEALER, player_id=0))
    s.apply(ev(Action.BLIND, player_id=1, amount=5000))  # SB
    s.apply(ev(Action.BLIND, player_id=2, amount=10000))  # BB
    assert s.pot == 15000
    assert s.live == {0, 1, 2}

    s.apply(ev(Action.YOUR_TURN))
    assert s.hero_to_act is True
    s.apply(ev(Action.RAISE, player_id=0, amount=30000))  # герой рейзит to 30k
    assert s.hero_id == 0
    assert s.hero_to_act is False
    assert s.pot == 45000  # 15000 + (30000 - 0)
    assert s.to_call == 0  # герой сам на верхнем уровне

    s.apply(ev(Action.FOLD, player_id=1))
    s.apply(ev(Action.CALL, player_id=2, amount=30000))
    assert s.live == {0, 2}
    assert s.num_opponents == 1  # {0,2} минус герой 0
    assert s.pot == 65000  # 45000 + (30000 - 10000)


def test_hero_is_actor_right_after_your_turn() -> None:
    s = HandState()
    s.hero_id = 5  # допустим, раньше определили ошибочно
    s.apply(ev(Action.YOUR_TURN))
    s.apply(ev(Action.CALL, player_id=2, amount=100))
    assert s.hero_id == 2  # строка после «Ваш ход» — всегда герой, переопределили


def test_streets_and_board() -> None:
    s = HandState()
    s.apply(ev(Action.HAND, cards=("A♠", "K♠")))
    assert s.hero_cards == ("A♠", "K♠")
    assert s.street == "preflop"

    s.apply(ev(Action.DEAL, amount=65000))  # банк после префлопа
    s.apply(ev(Action.TABLE, cards=("7♦", "5♣", "10♦")))
    assert s.street == "flop"
    assert s.board == ("7♦", "5♣", "10♦")
    assert s.pot == 65000
    assert s.to_call == 0  # новая улица — ставки обнулены

    s.apply(ev(Action.TABLE, cards=("9♥",)))
    assert s.street == "turn"
    s.apply(ev(Action.TABLE, cards=("A♣",)))
    assert s.street == "river"
    assert s.board == ("7♦", "5♣", "10♦", "9♥", "A♣")


def test_board_flop_replaces_not_appends() -> None:
    # Дубль чтения флопа / перенос борда прошлой раздачи (поздний Win не сбросил): новый
    # флоп ЗАМЕНЯЕТ борд, а не дозаписывает — иначе борд >5 ронял advise.
    s = HandState()
    s.apply(ev(Action.TABLE, cards=("7♦", "5♣", "10♦")))
    s.apply(ev(Action.TABLE, cards=("7♦", "5♣", "10♦")))  # дубль того же флопа
    assert s.board == ("7♦", "5♣", "10♦")  # не задвоился до 6 карт
    s.apply(ev(Action.TABLE, cards=("6♦", "6♠", "10♥")))  # флоп НОВОЙ раздачи (Win не сбросил)
    assert s.board == ("6♦", "6♠", "10♥")  # старый борд заменён, не 6 карт


def test_board_stray_single_card_ignored() -> None:
    # Внеочередная одиночная карта (поздняя строка прошлой раздачи) к невалидному борду
    # игнорируется: тёрн/ривер дозаписываются ТОЛЬКО к борду из 3/4 карт (3→4→5).
    s = HandState()
    s.apply(ev(Action.TABLE, cards=("9♥",)))  # тёрн без флопа — некуда дозаписать
    assert s.board == ()
    s.apply(ev(Action.TABLE, cards=("7♦", "5♣", "10♦")))  # флоп
    s.apply(ev(Action.TABLE, cards=("9♥",)))  # тёрн
    s.apply(ev(Action.TABLE, cards=("A♣",)))  # ривер
    assert s.board == ("7♦", "5♣", "10♦", "9♥", "A♣")
    s.apply(ev(Action.TABLE, cards=("2♠",)))  # лишняя карта к полному борду — игнор
    assert s.board == ("7♦", "5♣", "10♦", "9♥", "A♣")  # остался 5 карт, не 6


def test_new_hand_resets_stale_board_on_dealer_then_hand() -> None:
    # Новая раздача = Dealer + сразу ДВЕ строки Hand. Сброс прошлой (борд/карты) — на первой
    # Hand, НЕ на Dealer (живой баг 2026-06-14: новая рука героя + старый борд 6♥2♦9♣A♣A♠;
    # Win прошлой мог не прочитаться, первая Dealer-строка после захода идёт без времени).
    s = HandState()
    s.apply(ev(Action.HAND, cards=("Q♠", "Q♦")))  # карты ПРОШЛОЙ раздачи
    s.apply(ev(Action.TABLE, cards=("6♥", "2♦", "9♣", "A♣", "A♠")))  # борд прошлой раздачи
    s.apply(ev(Action.DEALER, player_id=7))  # Dealer НОВОЙ раздачи (Win не прочитан)
    assert s.board == ("6♥", "2♦", "9♣", "A♣", "A♠")  # ещё НЕ сброшен — ждём Hand
    s.apply(ev(Action.HAND, cards=("K♠",)))  # первая карта новой раздачи → сброс прошлой
    assert s.board == ()  # стейл-борд сброшен
    s.apply(ev(Action.HAND, cards=("K♦",)))  # вторая карта
    assert s.hero_cards == ("K♠", "K♦")  # чистые карты новой раздачи, без Q♠Q♦ прошлой


def test_midhand_dealer_reassign_keeps_board() -> None:
    # Прошлый дилер вышел/кикнут мид-хенд → новый Dealer БЕЗ последующих Hand: борд и банк
    # СОХРАНЯЮТСЯ — это та же раздача (правка 2026-06-14 по подсказке: сброс на Dealer+Hand,
    # а не на одном Dealer). Иначе мид-хенд реассайн дилера рушил бы текущую раздачу.
    s = HandState()
    s.apply(ev(Action.TABLE, cards=("6♥", "2♦", "9♣")))  # флоп текущей раздачи
    s.apply(ev(Action.BET, player_id=1, amount=100))
    s.apply(ev(Action.DEALER, player_id=5))  # мид-хенд реассайн дилера (Hand НЕ следует)
    assert s.board == ("6♥", "2♦", "9♣")  # борд цел
    assert s.pot == 100  # банк цел — та же раздача


def test_button_survives_new_hand_reset() -> None:
    # Регресс 2026-06-14: сброс на первой Hand звал полный _reset, стиравший _button_id,
    # выставленный Dealer ЭТОЙ раздачи → позиция не определялась (поз=—) → префлоп-чарты молчали,
    # QK фолдился откатной эвристикой. Баттон новой раздачи должен ПЕРЕЖИТЬ сброс.
    s = HandState()
    s.apply(ev(Action.TABLE, cards=("6♥", "2♦", "9♣")))  # стейл-борд прошлой раздачи
    s.apply(ev(Action.DEALER, player_id=3))  # Dealer новой раздачи → баттон 3
    s.apply(ev(Action.HAND, cards=("A♠",)))  # сброс стейл-борда, баттон должен остаться
    s.apply(ev(Action.HAND, cards=("K♠",)))
    assert s.board == ()  # стейл-борд сброшен
    assert s._button_id == 3  # баттон новой раздачи НЕ потерян (позиция определится)


def test_to_call_at_decision_point() -> None:
    s = HandState()
    s.apply(ev(Action.BET, player_id=1, amount=200))
    s.apply(ev(Action.YOUR_TURN))
    assert s.to_call == 200  # герой ещё не вносил — доколлить весь уровень
    s.apply(ev(Action.CALL, player_id=0, amount=200))  # герой коллит
    assert s.hero_id == 0
    assert s.to_call == 0


def test_dealer_change_does_not_reset_but_win_then_dealer_does() -> None:
    s = HandState()
    s.apply(ev(Action.DEALER, player_id=0))
    s.apply(ev(Action.BET, player_id=0, amount=100))
    assert s.pot == 100

    # Смена дилера БЕЗ Win — та же раздача, не сбрасываем.
    s.apply(ev(Action.DEALER, player_id=1))
    assert s.pot == 100
    assert s.live == {0, 1}

    # Win, затем Dealer — новая раздача (сброс).
    s.apply(ev(Action.WIN, player_id=0))
    s.apply(ev(Action.DEALER, player_id=2))
    assert s.pot == 0
    assert s.board == ()
    assert s.live == {2}
    assert s.hero_id is None


def test_hero_wins_settles_profit() -> None:
    s = HandState()
    s.apply(ev(Action.BLIND, player_id=0, amount=50, is_hero=True))  # вложил 50
    s.apply(ev(Action.CALL, player_id=1, amount=50))
    s.apply(ev(Action.DEAL, amount=100))  # банк после префлопа, новая улица
    s.apply(ev(Action.BET, player_id=0, amount=80, is_hero=True))  # вложил ещё 80
    s.apply(ev(Action.FOLD, player_id=1))
    s.apply(ev(Action.WIN, player_id=0, amount=180, is_hero=True))  # забрал банк 180
    assert s.last_settled == ("00:00:00", 50)  # выигрыш 180 − вложено 130


def test_hero_loses_settles_negative() -> None:
    s = HandState()
    s.apply(ev(Action.BLIND, player_id=0, amount=50, is_hero=True))
    s.apply(ev(Action.RAISE, player_id=1, amount=200))
    s.apply(ev(Action.CALL, player_id=0, amount=200, is_hero=True))  # доколлил до 200
    s.apply(ev(Action.WIN, player_id=1, amount=400, is_hero=False))  # выиграл оппонент
    assert s.last_settled == ("00:00:00", -200)  # только списание вложенного


def test_settle_falls_back_to_pot_without_win_amount() -> None:
    s = HandState()
    s.apply(ev(Action.BET, player_id=0, amount=100, is_hero=True))
    s.apply(ev(Action.FOLD, player_id=1))
    s.apply(ev(Action.WIN, player_id=0, is_hero=True))  # сумма из строки не прочиталась
    assert s.last_settled == ("00:00:00", 0)  # выигрыш = банк 100 − вложено 100


def test_split_pot_subtracts_invested_once() -> None:
    s = HandState()
    s.apply(ev(Action.BET, player_id=0, amount=100, is_hero=True))  # герой вложил 100
    s.apply(ev(Action.CALL, player_id=1, amount=100))
    s.apply(ev(Action.WIN, player_id=1, amount=100, is_hero=False))  # сплит: первая Win — оппонент
    assert s.last_settled == ("00:00:00", -100)  # списали вложенное один раз
    s.apply(ev(Action.WIN, player_id=0, amount=100, is_hero=True))  # вторая Win — герой
    assert s.last_settled == ("00:00:00", 100)  # только выигрыш (вложенное уже списано)


def test_last_settled_set_only_on_win() -> None:
    s = HandState()
    s.apply(ev(Action.BET, player_id=0, amount=100, is_hero=True))
    assert s.last_settled is None  # не Win — расчёта нет
    s.apply(ev(Action.WIN, player_id=0, amount=100, is_hero=True))
    assert s.last_settled is not None
    s.apply(ev(Action.DEALER, player_id=1))  # новая раздача — снова None
    assert s.last_settled is None


def test_opponent_tiers_by_action() -> None:
    s = HandState()
    s.hero_id = 0  # герой — не оппонент
    s.apply(ev(Action.RAISE, player_id=1, amount=300))  # агрессор → 2
    s.apply(ev(Action.CALL, player_id=2, amount=300))  # коллер → 1
    s.apply(ev(Action.BLIND, player_id=3, amount=100))  # только блайнд → пас (0)
    assert s.opponent_tiers() == [2, 1, 0]  # по id 1,2,3


def test_tier_keeps_max_aggression() -> None:
    s = HandState()
    s.apply(ev(Action.CALL, player_id=1, amount=100))  # сперва коллер
    s.apply(ev(Action.RAISE, player_id=1, amount=300))  # затем рейз — тир растёт до 2
    assert s.opponent_tiers() == [2]


def test_self_call_warns_merge(caplog) -> None:  # type: ignore[no-untyped-def]
    # игрок «коллирует» собственную верхнюю ставку → под одним id слиплись два игрока
    import logging

    s = HandState()
    s.apply(ev(Action.RAISE, player_id=1, amount=300))  # P1 — верхняя ставка
    with caplog.at_level(logging.WARNING):
        s.apply(ev(Action.CALL, player_id=1, amount=300))  # «колл» своей же ставки
    assert any("СКЛЕЙКА" in r.message for r in caplog.records)


def test_legit_call_does_not_warn(caplog) -> None:  # type: ignore[no-untyped-def]
    import logging

    s = HandState()
    s.apply(ev(Action.RAISE, player_id=1, amount=300))  # P1 ставит
    with caplog.at_level(logging.WARNING):
        s.apply(ev(Action.CALL, player_id=2, amount=300))  # P2 законно коллирует
    assert not any("СКЛЕЙКА" in r.message for r in caplog.records)


def test_tiers_reset_on_new_hand() -> None:
    s = HandState()
    s.apply(ev(Action.RAISE, player_id=1, amount=300))
    s.apply(ev(Action.WIN, player_id=1))
    s.apply(ev(Action.DEALER, player_id=2))  # новая раздача — тиры сброшены
    assert s.opponent_tiers() == [0]  # live = {2}, тир игрока 1 (=2) стёрт, у 2 действий нет


def _seated(button: int, blinds: list[int], actions: list[int]) -> HandState:
    """Состояние: Dealer(button), блайнды по порядку, затем действия (любые) в порядке мест."""
    s = HandState()
    s.apply(ev(Action.DEALER, player_id=button))
    for pid in blinds:
        s.apply(ev(Action.BLIND, player_id=pid, amount=10))
    for pid in actions:
        s.apply(ev(Action.FOLD, player_id=pid))  # фолд тоже фиксирует место
    return s


def test_position_button_and_blinds() -> None:
    s = _seated(button=0, blinds=[1, 2], actions=[3, 4, 5])
    s.hero_id = 0
    assert s.hero_position() == "BTN"  # баттон = Dealer
    s.hero_id = 1
    assert s.hero_position() == "SB"  # первый блайнд
    s.hero_id = 2
    assert s.hero_position() == "BB"  # второй блайнд


def test_position_blinds_by_amount_when_log_reversed() -> None:
    # Лог изредка отдаёт строки блайндов в обратном порядке (BB раньше SB). Позиция
    # определяется по СУММЕ (BB = больший блайнд), а не по порядку появления.
    s = HandState()
    s.apply(ev(Action.DEALER, player_id=0))
    s.apply(ev(Action.BLIND, player_id=2, amount=50))  # пришёл первым, но это BB (больше)
    s.apply(ev(Action.BLIND, player_id=1, amount=25))  # пришёл вторым, но это SB (меньше)
    s.apply(ev(Action.FOLD, player_id=3))
    s.hero_id = 0
    assert s.hero_position() == "BTN"
    s.hero_id = 1
    assert s.hero_position() == "SB"  # меньший блайнд — SB, хоть и пришёл вторым
    s.hero_id = 2
    assert s.hero_position() == "BB"  # больший блайнд — BB, хоть и пришёл первым


def test_position_ep_co_with_table_size() -> None:
    s = _seated(button=0, blinds=[1, 2], actions=[3, 4, 5])
    s.table_size = 6  # известен из прошлых раздач
    s.hero_id = 3
    assert s.hero_position() == "EP"  # UTG, offset 3 < n-1
    s.hero_id = 5
    assert s.hero_position() == "CO"  # offset 5 == n-1
    s.hero_id = 4
    assert s.hero_position() == "EP"


def test_position_open_before_hero_acts() -> None:
    # герой ещё не ходил — позиция по числу уже сходивших добровольцев
    s = _seated(button=0, blinds=[1, 2], actions=[3])  # UTG сходил, герой — следующий
    s.table_size = 6
    s.hero_id = 4
    assert s.hero_position() == "EP"  # offset 3+1=4
    s2 = _seated(button=0, blinds=[1, 2], actions=[3, 4])
    s2.table_size = 6
    s2.hero_id = 5
    assert s2.hero_position() == "CO"  # offset 3+2=5 == n-1


def test_position_none_when_table_size_unknown() -> None:
    s = _seated(button=0, blinds=[1, 2], actions=[])  # видно только баттон + 2 блайнда = 3
    s.hero_id = 3  # UTG, ещё не ходил; offset 3, table_size 3 → не отличить CO от EP
    assert s.hero_position() is None


def test_position_none_without_blinds_or_button() -> None:
    s = HandState()
    s.hero_id = 0
    assert s.hero_position() is None  # нет Dealer/блайндов
    s.apply(ev(Action.TABLE, cards=("7♦", "5♣", "2♠")))  # есть борд → не префлоп
    assert s.hero_position() is None


def test_table_size_persists_across_hands() -> None:
    s = _seated(button=0, blinds=[1, 2], actions=[3, 4, 5])  # 6 игроков замечено
    assert s.table_size == 6
    s.apply(ev(Action.WIN, player_id=0))
    s.apply(ev(Action.DEALER, player_id=1))  # новая раздача
    assert s.table_size == 6  # размер стола не сбрасывается


def test_position_straddle_kept_out_of_blinds() -> None:
    s = HandState()
    s.apply(ev(Action.DEALER, player_id=0))
    s.apply(ev(Action.BLIND, player_id=1, amount=5))  # SB
    s.apply(ev(Action.BLIND, player_id=2, amount=10))  # BB
    s.apply(ev(Action.BLIND, player_id=3, amount=20))  # страддл (UTG) — не блайнд по позиции
    s.apply(ev(Action.FOLD, player_id=4))  # HJ фолд
    s.table_size = 6
    assert s._blind_order == [1, 2]  # страддл не попал в блайнды
    s.hero_id = 5
    assert s.hero_position() == "CO"  # страддл учтён как место → CO не сломан
    s.hero_id = 3
    assert s.hero_position() == "EP"  # сам страддлер — ранняя позиция


def test_table_size_robust_to_spurious_dealer() -> None:
    s = _seated(button=0, blinds=[1, 2], actions=[3, 4, 5])  # table_size = 6
    s.apply(ev(Action.DEALER, player_id=99))  # ложный/мис-ридный Dealer mid-hand
    assert s.table_size == 6  # не раздулся фантомным игроком
    assert s._button_id == 0  # баттон не перезаписан mid-hand


def test_hero_in_position_button_and_last_to_act() -> None:
    s = HandState()
    s.hero_id = 0
    s._button_id = 0
    assert s.hero_in_position()  # баттон — всегда в позиции
    # герой ходит последним среди живых по кольцу мест → в позиции
    s2 = HandState()
    s2.hero_id = 9
    s2._button_id = 1
    s2.live = {1, 5, 9}
    s2._seen_order = [1, 5, 9]
    assert s2.hero_in_position()
    # тот же стол, но герой ходит раньше → вне позиции
    s2._seen_order = [9, 5, 1]
    assert not s2.hero_in_position()


def test_hero_in_position_unknown_is_oop() -> None:
    s = HandState()  # данных нет — консервативно вне позиции
    assert not s.hero_in_position()


def test_hero_remaining_from_stack_minus_invested() -> None:
    s = HandState(hero_cards=("A♠", "K♠"))
    assert s.hero_remaining is None  # без снимка стека
    s.hero_stack = 1000.0  # стек на начало раздачи (снимок + P&L)
    s._hero_invested = 150.0
    assert s.hero_remaining == 850.0  # остаток = стек − вложено


def test_min_raise_levels_for_autoclick() -> None:
    s = HandState()
    s.apply(ev(Action.DEALER, player_id=0))
    s.apply(ev(Action.BLIND, player_id=1, amount=25))
    s.apply(ev(Action.BLIND, player_id=2, amount=50))
    assert s.big_blind == 50
    assert s.min_raise_to == 100  # префлоп: мин-рейз = 2×ББ
    s.apply(ev(Action.RAISE, player_id=3, amount=150))  # рейз до 150 (шаг 100)
    assert s.min_raise_to == 250  # 150 + 100
    s.apply(ev(Action.ALL_IN, player_id=4, amount=200))  # короткий олл-ин (шаг 50 < 100)
    assert s.min_raise_to == 300  # 200 + прежний шаг 100 (недобор шаг не уменьшает)
    s.apply(ev(Action.DEAL, amount=500))  # новая улица
    assert s.min_raise_to == 50  # ставок нет — мин-бет = ББ
    s.apply(ev(Action.BET, player_id=3, amount=100))
    assert s.min_raise_to == 200  # 100 + 100


def test_big_blind_swapped_misread_and_straddle() -> None:
    s = HandState()
    s.apply(ev(Action.BLIND, player_id=1, amount=50))  # мисрид: сначала «большая» сумма
    s.apply(ev(Action.BLIND, player_id=2, amount=25))
    assert s.big_blind == 50  # max первых двух — устойчиво к перестановке SB/BB
    s.apply(ev(Action.BLIND, player_id=3, amount=100))  # страддл — не ББ
    assert s.big_blind == 50


def test_position_inferred_on_your_turn_without_hero_id() -> None:
    # герой не блайнд/не баттон и ещё не ходил (id неизвестен), но сейчас «Ваш ход»:
    # он — следующий ходящий → позиция выводится из порядка действий (чарты не молчат)
    s = HandState()
    s.apply(ev(Action.DEALER, player_id=0))
    s.apply(ev(Action.BLIND, player_id=1, amount=25))
    s.apply(ev(Action.BLIND, player_id=2, amount=50))
    s.apply(ev(Action.CALL, player_id=3, amount=50))  # лимпер перед героем
    s.apply(ev(Action.YOUR_TURN))
    s.table_size = 9
    assert s.hero_id is None
    assert s.hero_position() == "EP"  # следующий доброволец после лимпера
    s.hero_to_act = False  # вне своего хода без id позицию не угадываем
    assert s.hero_position() is None


def test_all_in_to_level_includes_street_contribution() -> None:
    s = HandState()
    s.hero_id = 0
    assert s.all_in_to is None  # без снимка стека
    s.hero_stack = 1000.0
    s._hero_invested = 50.0  # герой поставил блайнд 50
    s._street_in[0] = 50.0
    assert s.all_in_to == 1000.0  # уровень олл-ина = внесённое 50 + остаток 950


def test_player_position_for_each_seat() -> None:
    s = _seated(button=0, blinds=[1, 2], actions=[3, 4, 5])  # table_size = 6
    assert s.player_position(0) == "BTN"
    assert s.player_position(1) == "SB"
    assert s.player_position(2) == "BB"
    assert s.player_position(5) == "CO"  # последний доброволец
    assert s.player_position(3) == "EP"
    assert s.player_position(99) is None  # незамеченный игрок
    s.board = ("3♥", "6♥", "10♥")  # постфлоп — позиции не выводим
    assert s.player_position(0) is None


def test_opponent_context_classification() -> None:
    """Префлоп-контекст по первому добровольному действию (блайнды не в счёт)."""
    from poker_analyzer.parsing.events import Action, LogEvent

    s = HandState()
    for e in (
        LogEvent("00:00:01", Action.DEALER, player_id=0),  # BTN
        LogEvent("00:00:02", Action.BLIND, player_id=1, amount=10),  # SB
        LogEvent("00:00:03", Action.BLIND, player_id=2, amount=20),  # BB
        LogEvent("00:00:04", Action.RAISE, player_id=3, amount=60),  # опен (EP)
        LogEvent("00:00:05", Action.CALL, player_id=4, amount=60),  # колл опена
        LogEvent("00:00:06", Action.RAISE, player_id=0, amount=180),  # BTN 3-бет
        LogEvent("00:00:07", Action.CALL, player_id=2, amount=180),  # BB защита
        LogEvent(
            "00:00:08", Action.CALL, player_id=1, amount=180
        ),  # SB лимп-колл? нет — колл рейза
    ):
        s.apply(e)
    s.table_size = 6
    assert s.opponent_context(3) == "open_early"  # открыл рейзом из ранней
    assert s.opponent_context(4) == "call_vs_open"  # доколлил опен
    assert s.opponent_context(0) == "3bet"  # рейз поверх рейза
    assert s.opponent_context(2) == "bb_defend"  # BB доколлил рейз
    assert s.opponent_context(1) == "call_vs_open"  # SB доколлил рейз (не BB)


def test_opponent_context_limp_and_unknown() -> None:
    from poker_analyzer.parsing.events import Action, LogEvent

    s = HandState()
    for e in (
        LogEvent("00:00:01", Action.DEALER, player_id=0),
        LogEvent("00:00:02", Action.BLIND, player_id=1, amount=10),
        LogEvent("00:00:03", Action.BLIND, player_id=2, amount=20),
        LogEvent("00:00:04", Action.CALL, player_id=0, amount=20),  # лимп (нет рейза до)
        LogEvent("00:00:05", Action.CHECK, player_id=2),  # BB чек опции — не диапазон
    ):
        s.apply(e)
    s.table_size = 6
    assert s.opponent_context(0) == "limp"
    assert s.opponent_context(2) == "unknown"  # только чек → откат на тир
    assert s.opponent_context(9) == "unknown"  # не участвовал


# --- shown_hole_cards: вскрытая карманка = комбинация − борд (этап D, стадия 2) -------------


def test_shown_hole_two_cards() -> None:
    # EFG: пара восьмёрок + кикеры; борд 9♠K♥Q♦6♠8♠ → обе карманные играют (8♦ пара, J♠ кикер).
    combo = ("8♦", "8♠", "K♥", "Q♦", "J♠")
    board = ("9♠", "K♥", "Q♦", "6♠", "8♠")
    assert shown_hole_cards(combo, board) == ("8♦", "J♠")


def test_shown_hole_straight_prefers_player_cards() -> None:
    # Стрит A-K-Q-J-10; борд K♥10♠5♦J♥A♥. 10♣ ≠ бордовой 10♠ (игра показывает карту ИГРОКА) →
    # карманка Q♠ 10♣ (обе вошли в стрит, хотя 10 дублирует бордовую по рангу).
    combo = ("A♥", "K♥", "Q♠", "J♥", "10♣")
    board = ("K♥", "10♠", "5♦", "J♥", "A♥")
    assert shown_hole_cards(combo, board) == ("Q♠", "10♣")


def test_shown_hole_one_card() -> None:
    # Играет ОДНА карманная (вторая не вошла в лучшую руку и не показана) → 1 карта.
    combo = ("A♥", "K♥", "Q♠", "J♥", "10♠")  # 10♠ — бордовая
    board = ("K♥", "10♠", "5♦", "J♥", "A♥")
    assert shown_hole_cards(combo, board) == ("Q♠",)


def test_shown_hole_board_plays_zero() -> None:
    # Играет борд (карманка в комбинацию не вошла) → 0 карт.
    board = ("A♥", "K♥", "Q♥", "J♥", "10♥")
    assert shown_hole_cards(("A♥", "K♥", "Q♥", "J♥", "10♥"), board) == ()


def test_shown_hole_more_than_two_unreliable() -> None:
    # Рассинхрон распознавания масти борда/комбинации → >2 «карманных» → ненадёжно → ().
    combo = ("8♦", "8♠", "K♥", "Q♦", "J♠")
    board = ("9♠", "K♣", "Q♣", "6♠", "8♥")  # ни одна масть не совпала с комбо
    assert shown_hole_cards(combo, board) == ()


def test_shown_hole_empty_combo_folded() -> None:
    # Все сфолдили — победитель не вскрыт (карт нет) → ().
    assert shown_hole_cards((), ("A♥", "K♥", "Q♦")) == ()


# --- HandState: запись вскрытой карманки на Win -----------------------------------------------


def _board_then_win(*wins: LogEvent) -> HandState:
    """Борд 9♠K♥Q♦6♠8♠ (флоп/тёрн/ривер), затем переданные Win-события."""
    s = HandState()
    s.apply(ev(Action.TABLE, cards=("9♠", "K♥", "Q♦")))  # флоп заменяет борд
    s.apply(ev(Action.TABLE, cards=("6♠",)))  # тёрн
    s.apply(ev(Action.TABLE, cards=("8♠",)))  # ривер → борд полный
    for w in wins:
        s.apply(w)
    return s


def test_handstate_records_shown_hole_on_win() -> None:
    s = _board_then_win(
        ev(Action.WIN, player_id=3, amount=918, session_id=7, cards=("8♦", "8♠", "K♥", "Q♦", "J♠"))
    )
    assert s.shown_hole == {7: ("8♦", "J♠")}


def test_handstate_split_pot_records_each_winner() -> None:
    # Сплит/сайд-пот = несколько Win → запись на КАЖДОГО победителя по его session_id.
    s = _board_then_win(
        ev(Action.WIN, player_id=3, amount=500, session_id=7, cards=("8♦", "8♠", "K♥", "Q♦", "J♠")),
        ev(Action.WIN, player_id=4, amount=500, session_id=9, cards=("9♦", "9♣", "K♥", "Q♦", "8♠")),
    )
    assert s.shown_hole == {7: ("8♦", "J♠"), 9: ("9♦", "9♣")}


def test_handstate_no_session_id_not_recorded() -> None:
    # Без session_id привязать карманку некуда → не пишем.
    s = _board_then_win(
        ev(Action.WIN, player_id=3, amount=918, cards=("8♦", "8♠", "K♥", "Q♦", "J♠"))
    )
    assert s.shown_hole == {}


def test_handstate_folded_win_no_cards_not_recorded() -> None:
    # Все сфолдили — на Win карт нет → не пишем.
    s = _board_then_win(ev(Action.WIN, player_id=3, amount=918, session_id=7))
    assert s.shown_hole == {}


def test_handstate_shown_hole_reset_between_hands() -> None:
    s = _board_then_win(
        ev(Action.WIN, player_id=3, amount=918, session_id=7, cards=("8♦", "8♠", "K♥", "Q♦", "J♠"))
    )
    assert s.shown_hole == {7: ("8♦", "J♠")}
    s.apply(ev(Action.DEALER, player_id=0))  # Win→Dealer = новая раздача → сброс
    assert s.shown_hole == {}
