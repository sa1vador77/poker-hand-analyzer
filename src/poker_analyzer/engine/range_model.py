"""Линия-осведомлённое сужение диапазона оппонента (тег-слой над ``narrow_range``).

Один порог «эквити-vs-random» за улицу теряет ВСЮ последовательность действий: чек-колл,
донк, чек-рейз и дабл-баррель сводятся в один тир агрессии. Здесь по потоку событий
раздачи восстанавливаем ЛИНИЮ каждого игрока — на скольких постфлоп-улицах он проявлял
агрессию (баррели), был ли чек-рейз, был ли донк — и переводим её в добавку к порогу
сужения: чем «тяжелее» линия, тем уже и сильнее диапазон.

Слой не штрафует и не считает эквити — только форму диапазона (advisor поверх делает
доминацию/EV). Новых входов со скрина не требует: всё из ``Action`` / ``Deal``-границ улиц.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass

from poker_analyzer.config import LINE_NARROWING
from poker_analyzer.parsing.events import Action, LogEvent

# Агрессивные действия (двигают диапазон в сторону value).
_AGGRO = frozenset({Action.BET, Action.RAISE, Action.ALL_IN})


@dataclass(frozen=True, slots=True)
class LineTag:
    """Линия игрока на постфлопе (для сужения его диапазона)."""

    barrels: int  # на скольких постфлоп-улицах он бет/рейзил
    check_raised: bool  # был рейз поверх чужой ставки — очень сильная линия
    donked: bool  # бет в агрессора прошлой улицы — часто полярнее/слабее


def classify_lines(events: list[LogEvent]) -> dict[int, LineTag]:
    """Линии всех игроков по потоку событий раздачи (ключ — ``player_id``).

    Улицы считаем по ``Deal`` (закрывает круг ставок). В теги попадает только постфлоп
    (``street ≥ 1``): префлоп-рейз не должен выглядеть как постфлоп чек-рейз. Агрессор
    прошлой улицы запоминается для распознавания донка на следующей.
    """
    street = 0
    aggro_streets: dict[int, set[int]] = defaultdict(set)
    check_raised: set[int] = set()
    donked: set[int] = set()
    first_bettor: int | None = None  # первый, кто поставил на текущей улице
    prev_aggressor: int | None = None  # последний агрессор прошлой улицы
    last_aggressor: int | None = None  # последний агрессор текущей улицы

    for ev in events:
        if ev.action is Action.DEAL:  # граница улицы
            prev_aggressor = last_aggressor
            street += 1
            first_bettor = None
            last_aggressor = None
            continue
        pid = ev.player_id
        if pid is None:
            continue
        if ev.action in _AGGRO:
            if street >= 1:  # учитываем только постфлоп
                aggro_streets[pid].add(street)
                if ev.action is Action.RAISE:
                    check_raised.add(pid)  # рейз = поверх чьего-то бета на этой улице
                elif (
                    ev.action is Action.BET
                    and first_bettor is None
                    and prev_aggressor is not None
                    and pid != prev_aggressor
                ):
                    donked.add(pid)  # первым ставит не-агрессор прошлой улицы = донк
            if first_bettor is None:
                first_bettor = pid
            last_aggressor = pid

    return {
        pid: LineTag(len(streets), pid in check_raised, pid in donked)
        for pid, streets in aggro_streets.items()
    }


def line_threshold_delta(tag: LineTag | None) -> float:
    """Добавка к порогу сужения по линии: «тяжёлая» линия → уже диапазон.

    Чек-рейз и баррели сужают (диапазон поляризован во value), донк чуть расширяет (больше
    блефов). Ограничено ``[-donk_relief, cap]``, чтобы не схлопнуть диапазон в ноль.
    """
    if tag is None:
        return 0.0
    delta = LINE_NARROWING.per_extra_barrel * max(0, tag.barrels - 1)
    if tag.check_raised:
        delta += LINE_NARROWING.check_raise
    if tag.donked:
        delta -= LINE_NARROWING.donk_relief
    return max(-LINE_NARROWING.donk_relief, min(LINE_NARROWING.cap, delta))
