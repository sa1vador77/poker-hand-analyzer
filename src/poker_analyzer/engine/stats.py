"""Профили игроков по ``session_id``: статистика поведения → поправка диапазонов.

Этап B: на конце каждой раздачи (:meth:`ProfileBook.observe_hand`) по потоку её
событий копятся VPIP/PFR/пуши/агрессия каждого участника (герой пропускается — его
карты и так известны, профиль нужен для диапазонов ОППОНЕНТОВ). Этап C: лузовость
профиля — сглаженный VPIP со шринкеджем к пулу — превращается в поправку порога
сужения диапазона конкретного оппонента (:meth:`ProfileBook.threshold_delta`):
лузовый игрок → порог ниже (диапазон шире), тайтовый → выше. При выборке меньше
~``PROFILE_SHRINK_HANDS`` рук поправка почти нулевая: мало данных — верим пулу.

Реестр живёт у вызывающего кода (одна сессия анализа), в советник попадает
ссылкой ``HandState.profiles``.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

from poker_analyzer.config import (
    POOL_VPIP,
    PROFILE_NARROW_CAP,
    PROFILE_NARROW_SCALE,
    PROFILE_SHRINK_HANDS,
)
from poker_analyzer.parsing.events import Action, LogEvent

# Добровольное вложение денег (VPIP): блайнд не в счёт, чек/фолд — тоже.
_VPIP_ACTIONS = frozenset({Action.CALL, Action.BET, Action.RAISE, Action.ALL_IN})
_AGGRO_ACTIONS = frozenset({Action.BET, Action.RAISE, Action.ALL_IN})
_PASSIVE_ACTIONS = frozenset({Action.CALL, Action.CHECK})


@dataclass(slots=True)
class PlayerProfile:
    """Накопленная статистика одного игрока (ключ — его ``session_id``)."""

    hands: int = 0  # раздач, где игрок замечен
    vpip: int = 0  # раздач с добровольным вложением на префлопе (блайнд не в счёт)
    pfr: int = 0  # раздач с префлоп-рейзом/пушем
    jams: int = 0  # раздач с пушем (All-in, любая улица)
    aggro: int = 0  # агрессивных действий суммарно (bet/raise/all-in)
    passive: int = 0  # пассивных действий суммарно (call/check)
    wins: int = 0  # выигранных раздач (строки Win)

    def looseness(self) -> float:
        """Сглаженный VPIP: шринкедж к лузовости пула при малой выборке."""
        prior = PROFILE_SHRINK_HANDS * POOL_VPIP
        return (self.vpip + prior) / (self.hands + PROFILE_SHRINK_HANDS)


class ProfileBook:
    """Реестр профилей за сессию: ``session_id → PlayerProfile``."""

    def __init__(self) -> None:
        self._profiles: dict[int, PlayerProfile] = {}

    def get(self, session_id: int | None) -> PlayerProfile | None:
        """Профиль игрока или ``None`` (нет session_id / игрок ещё не встречался)."""
        if session_id is None:
            return None
        return self._profiles.get(session_id)

    # --- этап B: сбор --------------------------------------------------------------

    def observe_hand(self, events: Iterable[LogEvent]) -> list[int]:
        """Скармливает ЗАВЕРШЁННУЮ раздачу: обновляет профили всех её участников.

        Префлоп — события до первой карты борда (``Table``). Счётчики «раз за
        раздачу» (VPIP/PFR/пуш/победа) не задваиваются внутри одной руки. Возвращает
        ``session_id`` обновлённых игроков (для строки лога).
        """
        preflop = True
        marks: dict[int, set[str]] = {}  # session_id → какие пер-раздачные счётчики уже взяты
        for e in events:
            if e.action is Action.TABLE:
                preflop = False
            if e.session_id is None or e.player_id is None or e.is_hero:
                continue
            profile = self._profiles.setdefault(e.session_id, PlayerProfile())
            taken = marks.setdefault(e.session_id, set())
            if "hand" not in taken:
                taken.add("hand")
                profile.hands += 1
            if preflop and e.action in _VPIP_ACTIONS and "vpip" not in taken:
                taken.add("vpip")
                profile.vpip += 1
            if preflop and e.action in (Action.RAISE, Action.ALL_IN) and "pfr" not in taken:
                taken.add("pfr")
                profile.pfr += 1
            if e.action is Action.ALL_IN and "jam" not in taken:
                taken.add("jam")
                profile.jams += 1
            if e.action is Action.WIN and "win" not in taken:
                taken.add("win")
                profile.wins += 1
            if e.action in _AGGRO_ACTIONS:
                profile.aggro += 1
            elif e.action in _PASSIVE_ACTIONS:
                profile.passive += 1
        return sorted(marks)

    # --- этап C: профиль → советы ---------------------------------------------------

    def threshold_delta(self, session_id: int | None) -> float:
        """Поправка порога сужения диапазона оппонента по лузовости его профиля.

        Лузовее пула → отрицательная (порог ниже, диапазон ШИРЕ — его агрессия стоит
        меньше), тайтовее → положительная. Кап не даёт профилю доминировать над
        улицей/линией/поляризацией; шринкедж гасит поправку на малой выборке.
        """
        profile = self.get(session_id)
        if profile is None or profile.hands == 0:
            return 0.0
        delta = PROFILE_NARROW_SCALE * (POOL_VPIP - profile.looseness())
        return max(-PROFILE_NARROW_CAP, min(PROFILE_NARROW_CAP, delta))

    def brief(self, session_id: int | None) -> str | None:
        """Сводка для лога: ``v45·p21·14h`` (VPIP%, PFR%, рук) или ``None`` без данных."""
        profile = self.get(session_id)
        if profile is None or profile.hands == 0:
            return None
        vpip = 100.0 * profile.vpip / profile.hands
        pfr = 100.0 * profile.pfr / profile.hands
        return f"v{vpip:.0f}·p{pfr:.0f}·{profile.hands}h"
