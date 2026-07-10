"""Тесты идентификации игроков: герой, реестр раздачи и сессионный реестр."""

from __future__ import annotations

import numpy as np
import pytest

from poker_analyzer.identity import players as players_module
from poker_analyzer.identity.players import PlayerRegistry, SessionPlayers
from poker_analyzer.vision.recognition import find_nick_region


def _nick(pattern: tuple[tuple[int, int], ...], width: int = 80) -> np.ndarray:
    """Кроп ника (BGR, как из find_nick_region): тёмные «штрихи» текста на фоне."""
    gray = np.full((37, width), 30, np.uint8)
    for x0, x1 in pattern:
        gray[10:25, x0:x1] = 210
    return np.stack([gray, gray, gray], axis=-1)


def test_matches_hero_true_for_same_nick() -> None:
    hero_gray = _nick(((5, 30), (35, 70)))[:, :, 0]  # шаблон героя (grayscale)
    reg = PlayerRegistry(hero_templates=[hero_gray])
    same = _nick(((6, 31), (36, 71)))  # тот же ник, чуть сдвинут (как между кадрами)
    assert reg.matches_hero(same) is True


def test_matches_hero_false_for_other_nick() -> None:
    hero_gray = _nick(((5, 30), (35, 70)))[:, :, 0]
    reg = PlayerRegistry(hero_templates=[hero_gray])
    other = _nick(((2, 8),))  # совсем другой, почти пустой ник
    assert reg.matches_hero(other) is False


def test_matches_hero_false_without_template() -> None:
    reg = PlayerRegistry()  # шаблона героя нет
    assert reg.matches_hero(_nick(((5, 70),))) is False


# --- различение игроков (identify): анти-слияние и Win без регистрации ---------


def test_identify_same_nick_same_id() -> None:
    reg = PlayerRegistry()
    a = _nick(((5, 30), (35, 70)), width=80)
    b = _nick(((6, 31), (36, 71)), width=80)  # тот же ник, сдвиг между кадрами
    assert reg.identify(a) == reg.identify(b) == 0  # один игрок


def test_identify_width_mismatch_not_merged() -> None:
    reg = PlayerRegistry()
    wide = reg.identify(_nick(((5, 30), (35, 75)), width=90))
    narrow = reg.identify(_nick(((2, 10),), width=30))  # узкий ник другого игрока
    assert wide == 0 and narrow == 1  # сильно разная ширина → не слились в один id


def test_identify_register_false_skips_new_player() -> None:
    reg = PlayerRegistry()
    assert reg.identify(_nick(((5, 30), (35, 70)), width=80)) == 0  # обычная строка регистрирует
    stranger = _nick(((2, 8),), width=30)  # никого из реестра не напоминает
    assert reg.identify(stranger, register=False) is None  # Win-строка не плодит игроков
    assert reg.identify(stranger, register=False) is None  # и повторно — тоже None


# --- сессионный реестр (SessionPlayers): постоянные id между раздачами ---------


def test_session_id_stable_across_hands() -> None:
    """Один ник в двух раздачах → один session_id, хотя реестр раздачи сбрасывался."""
    reg = PlayerRegistry(session=SessionPlayers())
    nick_a = _nick(((5, 30), (35, 70)), width=80)
    nick_b = _nick(((2, 10),), width=30)  # другой игрок (узкий ник)
    pid_a = reg.identify(nick_a)
    pid_b = reg.identify(nick_b)
    sid_a, sid_b = reg.session_of(pid_a), reg.session_of(pid_b)
    assert sid_a != sid_b
    reg.on_keyword("win")  # конец раздачи …
    reg.on_keyword("dealer")  # … новая раздача → реестр раздачи сброшен
    pid_a2 = reg.identify(_nick(((6, 31), (36, 71)), width=80))  # тот же ник A, сдвиг
    assert pid_a2 == 0  # внутрираздачный id начался заново
    assert reg.session_of(pid_a2) == sid_a  # а сессионный — тот же


def test_session_of_none_without_session_registry() -> None:
    reg = PlayerRegistry()  # без сессионного реестра
    pid = reg.identify(_nick(((5, 30), (35, 70))))
    assert reg.session_of(pid) is None
    assert reg.session_of(None) is None


def _patch_correlation(monkeypatch: pytest.MonkeyPatch, scores: dict[tuple[int, int], float]):
    """Подменяет корреляцию ников управляемыми оценками по «маркеру» в пикселе [0,0]."""

    def fake(a: np.ndarray, b: np.ndarray, *, min_width_ratio: float | None = None) -> float:
        return scores[(int(a[0, 0]), int(b[0, 0]))]

    monkeypatch.setattr(players_module, "_nick_correlation", fake)


def _marked(marker: int) -> np.ndarray:
    crop = np.zeros((10, 40), np.uint8)
    crop[0, 0] = marker
    return crop


def test_session_ambiguous_match_registers_new_player(monkeypatch: pytest.MonkeyPatch) -> None:
    """Два близких кандидата выше порога → «при сомнении — новый игрок», не слияние."""
    scores = {
        (2, 1): 0.0,  # B не похож на A → регистрируются раздельно
        (3, 1): 0.85,  # запрос Q похож и на A …
        (3, 2): 0.83,  # … и на B (оба ≥ 0.8, отрыв 0.02 < margin 0.05) → сомнение
    }
    _patch_correlation(monkeypatch, scores)
    session = SessionPlayers(match_threshold=0.8, margin=0.05)
    assert session.identify(_marked(1)) == 0  # A
    assert session.identify(_marked(2)) == 1  # B
    assert session.identify(_marked(3)) == 2  # Q — новый игрок, а не слияние с A


def test_session_clear_winner_matches(monkeypatch: pytest.MonkeyPatch) -> None:
    """Один кандидат с заметным отрывом → совпадение (без дробления)."""
    scores = {
        (2, 1): 0.0,
        (3, 1): 0.9,  # Q явно A …
        (3, 2): 0.5,  # … а не B (второй ниже порога)
    }
    _patch_correlation(monkeypatch, scores)
    session = SessionPlayers(match_threshold=0.8, margin=0.05)
    assert session.identify(_marked(1)) == 0
    assert session.identify(_marked(2)) == 1
    assert session.identify(_marked(3)) == 0  # узнан как A


def test_session_lru_eviction(monkeypatch: pytest.MonkeyPatch) -> None:
    """Переполнение реестра вытесняет давно не виденного; его ник потом получает НОВЫЙ id."""
    scores = {(m, r): 1.0 if m == r else 0.0 for m in (1, 2, 3) for r in (1, 2, 3)}
    _patch_correlation(monkeypatch, scores)
    session = SessionPlayers(match_threshold=0.8, max_players=2)
    assert session.identify(_marked(1)) == 0
    assert session.identify(_marked(2)) == 1
    assert session.identify(_marked(3)) == 2  # переполнение → вытеснен A (самый давний)
    assert len(session) == 2
    assert session.identify(_marked(1)) == 3  # A забыт — получает новый id, не чужой


# --- шаг 2b: строка-ключ как первичный признак (корреляция тай-брейкером) -------

# Два кропа, которые корреляция НЕ сольёт (разная форма штрихов) — для проверки,
# что совпадение строки лечит сплит, а расхождение строки ветирует склейку.
_CROP_A = _nick(((5, 30), (35, 70)))  # две широкие полосы
_CROP_B = _nick(((5, 11), (22, 28), (44, 50), (64, 72)))  # четыре тонкие — другой рисунок


def test_string_key_rescues_split_across_poor_correlation() -> None:
    # контроль: без ключа разные кропы → разные id (корреляция их не слила)
    base = PlayerRegistry()
    assert base.identify(_CROP_A) != base.identify(_CROP_B)
    # со совпадающим ключом → ОДИН игрок (строка лечит сплит корреляции)
    reg = PlayerRegistry()
    id_a = reg.identify(_CROP_A, key="Furkatbek lrisov")
    id_b = reg.identify(_CROP_B, key="Furkatbek lr�sov")  # тот же ник, дырка
    assert id_a == id_b


def test_string_key_vetoes_wrong_merge() -> None:
    # ОДИН и тот же кроп (корреляция ~1.0), но РАЗНЫЕ ключи → не сливаем (склейка CCCEDlTS+Никита)
    reg = PlayerRegistry()
    id0 = reg.identify(_CROP_A, key="CCCEDlTS")
    id1 = reg.identify(_CROP_A, key="Hикитa")  # тот же кроп, другой игрок по строке
    assert id0 == 0
    assert id1 == 1


def test_session_string_recognizes_across_poor_correlation() -> None:
    session = SessionPlayers()
    s_a = session.identify(_CROP_A[:, :, 0], key="Furkatbek lrisov")
    s_b = session.identify(_CROP_B[:, :, 0], key="Furkatbek lrisov")  # другой кроп, тот же ключ
    assert s_a == s_b


def test_session_string_vetoes_merge_of_distinct_nicks() -> None:
    session = SessionPlayers()
    s0 = session.identify(_CROP_A[:, :, 0], key="CCCEDlTS")
    s1 = session.identify(_CROP_A[:, :, 0], key="Hикитa")  # тот же кроп, другой игрок
    assert s0 != s1


def test_find_nick_region_isolates_rightmost_cluster_on_win() -> None:
    # строка Win: тёмные «карты» слева, широкий разрыв (тире-сепаратор), тёмный «ник» справа.
    # Карты отрезаются ТОЛЬКО на Win (has_leading_cards=True).
    strip = np.full((37, 200, 3), 200, np.uint8)  # светлый фон
    strip[10:25, 10:60] = 20  # «карты» (тёмные) — должны быть исключены
    strip[10:25, 120:180] = 20  # «ник» (тёмный) — крайний справа
    region = find_nick_region(strip, has_leading_cards=True)
    assert region is not None
    assert region.shape[1] < 90  # вернулся правый кластер (ник ~66px), а не весь диапазон 10..180


def test_find_nick_region_keeps_all_words_by_default() -> None:
    # обычная строка (не Win): двусловный ник, межсловный пробел НЕ должен срезать первое слово
    strip = np.full((37, 200, 3), 200, np.uint8)
    strip[10:25, 40:90] = 20  # первое слово
    strip[10:25, 105:170] = 20  # второе слово (пробел ~15px между словами)
    region = find_nick_region(strip)  # has_leading_cards=False — берём весь кластер
    assert region is not None
    assert region.shape[1] > 120  # взят весь ник от первого слова до второго, а не только второе
