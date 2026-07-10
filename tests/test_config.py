"""Тесты загрузчика ``.env`` (config._load_dotenv / _strip_inline_comment)."""

from __future__ import annotations

from poker_analyzer.config import _strip_inline_comment


def test_strips_inline_comment_after_space() -> None:
    # Главный краш-кейс: строковый диапазон с комментарием в той же строке (как в .env.example).
    v = _strip_inline_comment("TT+, AQs+, AKo, AJs  # премиум при поз=— → рейз-для-вэлью")
    assert v == "TT+, AQs+, AKo, AJs"


def test_strips_inline_comment_numeric() -> None:
    assert _strip_inline_comment("0.06      # запас над break-even") == "0.06"


def test_keeps_value_without_comment() -> None:
    assert _strip_inline_comment("99+, ATs+, KQs, AJo+") == "99+, ATs+, KQs, AJo+"
    assert _strip_inline_comment("0.5") == "0.5"


def test_hash_without_leading_space_is_part_of_value() -> None:
    # «#» вплотную к значению (без пробела) — часть значения, не комментарий.
    assert _strip_inline_comment("a#b") == "a#b"


def test_quoted_value_kept_whole() -> None:
    # Закавыченное значение берём целиком (комментарий внутри кавычек — часть строки).
    assert _strip_inline_comment('"a # b"') == '"a # b"'


def test_env_example_parses_as_real_env(tmp_path: object) -> None:
    # .env.example должен копироваться в .env и грузиться БЕЗ падений (инвариант «скопируй в .env»):
    # каждый строковый ключ-диапазон проходит parse_range, числовой — float/int.
    import os

    from poker_analyzer.config import PROJECT_ROOT, _load_dotenv
    from poker_analyzer.engine.ranges import parse_range

    example = PROJECT_ROOT / ".env.example"
    if not example.exists():
        return  # шаблона нет — пропускаем
    saved = dict(os.environ)
    try:
        for k in list(os.environ):
            if k.startswith(("OPP_", "MW_VALUE", "COMMIT_", "PREMIUM_", "NARROW_")):
                del os.environ[k]
        _load_dotenv(example)
        # Ключи-диапазоны: распарситься без ошибок (значение без комментария).
        for key in ("OPP_STACKOFF", "PREMIUM_FLOOR", "OPP_JAMMER"):
            if key in os.environ:
                assert parse_range(os.environ[key])  # не должно бросать
        for key in ("MW_VALUE_MARGIN", "MW_VALUE_FLOOR", "COMMIT_RAISE_FRAC"):
            if key in os.environ:
                assert float(os.environ[key]) >= 0.0  # число, не «число # коммент»
    finally:
        os.environ.clear()
        os.environ.update(saved)
