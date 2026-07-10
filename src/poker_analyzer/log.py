"""Настройка логирования проекта.

Логирование — основной способ наблюдать за пайплайном, поэтому вынесено в отдельный
модуль с единым форматом. Сообщения логов — на русском. Помимо вывода в stderr пишем в
файл ``debug/poker.log`` (с ротацией) — чтобы было что приложить для разбора советов.
"""

from __future__ import annotations

import logging
import logging.handlers
import sys

from poker_analyzer.config import DEBUG_DIR

_LOG_FORMAT = "%(asctime)s.%(msecs)03d [%(levelname)s] %(name)s: %(message)s"
_DATE_FORMAT = "%H:%M:%S"
LOG_FILE = DEBUG_DIR / "poker.log"


def setup_logging(level: int = logging.INFO) -> None:
    """Настраивает корневой логгер: stderr + файл ``debug/poker.log``, единый формат.

    Вызывается один раз при старте приложения (см. :mod:`cli`). Файл
    ОБНУЛЯЕТСЯ на каждом запуске (плюс чистятся бэкапы ротации) — разбор всегда по
    текущей сессии, логи прошлых запусков не копятся. Ротация (≤ 5 МБ × 3) ограничивает
    рост внутри одной сессии.

    Уровни: stderr — INFO (события), ФАЙЛ — DEBUG с миллисекундами — полная трасса
    разбора для отладки. DEBUG включён только для пакета ``poker_analyzer``,
    сторонние библиотеки не шумят.
    """
    formatter = logging.Formatter(_LOG_FORMAT, datefmt=_DATE_FORMAT)

    stream = logging.StreamHandler(sys.stderr)
    stream.setFormatter(formatter)
    stream.setLevel(level)

    root = logging.getLogger()
    root.handlers.clear()  # на случай повторного вызова не плодим хендлеры
    root.addHandler(stream)
    root.setLevel(level)
    # DEBUG-записи рождаются только у нашего пакета; до stderr их не пускает уровень
    # stream-хендлера, в файл (DEBUG) — проходят.
    logging.getLogger("poker_analyzer").setLevel(logging.DEBUG)

    # Файл-лог — чтобы было что прислать для разбора. Файл и бэкапы ротации прошлых
    # сессий сносим РУКАМИ: mode="w" не работает — RotatingFileHandler при maxBytes>0
    # молча принудительно открывает в "a" (защита от потери логов в стдлибе).
    try:
        DEBUG_DIR.mkdir(parents=True, exist_ok=True)
        LOG_FILE.unlink(missing_ok=True)
        for backup in LOG_FILE.parent.glob(LOG_FILE.name + ".*"):
            backup.unlink(missing_ok=True)
        file_handler = logging.handlers.RotatingFileHandler(
            LOG_FILE, maxBytes=5_000_000, backupCount=2, encoding="utf-8"
        )
        file_handler.setFormatter(formatter)
        file_handler.setLevel(logging.DEBUG)  # полная телеметрия — только в файл
        root.addHandler(file_handler)
        logging.getLogger(__name__).info("Лог пишется в файл: %s", LOG_FILE)
    except OSError:
        logging.getLogger(__name__).warning("Не удалось открыть файл-лог: %s", LOG_FILE)
