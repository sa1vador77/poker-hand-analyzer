"""Точка входа для запуска пакета как модуля: ``python -m poker_analyzer``."""

import sys

from poker_analyzer.cli import main

if __name__ == "__main__":
    sys.exit(main())
