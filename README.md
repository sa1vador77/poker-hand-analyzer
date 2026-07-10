# poker-hand-analyzer

An offline Texas Hold'em hand analyzer for study and review — in the spirit of hand
trackers and solvers. Feed it a screenshot of a hand log (or a ready-made hand state)
and it reconstructs the hand, computes equities against modeled opponent ranges and
suggests a decision for the spot.

No live table connection: everything works on static input.

## Features

- **Native hand evaluator (C++ / nanobind).** A 7-card evaluator with Python bindings,
  roughly 10× faster than a pure-Python implementation. Equity is computed by exact
  board enumeration for small problems and Monte-Carlo sampling (fixed seed, deterministic)
  for large ones, switching automatically by problem size.
- **Opponent range model.** Ranges in standard notation (`"TT+, AQs+, KQs"`), assigned per
  opponent from their observed actions (aggressor / caller / limper / …) and preflop context
  (open from early/late, 3-bet, call vs open, blind defense). Postflop the range is narrowed
  street by street by combo strength on the actual board, adjusted for bet sizing
  (polarization), betting line (barrels, check-raise, donk) and board texture (wet/dry).
- **Positional preflop charts** (open / call / 3-bet / 4-bet by position) with
  stack-depth awareness: a raise that commits the stack is only advised when the hand
  is ahead of the range that actually continues for stacks.
- **Decision advisor.** Heads-up postflop: EV-based bet sizing with fold equity.
  Multiway: realized equity vs pot odds, with a value-bet threshold derived from equity
  math (`eq > 1/(N+1)` vs N callers), domination / reverse-implied-odds penalties, and
  a hand simulator (`engine/sim.py`) used as independent ground truth for calibration.
- **CV table-state recognition from screenshots.** Template matching (OpenCV), not OCR:
  the log font is fixed and the vocabulary is closed, so `cv2.matchTemplate` reads
  timestamps, keywords, card ranks/suits and amounts reliably. Players are told apart by
  the *image* of their nickname (correlation + glyph-by-glyph reading against a font
  atlas) — no text recognition of names is needed, only stable per-player identity.
- **Board texture analysis** (monotone / two-tone / connectedness / paired / broadway)
  feeding the range-narrowing thresholds.
- **Live calibration via `.env`.** All poker constants (ranges, narrowing thresholds,
  realization multipliers, value-bet margins) have sane defaults in code and can be
  overridden without touching code — see `.env.example` for the full annotated key list.

## Architecture

```
screenshot ─→ vision ─→ parsing ─→ identity ─→ engine ─→ advice
```

| Layer | What it does |
|---|---|
| `vision/` | row cropping, template matching, recognition of time/keywords/cards/amounts, nickname glyph reading |
| `parsing/` | splitting the log into rows, mapping recognized rows to typed `LogEvent`s |
| `identity/` | distinguishing players by nickname image; session-stable player ids |
| `engine/` | hand state accumulation, equity (C++), range model, texture, preflop charts, the advisor |
| `pipeline.py` | the per-row conveyor: screenshot → `RowResult` per log line |
| `cli.py` | entry point: analyze a screenshot file |

## Getting started

Requires Python 3.13+, [uv](https://docs.astral.sh/uv/) and a C++17 compiler
(the native module is built automatically by CMake via scikit-build-core).

```bash
uv sync                                  # environment + native _equity module build
uv run poker-analyzer screenshot.png     # parse a hand-log screenshot
uv run poker-analyzer screenshot.png --dump   # dump per-stage crops to debug/ (calibration)
```

Layout coordinates and matching thresholds live in `src/poker_analyzer/config.py`
(`Layout`, `Thresholds`) and are calibrated per log window size. Recognition templates
(digits, keywords, ranks, suits, nickname glyph atlas) live in `data/templates/`.

### Using the engine directly

The engine layer is independent of vision — you can drive it from Python with a
ready-made hand state:

```python
from poker_analyzer.engine.equity import cards, equity_vs_ranges
from poker_analyzer.engine.ranges import parse_range

r = equity_vs_ranges(
    cards("As Ks"),                       # hero
    cards("Kh 7d 2c"),                    # board
    [parse_range("22+, A2s+, KTs+, QJs, ATo+")],  # one opponent range
)
print(r.equity)
```

### Calibrating the value-bet threshold

`scripts/tune_value_thresholds.py` runs a panel of canonical spots through the actual
advisor functions, scores each candidate margin with the independent simulator
(including a reverse-implied-odds discount) and reports per-iteration metrics until
the optimum converges:

```bash
uv run python scripts/tune_value_thresholds.py
```

## Development

```bash
uv run ruff format && uv run ruff check   # style
uv run ty check                           # types
uv run pytest                             # 226 tests: engine, vision, parsing, pipeline
```

Editing `.py` files is picked up immediately; editing `native/*.cpp` requires `uv sync`
to rebuild the module.

## Notes

- Code comments and log messages are in Russian; identifiers are English.
- The recognition layer expects the specific log format it was calibrated for
  (fixed-pitch rows, red amounts, dark nicknames). Adapting to another format means
  re-cutting templates and recalibrating `Layout` — the pipeline itself is format-agnostic.
