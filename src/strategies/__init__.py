"""Strategy package.

`STRATEGIES` maps the names used on the command line / in config.yaml to the
strategy classes. `DISPLAY_NAMES` are the friendlier labels shown in tables and
chart legends.
"""

from .buy_and_hold import BuyAndHold
from .trend_filter import TrendFilter
from .sma_crossover import SmaCrossover
from .donchian_breakout import DonchianBreakout

STRATEGIES = {
    "buy_and_hold": BuyAndHold,
    "trend_filter": TrendFilter,
    "sma_crossover": SmaCrossover,
    "donchian_breakout": DonchianBreakout,
}

DISPLAY_NAMES = {
    "buy_and_hold": "Buy & Hold",
    "trend_filter": "Trend Filter",
    "sma_crossover": "SMA Crossover",
    "donchian_breakout": "Donchian Breakout",
}
