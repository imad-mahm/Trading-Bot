"""Donchian Channel Breakout (a classic trend-following 'turtle' system).

  * ENTRY: buy when price makes a new N-day HIGH (default N=55).
  * EXIT:  sell when price makes a new M-day LOW (default M=20).

The ATR stop-loss and 2%-risk position sizing from BaseStrategy still apply, so
a trade can also be closed early if its stop is hit.

Avoiding lookahead: the channel is built from the PREVIOUS N (or M) bars only —
we `.shift(1)` so today's own high/low isn't part of the level it has to break.
Otherwise "today's high >= the max-including-today" would be trivially true.
"""

import pandas as pd

from src.strategies.base import BaseStrategy


def _prior_high(values, period):
    """Highest value over the previous `period` bars (excluding the current one)."""
    return pd.Series(values).rolling(period).max().shift(1).to_numpy()


def _prior_low(values, period):
    """Lowest value over the previous `period` bars (excluding the current one)."""
    return pd.Series(values).rolling(period).min().shift(1).to_numpy()


class DonchianBreakout(BaseStrategy):
    entry_period = 55  # N: look-back for the breakout high (in candles)
    exit_period = 20   # M: look-back for the breakdown low (in candles)

    def init(self):
        super().init()  # ATR (+ regime SMA if gated)
        self.upper = self.I(_prior_high, self.data.High, self.entry_period, name="DonchUp")
        self.lower = self.I(_prior_low, self.data.Low, self.exit_period, name="DonchLow")

    def on_bar(self):
        # NaN comparisons are always False, so the warm-up bars are handled
        # automatically (no trades fire until the channels are valid).
        if not self.position:
            if self.data.High[-1] > self.upper[-1]:
                self.enter_long()
        else:
            if self.data.Low[-1] < self.lower[-1]:
                self.position.close()
