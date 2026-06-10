"""Trend Regime Filter — a deliberately tiny, slow strategy.

The whole rule:
  * Be fully LONG while price closes ABOVE its 200-day SMA.
  * Move to CASH while it closes BELOW.

That's it. It trades rarely (only when the long-term trend flips), which is the
point — almost no fees, and it sidesteps the deepest bear-market drawdowns by
sitting in cash. The 200-day SMA itself is the exit, so there's no ATR stop and
no 2%-risk sizing here; it simply holds (nearly) all-in when the trend is up.

Optional `buffer`: require price to be a little BEYOND the SMA to flip, which
reduces "whipsaw" (flip-flopping when price hovers right on the line). With
buffer=0 it flips exactly at the SMA. With buffer=0.01 it needs to be 1% above
to go long and 1% below to go to cash (a small dead-band / hysteresis).
"""

from backtesting import Strategy

from src.strategies.base import _sma_adapter


class TrendFilter(Strategy):
    sma_period = 200   # in candles (run.py converts 200 days -> candles)
    buffer = 0.0       # 0.01 = require 1% beyond the SMA to flip

    def init(self):
        self.sma = self.I(_sma_adapter, self.data.Close, self.sma_period, name="SMA")

    def next(self):
        price = self.data.Close[-1]
        sma = self.sma[-1]
        if not (sma > 0):  # SMA warm-up
            return

        upper = sma * (1 + self.buffer)  # must clear this to go long
        lower = sma * (1 - self.buffer)  # must break this to go to cash

        if not self.position:
            if price > upper:
                self.buy(size=0.9999)  # go (nearly) all-in
        else:
            if price < lower:
                self.position.close()  # back to cash
