"""SMA Crossover strategy (a classic trend follower).

  * Go LONG when the fast SMA crosses ABOVE the slow SMA (uptrend starting).
  * EXIT when the fast crosses back BELOW the slow (trend fading).

On daily candles the default is the well-known 50/200 "golden cross". Position
sizing, the ATR stop, and the optional regime gate all come from BaseStrategy.
"""

from backtesting.lib import crossover

from src.strategies.base import BaseStrategy, _sma_adapter


class SmaCrossover(BaseStrategy):
    # Periods in candles (run.py converts the config's day-values).
    fast = 50
    slow = 200

    def init(self):
        super().init()  # ATR (+ regime SMA if gated)
        self.sma_fast = self.I(_sma_adapter, self.data.Close, self.fast, name="SMA_fast")
        self.sma_slow = self.I(_sma_adapter, self.data.Close, self.slow, name="SMA_slow")

    def on_bar(self):
        # crossover(a, b) is True on the bar where a crosses above b.
        if crossover(self.sma_fast, self.sma_slow):
            if not self.position:
                self.enter_long()
        elif crossover(self.sma_slow, self.sma_fast):
            if self.position:
                self.position.close()
