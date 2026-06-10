"""Buy & Hold — the benchmark.

Buy once at the very start with (almost) all the cash and never sell. Every
other strategy has to BEAT this after fees to be worth running. It does not use
stops or position sizing — it's deliberately the simplest possible approach.
"""

from backtesting import Strategy


class BuyAndHold(Strategy):
    def init(self):
        # No indicators needed.
        pass

    def next(self):
        # On the first bar where we have no position, go (nearly) all-in.
        # 0.9999 leaves a sliver of cash so fees on the entry don't cause a
        # rejected order.
        if not self.position:
            self.buy(size=0.9999)
