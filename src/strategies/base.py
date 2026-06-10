"""Shared strategy base class.

Real signal strategies (SMA crossover, Donchian breakout) inherit from
`BaseStrategy`, which wraps backtesting.py's `Strategy` and adds, in ONE place:

  * the ATR stop-loss + 2%-risk position sizing (from src/risk.py), and
  * an optional **regime gate** (the 200-day trend filter).

A concrete strategy only implements `on_bar()` — its entry/exit signal logic.
The "how much to buy", "where's the stop", and "are we allowed to be long right
now" decisions are handled here, using the exact functions the live bot reuses.

The regime gate (when `use_regime_gate=True`):
  * blocks new long entries while price is below the regime SMA, and
  * force-closes any open position as soon as price closes below it.
Because the engine fills at the NEXT bar's open, a "close below" on bar N exits
at the open of bar N+1 — still no lookahead.

No-lookahead note: we never set trade_on_close=True, so a signal from bar N's
close trades at bar N+1's open. Verified in tests/test_no_lookahead.py.
"""

import pandas as pd
from backtesting import Strategy

from src import indicators, risk


# --- Adapters so backtesting.py's self.I() (which passes numpy arrays) can call
# our pandas-based indicators. Shared by the subclasses too.

def _atr_adapter(high, low, close, period):
    return indicators.atr(
        pd.Series(high), pd.Series(low), pd.Series(close), period
    ).to_numpy()


def _sma_adapter(values, period):
    return indicators.sma(pd.Series(values), period).to_numpy()


class BaseStrategy(Strategy):
    """Base class holding risk + regime parameters and the shared entry helper.

    All of these are class attributes so backtesting.py can override them per
    run (e.g. `bt.run(fast=50, use_regime_gate=True)`).
    """

    # Risk settings (overridden from config.yaml at run time).
    risk_per_trade = 0.02
    atr_period = 14
    atr_stop_multiplier = 2.0

    # Regime gate settings.
    use_regime_gate = False
    regime_sma_period = 200  # in candles (run.py converts 200 days -> candles)

    def init(self):
        """Pre-compute shared indicators. Subclasses call super().init() first,
        then add their own signal indicators on top."""
        self.atr = self.I(
            _atr_adapter,
            self.data.High,
            self.data.Low,
            self.data.Close,
            self.atr_period,
            name="ATR",
            overlay=False,
        )
        # Only compute the regime SMA if the gate is actually switched on.
        if self.use_regime_gate:
            self.regime_sma = self.I(
                _sma_adapter, self.data.Close, self.regime_sma_period,
                name="RegimeSMA",
            )

    def next(self):
        """Runs every bar. Applies the regime gate, then the strategy's signal."""
        # Regime gate: if we're long but price has closed below the trend SMA,
        # get out now (and don't look for new entries this bar).
        if self.use_regime_gate and self.position and not self.regime_above():
            self.position.close()
            return
        self.on_bar()

    def on_bar(self):
        """Strategy-specific entry/exit logic. Subclasses MUST implement this."""
        raise NotImplementedError("Subclasses must implement on_bar().")

    def regime_above(self) -> bool:
        """True if price is currently above the regime SMA (or the gate is off).
        During the SMA warm-up (NaN) we treat the regime as 'not above'."""
        if not self.use_regime_gate:
            return True
        sma = self.regime_sma[-1]
        if not (sma > 0):
            return False
        return self.data.Close[-1] > sma

    def enter_long(self):
        """Open a long sized by the 2%-risk rule with an ATR stop.

        The ONLY place strategies should open trades, so risk + the regime gate
        always apply.
        """
        # Respect the regime gate: no new longs while below the trend SMA.
        if self.use_regime_gate and not self.regime_above():
            return

        price = self.data.Close[-1]
        atr_value = self.atr[-1]
        if not (atr_value > 0):  # skip the ATR warm-up period
            return

        stop_price = risk.atr_stop_price(
            price, atr_value, self.atr_stop_multiplier, direction="long"
        )
        fraction = risk.position_size_fraction(self.risk_per_trade, price, stop_price)
        if fraction > 0:
            self.buy(size=fraction, sl=stop_price)
