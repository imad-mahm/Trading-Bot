"""Tests for the regime gate (the 200-day trend filter applied to a strategy).

The gate has two jobs:
  * it must FORCE AN EXIT promptly once price closes below the regime SMA, and
  * it must keep the strategy out of the market more than the ungated version.

We build data with a clear up-then-crash shape, run SMA crossover both gated and
ungated, and check both behaviours.
"""

import numpy as np
import pandas as pd

from src import indicators
from src.backtest import run_backtest
from src.data_loader import to_backtesting_format
from src.strategies.sma_crossover import SmaCrossover

REGIME_PERIOD = 30  # short, so the test data doesn't need to be huge


def _up_then_crash(n=300, seed=3) -> pd.DataFrame:
    """Rise for the first half (gets us long), then crash through the SMA."""
    rng = np.random.default_rng(seed)
    up = np.linspace(100, 220, n // 2)
    down = np.linspace(220, 80, n - n // 2)
    close = pd.Series(np.concatenate([up, down]) + rng.standard_normal(n) * 0.5)
    open_ = close.shift(1).fillna(close.iloc[0])
    high = pd.concat([open_, close], axis=1).max(axis=1) + 0.5
    low = pd.concat([open_, close], axis=1).min(axis=1) - 0.5
    index = pd.date_range("2021-01-01", periods=n, freq="D")
    return pd.DataFrame(
        {"open": open_.values, "high": high.values, "low": low.values,
         "close": close.values, "volume": 1000.0},
        index=index,
    )


def _held_at_close_mask(stats, n_bars: int) -> np.ndarray:
    """Bars at whose CLOSE a position was open. A trade fills at EntryBar's open
    and exits at ExitBar's open, so it is held at the close of bars
    EntryBar .. ExitBar-1."""
    held = np.zeros(n_bars, dtype=bool)
    for _, t in stats["_trades"].iterrows():
        held[int(t["EntryBar"]):int(t["ExitBar"])] = True
    return held


def test_gate_forces_exit_within_one_bar_of_regime_flip():
    df = _up_then_crash()
    params = {"fast": 5, "slow": 15, "use_regime_gate": True,
              "regime_sma_period": REGIME_PERIOD}
    _, stats = run_backtest(df, SmaCrossover, cash=1_000_000, commission=0.0, params=params)

    data = to_backtesting_format(df)
    close = data["Close"].to_numpy()
    sma = indicators.sma(data["Close"], REGIME_PERIOD).to_numpy()
    held = _held_at_close_mask(stats, len(close))

    # The gate must never let us sit long through TWO consecutive below-SMA
    # closes — it has to bail out on the next bar after the regime turns down.
    for i in range(1, len(close)):
        below_now = close[i] < sma[i]
        below_prev = close[i - 1] < sma[i - 1]
        if held[i] and below_now and below_prev:
            raise AssertionError(
                f"Position still held at bar {i} after two below-SMA closes — "
                "the regime gate failed to force an exit."
            )


def test_gate_reduces_time_in_market():
    df = _up_then_crash()
    common = {"fast": 5, "slow": 15}
    _, ungated = run_backtest(df, SmaCrossover, cash=1_000_000, commission=0.0,
                              params=common)
    _, gated = run_backtest(df, SmaCrossover, cash=1_000_000, commission=0.0,
                            params={**common, "use_regime_gate": True,
                                    "regime_sma_period": REGIME_PERIOD})
    assert gated["Exposure Time [%]"] <= ungated["Exposure Time [%]"], (
        "the regime gate should keep us in the market no more than the ungated run"
    )
