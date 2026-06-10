"""Tests for the Donchian breakout channel logic.

Two things to prove:
  1. The channel uses the PREVIOUS N bars only (it `.shift(1)`s), so today's own
     high/low never counts toward the level it must break.
  2. A real breakout actually triggers an entry, and that entry's signal bar had
     price breaking above the prior N-day high (i.e. the rule fired correctly).
"""

import numpy as np
import pandas as pd

from src.backtest import run_backtest
from src.data_loader import to_backtesting_format
from src.strategies.donchian_breakout import (
    DonchianBreakout, _prior_high, _prior_low,
)


def test_prior_high_excludes_current_bar():
    highs = [1, 2, 3, 4, 5]
    result = _prior_high(highs, 3)
    # First 3 are NaN (need 3 prior bars + the shift). Then:
    assert np.isnan(result[0]) and np.isnan(result[1]) and np.isnan(result[2])
    assert result[3] == 3.0   # max(1,2,3), NOT including bar 3's value (4)
    assert result[4] == 4.0   # max(2,3,4), NOT including bar 4's value (5)


def test_prior_low_excludes_current_bar():
    lows = [5, 4, 3, 2, 1]
    result = _prior_low(lows, 3)
    assert result[3] == 3.0   # min(5,4,3), excludes bar 3's value (2)
    assert result[4] == 2.0   # min(4,3,2), excludes bar 4's value (1)


def _breakout_data(n=200, seed=1) -> pd.DataFrame:
    """A long rise (forces a new-high breakout) then a fall (forces a new low)."""
    rng = np.random.default_rng(seed)
    up = np.linspace(100, 200, n // 2)
    down = np.linspace(200, 120, n - n // 2)
    close = pd.Series(np.concatenate([up, down]) + rng.standard_normal(n) * 0.3)
    open_ = close.shift(1).fillna(close.iloc[0])
    high = pd.concat([open_, close], axis=1).max(axis=1) + 0.5
    low = pd.concat([open_, close], axis=1).min(axis=1) - 0.5
    index = pd.date_range("2021-01-01", periods=n, freq="D")
    return pd.DataFrame(
        {"open": open_.values, "high": high.values, "low": low.values,
         "close": close.values, "volume": 1000.0},
        index=index,
    )


def test_breakout_triggers_entry_on_new_high():
    df = _breakout_data()
    _, stats = run_backtest(
        df, DonchianBreakout, cash=1_000_000, commission=0.0,
        params={"entry_period": 20, "exit_period": 10},
    )
    trades = stats["_trades"]
    assert len(trades) > 0, "a clear breakout should produce at least one trade"

    data = to_backtesting_format(df)
    prior_high = pd.Series(_prior_high(data["High"].to_numpy(), 20))

    # An entry fills at the bar AFTER its signal, so the signal bar is EntryBar-1.
    first_entry_bar = int(trades.iloc[0]["EntryBar"])
    signal_bar = first_entry_bar - 1
    assert data["High"].iloc[signal_bar] > prior_high.iloc[signal_bar], (
        "the entry's signal bar should have broken above the prior 20-bar high"
    )
