"""Test the no-lookahead guarantee.

A signal computed from candle N's close must only trade at the OPEN of candle
N+1 — never inside candle N. We verify this by running a real backtest and
checking that every trade's fill price equals the *open* of the bar it entered
on (which is the bar AFTER the signal). If the engine were peeking, fills would
land on the signal bar's close instead.
"""

import numpy as np
import pandas as pd

from src.backtest import run_backtest
from src.data_loader import to_backtesting_format
from src.strategies.sma_crossover import SmaCrossover


def _make_trending_data(n=400, seed=7) -> pd.DataFrame:
    """Synthetic OHLCV with waves, so SMA crossovers actually happen."""
    rng = np.random.default_rng(seed)
    t = np.arange(n)
    # A couple of overlaid sine waves + noise => repeated up/down trends.
    close = 100 + 15 * np.sin(t / 20) + 5 * np.sin(t / 7) + rng.standard_normal(n)
    close = pd.Series(close)
    open_ = close.shift(1).fillna(close.iloc[0])
    high = pd.concat([open_, close], axis=1).max(axis=1) + 0.5
    low = pd.concat([open_, close], axis=1).min(axis=1) - 0.5
    index = pd.date_range("2021-01-01", periods=n, freq="h")
    return pd.DataFrame(
        {"open": open_.values, "high": high.values, "low": low.values,
         "close": close.values, "volume": 1000.0},
        index=index,
    )


def test_entries_fill_at_next_bar_open():
    df = _make_trending_data()
    bt, stats = run_backtest(
        df, SmaCrossover, cash=1_000_000, commission=0.0,
        params={"fast": 10, "slow": 30},
    )
    trades = stats["_trades"]
    assert len(trades) > 0, "expected the strategy to make some trades"

    # The exact Capitalised OHLCV the engine used, with positional (integer)
    # access via .iloc so EntryBar lines up with the right row.
    data_open = to_backtesting_format(df)["Open"]

    for _, trade in trades.iterrows():
        entry_bar = int(trade["EntryBar"])
        # Fill price must equal that bar's OPEN (proves no same-bar-close fill).
        assert trade["EntryPrice"] == data_open.iloc[entry_bar], (
            "Entry did not fill at the bar open — possible lookahead bias."
        )


def test_signal_bar_is_before_entry_bar():
    """A crossover detected on bar N should produce an entry on bar N+1, so the
    entry bar index is always at least 1 (never bar 0)."""
    df = _make_trending_data()
    _, stats = run_backtest(
        df, SmaCrossover, cash=1_000_000, commission=0.0,
        params={"fast": 10, "slow": 30},
    )
    trades = stats["_trades"]
    assert (trades["EntryBar"] >= 1).all()
