"""Thin wrapper around backtesting.py that wires in our realism settings.

Realism baked in here (unchanged across phases):
  * Fees + slippage are charged on every trade (entry AND exit) via the engine's
    `commission` argument. We combine them because both scale with trade size.
  * `trade_on_close=False` (default) -> orders fill at the NEXT bar's open, never
    inside the signal bar. This is the no-lookahead guarantee.
  * `exclusive_orders=True` -> one position at a time.
"""

from __future__ import annotations

import pandas as pd
from backtesting import Backtest

from src.data_loader import to_backtesting_format


def effective_commission(fees: dict) -> float:
    """Combine the taker fee and slippage into one per-side cost fraction."""
    return float(fees.get("taker_fee", 0.0)) + float(fees.get("slippage", 0.0))


def make_backtest(df: pd.DataFrame, strategy_cls, cash: float, commission: float) -> Backtest:
    """Build a Backtest object from our (lower-case) OHLCV DataFrame."""
    data = to_backtesting_format(df)
    return Backtest(
        data,
        strategy_cls,
        cash=cash,
        commission=commission,
        trade_on_close=False,    # fill at next bar's open -> no lookahead
        exclusive_orders=True,   # one position at a time
        finalize_trades=True,    # close any open trade at the end for fair stats
    )


def run_backtest(
    df: pd.DataFrame,
    strategy_cls,
    cash: float,
    commission: float,
    params: dict | None = None,
):
    """Run one strategy over one DataFrame and return (Backtest, stats)."""
    bt = make_backtest(df, strategy_cls, cash, commission)
    stats = bt.run(**(params or {}))
    return bt, stats
