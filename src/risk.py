"""Position sizing and stop-loss rules.

This module is intentionally PURE (plain functions, no backtest-engine code) so
the exact same rules can be imported by the live trading bot in a later phase.
Test it, trust it, reuse it everywhere.
"""

from __future__ import annotations


def atr_stop_price(
    entry_price: float,
    atr_value: float,
    multiplier: float = 2.0,
    direction: str = "long",
) -> float:
    """Where to place the stop-loss, based on volatility (ATR).

    For a long trade the stop sits `multiplier x ATR` BELOW the entry price.
    More volatile market -> bigger ATR -> wider stop (so normal wiggles don't
    knock us out). This phase is long-only, but the short branch is included so
    the live bot can reuse it untouched.
    """
    distance = multiplier * atr_value
    if direction == "long":
        return entry_price - distance
    elif direction == "short":
        return entry_price + distance
    raise ValueError("direction must be 'long' or 'short'")


def position_size_units(
    equity: float,
    risk_per_trade: float,
    entry_price: float,
    stop_price: float,
) -> float:
    """How many UNITS of the asset to buy so that hitting the stop loses no more
    than `risk_per_trade` (e.g. 0.02 = 2%) of current equity.

    The maths:
        risk_dollars   = equity * risk_per_trade        # most we'll lose
        risk_per_unit  = entry_price - stop_price        # loss per unit if stopped
        units          = risk_dollars / risk_per_unit

    Returns 0 if the stop is not below the entry (which would be invalid).
    The live bot uses THIS function to know how much coin to order.
    """
    risk_per_unit = entry_price - stop_price
    if risk_per_unit <= 0:
        return 0.0
    risk_dollars = equity * risk_per_trade
    return risk_dollars / risk_per_unit


def position_size_fraction(
    risk_per_trade: float,
    entry_price: float,
    stop_price: float,
    max_fraction: float = 0.99,
) -> float:
    """The same idea as above, but expressed as a FRACTION of equity to deploy.

    The backtesting.py engine wants a fraction (a number between 0 and 1), so we
    convert. Notice equity cancels out of the maths entirely:

        fraction = risk_per_trade / stop_distance_pct

    where stop_distance_pct = (entry - stop) / entry.

    We cap the result at `max_fraction` (default 0.99) because we use NO
    leverage — you can never deploy more than the cash you have.
    """
    stop_distance = entry_price - stop_price
    if stop_distance <= 0:
        return 0.0
    stop_distance_pct = stop_distance / entry_price
    fraction = risk_per_trade / stop_distance_pct
    return min(fraction, max_fraction)
