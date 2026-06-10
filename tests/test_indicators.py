"""Tests for the indicator maths and the risk rules.

These check the *numbers*, so a beginner can trust the building blocks before
worrying about the backtest engine.
"""

import numpy as np
import pandas as pd
import pytest

from src import indicators, risk


def test_sma_matches_manual_average():
    s = pd.Series([1, 2, 3, 4, 5], dtype=float)
    result = indicators.sma(s, 3)
    # First two are NaN (not enough data); then rolling averages.
    assert np.isnan(result.iloc[0]) and np.isnan(result.iloc[1])
    assert result.iloc[2] == pytest.approx(2.0)   # (1+2+3)/3
    assert result.iloc[3] == pytest.approx(3.0)   # (2+3+4)/3
    assert result.iloc[4] == pytest.approx(4.0)   # (3+4+5)/3


def test_ema_first_value_equals_first_price():
    s = pd.Series([10, 11, 12, 13], dtype=float)
    result = indicators.ema(s, 2)
    # With adjust=False the EMA seeds on the first value.
    assert result.iloc[0] == pytest.approx(10.0)
    assert result.iloc[-1] > result.iloc[0]       # rising series -> rising EMA


def test_rsi_is_100_when_price_only_rises():
    # A monotonically rising series has no losses, so RSI saturates at 100.
    s = pd.Series(np.arange(1, 30), dtype=float)
    result = indicators.rsi(s, 14)
    assert result.dropna().iloc[-1] == pytest.approx(100.0)


def test_rsi_stays_within_bounds():
    rng = np.random.default_rng(42)
    s = pd.Series(100 + rng.standard_normal(200).cumsum())
    result = indicators.rsi(s, 14).dropna()
    assert result.min() >= 0.0
    assert result.max() <= 100.0


def test_atr_is_positive_and_tracks_volatility():
    n = 100
    high = pd.Series(np.linspace(10, 20, n) + 1.0)
    low = pd.Series(np.linspace(10, 20, n) - 1.0)
    close = pd.Series(np.linspace(10, 20, n))
    result = indicators.atr(high, low, close, 14).dropna()
    assert (result > 0).all()


def test_position_size_fraction_matches_risk_rule():
    # Stop is 5% below entry; risking 2% of equity => deploy 2%/5% = 40%.
    frac = risk.position_size_fraction(
        risk_per_trade=0.02, entry_price=100.0, stop_price=95.0
    )
    assert frac == pytest.approx(0.40)


def test_position_size_fraction_is_capped_at_no_leverage():
    # A very tight stop would imply >100% deployment; must be capped below 1.
    frac = risk.position_size_fraction(
        risk_per_trade=0.02, entry_price=100.0, stop_price=99.9, max_fraction=0.99
    )
    assert frac <= 0.99


def test_atr_stop_price_sits_below_entry_for_longs():
    stop = risk.atr_stop_price(entry_price=100.0, atr_value=2.0, multiplier=2.0)
    assert stop == pytest.approx(96.0)  # 100 - 2*2


def test_position_size_units_zero_when_stop_above_entry():
    units = risk.position_size_units(10000, 0.02, entry_price=100, stop_price=105)
    assert units == 0.0
