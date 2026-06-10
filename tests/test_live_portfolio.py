"""Tests for the paper portfolio's accounting (fills, fees, P&L)."""

import pytest

from src.live.portfolio import Portfolio

COMMISSION = 0.0015  # taker_fee + slippage, same as the backtest


def _fresh():
    return Portfolio(symbol="BTC/USDT", start_time="2024-01-01T00:00:00+00:00",
                     start_equity=10000.0, cash=10000.0, peak_equity=10000.0)


def test_open_position_charges_notional_plus_fee():
    p = _fresh()
    p.open_position(units=1.0, price=1000.0, time_iso="t0", stop_price=900.0,
                    commission_rate=COMMISSION, atr=50.0)
    # Spent 1000 notional + 1.5 fee.
    assert p.cash == pytest.approx(10000 - 1000 - 1.5)
    assert p.fees_paid == pytest.approx(1.5)
    assert p.position.units == 1.0
    assert p.position.stop_price == 900.0


def test_close_position_pnl_includes_both_fees():
    p = _fresh()
    p.open_position(1.0, 1000.0, "t0", 900.0, COMMISSION, 50.0)
    trade = p.close_position(1100.0, "t1", COMMISSION, reason="signal")
    # Gross gain 100; fees = 1.5 (entry) + 1.65 (exit) = 3.15; net = 96.85.
    assert trade["pnl"] == pytest.approx(100 - 1.5 - 1.65)
    assert trade["reason"] == "signal"
    assert p.position is None
    assert p.realized_pnl == pytest.approx(trade["pnl"])
    # Cash back to 10000 + net pnl.
    assert p.cash == pytest.approx(10000 + trade["pnl"])


def test_equity_and_unrealized_pnl():
    p = _fresh()
    p.open_position(2.0, 100.0, "t0", 90.0, COMMISSION, 5.0)
    # Cash = 10000 - 200 - 0.3 = 9799.7; position worth 2*110 = 220 at price 110.
    assert p.equity(110.0) == pytest.approx(9799.7 + 220.0)
    assert p.unrealized_pnl(110.0) == pytest.approx(20.0)


def test_max_drawdown_tracks_worst_dip():
    p = _fresh()
    p.open_position(1.0, 100.0, "t0", 90.0, COMMISSION, 5.0)
    p.record_equity(100.0, "2024-01-01T00:00:00+00:00")  # ~baseline
    p.record_equity(80.0, "2024-01-01T01:00:00+00:00")   # big dip
    p.record_equity(95.0, "2024-01-01T02:00:00+00:00")   # partial recovery
    assert p.max_drawdown < 0  # we recorded a drawdown
    # Drawdown is measured against the running peak (the $10,000 starting peak).
    assert p.max_drawdown == pytest.approx(p.equity(80.0) / p.peak_equity - 1, abs=1e-9)
