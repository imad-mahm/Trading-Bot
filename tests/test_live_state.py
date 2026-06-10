"""Test that state survives a save/reload round-trip (crash safety)."""

from src.live.portfolio import Portfolio, Position
from src.live.state import State


def test_state_roundtrip(tmp_path):
    path = tmp_path / "live_state.json"
    state = State(path)
    state.halted = True
    state.halt_reason = "test halt"

    port = Portfolio(symbol="ETH/USDT", start_time="2024-01-01T00:00:00+00:00",
                     start_equity=10000.0, cash=8000.0, peak_equity=10500.0)
    port.position = Position(units=1.5, entry_price=2000.0, entry_time="t0",
                             stop_price=1800.0, entry_fee=4.5, atr_at_entry=80.0)
    port.pending = {"action": "EXIT", "signal_time": "t1", "reason": "exit signal"}
    port.trades = [{"symbol": "ETH/USDT", "pnl": 123.45, "reason": "stop"}]
    port.last_processed_close = "2024-03-01T00:00:00+00:00"
    port.max_drawdown = -0.12
    port.total_periods = 60
    port.time_in_market_periods = 25
    state.portfolios["ETH/USDT"] = port
    state.save()

    # Reload into a brand-new object, as a restart would.
    loaded = State.load(path)
    assert loaded.halted is True
    assert loaded.halt_reason == "test halt"
    lp = loaded.portfolios["ETH/USDT"]
    assert lp.cash == 8000.0
    assert lp.position.units == 1.5
    assert lp.position.stop_price == 1800.0
    assert lp.pending["action"] == "EXIT"
    assert lp.trades[0]["pnl"] == 123.45
    assert lp.last_processed_close == "2024-03-01T00:00:00+00:00"
    assert lp.max_drawdown == -0.12
    assert lp.total_periods == 60
    assert lp.time_in_market_periods == 25


def test_load_missing_file_returns_empty(tmp_path):
    state = State.load(tmp_path / "does_not_exist.json")
    assert state.halted is False
    assert state.portfolios == {}
