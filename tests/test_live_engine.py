"""Engine tests with synthetic candles (no network).

Covers: next-open fills, missed-candle catch-up, stop-loss triggering, and the
daily-loss-limit halt.
"""

import pandas as pd

from src.live import engine
from src.live.portfolio import Portfolio, Position
from src.live.state import State

# Donchian with tiny periods so small synthetic data triggers signals.
PARAMS = {"entry_period": 3, "exit_period": 2, "regime_gate": False}
RISK = {"risk_per_trade": 0.02, "atr_period": 2, "atr_stop_multiplier": 2.0}
COMMISSION = 0.0015


class DummyNotifier:
    def __init__(self):
        self.messages = []

    def send(self, text):
        self.messages.append(text)
        return True


def _candles(rows, start="2023-01-01"):
    idx = pd.date_range(start, periods=len(rows), freq="D", tz="UTC")
    df = pd.DataFrame(rows, columns=["open", "high", "low", "close"], index=idx)
    df["volume"] = 1.0
    return df


# A flat run, then a breakout high on candle 4 (index 4).
_BREAKOUT_ROWS = [
    (100, 101, 99, 100),
    (100, 101, 99, 100),
    (100, 101, 99, 100),
    (100, 101, 99, 100),
    (100, 110, 99, 108),   # 4: new 3-day high -> ENTER signal
    (108, 112, 107, 110),  # 5: fill at open 108
    (110, 113, 109, 112),  # 6
    (112, 114, 110, 113),  # 7: forming
]


def _fresh_port(last_idx):
    p = Portfolio(symbol="BTC/USDT", start_time="2023-01-01T00:00:00+00:00",
                  start_equity=10000.0, cash=10000.0, peak_equity=10000.0)
    p.last_processed_close = last_idx.isoformat()
    return p


def test_entry_fills_at_next_candle_open():
    df = _candles(_BREAKOUT_ROWS)
    closed, forming = df.iloc[:-1], df.iloc[-1]
    port = _fresh_port(closed.index[3])  # start acting from candle 4

    engine.advance_symbol(port, closed, forming, "donchian_breakout", PARAMS,
                          RISK, COMMISSION, halted=False, notifier=DummyNotifier())

    assert port.position is not None
    # Signal was on candle 4; the fill must be at candle 5's OPEN (108).
    assert port.position.entry_price == 108.0
    assert port.position.stop_price < 108.0   # ATR stop sits below entry
    assert port.position.units > 0
    assert port.cash < 10000.0
    assert len(port.trades) == 0              # still holding


def test_catch_up_processes_all_missed_candles():
    df = _candles(_BREAKOUT_ROWS)
    closed, forming = df.iloc[:-1], df.iloc[-1]
    # Pretend the bot was down since candle 1 — it must replay 2,3,4,5,6 in order.
    port = _fresh_port(closed.index[1])

    engine.advance_symbol(port, closed, forming, "donchian_breakout", PARAMS,
                          RISK, COMMISSION, halted=False, notifier=DummyNotifier())

    assert port.total_periods == 5           # candles 2,3,4,5,6 processed
    assert port.position is not None          # the missed breakout was caught up
    assert port.last_processed_close == closed.index[-1].isoformat()


def test_stop_loss_triggers_on_intra_candle_low():
    rows = list(_BREAKOUT_ROWS)
    # Candle 6 crashes below the ATR stop. High stays under the 3-day high (112)
    # so it does NOT also trigger a fresh breakout re-entry.
    rows[6] = (108, 108, 50, 60)
    df = _candles(rows)
    closed, forming = df.iloc[:-1], df.iloc[-1]
    port = _fresh_port(closed.index[3])

    notifier = DummyNotifier()
    engine.advance_symbol(port, closed, forming, "donchian_breakout", PARAMS,
                          RISK, COMMISSION, halted=False, notifier=notifier)

    assert port.position is None
    assert len(port.trades) == 1
    assert port.trades[0]["reason"] == "stop"
    # Stop fills exactly at the recorded stop price.
    assert port.trades[0]["exit_price"] == port.trades[0]["stop_price"]
    assert any("STOP" in m for m in notifier.messages)


def test_halt_suppresses_new_entries():
    df = _candles(_BREAKOUT_ROWS)
    closed, forming = df.iloc[:-1], df.iloc[-1]
    port = _fresh_port(closed.index[3])

    engine.advance_symbol(port, closed, forming, "donchian_breakout", PARAMS,
                          RISK, COMMISSION, halted=True, notifier=DummyNotifier())

    assert port.position is None  # halted -> no entry taken


def test_daily_loss_limit_flattens_and_halts():
    state = State(":memory:")
    port = Portfolio(symbol="BTC/USDT", start_time="2024-01-01T00:00:00+00:00",
                     start_equity=10000.0, cash=0.0, peak_equity=10000.0)
    # Fully invested: 100 units bought at 100 (=10000 notional).
    port.position = Position(units=100.0, entry_price=100.0, entry_time="t0",
                             stop_price=80.0, entry_fee=15.0, atr_at_entry=10.0)
    port.equity_snapshots = [["2024-01-01T00:00:00+00:00", 10000.0]]
    state.portfolios["BTC/USDT"] = port

    notifier = DummyNotifier()
    # Price drops to 90 -> equity 9000, a 10% fall from the 24h peak of 10000.
    engine._enforce_daily_loss(state, port, cur_price=90.0,
                               now_iso="2024-01-01T06:00:00+00:00",
                               limit=0.05, commission=COMMISSION, notifier=notifier)

    assert state.halted is True
    assert port.position is None                      # flattened
    assert any("DAILY LOSS" in m for m in notifier.messages)
