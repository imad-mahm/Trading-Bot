"""Exchange-engine tests against a simulated Binance (no network, no keys).

FakeBinance mimics the parts of ccxt's Binance spot API the broker uses,
including the awkward real-world details: buy fees taken in the coin, coins
locked under a resting stop, stops that trigger on price moves, limit prices a
gap can jump over, and responses lost AFTER an order really went through.
"""

import ccxt
import pandas as pd
import pytest

from src.live.broker import BinanceBroker, BrokerError, check_live_allowed
from src.live.exchange_engine import ExchangeEngine
from src.live.state import State

FEE = 0.001


class FakeBinance:
    def __init__(self, usdt=10_000.0, price=100.0):
        self.prices = {"BTC/USDT": price}
        self.free = {"USDT": usdt, "BTC": 0.0}
        self.used = {"USDT": 0.0, "BTC": 0.0}
        self.orders = {}
        self.next_id = 1
        self.lose_next_response = False
        self.reject_stops = False
        self.created = []

    # --- metadata ---
    def load_markets(self):
        return {"BTC/USDT": self.market("BTC/USDT")}

    def market(self, symbol):
        return {"symbol": symbol, "base": "BTC", "quote": "USDT",
                "precision": {"amount": 0.00001, "price": 0.01},
                "limits": {"amount": {"min": 0.00001}, "cost": {"min": 5.0}},
                "info": {"orderTypes": ["LIMIT", "MARKET", "STOP_LOSS_LIMIT"]}}

    def amount_to_precision(self, symbol, amount):
        return f"{int(amount * 100000) / 100000:.5f}"

    def price_to_precision(self, symbol, price):
        return f"{round(price, 2):.2f}"

    # --- reads ---
    def fetch_balance(self):
        total = {a: self.free[a] + self.used[a] for a in self.free}
        return {"free": dict(self.free), "used": dict(self.used), "total": total}

    def fetch_ticker(self, symbol):
        return {"last": self.prices[symbol]}

    def _find(self, order_id, params):
        cid = (params or {}).get("origClientOrderId")
        for o in self.orders.values():
            if (order_id and o["id"] == str(order_id)) or (cid and o["clientOrderId"] == cid):
                return o
        raise ccxt.OrderNotFound("Order does not exist.")

    def fetch_order(self, order_id, symbol=None, params=None):
        return dict(self._find(order_id, params))

    # --- writes ---
    def create_order(self, symbol, type_, side, amount, price=None, params=None):
        params = params or {}
        amount = float(amount)
        oid = str(self.next_id)
        self.next_id += 1
        o = {"id": oid, "clientOrderId": params.get("newClientOrderId"), "symbol": symbol,
             "type": type_, "side": side, "amount": amount, "filled": 0.0, "fees": [],
             "status": "open", "average": None, "cost": 0.0}
        p = self.prices[symbol]
        if type_ == "market":
            if side == "buy":
                cost = amount * p
                if cost > self.free["USDT"] + 1e-9:
                    raise ccxt.InsufficientFunds("Account has insufficient balance.")
                self.free["USDT"] -= cost
                self.free["BTC"] += amount * (1 - FEE)
                o["fees"] = [{"currency": "BTC", "cost": amount * FEE}]
            else:
                if amount > self.free["BTC"] + 1e-9:
                    raise ccxt.InsufficientFunds("Account has insufficient balance.")
                self.free["BTC"] -= amount
                self.free["USDT"] += amount * p * (1 - FEE)
                o["fees"] = [{"currency": "USDT", "cost": amount * p * FEE}]
            o.update(filled=amount, average=p, cost=amount * p, status="closed")
        elif type_ == "STOP_LOSS_LIMIT":
            if self.reject_stops:
                raise ccxt.InvalidOrder("Stop price would trigger immediately.")
            if amount > self.free["BTC"] + 1e-9:
                raise ccxt.InsufficientFunds("Account has insufficient balance.")
            self.free["BTC"] -= amount
            self.used["BTC"] += amount
            o.update(stopPrice=float(params["stopPrice"]), price=float(price))
        self.orders[oid] = o
        self.created.append(o)
        if self.lose_next_response:
            self.lose_next_response = False
            raise ccxt.NetworkError("Connection reset by peer")
        return dict(o)

    def cancel_order(self, order_id, symbol=None):
        o = self._find(order_id, None)
        if o["status"] != "open":
            raise ccxt.OrderNotFound("Unknown order sent.")
        o["status"] = "canceled"
        rest = o["amount"] - o["filled"]
        self.used["BTC"] -= rest
        self.free["BTC"] += rest
        return dict(o)

    # --- simulation ---
    def set_price(self, symbol, p):
        """Move the market; trigger any resting stop whose limit is reachable."""
        self.prices[symbol] = p
        for o in self.orders.values():
            if o["type"] == "STOP_LOSS_LIMIT" and o["status"] == "open" and p <= o["stopPrice"]:
                if p >= o["price"]:  # the limit is fillable
                    fill = max(p, o["price"])
                    self.used["BTC"] -= o["amount"]
                    self.free["USDT"] += o["amount"] * fill * (1 - FEE)
                    o.update(filled=o["amount"], average=fill, cost=o["amount"] * fill,
                             status="closed",
                             fees=[{"currency": "USDT", "cost": o["amount"] * fill * FEE}])


class Notes:
    def __init__(self):
        self.messages = []

    def send(self, text):
        self.messages.append(text)
        return True

    def any(self, word):
        return any(word.lower() in m.lower() for m in self.messages)


# Flat, then a 3-day-high breakout on row 4, then a slide to a 2-day low on row 7.
ROWS = [
    (100, 101, 99, 100), (100, 101, 99, 100), (100, 101, 99, 100), (100, 101, 99, 100),
    (100, 110, 99, 108),   # 4: breakout -> ENTER
    (108, 112, 107, 110),  # 5
    (110, 113, 109, 112),  # 6
    (112, 112, 100, 101),  # 7: new 2-day low -> EXIT
]


def _candles(n):
    idx = pd.date_range("2026-01-01", periods=len(ROWS), freq="D", tz="UTC")
    df = pd.DataFrame(ROWS, columns=["open", "high", "low", "close"], index=idx)
    df["volume"] = 1.0
    return df.iloc[:n]


def _config(tmp_path, **ex):
    return {
        "live": {"strategy": "donchian_breakout", "timeframe": "1d", "symbols": ["BTC/USDT"],
                 "params": {"entry_days": 3, "exit_days": 2, "regime_gate": False},
                 "daily_loss_limit": 0.5, "data_anomaly_threshold": 0.5},
        "risk": {"risk_per_trade": 0.02, "atr_period": 2, "atr_stop_multiplier": 2.0},
        "regime": {"sma_period_days": 200},
        "exchange_trading": {"environment": "testnet", "allocation_usd": 1000,
                             "max_position_usd": 1000, "stop_limit_buffer": 0.01,
                             "max_signal_age_hours": 6,
                             "state_file": str(tmp_path / "s.json"), **ex},
    }


class Harness:
    """Drive the engine one 'day' at a time against the fake exchange."""

    def __init__(self, tmp_path, price=100.0, **ex):
        self.fake = FakeBinance(price=price)
        self.config = _config(tmp_path, **ex)
        self.state = State(self.config["exchange_trading"]["state_file"])
        self.notes = Notes()
        self.n = 4
        self.extra_hours = 1
        self.engine = ExchangeEngine(
            self.config, self.state, BinanceBroker(self.fake, retry_wait=0), self.notes,
            fetch_fn=lambda s, t, l: _candles(self.n), now_fn=self._now)

    def _now(self):
        last = _candles(self.n).index[-1]
        return last + pd.Timedelta(days=1, hours=self.extra_hours)

    def tick(self, n=None):
        if n is not None:
            self.n = n
        self.engine.run_once()
        return self.state.portfolios["BTC/USDT"]

    def enter(self):
        self.tick(4)   # first run: initialise, flat
        return self.tick(5)  # breakout candle closed -> buy + stop


def test_entry_buys_and_places_exchange_stop(tmp_path):
    h = Harness(tmp_path)
    port = h.enter()
    pos = port.position
    assert pos is not None
    assert pos.entry_price == 100.0
    # Binance took the buy fee in BTC: the bot must track what it really holds.
    buy = h.fake.created[0]
    assert pos.units == pytest.approx(buy["amount"] * (1 - FEE))
    assert pos.entry_fee == pytest.approx(buy["amount"] * FEE * 100.0)
    # A real stop order rests on the exchange, locking the coins.
    stop = h.fake.orders[pos.stop_order_id]
    assert stop["type"] == "STOP_LOSS_LIMIT" and stop["status"] == "open"
    assert stop["stopPrice"] == pytest.approx(round(pos.stop_price, 2))
    assert stop["price"] < stop["stopPrice"]
    assert h.fake.used["BTC"] > 0
    assert stop["amount"] <= pos.units
    # Never spends more than the allocation.
    assert buy["amount"] * 100.0 <= 1000.0


def test_stop_fill_on_exchange_is_recorded(tmp_path):
    h = Harness(tmp_path)
    port = h.enter()
    stop_trigger = h.fake.orders[port.position.stop_order_id]["stopPrice"]
    h.fake.set_price("BTC/USDT", stop_trigger - 0.1)  # fires while the bot is "asleep"
    port = h.tick()
    assert port.position is None
    t = port.trades[-1]
    assert t["reason"] == "stop"
    assert t["exit_price"] == pytest.approx(stop_trigger - 0.1)
    assert t["pnl"] < 0
    assert h.notes.any("STOP")


def test_lost_buy_response_never_double_buys(tmp_path):
    h = Harness(tmp_path)
    h.tick(4)
    h.fake.lose_next_response = True        # order goes through, reply is lost
    port = h.tick(5)
    assert port.position is None and port.inflight is not None
    port = h.tick()                          # next cycle verifies by client id
    buys = [o for o in h.fake.created if o["type"] == "market" and o["side"] == "buy"]
    assert len(buys) == 1
    assert port.position is not None and port.inflight is None
    assert port.position.stop_order_id is not None


def test_order_that_never_arrived_is_cleared(tmp_path):
    h = Harness(tmp_path)
    port = h.tick(4)
    port.inflight = {"kind": "entry", "client_id": "tb-BTC-in-ghost", "atr": 2.0}
    port = h.tick()
    assert port.inflight is None and port.position is None


def test_rejected_order_clears_write_ahead(tmp_path):
    h = Harness(tmp_path)
    h.tick(4)
    h.fake.free["USDT"] = 0.0
    port = h.tick(5)
    assert port.position is None and port.inflight is None
    assert h.notes.any("skipped") or h.notes.any("rejected")


def test_stale_entry_is_not_chased(tmp_path):
    h = Harness(tmp_path)
    h.tick(4)
    h.extra_hours = 30                       # bot was down for a day after the breakout
    port = h.tick(5)
    assert port.position is None
    assert not [o for o in h.fake.created if o["side"] == "buy"]
    assert h.notes.any("not chasing")


def test_exit_signal_cancels_stop_then_sells(tmp_path):
    h = Harness(tmp_path)
    port = h.enter()
    stop_id = port.position.stop_order_id
    h.tick(6)
    h.fake.set_price("BTC/USDT", 101.0)
    port = h.tick(8)                         # row 7 makes a new 2-day low -> EXIT
    assert port.position is None
    assert port.trades[-1]["reason"] == "signal"
    assert h.fake.orders[stop_id]["status"] == "canceled"
    assert h.fake.used["BTC"] == pytest.approx(0.0, abs=1e-9)
    assert h.fake.free["BTC"] < 0.0001       # only rounding dust remains


def test_missing_stop_is_replaced(tmp_path):
    h = Harness(tmp_path)
    port = h.enter()
    old = port.position.stop_order_id
    h.fake.cancel_order(old)                 # someone cancelled it in the Binance app
    port = h.tick()
    assert port.position is not None
    assert port.position.stop_order_id not in (None, old)
    assert h.fake.orders[port.position.stop_order_id]["status"] == "open"
    assert h.notes.any("Re-placing")


def test_gap_through_stop_limit_sells_at_market(tmp_path):
    h = Harness(tmp_path)
    port = h.enter()
    limit = port.position.stop_limit_price
    h.fake.set_price("BTC/USDT", limit * 0.9)  # crash straight through the limit
    port = h.tick()
    assert port.position is None
    assert "gap" in port.trades[-1]["reason"]
    assert port.trades[-1]["exit_price"] == pytest.approx(limit * 0.9)


def test_coins_moved_outside_bot_halts(tmp_path):
    h = Harness(tmp_path)
    port = h.enter()
    h.fake.used["BTC"] = 0.0                 # e.g. sold by hand
    h.tick()
    assert h.state.halted
    assert h.notes.any("outside the bot")


def test_unprotectable_position_is_sold(tmp_path):
    h = Harness(tmp_path)
    h.tick(4)
    h.fake.reject_stops = True
    port = h.tick(5)
    assert port.position is None
    assert port.trades[-1]["reason"] == "no stop could be placed"
    assert h.notes.any("Could not place a stop")


def test_state_survives_restart(tmp_path):
    h = Harness(tmp_path)
    h.enter()
    reloaded = State.load(h.config["exchange_trading"]["state_file"])
    pos = reloaded.portfolios["BTC/USDT"].position
    assert pos is not None and pos.stop_order_id is not None


def test_live_requires_double_key(monkeypatch):
    monkeypatch.delenv("LIVE_TRADING_CONFIRM", raising=False)
    with pytest.raises(BrokerError):
        check_live_allowed({"environment": "live", "max_position_usd": 100})
    monkeypatch.setenv("LIVE_TRADING_CONFIRM", "yes-real-money")
    with pytest.raises(BrokerError):
        check_live_allowed({"environment": "live"})          # no position cap
    check_live_allowed({"environment": "live", "max_position_usd": 100})
    check_live_allowed({"environment": "testnet"})           # fake money: always fine
