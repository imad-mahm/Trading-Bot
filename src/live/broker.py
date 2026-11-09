"""Binance broker — the ONLY module that sends real orders.

Everything exchange-specific lives here, behind a small interface the exchange
engine uses: buy, sell, place/check/cancel a stop-loss, read balances. The
strategy, indicators and risk sizing stay exactly where they were.

Environments (``exchange_trading.environment`` in config.yaml):
  * ``testnet`` — Binance Spot Testnet (testnet.binance.vision). Fake funds,
    keys from logging in with GitHub. Prices can differ from the real market.
  * ``demo``    — Binance Demo Trading (demo.binance.com). Fake funds, but the
    order book mirrors the real market, so fills and stops behave realistically.
  * ``live``    — REAL MONEY. Refused unless the double-key check passes (see
    ``check_live_allowed``).

Stops are real exchange orders: right after every entry the bot places a
STOP_LOSS_LIMIT sell on Binance, so the exchange enforces the stop 24/7 even if
the bot is down. The limit sits a small buffer below the trigger so it fills in
fast markets; if price ever gaps through the limit, the engine notices and sells
at market.

API keys come ONLY from environment variables (or .env), never config.yaml:
  BINANCE_API_KEY / BINANCE_API_SECRET
"""

from __future__ import annotations

import logging
import math
import os
import time
from dataclasses import dataclass

import ccxt

log = logging.getLogger("trade.broker")

ENVIRONMENTS = ("testnet", "demo", "live")
LIVE_CONFIRM_VAR = "LIVE_TRADING_CONFIRM"
LIVE_CONFIRM_VALUE = "yes-real-money"


class BrokerError(RuntimeError):
    """A broker action failed in a way the engine must handle (not a crash)."""


@dataclass
class Fill:
    """The result of a completed market order, in plain numbers."""
    order_id: str
    client_id: str | None
    units: float        # units we actually hold/sold after any fee taken in coin
    price: float        # average fill price
    fee_usd: float      # total fee converted to the quote currency (USDT)


def check_live_allowed(cfg: dict) -> None:
    """Refuse real-money trading unless BOTH keys are turned:
    config says ``environment: live`` AND the env var LIVE_TRADING_CONFIRM is set
    to the exact confirmation phrase. Also requires a per-order dollar cap."""
    if cfg.get("environment") != "live":
        return
    if os.environ.get(LIVE_CONFIRM_VAR) != LIVE_CONFIRM_VALUE:
        raise BrokerError(
            f"environment is 'live' (REAL MONEY) but {LIVE_CONFIRM_VAR} is not set to "
            f"'{LIVE_CONFIRM_VALUE}'. Refusing to start.")
    if not cfg.get("max_position_usd"):
        raise BrokerError("Live trading requires exchange_trading.max_position_usd to be set.")


def make_exchange(cfg: dict) -> ccxt.Exchange:
    """Build an authenticated ccxt Binance client for the configured environment."""
    env = cfg.get("environment", "testnet")
    if env not in ENVIRONMENTS:
        raise BrokerError(f"Unknown environment '{env}'. Use one of {ENVIRONMENTS}.")
    check_live_allowed(cfg)

    key = os.environ.get("BINANCE_API_KEY", "")
    secret = os.environ.get("BINANCE_API_SECRET", "")
    if not key or not secret:
        raise BrokerError("BINANCE_API_KEY / BINANCE_API_SECRET are not set (see .env.example).")

    ex = ccxt.binance({
        "apiKey": key,
        "secret": secret,
        "enableRateLimit": True,
        "options": {
            "defaultType": "spot",
            # Spot only: the testnet has no futures endpoints, and we never use them.
            "fetchMarkets": {"types": ["spot"]},
            "adjustForTimeDifference": True,  # avoids "timestamp ahead of server" errors
        },
    })
    if env == "testnet":
        ex.set_sandbox_mode(True)
    elif env == "demo":
        ex.enable_demo_trading(True)
    return ex


class BinanceBroker:
    """Thin, testable wrapper around a ccxt exchange object."""

    def __init__(self, exchange, stop_limit_buffer: float = 0.01, retries: int = 3,
                 retry_wait: float = 2.0):
        self.ex = exchange
        self.stop_limit_buffer = stop_limit_buffer
        self.retries = retries
        self.retry_wait = retry_wait
        self._loaded = False

    # ----- plumbing -----

    def _call(self, fn, *args, _tries=None, **kwargs):
        """Retry transient network errors; let real rejections through at once."""
        tries = _tries or self.retries
        for attempt in range(1, tries + 1):
            try:
                return fn(*args, **kwargs)
            except (ccxt.NetworkError, ccxt.RequestTimeout) as exc:
                if attempt == tries:
                    raise BrokerError(f"network error after {attempt} tries: {exc}") from exc
                log.warning("Network error (%s), retry %d/%d", exc, attempt, tries)
                time.sleep(self.retry_wait)

    def load(self) -> None:
        if not self._loaded:
            self._call(self.ex.load_markets)
            self._loaded = True

    def market(self, symbol: str) -> dict:
        self.load()
        return self.ex.market(symbol)

    # ----- market rules -----

    def floor_amount(self, symbol: str, amount: float) -> float:
        """Round an amount DOWN to the exchange's step size (never oversell)."""
        if amount <= 0:
            return 0.0
        self.load()
        step = (self.market(symbol).get("precision") or {}).get("amount")
        if step:
            floored = math.floor(amount / step + 1e-9) * step
            return float(self.ex.amount_to_precision(symbol, floored)) if floored > 0 else 0.0
        return float(self.ex.amount_to_precision(symbol, amount))

    def min_order(self, symbol: str) -> tuple[float, float]:
        """(minimum units, minimum notional in USDT) for an order."""
        limits = self.market(symbol).get("limits") or {}
        min_amt = (limits.get("amount") or {}).get("min") or 0.0
        min_cost = (limits.get("cost") or {}).get("min") or 0.0
        return float(min_amt), float(min_cost)

    # ----- reads -----

    def balances(self) -> dict:
        """{asset: {"free": x, "used": y, "total": z}} for non-zero assets."""
        bal = self._call(self.ex.fetch_balance)
        out = {}
        for asset, total in (bal.get("total") or {}).items():
            if total:
                out[asset] = {"free": float(bal["free"].get(asset) or 0),
                              "used": float(bal["used"].get(asset) or 0),
                              "total": float(total)}
        return out

    def free(self, asset: str) -> float:
        bal = self._call(self.ex.fetch_balance)
        return float((bal.get("free") or {}).get(asset) or 0.0)

    def total(self, asset: str) -> float:
        bal = self._call(self.ex.fetch_balance)
        return float((bal.get("total") or {}).get(asset) or 0.0)

    def last_price(self, symbol: str) -> float:
        t = self._call(self.ex.fetch_ticker, symbol)
        price = t.get("last") or t.get("close")
        if not price or price <= 0:
            raise BrokerError(f"no valid price for {symbol}")
        return float(price)

    def fetch_order(self, symbol: str, order_id: str | None = None,
                    client_id: str | None = None) -> dict | None:
        """Look an order up by exchange id or by our client id. None if unknown."""
        params = {"origClientOrderId": client_id} if (client_id and not order_id) else {}
        try:
            return self._call(self.ex.fetch_order, order_id, symbol, params)
        except ccxt.OrderNotFound:
            return None

    # ----- fills -----

    def _fee_usd(self, order: dict, symbol: str, price: float) -> tuple[float, float]:
        """Return (fee in USDT, fee taken in the base coin)."""
        m = self.market(symbol)
        base, quote = m["base"], m["quote"]
        fees = order.get("fees") or ([order["fee"]] if order.get("fee") else [])
        usd, in_base = 0.0, 0.0
        for f in fees:
            cost = float(f.get("cost") or 0)
            cur = f.get("currency")
            if not cost:
                continue
            if cur == quote:
                usd += cost
            elif cur == base:
                usd += cost * price
                in_base += cost
            else:  # e.g. BNB fee discount: price it in USDT
                try:
                    usd += cost * self.last_price(f"{cur}/{quote}")
                except Exception:  # noqa: BLE001
                    usd += float(order.get("cost") or 0) * 0.001
        return usd, in_base

    def to_fill(self, order: dict, symbol: str) -> Fill:
        filled = float(order.get("filled") or 0)
        if filled <= 0:
            raise BrokerError(f"order {order.get('id')} has no fill (status {order.get('status')})")
        price = float(order.get("average") or 0) or float(order.get("cost") or 0) / filled
        fee_usd, fee_base = self._fee_usd(order, symbol, price)
        units = filled - fee_base if order.get("side") == "buy" else filled
        return Fill(order_id=str(order.get("id")), client_id=order.get("clientOrderId"),
                    units=units, price=price, fee_usd=fee_usd)

    def _ensure_closed(self, order: dict, symbol: str) -> dict:
        """Market orders fill immediately, but the create response can be thin;
        re-fetch until filled/closed (bounded)."""
        for _ in range(5):
            if order.get("status") in ("closed", "canceled", "expired", "rejected") and order.get("filled") is not None:
                return order
            time.sleep(self.retry_wait if self.retry_wait else 0)
            fetched = self.fetch_order(symbol, order_id=order.get("id"))
            if fetched:
                order = fetched
        return order

    def market_buy(self, symbol: str, units: float, client_id: str) -> Fill:
        # Never retried: a lost response may still mean the order went through.
        # The engine's write-ahead record resolves it by client id instead.
        order = self._call(self.ex.create_order, symbol, "market", "buy", units, None,
                           {"newClientOrderId": client_id}, _tries=1)
        return self.to_fill(self._ensure_closed(order, symbol), symbol)

    def market_sell(self, symbol: str, units: float, client_id: str) -> Fill:
        order = self._call(self.ex.create_order, symbol, "market", "sell", units, None,
                           {"newClientOrderId": client_id}, _tries=1)
        return self.to_fill(self._ensure_closed(order, symbol), symbol)

    # ----- stops -----

    def place_stop(self, symbol: str, units: float, stop_price: float, client_id: str) -> dict:
        """Place a resting stop-loss sell on the exchange. Returns
        {"id", "client_id", "stop_price", "limit_price"}."""
        stop_p = float(self.ex.price_to_precision(symbol, stop_price))
        limit_p = float(self.ex.price_to_precision(symbol, stop_price * (1 - self.stop_limit_buffer)))
        order = self._call(
            self.ex.create_order, symbol, "STOP_LOSS_LIMIT", "sell", units, limit_p,
            {"stopPrice": stop_p, "timeInForce": "GTC", "newClientOrderId": client_id},
            _tries=1)
        return {"id": str(order["id"]), "client_id": client_id,
                "stop_price": stop_p, "limit_price": limit_p}

    def cancel(self, symbol: str, order_id: str) -> None:
        try:
            self._call(self.ex.cancel_order, order_id, symbol)
        except ccxt.OrderNotFound:
            pass  # already gone (filled or cancelled) — the caller re-checks

    def withdrawals_enabled(self) -> bool | None:
        """True/False if the API key can withdraw funds (live only); None if
        the endpoint isn't available (testnet/demo)."""
        try:
            r = self.ex.sapi_get_account_apirestrictions()
            return bool(r.get("enableWithdrawals"))
        except Exception:  # noqa: BLE001
            return None
