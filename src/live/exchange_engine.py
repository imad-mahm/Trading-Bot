"""The exchange engine: the same strategy, but real orders on Binance.

REUSE, DON'T REWRITE — the decision logic is the paper engine's
``compute_signal`` (which itself reuses the backtest strategy's own channel
functions), sizing is ``risk.py`` via ``_size_and_stop``, and the stop is
``risk.atr_stop_price``. Only *execution* differs:

  * Paper mode replays missed candles and fills at historical opens. A real
    exchange can't trade in the past, so here a missed ENTRY is never chased
    (it's logged), while a missed EXIT is acted on immediately — the safe side.
  * Entries are market orders sized from the fill price actually available.
  * The stop is a real STOP_LOSS_LIMIT order resting on Binance, placed right
    after every entry, so it's enforced 24/7 even if this bot is offline.

Crash safety — a write-ahead record (``port.inflight``) is saved BEFORE any
order is sent. If the bot dies between sending an order and recording its
result, the next cycle looks the order up by its client id and finishes the
bookkeeping, so it can never double-buy or forget a position.

Every cycle also RECONCILES with the exchange: is our stop still there? did it
fire? does the account still hold the coins we think we hold? Any surprise
halts trading and alerts you rather than guessing.
"""

from __future__ import annotations

import logging

import ccxt
import pandas as pd

from src import indicators, risk
from src.data_loader import fetch_recent_ohlcv, split_forming, timeframe_seconds
from src.live.broker import BrokerError, Fill
from src.live.engine import (
    _entry_reason, _is_anomalous, _size_and_stop, _warmup_needed, compute_signal,
    resolve_params,
)

log = logging.getLogger("trade.engine")

FEE_HEADROOM = 0.003   # keep this much cash spare so fees never cause a rejection


# --------------------------------------------------------------------------- #
# Small helpers
# --------------------------------------------------------------------------- #

def _cid(symbol: str, kind: str, when: pd.Timestamp) -> str:
    """Deterministic client order id (Binance allows 36 chars of [A-Za-z0-9_-])."""
    base = symbol.split("/")[0][:6]
    return f"tb-{base}-{kind}-{when.strftime('%y%m%d%H%M%S')}"[:36]


def _merge_fills(a: Fill | None, b: Fill | None) -> Fill | None:
    """Combine two partial fills (e.g. a half-filled stop + a market sell)."""
    if a is None:
        return b
    if b is None:
        return a
    units = a.units + b.units
    price = (a.units * a.price + b.units * b.price) / units if units else b.price
    return Fill(order_id=f"{a.order_id}+{b.order_id}", client_id=b.client_id,
                units=units, price=price, fee_usd=a.fee_usd + b.fee_usd)


def _fill_to_dict(f: Fill | None) -> dict | None:
    return None if f is None else dict(f.__dict__)


def _fill_from_dict(d: dict | None) -> Fill | None:
    return None if not d else Fill(**d)


def _notify_entry(notifier, port, env):
    p = port.position
    notifier.send(
        f"🟢 <b>ENTER</b> {port.symbol} [{env}]\n"
        f"price ${p.entry_price:,.2f}  size {p.units:.6f} (~${p.units * p.entry_price:,.0f})\n"
        f"stop ${p.stop_price:,.2f}  fee ${p.entry_fee:,.2f}")


def _notify_exit(notifier, trade, label, env):
    emoji = "🛑" if "STOP" in label.upper() else "🔴"
    notifier.send(
        f"{emoji} <b>EXIT</b> {trade['symbol']} ({label}) [{env}]\n"
        f"price ${trade['exit_price']:,.2f}  pnl ${trade['pnl']:,.2f} "
        f"({trade['return_pct']:+.2f}%)")


def _halt(state, notifier, reason):
    state.halted = True
    state.halt_reason = reason
    notifier.send(f"⛔ <b>TRADING HALTED</b>\n{reason}")
    log.error(reason)


# --------------------------------------------------------------------------- #
# The engine
# --------------------------------------------------------------------------- #

class ExchangeEngine:
    def __init__(self, config, state, broker, notifier, fetch_fn=None, now_fn=None):
        self.config = config
        self.cfg = config["exchange_trading"]
        self.env = self.cfg.get("environment", "testnet")
        self.state = state
        self.broker = broker
        self.notifier = notifier
        live = config["live"]
        self.strategy = live["strategy"]
        self.timeframe = live["timeframe"]
        self.params = resolve_params(config)
        self.risk_cfg = config["risk"]
        chain = self.cfg.get("data_exchanges") or live.get("exchanges")
        self.fetch_fn = fetch_fn or (lambda s, t, l: fetch_recent_ohlcv(s, t, l, exchanges=chain))
        self.now_fn = now_fn or (lambda: pd.Timestamp.now(tz="UTC"))

    # ----- order lifecycle -----

    def _send(self, port, kind, client_id, extra, send):
        """Write-ahead, send, and return the Fill (or None if the outcome is
        unknown — the next cycle's `_resolve_inflight` will finish it)."""
        port.inflight = {"kind": kind, "client_id": client_id,
                         "sent_at": self.now_fn().isoformat(), **extra}
        self.state.save()
        try:
            return send()
        except (ccxt.InvalidOrder, ccxt.InsufficientFunds, ccxt.AuthenticationError,
                ccxt.PermissionDenied, ccxt.BadSymbol) as exc:
            # Definitely rejected: the order does not exist. Safe to forget.
            port.inflight = None
            self.state.save()
            self.notifier.send(f"⚠️ {kind.upper()} order rejected for {port.symbol}: {exc}")
            return None
        except Exception as exc:  # noqa: BLE001 — unknown outcome; keep the record
            self.notifier.send(
                f"⚠️ {kind.upper()} order for {port.symbol} outcome unknown ({exc}). "
                "Will verify with the exchange next cycle.")
            return None

    def _resolve_inflight(self, port) -> bool:
        """Finish bookkeeping for an order sent before a crash/timeout.
        Returns True if the symbol is clear to continue this cycle."""
        inf = port.inflight
        if not inf:
            return True
        order = self.broker.fetch_order(port.symbol, client_id=inf["client_id"])
        if order is None:
            log.info("%s: in-flight %s order %s never reached the exchange; clearing.",
                     port.symbol, inf["kind"], inf["client_id"])
            port.inflight = None
            self.state.save()
            return True
        if order.get("status") == "open":
            log.info("%s: in-flight order %s still open; waiting.", port.symbol, inf["client_id"])
            return False
        fill = None
        if float(order.get("filled") or 0) > 0:
            fill = self.broker.to_fill(order, port.symbol)
        if inf["kind"] == "entry":
            if fill:
                self._complete_entry(port, fill, inf["atr"])
            else:
                port.inflight = None
        else:  # exit
            prior = _fill_from_dict(inf.get("prior_fill"))
            total = _merge_fills(prior, fill)
            if total and port.position:
                self._complete_exit(port, total, inf.get("reason", "exit"))
            else:
                port.inflight = None
        self.state.save()
        return True

    def _complete_entry(self, port, fill: Fill, atr: float):
        now_iso = self.now_fn().isoformat()
        stop = risk.atr_stop_price(fill.price, atr, self.risk_cfg["atr_stop_multiplier"])
        port.open_position_filled(fill.units, fill.price, now_iso, stop, fill.fee_usd, atr)
        port.inflight = None
        self.state.save()
        _notify_entry(self.notifier, port, self.env)
        self._protect(port)

    def _complete_exit(self, port, fill: Fill, reason: str):
        trade = port.close_position_filled(fill.price, self.now_fn().isoformat(),
                                           fill.fee_usd, reason, units=fill.units)
        trade["order_ids"] = fill.order_id
        port.inflight = None
        self.state.save()
        _notify_exit(self.notifier, trade, reason, self.env)

    # ----- entries -----

    def _enter(self, port, hist, signal_ts):
        symbol = port.symbol
        atr = float(indicators.atr(hist["high"], hist["low"], hist["close"],
                                   self.risk_cfg["atr_period"]).iloc[-1])
        if not atr > 0:
            return
        quote = self.broker.market(symbol)["quote"]
        price = self.broker.last_price(symbol)
        units, stop = _size_and_stop(port.cash, self.risk_cfg, price, atr)
        if stop >= price:
            return
        cap = self.cfg.get("max_position_usd")
        if cap:
            units = min(units, float(cap) / price)
        budget = min(port.cash, self.broker.free(quote)) / (1 + FEE_HEADROOM)
        units = self.broker.floor_amount(symbol, min(units, budget / price))
        min_amt, min_cost = self.broker.min_order(symbol)
        if units <= 0 or units < min_amt or units * price < min_cost:
            self.notifier.send(
                f"⚠️ Entry signal for {symbol} skipped: order too small "
                f"({units:.8f} units ≈ ${units * price:,.2f}; exchange minimum "
                f"${min_cost:,.2f}). Check allocation and free {quote}.")
            return
        cid = _cid(symbol, "in", signal_ts)
        fill = self._send(port, "entry", cid,
                          {"atr": atr, "reason": _entry_reason(self.strategy, self.params)},
                          lambda: self.broker.market_buy(symbol, units, cid))
        if fill:
            self._complete_entry(port, fill, atr)

    # ----- stops -----

    def _protect(self, port):
        """Make sure an exchange stop order guards the open position."""
        pos = port.position
        symbol = port.symbol
        price = self.broker.last_price(symbol)
        if price <= pos.stop_price:
            # Already through the stop (fast market): don't place, just get out.
            self.exit_now(port, "stop (price already below stop)")
            return
        units = self.broker.floor_amount(symbol, pos.units)
        now = self.now_fn()
        last_exc = None
        for attempt in range(3):
            cid = _cid(symbol, f"sl{(pos.stop_replacements * 3 + attempt) % 1000}", now)
            try:
                o = self.broker.place_stop(symbol, units, pos.stop_price, cid)
                pos.stop_order_id, pos.stop_client_id = o["id"], o["client_id"]
                pos.stop_limit_price = o["limit_price"]
                pos.stop_replacements += 1
                self.state.save()
                log.info("%s: stop placed at %.2f (limit %.2f), order %s",
                         symbol, o["stop_price"], o["limit_price"], o["id"])
                return
            except Exception as exc:  # noqa: BLE001
                last_exc = exc
                log.warning("%s: placing stop failed (%s), attempt %d/3", symbol, exc, attempt + 1)
        # Never sit in an unprotected position.
        self.notifier.send(f"❗ Could not place a stop for {symbol} ({last_exc}). "
                           "Selling the position rather than holding it unprotected.")
        self.exit_now(port, "no stop could be placed")

    def _reconcile(self, port) -> None:
        """Check the stop order and the coin balance against what we believe."""
        pos = port.position
        if pos is None:
            return
        symbol = port.symbol
        if not pos.stop_order_id:
            self._protect(port)
            return
        order = self.broker.fetch_order(symbol, order_id=pos.stop_order_id)
        status = (order or {}).get("status")
        filled = float((order or {}).get("filled") or 0)

        if status == "closed" and filled > 0:
            fill = self.broker.to_fill(order, symbol)
            self._complete_exit(port, fill, "stop")
            return
        if order is None or status in ("canceled", "expired", "rejected"):
            if filled > 0:  # partially filled, then cancelled: sell the rest
                self.exit_now(port, "stop (partial)")
                return
            self.notifier.send(f"⚠️ Stop order for {symbol} is gone ({status or 'not found'}). Re-placing it.")
            pos.stop_order_id = pos.stop_client_id = None
            self._protect(port)
            return

        # Stop is resting. Is someone else moving our coins?
        base = self.broker.market(symbol)["base"]
        held = self.broker.total(base)
        if held < pos.units * 0.98 - filled:
            _halt(self.state, self.notifier,
                  f"{symbol}: account holds {held:.8f} {base} but the bot's position is "
                  f"{pos.units:.8f}. Coins were moved outside the bot. Investigate, then "
                  "`trade.py resume` (or `trade.py adopt-flat` if you sold them yourself).")
            return

        # Gap protection: price fell through the stop's limit, so it may never fill.
        price = self.broker.last_price(symbol)
        if pos.stop_limit_price and price < pos.stop_limit_price:
            self.notifier.send(f"⚠️ {symbol} gapped below the stop limit "
                               f"(${price:,.2f} < ${pos.stop_limit_price:,.2f}). Selling at market.")
            self.exit_now(port, "stop (gap, market exit)")

    # ----- exits -----

    def exit_now(self, port, reason: str) -> None:
        """Cancel any stop, then sell whatever remains at market."""
        pos = port.position
        if pos is None:
            return
        symbol = port.symbol
        prior = None
        if pos.stop_order_id:
            self.broker.cancel(symbol, pos.stop_order_id)
            o = self.broker.fetch_order(symbol, order_id=pos.stop_order_id)
            if o and float(o.get("filled") or 0) > 0:
                prior = self.broker.to_fill(o, symbol)
                if o.get("status") == "closed":  # the stop fired before we cancelled
                    self._complete_exit(port, prior, "stop")
                    return
            pos.stop_order_id = None
            self.state.save()
        base = self.broker.market(symbol)["base"]
        remaining = pos.units - (prior.units if prior else 0.0)
        units = self.broker.floor_amount(symbol, min(remaining, self.broker.free(base)))
        min_amt, min_cost = self.broker.min_order(symbol)
        price = self.broker.last_price(symbol)
        if units <= 0 or units < min_amt or units * price < min_cost:
            if prior:
                self._complete_exit(port, prior, reason)
            else:
                _halt(self.state, self.notifier,
                      f"{symbol}: tried to exit ({reason}) but only {units:.8f} {base} is sellable.")
            return
        cid = _cid(symbol, "out", self.now_fn())
        fill = self._send(port, "exit", cid,
                          {"reason": reason, "prior_fill": _fill_to_dict(prior)},
                          lambda: self.broker.market_sell(symbol, units, cid))
        if fill:
            self._complete_exit(port, _merge_fills(prior, fill), reason)

    # ----- the cycle -----

    def run_once(self) -> None:
        now = self.now_fn()
        limit = min(1000, max(400, _warmup_needed(self.params, self.risk_cfg) + 50))
        period = pd.Timedelta(seconds=timeframe_seconds(self.timeframe))
        max_age = pd.Timedelta(hours=float(self.cfg.get("max_signal_age_hours", 6)))

        for symbol in self.config["live"]["symbols"]:
            port = self.state.ensure_portfolio(symbol, now.isoformat(),
                                               float(self.cfg["allocation_usd"]))
            try:
                self._cycle_symbol(port, now, limit, period, max_age)
            except Exception as exc:  # noqa: BLE001 — one symbol must not stop the other
                log.exception("%s cycle error: %s", symbol, exc)
                self.notifier.send(f"❗ {symbol} cycle error (bot still running): {exc}")
            self.state.save()

    def _cycle_symbol(self, port, now, limit, period, max_age):
        symbol = port.symbol
        # 1. Finish anything left half-done, then check the exchange agrees with us.
        if not self._resolve_inflight(port):
            return
        self._reconcile(port)

        # 2. Market data (same public candles the paper bot and backtest use).
        try:
            recent = self.fetch_fn(symbol, self.timeframe, limit)
        except Exception as exc:  # noqa: BLE001
            self.notifier.send(f"⚠️ Data fetch failed for {symbol}: {exc}. Skipping cycle.")
            return
        closed, _ = split_forming(recent, self.timeframe)
        if len(closed) < 2:
            return
        if _is_anomalous(closed, self.config["live"]["data_anomaly_threshold"]):
            self.notifier.send(f"⚠️ Anomalous price data for {symbol}; skipped this cycle.")
            return
        if port.last_processed_close is None:  # first run: start flat, from now
            port.last_processed_close = closed.index[-1].isoformat()
            log.info("Initialised %s exchange portfolio ($%.2f allocation).", symbol, port.cash)
            return

        # 3. Signals for every candle that closed since we last looked.
        new_closed = closed[closed.index > pd.Timestamp(port.last_processed_close)]
        in_pos = port.position is not None
        exit_ts, enter = None, None
        for ts in new_closed.index:
            hist = closed.loc[:ts]
            action = compute_signal(self.strategy, self.params, hist, in_pos)
            port.total_periods += 1
            port.time_in_market_periods += int(in_pos)
            if action == "EXIT" and in_pos:
                exit_ts = ts
            elif action == "ENTER" and not in_pos:
                enter = (ts, hist)
        if len(new_closed):
            port.last_processed_close = new_closed.index[-1].isoformat()
            self.state.save()

        # 4. Act. Exits are never skipped; entries only on a FRESH signal.
        if exit_ts is not None and port.position is not None:
            self.exit_now(port, "signal")
        elif enter is not None and port.position is None:
            ts, hist = enter
            fresh = ts == closed.index[-1] and now - (ts + period) <= max_age
            if self.state.halted:
                log.info("%s: entry signal ignored — trading halted.", symbol)
            elif not fresh:
                self.notifier.send(f"ℹ️ {symbol}: missed a stale entry signal from {ts:%Y-%m-%d} "
                                   "(bot was offline); not chasing it.")
            else:
                self._enter(port, hist, ts)

        # 5. Equity, drawdown and the daily-loss limit (priced on the exchange).
        price = self.broker.last_price(symbol)
        port.record_equity(price, now.isoformat())
        peak = port.peak_equity_last_24h()
        limit_pct = self.config["live"]["daily_loss_limit"]
        if not self.state.halted and peak > 0 and port.equity(price) < peak * (1 - limit_pct):
            if port.position is not None:
                self.exit_now(port, "daily-loss flatten")
            _halt(self.state, self.notifier,
                  f"{symbol}: equity fell >{limit_pct:.0%} from its 24h peak ${peak:,.0f}. "
                  "`trade.py resume` to restart.")
        log.info("%s cycle done: equity $%.2f, %s", symbol, port.equity(price),
                 "LONG" if port.position else "flat")
