"""The live engine: fetch candle → ask the strategy → simulate the fill.

REUSE, DON'T REWRITE — this module imports:
  * the strategy's own channel functions (``_prior_high`` / ``_prior_low`` from
    ``donchian_breakout``), so the live entry/exit rule is byte-for-byte the
    backtest rule;
  * ``src.indicators`` for the ATR; and
  * ``src.risk`` for the 2%-risk position sizing and the ATR stop.

No-lookahead: a signal is computed only from CLOSED candles. The simulated fill
happens at the OPEN of the next candle — never inside the candle that produced
the signal — exactly like the backtester.

Costs: a combined commission = taker_fee + slippage is charged on every fill,
identical to the backtest, so live and backtest stay directly comparable.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from src import indicators, risk
from src.data_loader import days_to_bars, fetch_recent_ohlcv, split_forming
from src.strategies.donchian_breakout import _prior_high, _prior_low

log = logging.getLogger("paper.engine")


# --------------------------------------------------------------------------- #
# Signal — reuses the SAME pure functions the backtest strategy uses.
# --------------------------------------------------------------------------- #

def compute_signal(strategy: str, params: dict, hist: pd.DataFrame, in_position: bool) -> str:
    """Return 'ENTER', 'EXIT', or 'HOLD' for the latest CLOSED candle.

    `hist` is the closed-candle history (oldest first, newest last). The decision
    is made from `hist`'s final row; the engine fills it at the next candle's open.
    """
    close, high, low = hist["close"], hist["high"], hist["low"]

    # Optional regime gate (price must be above the long trend SMA to be long).
    regime_ok = True
    if params.get("regime_gate"):
        sma = indicators.sma(close, params["regime_period"]).to_numpy()
        regime_ok = bool(not np.isnan(sma[-1]) and close.iloc[-1] > sma[-1])
        if in_position and not regime_ok:
            return "EXIT"  # forced exit when the trend turns down

    if strategy == "donchian_breakout":
        upper = _prior_high(high.to_numpy(), params["entry_period"])[-1]
        lower = _prior_low(low.to_numpy(), params["exit_period"])[-1]
        if not in_position:
            if regime_ok and not np.isnan(upper) and high.iloc[-1] > upper:
                return "ENTER"
        else:
            if not np.isnan(lower) and low.iloc[-1] < lower:
                return "EXIT"
        return "HOLD"

    if strategy == "sma_crossover":
        fast = indicators.sma(close, params["fast"]).to_numpy()
        slow = indicators.sma(close, params["slow"]).to_numpy()
        if np.isnan(fast[-1]) or np.isnan(slow[-1]) or np.isnan(fast[-2]):
            return "HOLD"
        crossed_up = fast[-2] <= slow[-2] and fast[-1] > slow[-1]
        crossed_down = fast[-2] >= slow[-2] and fast[-1] < slow[-1]
        if not in_position and crossed_up and regime_ok:
            return "ENTER"
        if in_position and crossed_down:
            return "EXIT"
        return "HOLD"

    if strategy == "trend_filter":
        sma = indicators.sma(close, params["sma_period"]).to_numpy()
        if np.isnan(sma[-1]):
            return "HOLD"
        buf = params.get("buffer", 0.0)
        price = close.iloc[-1]
        if not in_position and price > sma[-1] * (1 + buf):
            return "ENTER"
        if in_position and price < sma[-1] * (1 - buf):
            return "EXIT"
        return "HOLD"

    raise ValueError(f"Unknown live strategy: {strategy}")


def _entry_reason(strategy: str, params: dict) -> str:
    if strategy == "donchian_breakout":
        return f"new {params['entry_period']}-candle high breakout"
    if strategy == "sma_crossover":
        return f"SMA {params['fast']}/{params['slow']} bullish cross"
    if strategy == "trend_filter":
        return f"close above {params['sma_period']}-candle SMA"
    return "entry signal"


# --------------------------------------------------------------------------- #
# Order execution helpers (all paper, all reuse risk.py).
# --------------------------------------------------------------------------- #

def _size_and_stop(cash, risk_cfg, entry_price, atr_value):
    """2%-risk position sizing + ATR stop, straight from risk.py (unchanged)."""
    stop = risk.atr_stop_price(entry_price, atr_value, risk_cfg["atr_stop_multiplier"])
    fraction = risk.position_size_fraction(risk_cfg["risk_per_trade"], entry_price, stop)
    units = (fraction * cash) / entry_price
    return units, stop


def _execute_pending(port, open_price, at_time, halted, risk_cfg, commission, notifier):
    """Fill a pending order at `open_price` (the next candle's open)."""
    p = port.pending
    at_iso = at_time.isoformat() if hasattr(at_time, "isoformat") else str(at_time)

    if p["action"] == "ENTER":
        port.pending = None
        if halted:
            log.info("Entry suppressed for %s — trading halted.", port.symbol)
            return
        if port.position is not None:
            return
        units, stop = _size_and_stop(port.cash, risk_cfg, open_price, p["atr"])
        if units <= 0:
            return
        port.open_position(units, open_price, at_iso, stop, commission, p["atr"])
        notifier.send(
            f"🟢 <b>ENTER</b> {port.symbol}\n"
            f"price ${open_price:,.2f}  size {units:.6f} (~${units * open_price:,.0f})\n"
            f"stop ${stop:,.2f}  reason: {p.get('reason', 'signal')}"
        )
    elif p["action"] == "EXIT":
        port.pending = None
        if port.position is None:
            return
        trade = port.close_position(open_price, at_iso, commission, reason="signal")
        _notify_exit(notifier, trade, "signal")


def _execute_stop(port, at_time_iso, commission, notifier):
    """Stop-loss fill at the stop price (combined commission folds in slippage)."""
    stop_price = port.position.stop_price
    trade = port.close_position(stop_price, at_time_iso, commission, reason="stop")
    port.pending = None  # cancel any pending exit; we're already out
    _notify_exit(notifier, trade, "STOP-LOSS hit")


def _notify_exit(notifier, trade, label):
    emoji = "🛑" if "STOP" in label else "🔴"
    notifier.send(
        f"{emoji} <b>EXIT</b> {trade['symbol']} ({label})\n"
        f"price ${trade['exit_price']:,.2f}  pnl ${trade['pnl']:,.2f} "
        f"({trade['return_pct']:+.2f}%)"
    )


def _set_pending_from_action(port, action, ts, hist, strategy, params, risk_cfg, halted):
    """Translate a signal into a pending order to fill at the next candle's open."""
    if action == "ENTER" and port.position is None and not halted:
        atr_value = indicators.atr(
            hist["high"], hist["low"], hist["close"], risk_cfg["atr_period"]
        ).iloc[-1]
        if atr_value > 0:
            port.pending = {
                "action": "ENTER", "signal_time": ts.isoformat(),
                "atr": float(atr_value), "reason": _entry_reason(strategy, params),
            }
    elif action == "EXIT" and port.position is not None:
        port.pending = {"action": "EXIT", "signal_time": ts.isoformat(), "reason": "exit signal"}


# --------------------------------------------------------------------------- #
# Core: advance one symbol's portfolio through any new closed candles.
# --------------------------------------------------------------------------- #

def advance_symbol(port, closed, forming, strategy, params, risk_cfg, commission,
                   halted, notifier):
    """Process every closed candle newer than `last_processed_close`, then act on
    the forming candle's open. This single function powers BOTH live ticking and
    missed-candle catch-up after a restart."""
    last_close = pd.Timestamp(port.last_processed_close)
    new_closed = closed[closed.index > last_close]

    for ts, row in new_closed.iterrows():
        # 1. Fill any pending order at THIS candle's open (it was set earlier).
        if port.pending and pd.Timestamp(port.pending["signal_time"]) < ts:
            _execute_pending(port, row["open"], ts, halted, risk_cfg, commission, notifier)

        # 2. Intra-candle stop check, using this candle's low (same as backtest).
        if port.position and row["low"] <= port.position.stop_price:
            _execute_stop(port, ts.isoformat(), commission, notifier)

        # 3. Time-in-market accounting for this completed period.
        port.total_periods += 1
        if port.position is not None:
            port.time_in_market_periods += 1

        # 4. Compute the signal from history up to and including this candle.
        hist = closed.loc[:ts]
        action = compute_signal(strategy, params, hist, port.position is not None)
        _set_pending_from_action(port, action, ts, hist, strategy, params, risk_cfg, halted)

        port.last_processed_close = ts.isoformat()

    # 5. Act promptly on the latest signal: fill its pending order at the forming
    #    candle's open (that IS "the open of candle N+1").
    if forming is not None and port.pending:
        if pd.Timestamp(port.pending["signal_time"]) < forming.name:
            _execute_pending(port, forming["open"], forming.name, halted,
                             risk_cfg, commission, notifier)


# --------------------------------------------------------------------------- #
# Config helpers
# --------------------------------------------------------------------------- #

def resolve_params(config: dict) -> dict:
    """Turn the live config (lookbacks in DAYS) into candle-count params."""
    live = config["live"]
    tf = live["timeframe"]
    p = live.get("params", {})
    strategy = live["strategy"]
    out = {"regime_gate": bool(p.get("regime_gate", False))}
    if out["regime_gate"]:
        out["regime_period"] = days_to_bars(config["regime"]["sma_period_days"], tf)
    if strategy == "donchian_breakout":
        out["entry_period"] = days_to_bars(p["entry_days"], tf)
        out["exit_period"] = days_to_bars(p["exit_days"], tf)
    elif strategy == "sma_crossover":
        out["fast"] = days_to_bars(p.get("fast_days", 50), tf)
        out["slow"] = days_to_bars(p.get("slow_days", 200), tf)
    elif strategy == "trend_filter":
        out["sma_period"] = days_to_bars(p.get("sma_period_days", 200), tf)
        out["buffer"] = float(p.get("buffer", 0.0))
    return out


def commission_rate(config: dict) -> float:
    fees = config["fees"]
    return float(fees["taker_fee"]) + float(fees["slippage"])


def _warmup_needed(params: dict, risk_cfg: dict) -> int:
    periods = [params.get("entry_period", 0), params.get("exit_period", 0),
               params.get("regime_period", 0), params.get("fast", 0),
               params.get("slow", 0), params.get("sma_period", 0),
               risk_cfg["atr_period"]]
    return max(periods) + 5


def _is_anomalous(closed: pd.DataFrame, threshold: float) -> bool:
    """True if the latest close jumped more than `threshold` vs the prior close,
    or any recent price is non-positive / NaN (garbage data)."""
    tail = closed["close"].tail(2)
    if tail.isna().any() or (tail <= 0).any():
        return True
    if len(tail) == 2:
        change = abs(tail.iloc[-1] / tail.iloc[-2] - 1)
        if change > threshold:
            return True
    return False


# --------------------------------------------------------------------------- #
# Orchestration: one full cycle across all symbols.
# --------------------------------------------------------------------------- #

def run_once(config, state, notifier, fetch_fn=None, now=None) -> None:
    """Do one complete cycle: fetch, sanity-check, advance, poll stops, enforce
    the daily-loss limit, and persist — for every live symbol."""
    fetch_fn = fetch_fn or fetch_recent_ohlcv
    live = config["live"]
    strategy, timeframe = live["strategy"], live["timeframe"]
    params = resolve_params(config)
    risk_cfg = config["risk"]
    commission = commission_rate(config)
    now_ts = now or pd.Timestamp.now(tz="UTC")
    now_iso = now_ts.isoformat()
    limit = min(1000, max(400, _warmup_needed(params, risk_cfg) + 50))

    for symbol in live["symbols"]:
        port = state.ensure_portfolio(symbol, now_iso, live["starting_balance"])

        # --- fetch (data errors must never crash the loop) ---
        try:
            recent = fetch_fn(symbol, timeframe, limit)
        except Exception as exc:  # noqa: BLE001
            log.warning("Fetch failed for %s: %s", symbol, exc)
            notifier.send(f"⚠️ Data fetch failed for {symbol}: {exc}. Skipping cycle.")
            continue

        closed, forming = split_forming(recent, timeframe)
        if len(closed) < 2:
            log.info("Not enough candles yet for %s.", symbol)
            continue

        # --- data sanity guard ---
        if _is_anomalous(closed, live["data_anomaly_threshold"]):
            log.warning("Anomalous data for %s — skipping cycle.", symbol)
            notifier.send(f"⚠️ Anomalous price data for {symbol}; skipped this cycle.")
            continue

        # --- first run for this symbol: start flat from NOW, don't replay history ---
        if port.last_processed_close is None:
            port.last_processed_close = closed.index[-1].isoformat()
            port.peak_equity = port.cash
            log.info("Initialised paper portfolio for %s (flat, $%s).",
                     symbol, live["starting_balance"])
            state.save()
            continue

        # --- process new closed candles + act on the forming candle ---
        advance_symbol(port, closed, forming, strategy, params, risk_cfg,
                       commission, state.halted, notifier)

        # --- intra-period (live) stop check on the forming candle's low ---
        cur_price = float(forming["close"]) if forming is not None else float(closed["close"].iloc[-1])
        if port.position is not None:
            low_now = float(forming["low"]) if forming is not None else float(closed["low"].iloc[-1])
            if low_now <= port.position.stop_price:
                _execute_stop(port, now_iso, commission, notifier)

        # --- equity snapshot, drawdown, and the daily-loss limit ---
        port.record_equity(cur_price, now_iso)
        _enforce_daily_loss(state, port, cur_price, now_iso, live["daily_loss_limit"],
                            commission, notifier)

        log.info("%s cycle done: equity $%.2f, %s", symbol, port.equity(cur_price),
                 "in position" if port.position else "flat")
        state.save()


def _enforce_daily_loss(state, port, cur_price, now_iso, limit, commission, notifier):
    """If equity has dropped more than `limit` from its 24h peak, flatten and halt."""
    peak_24h = port.peak_equity_last_24h()
    equity = port.equity(cur_price)
    if peak_24h > 0 and equity < peak_24h * (1 - limit):
        if port.position is not None:
            trade = port.close_position(cur_price, now_iso, commission, reason="daily-loss-flatten")
            _notify_exit(notifier, trade, "DAILY-LOSS flatten")
        state.halted = True
        state.halt_reason = (
            f"{port.symbol}: equity ${equity:,.0f} fell >{limit:.0%} from 24h peak "
            f"${peak_24h:,.0f}. Trading halted — `paper.py resume` to restart."
        )
        notifier.send(f"⛔ <b>DAILY LOSS LIMIT</b>\n{state.halt_reason}")
        log.warning(state.halt_reason)
