"""Performance reporting for the paper bot.

Two views:
  * `status`  — a quick snapshot: equity, open position, unrealized P&L.
  * `report`  — since-start performance for each symbol, shown side by side with
    (a) buy-and-hold over the same window and (b) the BACKTESTER run over the
    exact same window. That last column is the whole scientific point of Phase 2:
    if live and backtest diverge a lot, something is wrong with the bot, not the
    market.
"""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

from src import backtest as bt_engine
from src.data_loader import days_to_bars, fetch_recent_ohlcv, split_forming, to_backtesting_format
from src.live.engine import _warmup_needed, commission_rate, resolve_params
from src.strategies import DISPLAY_NAMES, STRATEGIES

log = logging.getLogger("paper.reporter")


def _current_price(config, symbol, fetch_fn=None):
    """Latest price for a symbol (forming candle's close), or None on failure."""
    fetch_fn = fetch_fn or fetch_recent_ohlcv
    try:
        recent = fetch_fn(symbol, config["live"]["timeframe"], 5)
        return float(recent["close"].iloc[-1])
    except Exception as exc:  # noqa: BLE001
        log.warning("Could not fetch current price for %s: %s", symbol, exc)
        return None


def gather_status(config, state, fetch_fn=None) -> list[dict]:
    rows = []
    for symbol in config["live"]["symbols"]:
        port = state.portfolios.get(symbol)
        price = _current_price(config, symbol, fetch_fn)
        if port is None:
            rows.append({"Symbol": symbol, "Status": "not started",
                         "Equity": config["live"]["starting_balance"]})
            continue
        ref_price = price if price is not None else (
            port.position.entry_price if port.position else port.cash)
        equity = port.equity(ref_price) if price is not None else port.cash + (
            port.position.units * port.position.entry_price if port.position else 0)
        rows.append({
            "Symbol": symbol,
            "Status": "LONG" if port.position else "flat",
            "Price": round(price, 2) if price else None,
            "Equity": round(equity, 2),
            "Cash": round(port.cash, 2),
            "Units": round(port.position.units, 6) if port.position else 0,
            "Entry": round(port.position.entry_price, 2) if port.position else None,
            "Stop": round(port.position.stop_price, 2) if port.position else None,
            "Unreal. P&L": round(port.unrealized_pnl(ref_price), 2) if (port.position and price) else 0.0,
            "Trades": len(port.trades),
        })
    return rows


def print_status(config, state, fetch_fn=None) -> None:
    rows = gather_status(config, state, fetch_fn)
    df = pd.DataFrame(rows).set_index("Symbol")
    print("\n=== PAPER TRADING STATUS ===")
    if state.halted:
        print(f"  ** HALTED ** {state.halt_reason}")
    print(df.to_string())
    print()


def _backtest_over_window(config, symbol, start_iso, fetch_fn=None):
    """Run the backtester over the live window (with proper warm-up before it)
    and return (return_pct, max_dd_pct) measured FROM start_iso onward."""
    fetch_fn = fetch_fn or fetch_recent_ohlcv
    live = config["live"]
    params = resolve_params(config)
    risk_cfg = config["risk"]
    warmup = _warmup_needed(params, risk_cfg)

    recent = fetch_fn(symbol, live["timeframe"], min(1000, warmup + 400))
    closed, _ = split_forming(recent, live["timeframe"])
    if len(closed) < warmup + 2:
        return None, None

    # Build the strategy params the backtest Strategy class expects.
    cls = STRATEGIES[live["strategy"]]
    bt_params = {
        "risk_per_trade": risk_cfg["risk_per_trade"],
        "atr_period": risk_cfg["atr_period"],
        "atr_stop_multiplier": risk_cfg["atr_stop_multiplier"],
    }
    if live["strategy"] == "donchian_breakout":
        bt_params["entry_period"] = params["entry_period"]
        bt_params["exit_period"] = params["exit_period"]
    if params.get("regime_gate"):
        bt_params["use_regime_gate"] = True
        bt_params["regime_sma_period"] = params["regime_period"]

    # Use a large cash so backtesting.py's whole-unit rounding is negligible —
    # the live bot trades FRACTIONAL units, so we compare PERCENTAGES, which are
    # scale-invariant. (Small cash here triggers spurious "insufficient margin".)
    _, stats = bt_engine.run_backtest(
        closed, cls, 1_000_000, commission_rate(config), bt_params)

    eq = stats["_equity_curve"]["Equity"]  # tz-naive datetime index
    start = pd.Timestamp(start_iso).tz_localize(None)
    window = eq[eq.index >= start]
    if len(window) < 2:
        return None, None
    ret = (window.iloc[-1] / window.iloc[0] - 1) * 100
    running_peak = window.cummax()
    max_dd = ((window - running_peak) / running_peak).min() * 100
    return round(ret, 2), round(max_dd, 2)


def build_report(config, state, fetch_fn=None, save=True) -> pd.DataFrame:
    """Per-symbol live performance vs buy-and-hold vs the backtester."""
    commission = commission_rate(config)
    rows = []
    for symbol in config["live"]["symbols"]:
        port = state.portfolios.get(symbol)
        if port is None or port.last_processed_close is None:
            continue
        price = _current_price(config, symbol, fetch_fn)
        if price is None:
            continue

        equity = port.equity(price)
        live_return = (equity / port.start_equity - 1) * 100

        # Buy & hold over the same window: buy at the first candle after start,
        # hold to now, paying one round-trip of commission (fair to the bot).
        bh_return = None
        try:
            recent = (fetch_fn or fetch_recent_ohlcv)(symbol, config["live"]["timeframe"], 1000)
            closed, _ = split_forming(recent, config["live"]["timeframe"])
            start = pd.Timestamp(port.start_time)
            after = closed[closed.index >= start]
            if len(after) >= 1:
                entry = float(after["open"].iloc[0])
                gross = price / entry
                bh_return = (gross * (1 - commission) ** 2 - 1) * 100
        except Exception as exc:  # noqa: BLE001
            log.warning("B&H calc failed for %s: %s", symbol, exc)

        bt_return, bt_maxdd = _backtest_over_window(config, symbol, port.start_time, fetch_fn)

        rows.append({
            "Symbol": symbol,
            "Live Return %": round(live_return, 2),
            "Live MaxDD %": round(port.max_drawdown * 100, 2),
            "Backtest Return %": bt_return,
            "Backtest MaxDD %": bt_maxdd,
            "Buy&Hold Return %": round(bh_return, 2) if bh_return is not None else None,
            "# Trades": len(port.trades),
            "Fees $": round(port.fees_paid, 2),
            "Time in Mkt %": round(port.time_in_market_pct(), 1),
            "Equity $": round(equity, 2),
        })

    df = pd.DataFrame(rows)
    print("\n=== PAPER TRADING REPORT (live vs backtest vs buy & hold) ===")
    if df.empty:
        print("  No active portfolios yet. Run `paper.py tick` first.")
        return df
    print(df.set_index("Symbol").to_string())
    print("\n  A large live-vs-backtest gap signals a bug; a small one is healthy.")

    if save:
        out = Path(config["reports_dir"]) / "paper_report.csv"
        out.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(out, index=False)
        print(f"  Saved to {out}")
    return df
