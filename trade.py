"""Exchange-trading CLI — the same strategy, with real orders on Binance.

Runs on Binance's Testnet or Demo Trading (fake money) by default. Real money
needs BOTH `exchange_trading.environment: live` in config.yaml AND the env var
LIVE_TRADING_CONFIRM=yes-real-money (see the README's "Phase 3" section).

Commands
--------
  python trade.py check     # connect, show environment, balances, rules — places NO orders
  python trade.py run       # loop forever (the always-on VM deployment)
  python trade.py tick      # do ONE cycle and exit
  python trade.py status    # equity, positions, the live stop order
  python trade.py halt [--flatten]  # stop new entries (optionally sell everything now)
  python trade.py resume    # clear a halt and continue
  python trade.py adopt-flat SYMBOL --yes  # you sold manually: record the position as closed

The paper bot (paper.py) is untouched and keeps its own state file.
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8")
    except (AttributeError, ValueError):
        pass

import pandas as pd  # noqa: E402

from paper import setup_logging  # noqa: E402  (same rotating-file + console logging)
from src.config import load_config  # noqa: E402
from src.env import load_dotenv  # noqa: E402
from src.live.broker import BinanceBroker, BrokerError, make_exchange  # noqa: E402
from src.live.exchange_engine import ExchangeEngine  # noqa: E402
from src.live.notifier import Notifier  # noqa: E402
from src.live.state import State  # noqa: E402

load_dotenv()
log = logging.getLogger("trade")


def build(config):
    cfg = config["exchange_trading"]
    broker = BinanceBroker(make_exchange(cfg),
                           stop_limit_buffer=float(cfg.get("stop_limit_buffer", 0.01)))
    broker.load()
    return broker


# --------------------------------------------------------------------------- #
# Commands
# --------------------------------------------------------------------------- #

def cmd_check(config, broker, state) -> None:
    cfg = config["exchange_trading"]
    env = cfg.get("environment", "testnet")
    print(f"\nEnvironment: {env.upper()}" + ("   <<< REAL MONEY >>>" if env == "live" else "   (fake funds)"))
    print(f"Symbols: {', '.join(config['live']['symbols'])}  "
          f"allocation ${float(cfg['allocation_usd']):,.0f} each"
          + (f", max ${float(cfg['max_position_usd']):,.0f} per position" if cfg.get("max_position_usd") else ""))
    w = broker.withdrawals_enabled()
    if w:
        print("\n!! This API key CAN WITHDRAW FUNDS. Create a key with withdrawals disabled.")
        sys.exit(2)
    elif w is False:
        print("API key: withdrawals disabled ✓")
    bal = broker.balances()
    print("\nBalances:")
    for asset in sorted(bal):
        b = bal[asset]
        print(f"  {asset:6} free {b['free']:>16,.8f}   in orders {b['used']:>14,.8f}")
    need = 0.0
    print("\nMarkets:")
    for symbol in config["live"]["symbols"]:
        m = broker.market(symbol)
        price = broker.last_price(symbol)
        min_amt, min_cost = broker.min_order(symbol)
        stop_types = [t for t in (m.get("info", {}).get("orderTypes") or []) if "STOP" in t]
        print(f"  {symbol}: price ${price:,.2f}  min order {min_amt} units / ${min_cost:,.2f}  "
              f"stop types {stop_types}")
        port = state.portfolios.get(symbol)
        if port is None or port.position is None:
            need += float(cfg["allocation_usd"]) if port is None else port.cash
    quote_free = bal.get("USDT", {}).get("free", 0.0)
    print(f"\nFree USDT ${quote_free:,.2f} vs ${need:,.2f} the bot may deploy "
          + ("✓" if quote_free >= need else "— entries will be scaled down to what's free"))
    print("\nNo orders were placed.\n")


def cmd_run(engine, config, notifier) -> None:
    poll = config["exchange_trading"].get("poll_seconds", 300)
    notifier.send(f"▶️ Exchange bot started [{engine.env}] ({config['live']['strategy']}, "
                  f"{', '.join(config['live']['symbols'])}). Polling every {poll}s.")
    try:
        while True:
            engine.run_once()
            time.sleep(poll)
    except KeyboardInterrupt:
        notifier.send("⏹️ Exchange bot stopped (manual).")


def cmd_status(config, broker, state) -> None:
    env = config["exchange_trading"].get("environment", "testnet")
    print(f"\n=== EXCHANGE TRADING STATUS [{env}] ===")
    if state.halted:
        print(f"  ** HALTED ** {state.halt_reason}")
    rows = []
    for symbol in config["live"]["symbols"]:
        port = state.portfolios.get(symbol)
        if port is None:
            rows.append({"Symbol": symbol, "Status": "not started"})
            continue
        price = broker.last_price(symbol)
        pos = port.position
        rows.append({
            "Symbol": symbol, "Status": "LONG" if pos else "flat",
            "Price": round(price, 2), "Equity": round(port.equity(price), 2),
            "Return %": round((port.equity(price) / port.start_equity - 1) * 100, 2),
            "Entry": round(pos.entry_price, 2) if pos else None,
            "Stop": round(pos.stop_price, 2) if pos else None,
            "Stop order": pos.stop_order_id if pos else None,
            "Trades": len(port.trades), "Fees $": round(port.fees_paid, 2),
        })
    print(pd.DataFrame(rows).set_index("Symbol").to_string())
    print()


def cmd_halt(engine, state, notifier, flatten: bool) -> None:
    state.halted = True
    state.halt_reason = "Manually halted via `trade.py halt`."
    state.save()
    if flatten:
        for port in state.portfolios.values():
            if port.position is not None:
                engine.exit_now(port, "manual flatten")
        state.save()
    notifier.send("⛔ Exchange trading halted manually." + (" Positions sold." if flatten else ""))
    print("Halted." + (" Positions sold." if flatten else ""))


def cmd_resume(state, notifier) -> None:
    state.halted = False
    state.halt_reason = None
    for port in state.portfolios.values():
        port.equity_snapshots = []  # fresh 24h window, so we don't instantly re-halt
    state.save()
    notifier.send("✅ Exchange trading resumed.")
    print("Resumed.")


def cmd_adopt_flat(broker, state, symbol, confirmed) -> None:
    port = state.portfolios.get(symbol)
    if port is None or port.position is None:
        print(f"{symbol} has no open position in the bot's records.")
        return
    if not confirmed:
        print(f"This records {symbol} as SOLD at the current price (use only if you sold it "
              "yourself on Binance). Re-run with --yes.")
        return
    if port.position.stop_order_id:
        broker.cancel(symbol, port.position.stop_order_id)
    price = broker.last_price(symbol)
    port.close_position_filled(price, pd.Timestamp.now(tz="UTC").isoformat(), 0.0,
                               "sold outside the bot")
    state.save()
    print(f"{symbol} recorded as flat at ~${price:,.2f}.")


# --------------------------------------------------------------------------- #

def main() -> None:
    parser = argparse.ArgumentParser(description="Exchange trading bot (Binance).")
    sub = parser.add_subparsers(dest="command", required=True)
    for name, help_ in [("check", "Connect and show balances; no orders."),
                        ("run", "Loop forever."), ("tick", "One cycle."),
                        ("status", "Positions and P&L."), ("resume", "Clear a halt.")]:
        sub.add_parser(name, help=help_)
    p_halt = sub.add_parser("halt", help="Stop new entries.")
    p_halt.add_argument("--flatten", action="store_true", help="Also sell all positions now.")
    p_adopt = sub.add_parser("adopt-flat", help="Record a manually-sold position as closed.")
    p_adopt.add_argument("symbol")
    p_adopt.add_argument("--yes", action="store_true")
    args = parser.parse_args()

    config = load_config()
    cfg = config["exchange_trading"]
    setup_logging(cfg.get("log_file", "logs/trade.log"))
    state = State.load(cfg["state_file"])
    notifier = Notifier(config)

    if args.command == "resume":
        cmd_resume(state, notifier)
        return
    try:
        broker = build(config)
    except BrokerError as exc:
        print(f"Cannot start: {exc}")
        sys.exit(1)

    engine = ExchangeEngine(config, state, broker, notifier)
    if args.command == "check":
        cmd_check(config, broker, state)
    elif args.command == "run":
        cmd_run(engine, config, notifier)
    elif args.command == "tick":
        engine.run_once()
    elif args.command == "status":
        cmd_status(config, broker, state)
    elif args.command == "halt":
        cmd_halt(engine, state, notifier, args.flatten)
    elif args.command == "adopt-flat":
        cmd_adopt_flat(broker, state, args.symbol, args.yes)


if __name__ == "__main__":
    main()
