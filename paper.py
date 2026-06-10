"""Paper-trading CLI — the Phase 2 live (fake-money) bot.

Commands
--------
  python paper.py run       # loop forever (always-on PC / Pi / free VM)
  python paper.py tick      # do ONE cycle and exit (for GitHub Actions cron)
  python paper.py status    # snapshot: equity, positions, P&L
  python paper.py report    # live vs backtest vs buy & hold
  python paper.py halt [--flatten]   # stop new entries (optionally sell everything)
  python paper.py resume    # clear the halt and continue
  python paper.py reset --yes        # wipe all paper state and start fresh

See the README's "Phase 2: live paper trading" section for setup + deployment.
"""

from __future__ import annotations

import argparse
import logging
import logging.handlers
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8")
    except (AttributeError, ValueError):
        pass

from src.config import load_config
from src.env import load_dotenv
from src.live import engine, reporter
from src.live.notifier import Notifier
from src.live.state import State

# Load any .env file in the project root BEFORE we read TELEGRAM_* variables.
# (Real OS env vars / GitHub secrets still win — see src/env.py.)
load_dotenv()


def setup_logging(log_file: str) -> None:
    """Log to a rotating file AND the console, so nothing important is lost."""
    Path(log_file).parent.mkdir(parents=True, exist_ok=True)
    handler = logging.handlers.RotatingFileHandler(
        log_file, maxBytes=2_000_000, backupCount=5, encoding="utf-8")
    fmt = logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s")
    handler.setFormatter(fmt)
    console = logging.StreamHandler()
    console.setFormatter(fmt)
    root = logging.getLogger()
    root.setLevel(logging.INFO)
    root.handlers.clear()
    root.addHandler(handler)
    root.addHandler(console)


def _current_prices(config, notifier, fetch_fn=None):
    prices = {}
    for symbol in config["live"]["symbols"]:
        prices[symbol] = reporter._current_price(config, symbol, fetch_fn)
    return prices


# --------------------------------------------------------------------------- #
# Commands
# --------------------------------------------------------------------------- #

def cmd_tick(config, state, notifier) -> None:
    engine.run_once(config, state, notifier)


def cmd_run(config, state, notifier) -> None:
    poll = config["live"]["poll_seconds"]
    notifier.send(f"▶️ Paper bot started ({config['live']['strategy']}, "
                  f"{config['live']['timeframe']}, symbols: "
                  f"{', '.join(config['live']['symbols'])}). Polling every {poll}s.")
    log = logging.getLogger("paper")
    try:
        while True:
            try:
                engine.run_once(config, state, notifier)
            except Exception as exc:  # noqa: BLE001 — a bad cycle must not kill the bot
                log.exception("Cycle error (continuing): %s", exc)
                notifier.send(f"❗ Cycle error (bot still running): {exc}")
            time.sleep(poll)
    except KeyboardInterrupt:
        notifier.send("⏹️ Paper bot stopped (manual).")
        log.info("Stopped by user.")


def cmd_status(config, state, notifier) -> None:
    reporter.print_status(config, state)


def cmd_report(config, state, notifier) -> None:
    reporter.build_report(config, state)


def cmd_halt(config, state, notifier, flatten: bool) -> None:
    state.halted = True
    state.halt_reason = "Manually halted via `paper.py halt`."
    commission = engine.commission_rate(config)
    if flatten:
        import pandas as pd
        prices = _current_prices(config, notifier)
        for symbol, port in state.portfolios.items():
            if port.position is not None and prices.get(symbol):
                now_iso = pd.Timestamp.now(tz="UTC").isoformat()
                trade = port.close_position(prices[symbol], now_iso, commission,
                                            reason="manual-flatten")
                engine._notify_exit(notifier, trade, "manual flatten")
    state.save()
    notifier.send("⛔ Trading halted manually." + (" Positions flattened." if flatten else ""))
    print("Halted." + (" Positions flattened." if flatten else ""))


def cmd_resume(config, state, notifier) -> None:
    state.halted = False
    state.halt_reason = None
    # Reset the 24h loss window so we don't instantly re-halt on stale snapshots.
    for port in state.portfolios.values():
        port.equity_snapshots = []
    state.save()
    notifier.send("✅ Trading resumed.")
    print("Resumed.")


def cmd_reset(config, state, notifier, confirmed: bool) -> None:
    path = Path(config["live"]["state_file"])
    if not confirmed:
        print(f"This will DELETE all paper state ({path}). Re-run with --yes to confirm.")
        return
    if path.exists():
        path.unlink()
    print("Paper state wiped. The next run starts fresh.")


# --------------------------------------------------------------------------- #
# Entry point
# --------------------------------------------------------------------------- #

def main() -> None:
    parser = argparse.ArgumentParser(description="Paper-trading bot (Phase 2).")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("run", help="Loop forever (always-on deployment).")
    sub.add_parser("tick", help="Do one cycle and exit (cron deployment).")
    sub.add_parser("status", help="Show equity, positions, P&L.")
    sub.add_parser("report", help="Live vs backtest vs buy & hold.")
    p_halt = sub.add_parser("halt", help="Stop new entries.")
    p_halt.add_argument("--flatten", action="store_true", help="Also sell all positions now.")
    sub.add_parser("resume", help="Clear the halt and continue.")
    p_reset = sub.add_parser("reset", help="Wipe all paper state.")
    p_reset.add_argument("--yes", action="store_true", help="Confirm the wipe.")
    args = parser.parse_args()

    config = load_config()
    setup_logging(config["live"]["log_file"])
    state = State.load(config["live"]["state_file"])
    notifier = Notifier(config)

    if args.command == "run":
        cmd_run(config, state, notifier)
    elif args.command == "tick":
        cmd_tick(config, state, notifier)
    elif args.command == "status":
        cmd_status(config, state, notifier)
    elif args.command == "report":
        cmd_report(config, state, notifier)
    elif args.command == "halt":
        cmd_halt(config, state, notifier, args.flatten)
    elif args.command == "resume":
        cmd_resume(config, state, notifier)
    elif args.command == "reset":
        cmd_reset(config, state, notifier, args.yes)


if __name__ == "__main__":
    main()
