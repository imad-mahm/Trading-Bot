"""Command-line entry point for the crypto backtester.

Usage
-----
  python run.py download                         # fetch + cache all timeframes
  python run.py backtest --all --timeframe 1d    # the Phase 2 main run
  python run.py backtest --strategy donchian_breakout --symbol BTC/USDT --timeframe 1d
  python run.py backtest --all --timeframe 1h    # rerun the matrix on hourly

`backtest --all` runs a whole MATRIX per symbol: Buy & Hold, the Trend Filter
(with and without a buffer), SMA crossover (plain and regime-gated), and the
Donchian breakout sweep (each N/M combo, plain and regime-gated). Results print
as tables sorted by out-of-sample Calmar, and charts are saved to reports/.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, field
from pathlib import Path

# Make sure `from src import ...` works no matter where we're run from.
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))

# Print UTF-8 so symbols like ✓/✗/— render on Windows consoles (cp1252) too.
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8")
    except (AttributeError, ValueError):
        pass

import pandas as pd

from src import backtest as bt_engine
from src import indicators, report
from src.config import load_config
from src.data_loader import (
    days_to_bars, load_data, split_in_out, to_backtesting_format, validate_data,
)
from src.market_data import load_market_data, validate_market
from src.strategies import (
    STRATEGIES, BuyAndHold, DonchianBreakout, SmaCrossover, TrendFilter,
)


# --------------------------------------------------------------------------- #
# A single thing to run: a strategy class + the exact parameters for it.
# --------------------------------------------------------------------------- #

@dataclass
class Variant:
    name: str          # display name shown in the report
    key: str           # strategy id, for --strategy filtering
    cls: type          # the backtesting.py Strategy subclass
    params: dict = field(default_factory=dict)
    gated: bool = False  # True if this variant uses the regime gate


def risk_params(config: dict) -> dict:
    risk = config["risk"]
    return {
        "risk_per_trade": risk["risk_per_trade"],
        "atr_period": risk["atr_period"],
        "atr_stop_multiplier": risk["atr_stop_multiplier"],
    }


def filter_params(strategy_cls, params: dict) -> dict:
    """Keep only params the class actually has (so we never pass, say, a risk
    param to Buy & Hold, which the engine would reject)."""
    return {k: v for k, v in params.items() if hasattr(strategy_cls, k)}


def build_variants(config: dict, timeframe: str):
    """Build the full run matrix for one timeframe. Periods in config are in
    DAYS; we convert them to candle counts for this timeframe here."""
    base_risk = risk_params(config)
    strat_cfg = config["strategies"]
    regime_period = days_to_bars(config["regime"]["sma_period_days"], timeframe)
    gate = {"use_regime_gate": True, "regime_sma_period": regime_period}

    variants: list[Variant] = []

    # 1) Benchmark.
    variants.append(Variant("Buy & Hold", "buy_and_hold", BuyAndHold, {}))

    # 2) Trend filter, standalone: without and with the anti-whipsaw buffer.
    tf = strat_cfg["trend_filter"]
    tf_period = days_to_bars(tf["sma_period_days"], timeframe)
    buf = tf.get("buffer_alt", 0.01)
    variants.append(Variant("Trend Filter 200d", "trend_filter", TrendFilter,
                            {"sma_period": tf_period, "buffer": 0.0}))
    variants.append(Variant(f"Trend Filter 200d (+{buf:.0%} buf)", "trend_filter",
                            TrendFilter, {"sma_period": tf_period, "buffer": buf}))

    # 3) SMA crossover: plain and regime-gated.
    sma = strat_cfg["sma_crossover"]
    sma_params = {
        **base_risk,
        "fast": days_to_bars(sma["fast_days"], timeframe),
        "slow": days_to_bars(sma["slow_days"], timeframe),
    }
    variants.append(Variant("SMA Crossover", "sma_crossover", SmaCrossover, sma_params))
    variants.append(Variant("SMA Crossover + Regime", "sma_crossover", SmaCrossover,
                            {**sma_params, **gate}, gated=True))

    # 4) Donchian breakout sweep: every N/M combo, plain and regime-gated.
    don = strat_cfg["donchian_breakout"]
    for n in don["entry_days_sweep"]:
        for m in don["exit_days_sweep"]:
            dp = {
                **base_risk,
                "entry_period": days_to_bars(n, timeframe),
                "exit_period": days_to_bars(m, timeframe),
            }
            variants.append(Variant(f"Donchian {n}/{m}", "donchian_breakout",
                                    DonchianBreakout, dp))
            variants.append(Variant(f"Donchian {n}/{m} + Regime", "donchian_breakout",
                                    DonchianBreakout, {**dp, **gate}, gated=True))

    return variants, regime_period


def run_variant(df_segment, variant: Variant, cash, commission):
    safe = filter_params(variant.cls, variant.params)
    _, stats = bt_engine.run_backtest(df_segment, variant.cls, cash, commission, safe)
    return stats



# --------------------------------------------------------------------------- #
# Universes: crypto (Binance, the original) or markets (ETFs: gold, oil, ...).
# --------------------------------------------------------------------------- #

@dataclass
class Universe:
    name: str
    symbols: list
    labels: dict
    fees: dict
    start_date: str
    end_date: str | None

    def load(self, config, symbol, timeframe, refresh=False, verbose=True):
        if self.name == "markets":
            return load_market_data(symbol, self.start_date, self.end_date,
                                    config["data_dir"], refresh=refresh, verbose=verbose)
        return load_data(symbol, timeframe, self.start_date, self.end_date,
                         config["data_dir"], refresh=refresh, verbose=verbose)


def get_universe(config: dict, name: str) -> Universe:
    if name == "markets":
        m = config["markets"]
        return Universe("markets", list(m["symbols"]), dict(m["symbols"]), m["fees"],
                        m["start_date"], m.get("end_date"))
    return Universe("crypto", list(config["symbols"]), {}, config["fees"],
                    config["start_date"], config["end_date"])


# The one configuration judged across markets: the strategy the live bot runs.
# Fixed in advance (not the best per market — that would be curve-fitting).
LIVE_VARIANT = "Donchian 20/20"


def passes_bar(row, bh) -> dict:
    """The Phase 2 success bar, applied to one out-of-sample result."""
    checks = {
        "shallower DD": abs(row["Max DD [%]"]) <= abs(bh["Max DD [%]"]) * 2 / 3,
        "Calmar > B&H": row["Calmar"] > bh["Calmar"],
        "trades < 100": row["# Trades"] < 100,
        "fees < 5%": row["Fees [%]"] < 5,
    }
    checks["PASS"] = all(checks.values())
    return checks


def print_cross_market_summary(rows: list, reports_dir: Path, universe: Universe, timeframe: str):
    if not rows:
        return
    table = pd.DataFrame(rows).set_index("Market")
    title = (f"CROSS-MARKET SUMMARY — {LIVE_VARIANT} vs Buy & Hold, OUT-OF-SAMPLE "
             f"(after in-sample end)")
    report.print_table(title, table)
    n_pass = int((table["Passes bar?"] == "YES ✓").sum())
    print(f"  {LIVE_VARIANT} clears the success bar on {n_pass} of {len(table)} markets.")
    print("  Bar = max DD ≤ ⅔ of Buy & Hold's, Calmar above Buy & Hold's, "
          "<100 trades, fees <5%.\n")
    report.save_table_csv(table, reports_dir / f"summary_{universe.name}_{timeframe}.csv")

# --------------------------------------------------------------------------- #
# Commands
# --------------------------------------------------------------------------- #

def cmd_download(config: dict, timeframe_arg: str | None, refresh: bool,
                 universe_name: str = "crypto") -> None:
    """Download + cache + validate data for every symbol and timeframe."""
    if universe_name == "markets":
        uni = get_universe(config, "markets")
        print(f"Downloading daily ETF data (adjusted) from {uni.start_date}\n")
        failed = []
        for ticker in uni.symbols:
            try:
                df = uni.load(config, ticker, "1d", refresh=refresh)
                validate_market(df, ticker)
            except Exception as exc:  # noqa: BLE001 — report all failures at the end
                print(f"  {ticker}: FAILED ({exc})")
                failed.append(ticker)
        print("Done." if not failed else f"Done, but failed: {', '.join(failed)}")
        if failed:
            sys.exit(1)
        return

    timeframes = [timeframe_arg] if timeframe_arg else config["timeframes"]
    print(f"Downloading market data for timeframes: {', '.join(timeframes)}\n")
    for timeframe in timeframes:
        for symbol in config["symbols"]:
            df = load_data(
                symbol, timeframe, config["start_date"], config["end_date"],
                config["data_dir"], refresh=refresh,
            )
            validate_data(df, timeframe)
            print()
    print("Done.")


def _regime_cash_mask(df, regime_period: int, cutoff: str):
    """For the OUT-OF-SAMPLE window, a boolean Series that is True when price is
    below the 200-day regime SMA (i.e. when the gate would sit in cash). The SMA
    is computed over the full history so it's already valid at the OOS start."""
    close = to_backtesting_format(df)["Close"]
    sma = indicators.sma(close, regime_period)
    cash = (close < sma).fillna(False)
    cutoff_ts = pd.Timestamp(cutoff)
    oos = close.index > cutoff_ts
    return close.index[oos], cash[oos].to_numpy()


def cmd_backtest(config: dict, args) -> None:
    cash = config["cash"]
    universe = get_universe(config, args.universe)
    commission = bt_engine.effective_commission(universe.fees)
    timeframe = "1d" if universe.name == "markets" else (args.timeframe or config["timeframe"])
    reports_dir = Path(config["reports_dir"])
    reports_dir.mkdir(parents=True, exist_ok=True)

    variants, regime_period = build_variants(config, timeframe)
    # Targeted mode: keep just the chosen strategy's variants (+ the benchmark).
    if args.strategy and not args.all:
        variants = [v for v in variants if v.key in (args.strategy, "buy_and_hold")]
    symbols = [args.symbol] if args.symbol else universe.symbols
    summary_rows: list = []

    print(f"Universe: {universe.name}   |   Timeframe: {timeframe}   |   starting cash: ${cash:,.0f}")
    print(f"Fees+slippage per side: {commission * 100:.3f}%  "
          f"(round-trip ≈ {commission * 200:.3f}%)")
    print(f"In-sample ends: {config['in_sample_end']}   |   "
          f"out-of-sample = after that, untouched\n")

    for symbol in symbols:
        df = universe.load(config, symbol, timeframe, refresh=args.refresh)
        in_df, out_df = split_in_out(df, config["in_sample_end"])
        label = f"{symbol} ({universe.labels[symbol]})" if symbol in universe.labels else symbol

        is_stats: dict = {}
        oos_stats: dict = {}
        for v in variants:
            is_stats[v.name] = run_variant(in_df, v, cash, commission)
            oos_stats[v.name] = run_variant(out_df, v, cash, commission)

        # Out-of-sample table sorted by Calmar (the headline). Use that same row
        # order for the in-sample table so they line up for easy comparison.
        oos_table = report.comparison_table(oos_stats, commission, cash)
        is_table = report.comparison_table(is_stats, commission, cash,
                                           order=list(oos_table.index))

        safe = symbol.replace("/", "")
        report.print_table(f"{label}  {timeframe}  —  IN-SAMPLE "
                           f"(order follows OOS Calmar)", is_table)
        report.print_table(f"{label}  {timeframe}  —  OUT-OF-SAMPLE "
                           f"(sorted by Calmar; this is the one that matters)", oos_table)
        report.save_table_csv(is_table, reports_dir / f"{safe}_{timeframe}_in-sample.csv")
        report.save_table_csv(oos_table, reports_dir / f"{safe}_{timeframe}_out-of-sample.csv")

        # Charts (out-of-sample).
        report.plot_equity_curves(
            oos_stats, f"{symbol} {timeframe} — OOS equity vs Buy & Hold",
            reports_dir / f"{safe}_{timeframe}_oos_equity.png")
        report.plot_drawdowns(
            oos_stats, f"{symbol} {timeframe} — OOS drawdown",
            reports_dir / f"{safe}_{timeframe}_oos_drawdown.png")

        # Regime-gated subset, with the in-cash band shaded.
        gated = {v.name: oos_stats[v.name] for v in variants if v.gated}
        if gated:
            gated["Buy & Hold"] = oos_stats["Buy & Hold"]  # reference line
            band_index, cash_mask = _regime_cash_mask(df, regime_period, config["in_sample_end"])
            report.plot_regime_equity(
                gated, band_index, cash_mask,
                f"{symbol} {timeframe} — OOS regime-gated equity "
                f"(grey = filter in cash)",
                reports_dir / f"{safe}_{timeframe}_oos_regime.png")

        print(f"  Tables + charts saved to {reports_dir}/ for {symbol}\n")

        if LIVE_VARIANT in oos_table.index:
            row, bh = oos_table.loc[LIVE_VARIANT], oos_table.loc["Buy & Hold"]
            is_row, is_bh = is_table.loc[LIVE_VARIANT], is_table.loc["Buy & Hold"]
            checks = passes_bar(row, bh)
            summary_rows.append({
                "Market": label,
                "B&H Ret %": bh["Return [%]"], "B&H DD %": bh["Max DD [%]"],
                "B&H Calmar": bh["Calmar"],
                "20/20 Ret %": row["Return [%]"], "20/20 DD %": row["Max DD [%]"],
                "20/20 Calmar": row["Calmar"], "Trades": row["# Trades"],
                "In-sample Calmar 20/20 vs B&H": f"{is_row['Calmar']} vs {is_bh['Calmar']}",
                "Passes bar?": "YES ✓" if checks["PASS"] else
                    "no: " + ", ".join(k for k, v in checks.items() if not v and k != "PASS"),
            })

    print_cross_market_summary(summary_rows, reports_dir, universe, timeframe)
    print("Backtest complete. The out-of-sample table (sorted by Calmar) is the verdict.")


# --------------------------------------------------------------------------- #
# Argument parsing
# --------------------------------------------------------------------------- #

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Crypto strategy backtester (Phase 2: trend following).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p_dl = sub.add_parser("download", help="Fetch + cache + validate data.")
    p_dl.add_argument("--timeframe", help="Only this timeframe (default: all in config).")
    p_dl.add_argument("--refresh", action="store_true", help="Ignore cache, re-download.")
    p_dl.add_argument("--universe", choices=["crypto", "markets"], default="crypto",
                      help="crypto (Binance) or markets (gold, oil, stocks... ETFs).")

    p_sel = sub.add_parser(
        "select",
        help="Pick the best Donchian config by IN-SAMPLE Calmar, then confirm "
             "it out-of-sample against the Phase 2 success bar.")
    p_sel.add_argument("--timeframe", help="Candle size, e.g. 1d (default: config).")

    p_bt = sub.add_parser("backtest", help="Run the strategy matrix and report.")
    p_bt.add_argument("--strategy", choices=list(STRATEGIES.keys()),
                      help="Run only this strategy's variants (vs Buy & Hold).")
    p_bt.add_argument("--symbol", help="Run a single symbol, e.g. BTC/USDT.")
    p_bt.add_argument("--all", action="store_true",
                      help="Run every variant on every symbol.")
    p_bt.add_argument("--timeframe", help="Candle size, e.g. 1d or 1h.")
    p_bt.add_argument("--refresh", action="store_true", help="Re-download first.")
    p_bt.add_argument("--universe", choices=["crypto", "markets"], default="crypto",
                      help="crypto (Binance) or markets (gold, oil, stocks... ETFs).")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    config = load_config()
    if args.command == "download":
        cmd_download(config, args.timeframe, args.refresh, args.universe)
    elif args.command == "select":
        from src.select import select_and_confirm
        select_and_confirm(config, args.timeframe or config["timeframe"])
    elif args.command == "backtest":
        cmd_backtest(config, args)


if __name__ == "__main__":
    main()
