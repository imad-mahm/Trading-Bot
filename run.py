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
# Commands
# --------------------------------------------------------------------------- #

def cmd_download(config: dict, timeframe_arg: str | None, refresh: bool) -> None:
    """Download + cache + validate data for every symbol and timeframe."""
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
    commission = bt_engine.effective_commission(config["fees"])
    timeframe = args.timeframe or config["timeframe"]
    reports_dir = Path(config["reports_dir"])
    reports_dir.mkdir(parents=True, exist_ok=True)

    variants, regime_period = build_variants(config, timeframe)
    # Targeted mode: keep just the chosen strategy's variants (+ the benchmark).
    if args.strategy and not args.all:
        variants = [v for v in variants if v.key in (args.strategy, "buy_and_hold")]
    symbols = [args.symbol] if args.symbol else config["symbols"]

    print(f"Timeframe: {timeframe}   |   starting cash: ${cash:,.0f}")
    print(f"Fees+slippage per side: {commission * 100:.3f}%  "
          f"(round-trip ≈ {commission * 200:.3f}%)")
    print(f"In-sample ends: {config['in_sample_end']}   |   "
          f"out-of-sample = after that, untouched\n")

    for symbol in symbols:
        df = load_data(
            symbol, timeframe, config["start_date"], config["end_date"],
            config["data_dir"], refresh=args.refresh, verbose=True,
        )
        in_df, out_df = split_in_out(df, config["in_sample_end"])

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
        report.print_table(f"{symbol}  {timeframe}  —  IN-SAMPLE "
                           f"(order follows OOS Calmar)", is_table)
        report.print_table(f"{symbol}  {timeframe}  —  OUT-OF-SAMPLE "
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
    return parser


def main() -> None:
    args = build_parser().parse_args()
    config = load_config()
    if args.command == "download":
        cmd_download(config, args.timeframe, args.refresh)
    elif args.command == "select":
        from src.select import select_and_confirm
        select_and_confirm(config, args.timeframe or config["timeframe"])
    elif args.command == "backtest":
        cmd_backtest(config, args)


if __name__ == "__main__":
    main()
