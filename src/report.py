"""Turn raw backtest stats into readable tables and charts.

New in Phase 2, the comparison table adds three columns that matter for a
trend-following strategy whose goal is *risk-adjusted* survival, not raw return:

  * Time in Mkt %  — fraction of candles with an open position. Matching the
    benchmark's return while invested only half the time is a real edge.
  * Calmar         — CAGR divided by the absolute max drawdown. Our headline
    "return per unit of pain" number. The table is SORTED by this (out-of-sample).
  * Fees %         — total fees + slippage paid, as a % of starting equity, so
    the cost bleed from over-trading is impossible to ignore.
"""

from __future__ import annotations

import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # render to files without a screen/GUI
import matplotlib.pyplot as plt
import pandas as pd

BENCHMARK_NAME = "Buy & Hold"


def _calmar(cagr: float, max_dd: float) -> float:
    """CAGR / |max drawdown|. Returns NaN if there was no drawdown to divide by."""
    if max_dd == 0 or math.isnan(max_dd):
        return float("nan")
    return cagr / abs(max_dd)


def _fees_pct(stats, commission: float, starting_cash: float) -> float:
    """Total fees+slippage paid, as a % of starting equity.

    The engine charges `commission` on every transaction (entry and exit), so we
    sum the traded notional on both sides and multiply by the commission rate.
    """
    trades = stats["_trades"]
    if len(trades) == 0:
        return 0.0
    entry_notional = (trades["Size"].abs() * trades["EntryPrice"]).sum()
    exit_notional = (trades["Size"].abs() * trades["ExitPrice"]).sum()
    total_fees = commission * (entry_notional + exit_notional)
    return total_fees / starting_cash * 100


def _metrics_row(name: str, stats, commission: float, starting_cash: float) -> dict:
    """Extract the reported metrics from one backtesting.py stats object."""
    cagr = float(stats.get("Return (Ann.) [%]", float("nan")))
    max_dd = float(stats["Max. Drawdown [%]"])
    return {
        "Strategy": name,
        "Return [%]": round(float(stats["Return [%]"]), 1),
        "CAGR [%]": round(cagr, 1),
        "Max DD [%]": round(max_dd, 1),
        "Calmar": round(_calmar(cagr, max_dd), 2),
        "Time in Mkt [%]": round(float(stats["Exposure Time [%]"]), 1),
        "Sharpe": round(float(stats["Sharpe Ratio"]), 2),
        "# Trades": int(stats["# Trades"]),
        "Fees [%]": round(_fees_pct(stats, commission, starting_cash), 2),
        "Final Equity": round(float(stats["Equity Final [$]"]), 0),
    }


def comparison_table(
    stats_by_strategy: dict,
    commission: float,
    starting_cash: float,
    order: list | None = None,
) -> pd.DataFrame:
    """Build the per-segment comparison table.

    `stats_by_strategy` maps display name -> stats, and must include the
    benchmark ("Buy & Hold"). If `order` is given, rows are reindexed to it;
    otherwise rows are sorted by Calmar (best first). A blunt "Beat B&H?" column
    flags whether each strategy beat Buy & Hold's raw return.
    """
    rows = [_metrics_row(n, s, commission, starting_cash) for n, s in stats_by_strategy.items()]
    table = pd.DataFrame(rows).set_index("Strategy")

    benchmark_return = table.loc[BENCHMARK_NAME, "Return [%]"]
    table["Beat B&H?"] = [
        "—" if name == BENCHMARK_NAME
        else ("YES ✓" if ret > benchmark_return else "no ✗")
        for name, ret in zip(table.index, table["Return [%]"])
    ]

    if order is not None:
        table = table.reindex(order)
    else:
        # Sort by Calmar, best first. NaNs (no drawdown) sink to the bottom.
        table = table.sort_values("Calmar", ascending=False, na_position="last")
    return table


def print_table(title: str, table: pd.DataFrame) -> None:
    """Pretty-print a comparison table to the console."""
    print()
    print("=" * len(title))
    print(title)
    print("=" * len(title))
    print(table.to_string())
    print()


def save_table_csv(table: pd.DataFrame, out_path: Path) -> None:
    table.to_csv(out_path)


# --------------------------------------------------------------------------- #
# Charts
# --------------------------------------------------------------------------- #

def _plot_equity_lines(ax, stats_by_strategy: dict) -> None:
    """Draw every strategy's equity curve; Buy & Hold as a dashed black line."""
    for name, stats in stats_by_strategy.items():
        equity = stats["_equity_curve"]["Equity"]
        if name == BENCHMARK_NAME:
            ax.plot(equity.index, equity.values, label=name, linewidth=2.5,
                    color="black", linestyle="--")
        else:
            ax.plot(equity.index, equity.values, label=name, linewidth=1.4)


def plot_equity_curves(stats_by_strategy: dict, title: str, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(11, 6))
    _plot_equity_lines(ax, stats_by_strategy)
    ax.set_title(title)
    ax.set_xlabel("Date")
    ax.set_ylabel("Account equity ($)")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def plot_drawdowns(stats_by_strategy: dict, title: str, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(11, 5))
    for name, stats in stats_by_strategy.items():
        dd = stats["_equity_curve"]["DrawdownPct"] * 100  # fraction -> %
        ax.plot(dd.index, -dd.values, label=name, linewidth=1.1)
    ax.set_title(title)
    ax.set_xlabel("Date")
    ax.set_ylabel("Drawdown (%)")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def _shade_cash_periods(ax, index, cash_mask) -> None:
    """Shade contiguous date ranges where `cash_mask` is True (regime = cash)."""
    in_region = False
    start = None
    labelled = False
    for timestamp, is_cash in zip(index, cash_mask):
        if is_cash and not in_region:
            in_region, start = True, timestamp
        elif not is_cash and in_region:
            ax.axvspan(start, timestamp, color="grey", alpha=0.15,
                       label=None if labelled else "Regime: in cash")
            labelled, in_region = True, False
    if in_region:
        ax.axvspan(start, index[-1], color="grey", alpha=0.15,
                   label=None if labelled else "Regime: in cash")


def plot_regime_equity(
    stats_by_strategy: dict, band_index, cash_mask, title: str, out_path: Path
) -> None:
    """Equity curves for the regime-gated variants, with grey bands marking the
    stretches when the 200-day filter had the strategies in cash."""
    fig, ax = plt.subplots(figsize=(11, 6))
    _shade_cash_periods(ax, band_index, cash_mask)
    _plot_equity_lines(ax, stats_by_strategy)
    ax.set_title(title)
    ax.set_xlabel("Date")
    ax.set_ylabel("Account equity ($)")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
