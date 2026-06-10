"""Honest candidate selection for the Donchian breakout family.

The Phase 2 report is sorted by out-of-sample Calmar, which makes it tempting to
just eyeball the OOS winner — but that's a subtle form of cheating (you're
choosing based on the exam answers). This module does it the disciplined way:

  1. Score every Donchian (N, M, gated?) combo by its IN-SAMPLE Calmar,
     averaged across all symbols so we end up with ONE config for the whole book.
  2. Pick the best purely on that in-sample score.
  3. Only THEN look at its out-of-sample results, and check them against the
     Phase 2 success bar.

If the in-sample-chosen config still clears the bar out-of-sample, the candidate
is real — not a lucky cherry-pick.
"""

from __future__ import annotations

import math

import pandas as pd

from src import backtest as bt_engine
from src import report
from src.data_loader import days_to_bars, load_data, split_in_out
from src.strategies import BuyAndHold, DonchianBreakout

# Phase 2 success bar (all must hold, out-of-sample, on EVERY symbol).
MAX_TRADES = 100          # "dozens, not hundreds"
MAX_FEES_PCT = 5.0        # fees under ~5% of starting equity


def _calmar(stats) -> float:
    cagr = float(stats.get("Return (Ann.) [%]", float("nan")))
    dd = float(stats["Max. Drawdown [%]"])
    if dd == 0 or math.isnan(dd):
        return float("nan")
    return cagr / abs(dd)


def _candidates(config: dict, timeframe: str) -> list[dict]:
    """All Donchian (N, M) combos, plain and regime-gated."""
    risk = config["risk"]
    base_risk = {
        "risk_per_trade": risk["risk_per_trade"],
        "atr_period": risk["atr_period"],
        "atr_stop_multiplier": risk["atr_stop_multiplier"],
    }
    regime_period = days_to_bars(config["regime"]["sma_period_days"], timeframe)
    don = config["strategies"]["donchian_breakout"]

    out = []
    for n in don["entry_days_sweep"]:
        for m in don["exit_days_sweep"]:
            params = {
                **base_risk,
                "entry_period": days_to_bars(n, timeframe),
                "exit_period": days_to_bars(m, timeframe),
            }
            for gated in (False, True):
                p = dict(params)
                tag = ""
                if gated:
                    p["use_regime_gate"] = True
                    p["regime_sma_period"] = regime_period
                    tag = " + Regime"
                out.append({"name": f"Donchian {n}/{m}{tag}", "params": p})
    return out


def select_and_confirm(config: dict, timeframe: str) -> dict:
    """Run the full select-on-in-sample, confirm-on-out-of-sample analysis.

    Returns a dict summarising the chosen config and whether it cleared the bar.
    Also prints a human-readable report.
    """
    cash = config["cash"]
    commission = bt_engine.effective_commission(config["fees"])
    symbols = config["symbols"]

    # Load + split each symbol once.
    splits = {}
    for symbol in symbols:
        df = load_data(symbol, timeframe, config["start_date"], config["end_date"],
                       config["data_dir"], verbose=False)
        splits[symbol] = split_in_out(df, config["in_sample_end"])

    candidates = _candidates(config, timeframe)

    # ---- Step 1+2: score by IN-SAMPLE Calmar, averaged across symbols. ----
    print(f"\nSelecting on IN-SAMPLE Calmar only (≤ {config['in_sample_end']}), "
          f"averaged across {len(symbols)} symbols, timeframe {timeframe}:")
    scored = []
    for cand in candidates:
        calmars = []
        for symbol in symbols:
            in_df, _ = splits[symbol]
            _, stats = bt_engine.run_backtest(in_df, DonchianBreakout, cash,
                                              commission, cand["params"])
            calmars.append(_calmar(stats))
        avg = float(pd.Series(calmars).mean())
        scored.append({**cand, "is_calmar_avg": avg})
        print(f"  {cand['name']:<26}  in-sample avg Calmar = {avg:5.2f}")

    scored.sort(key=lambda c: (math.isnan(c["is_calmar_avg"]), -c["is_calmar_avg"]))
    winner = scored[0]
    print(f"\n>>> Chosen on in-sample: {winner['name']} "
          f"(avg Calmar {winner['is_calmar_avg']:.2f})")

    # ---- Step 3: confirm on OUT-OF-SAMPLE, check the success bar. ----
    print("\nConfirming that exact config on UNTOUCHED out-of-sample data:")
    rows = []
    clears_all = True
    for symbol in symbols:
        in_df, out_df = splits[symbol]
        _, stats = bt_engine.run_backtest(out_df, DonchianBreakout, cash,
                                          commission, winner["params"])
        _, bh = bt_engine.run_backtest(out_df, BuyAndHold, cash, commission, {})

        calmar = _calmar(stats)
        bh_calmar = _calmar(bh)
        dd = float(stats["Max. Drawdown [%]"])
        bh_dd = float(bh["Max. Drawdown [%]"])
        trades = int(stats["# Trades"])
        fees = report._fees_pct(stats, commission, cash)

        # Each criterion of the Phase 2 bar.
        shallower_dd = abs(dd) < abs(bh_dd)
        better_calmar = calmar > bh_calmar
        few_trades = trades < MAX_TRADES
        cheap = fees < MAX_FEES_PCT
        passed = shallower_dd and better_calmar and few_trades and cheap
        clears_all = clears_all and passed

        rows.append({
            "Symbol": symbol,
            "OOS Calmar": round(calmar, 2),
            "B&H Calmar": round(bh_calmar, 2),
            "OOS MaxDD%": round(dd, 1),
            "B&H MaxDD%": round(bh_dd, 1),
            "# Trades": trades,
            "Fees%": round(fees, 2),
            "Clears bar?": "YES ✓" if passed else "no ✗",
        })

    table = pd.DataFrame(rows).set_index("Symbol")
    print(table.to_string())

    verdict = ("PAPER-TRADE CANDIDATE ✓ — cleared the bar on every symbol "
               "out-of-sample." if clears_all else
               "Not yet — failed the bar on at least one symbol out-of-sample.")
    print(f"\nVERDICT: {winner['name']} -> {verdict}\n")

    return {"winner": winner["name"], "params": winner["params"],
            "clears_all": clears_all, "table": table}
