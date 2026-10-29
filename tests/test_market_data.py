"""Market-data (ETF) loader tests — no network."""

import numpy as np
import pandas as pd

from src.market_data import _clean, load_market_data, market_cache_path, validate_market
from run import get_universe, passes_bar
from src.config import load_config


def _yahoo_like(n=10):
    """A frame shaped like yfinance's output: MultiIndex columns, naive dates."""
    idx = pd.bdate_range("2024-01-01", periods=n)
    cols = pd.MultiIndex.from_product([["Close", "High", "Low", "Open", "Volume"], ["GLD"]])
    data = np.column_stack([np.linspace(100, 110, n), np.linspace(101, 111, n),
                            np.linspace(99, 109, n), np.linspace(100, 110, n), np.full(n, 1e6)])
    return pd.DataFrame(data, index=idx, columns=cols)


def test_clean_normalises_yfinance_frame():
    df = _clean(_yahoo_like())
    assert list(df.columns) == ["open", "high", "low", "close", "volume"]
    assert str(df.index.tz) == "UTC"
    assert df.index.is_monotonic_increasing


def test_clean_drops_bad_rows():
    raw = _yahoo_like()
    raw.iloc[3, 0] = np.nan                    # missing close
    raw.iloc[5, 1] = -1.0                      # impossible price
    raw = pd.concat([raw, raw.iloc[[2]]])      # duplicate date
    df = _clean(raw)
    assert len(df) == 8
    assert not df.index.duplicated().any()


def test_cache_round_trip_and_date_filter(tmp_path):
    df = _clean(_yahoo_like(30))
    df.to_parquet(market_cache_path(tmp_path, "GLD"))
    out = load_market_data("GLD", "2024-01-10", "2024-01-31", tmp_path, verbose=False)
    assert out.index.min() >= pd.Timestamp("2024-01-10", tz="UTC")
    assert out.index.max() <= pd.Timestamp("2024-01-31", tz="UTC")


def test_validate_flags_suspicious_jumps():
    df = _clean(_yahoo_like(20))
    df.iloc[10:, :4] *= 8                      # e.g. an unadjusted 1-for-8 split
    assert validate_market(df, "USO")["moves_over_25pct"] == 1
    assert validate_market(_clean(_yahoo_like(20)), "GLD")["moves_over_25pct"] == 0


def test_markets_universe_uses_its_own_fees_and_symbols():
    config = load_config()
    uni = get_universe(config, "markets")
    assert "GLD" in uni.symbols and "USO" in uni.symbols
    assert uni.fees == config["markets"]["fees"]
    assert get_universe(config, "crypto").symbols == config["symbols"]


def test_success_bar():
    bh = {"Max DD [%]": -30.0, "Calmar": 0.3}
    good = {"Max DD [%]": -12.0, "Calmar": 0.6, "# Trades": 20, "Fees [%]": 1.0}
    assert passes_bar(good, bh)["PASS"]
    shallow_not_enough = {**good, "Max DD [%]": -25.0}
    assert not passes_bar(shallow_not_enough, bh)["PASS"]
    assert not passes_bar({**good, "Calmar": 0.2}, bh)["PASS"]


def test_zero_volume_ghost_print_is_dropped_but_real_crash_kept():
    df = _clean(_yahoo_like(20))
    ghost_day, crash_day = df.index[5], df.index[12]
    df.loc[ghost_day, ["open", "high", "low", "close"]] *= 0.67   # bogus print...
    df.loc[ghost_day, "volume"] = 0.0                              # ...nothing traded
    df.loc[crash_day:, ["open", "high", "low", "close"]] *= 0.7    # real crash, with volume
    out = _clean(df)
    assert ghost_day not in out.index
    assert crash_day in out.index
