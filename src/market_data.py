"""Daily price history for NON-crypto markets (gold, silver, oil, indices...).

We use exchange-traded funds (ETFs) rather than raw futures because:
  * they're what an ordinary brokerage account can actually buy, with no
    leverage and no futures "rolls" to model; and
  * their history is clean and long (most go back to 2006-2008).

Prices are DIVIDEND- AND SPLIT-ADJUSTED (Yahoo Finance `auto_adjust=True`), so
a backtest's return includes dividends (matters a lot for bonds like TLT) and
isn't broken by splits (USO did a 1-for-8 reverse split in 2020).

ETFs trade ~252 days a year (no weekends), not 365 like crypto. Lookbacks in
config are counted in TRADING days for these markets, which is the standard
convention (a "20-day breakout" on gold = 20 trading days), and backtesting.py
annualises with 252 automatically when there are no weekend bars.

Source: Yahoo Finance via `yfinance` (free, no key). Results are cached as
Parquet in the data folder, same as the crypto loader.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

OHLCV = ["open", "high", "low", "close", "volume"]


def market_cache_path(data_dir: str | Path, ticker: str) -> Path:
    return Path(data_dir) / f"mkt_{ticker.upper()}_1d.parquet"


def _clean(df: pd.DataFrame) -> pd.DataFrame:
    """Normalise to our lower-case OHLCV frame with a UTC DatetimeIndex."""
    if isinstance(df.columns, pd.MultiIndex):  # yfinance: (field, ticker)
        df = df.droplevel(-1, axis=1)
    df = df.rename(columns=str.lower)[OHLCV].copy()
    idx = pd.to_datetime(df.index)
    df.index = idx.tz_localize("UTC") if idx.tz is None else idx.tz_convert("UTC")
    df.index.name = "datetime"
    df = df[~df.index.duplicated(keep="last")].sort_index()
    df = df.dropna(subset=["open", "high", "low", "close"])
    df = df[(df[["open", "high", "low", "close"]] > 0).all(axis=1)]
    df["volume"] = df["volume"].fillna(0.0)
    return df.astype(float)


def download_market(ticker: str, start_date: str, end_date: str | None = None) -> pd.DataFrame:
    """Download adjusted daily OHLCV for one ETF/stock ticker from Yahoo."""
    import yfinance as yf  # imported lazily: only needed for this command

    raw = yf.download(ticker, start=start_date, end=end_date, interval="1d",
                      auto_adjust=True, progress=False, threads=False)
    if raw is None or raw.empty:
        raise RuntimeError(f"No data returned for {ticker} (Yahoo unreachable or bad ticker).")
    return _clean(raw)


def load_market_data(ticker: str, start_date: str, end_date: str | None,
                     data_dir: str | Path, refresh: bool = False,
                     verbose: bool = True) -> pd.DataFrame:
    """Return cached data for `ticker`, downloading it first if needed."""
    path = market_cache_path(data_dir, ticker)
    if path.exists() and not refresh:
        df = pd.read_parquet(path)
        if verbose:
            print(f"  {ticker}: loaded from cache ({path.name})")
    else:
        df = download_market(ticker, start_date, end_date)
        path.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(path)
        if verbose:
            print(f"  {ticker}: saved {len(df):,} daily bars to {path.name}")
    start = pd.Timestamp(start_date, tz="UTC")
    df = df[df.index >= start]
    if end_date:
        df = df[df.index <= pd.Timestamp(end_date, tz="UTC")]
    return df


def validate_market(df: pd.DataFrame, ticker: str) -> dict:
    """Quality checks suited to exchange-traded data (weekends/holidays are
    normal gaps, so we flag only gaps longer than 5 calendar days and any
    single-day move over 25%, which usually means bad data or an unadjusted
    split)."""
    gaps = df.index.to_series().diff().dt.days
    big_gaps = int((gaps > 5).sum())
    moves = df["close"].pct_change().abs()
    jumps = moves[moves > 0.25]
    summary = {"ticker": ticker, "rows": len(df), "start": df.index.min(),
               "end": df.index.max(), "gaps_over_5d": big_gaps,
               "moves_over_25pct": len(jumps)}
    print(f"    {ticker}: {len(df):,} bars  {df.index.min():%Y-%m-%d} → {df.index.max():%Y-%m-%d}"
          f"  gaps>5d: {big_gaps}  daily moves>25%: {len(jumps)}"
          + (f"  ({', '.join(f'{d:%Y-%m-%d} {v:+.0%}' for d, v in df['close'].pct_change()[jumps.index].items())})"
             if len(jumps) else ""))
    return summary
