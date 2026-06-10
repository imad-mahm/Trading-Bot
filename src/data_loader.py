"""Download historical candles from Binance (via ccxt) and cache them on disk.

Public market data needs NO API key. We download in a pagination loop because
Binance returns at most ~1000 candles per request, then we save the result as a
Parquet file so future runs are instant.
"""

from __future__ import annotations

import logging
import time
from pathlib import Path

import ccxt
import pandas as pd

log = logging.getLogger("paper.data")

# Standard OHLCV column names used everywhere in this project (lower-case).
OHLCV_COLUMNS = ["open", "high", "low", "close", "volume"]

# Live-price source preference. Binance is first because that's what the strategy
# was validated on, but Binance returns HTTP 451 to US IPs (e.g. GitHub Actions
# runners), so we fall back to Kraken then OKX — both serve the same BTC/USDT and
# ETH/USDT spot candles and are reachable from US/cloud IPs. Daily candles close
# at 00:00 UTC on all three, so they line up. (Override via live.exchanges.)
DEFAULT_EXCHANGE_CHAIN = ["binance", "kraken", "okx"]

# Reuse one ccxt object per exchange (creating them is comparatively expensive).
_EXCHANGE_CACHE: dict[str, ccxt.Exchange] = {}

# Seconds per unit, for converting timeframe strings like "1h" or "1d".
_UNIT_SECONDS = {"m": 60, "h": 3600, "d": 86400, "w": 604800}


def timeframe_seconds(timeframe: str) -> int:
    """How many seconds one candle of `timeframe` covers (e.g. '1h' -> 3600)."""
    number = int(timeframe[:-1])
    unit = timeframe[-1]
    return number * _UNIT_SECONDS[unit]


def bars_per_day(timeframe: str) -> float:
    """How many candles of `timeframe` fit in a calendar day.

    Used to convert strategy lookbacks written in DAYS into candle counts for
    whatever timeframe we're testing: 1d -> 1, 1h -> 24, 4h -> 6, etc.
    """
    return 86400 / timeframe_seconds(timeframe)


def days_to_bars(days: float, timeframe: str) -> int:
    """Convert a lookback in days to a whole number of candles (min 1)."""
    return max(1, round(days * bars_per_day(timeframe)))


def get_exchange(exchange_name: str = "binance") -> ccxt.Exchange:
    """Return a cached ccxt exchange object with rate-limiting turned on (so we
    don't hammer the API and get temporarily blocked)."""
    if exchange_name not in _EXCHANGE_CACHE:
        exchange_class = getattr(ccxt, exchange_name)
        _EXCHANGE_CACHE[exchange_name] = exchange_class({"enableRateLimit": True})
    return _EXCHANGE_CACHE[exchange_name]


def _safe_symbol(symbol: str) -> str:
    """Turn 'BTC/USDT' into 'BTCUSDT' so it is safe to use in a filename."""
    return symbol.replace("/", "")


def cache_path(data_dir: str | Path, symbol: str, timeframe: str) -> Path:
    """Where a given symbol/timeframe is cached on disk."""
    return Path(data_dir) / f"{_safe_symbol(symbol)}_{timeframe}.parquet"


def download_ohlcv(
    symbol: str,
    timeframe: str,
    start_date: str,
    end_date: str | None = None,
    exchange_name: str = "binance",
    verbose: bool = True,
) -> pd.DataFrame:
    """Download every candle from `start_date` to `end_date` (or now).

    Returns a DataFrame indexed by UTC timestamp with columns
    open, high, low, close, volume.
    """
    exchange = get_exchange(exchange_name)

    # Convert dates to milliseconds since 1970 (the format ccxt expects).
    since = exchange.parse8601(f"{start_date}T00:00:00Z")
    if end_date:
        end_ts = exchange.parse8601(f"{end_date}T00:00:00Z")
    else:
        end_ts = exchange.milliseconds()  # right now

    # How many milliseconds one candle covers (e.g. 1h -> 3,600,000).
    timeframe_ms = exchange.parse_timeframe(timeframe) * 1000
    limit = 1000  # Binance's max candles per request

    all_rows: list[list] = []
    while since < end_ts:
        batch = exchange.fetch_ohlcv(symbol, timeframe, since=since, limit=limit)
        if not batch:
            break  # no more data returned

        all_rows.extend(batch)

        # Advance `since` to just past the last candle we received.
        last_timestamp = batch[-1][0]
        since = last_timestamp + timeframe_ms

        if verbose:
            last_dt = pd.to_datetime(last_timestamp, unit="ms", utc=True)
            print(f"  {symbol} {timeframe}: downloaded up to {last_dt:%Y-%m-%d} "
                  f"({len(all_rows):,} candles)", end="\r")

        # If we got a short batch, we've reached the latest available candle.
        if len(batch) < limit:
            break

        # Be polite to the API.
        time.sleep(exchange.rateLimit / 1000)

    if verbose:
        print()  # finish the progress line

    # Build a clean DataFrame.
    df = pd.DataFrame(all_rows, columns=["timestamp", *OHLCV_COLUMNS])
    df = df.drop_duplicates(subset="timestamp").sort_values("timestamp")
    df["datetime"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
    df = df.set_index("datetime")[OHLCV_COLUMNS]

    # Trim anything at/after the requested end time.
    end_dt = pd.to_datetime(end_ts, unit="ms", utc=True)
    df = df[df.index < end_dt]
    return df


def fetch_recent_ohlcv(
    symbol: str,
    timeframe: str,
    limit: int = 400,
    exchanges: list[str] | None = None,
) -> pd.DataFrame:
    """Fetch the most recent `limit` candles for live use (no caching).

    Tries each exchange in `exchanges` (default: Binance -> Kraken -> OKX) and
    returns the first one that answers. This is what makes the bot run in the
    cloud: if Binance is geoblocked (HTTP 451 from US IPs like GitHub Actions),
    it transparently falls back to an exchange that isn't.

    The LAST row is usually the still-forming candle (its open is known, but its
    high/low/close are still updating). The live engine separates that out so it
    never trades on an unfinished candle. Returns a UTC-indexed OHLCV DataFrame.
    """
    chain = exchanges or DEFAULT_EXCHANGE_CHAIN
    errors: list[str] = []
    for name in chain:
        try:
            exchange = get_exchange(name)
            raw = exchange.fetch_ohlcv(symbol, timeframe, limit=limit)
            if not raw:
                raise ValueError("no candles returned")
            df = pd.DataFrame(raw, columns=["timestamp", *OHLCV_COLUMNS])
            df = df.drop_duplicates(subset="timestamp").sort_values("timestamp")
            df["datetime"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
            if name != chain[0]:
                log.info("Fetched %s %s from fallback exchange '%s'.", symbol, timeframe, name)
            return df.set_index("datetime")[OHLCV_COLUMNS]
        except Exception as exc:  # noqa: BLE001 — try the next exchange
            errors.append(f"{name}: {exc}")
            log.warning("Fetch of %s from '%s' failed (%s); trying next source.", symbol, name, exc)
    raise RuntimeError(
        f"All exchanges failed for {symbol} {timeframe}: " + " | ".join(errors))


def split_forming(df: pd.DataFrame, timeframe: str):
    """Split a recent-OHLCV frame into (closed_candles, forming_candle_or_None).

    A candle is still "forming" if its period hasn't elapsed yet — i.e. its open
    time plus one timeframe is still in the future.
    """
    if len(df) == 0:
        return df, None
    period = pd.Timedelta(seconds=timeframe_seconds(timeframe))
    now = pd.Timestamp.now(tz="UTC")
    last_open = df.index[-1]
    if last_open + period > now:
        return df.iloc[:-1], df.iloc[-1]
    return df, None


def load_data(
    symbol: str,
    timeframe: str,
    start_date: str,
    end_date: str | None,
    data_dir: str | Path,
    refresh: bool = False,
    exchange_name: str = "binance",
    verbose: bool = True,
) -> pd.DataFrame:
    """Return candles for a symbol, downloading + caching if needed.

    Set `refresh=True` to ignore any cache and re-download.
    """
    path = cache_path(data_dir, symbol, timeframe)

    if path.exists() and not refresh:
        if verbose:
            print(f"  {symbol} {timeframe}: loaded from cache ({path.name})")
        return pd.read_parquet(path)

    df = download_ohlcv(symbol, timeframe, start_date, end_date, exchange_name, verbose)
    Path(data_dir).mkdir(parents=True, exist_ok=True)
    df.to_parquet(path)
    if verbose:
        print(f"  {symbol} {timeframe}: saved {len(df):,} candles to {path.name}")
    return df


def validate_data(df: pd.DataFrame, timeframe: str) -> dict:
    """Run basic data-quality checks and print a short summary.

    Checks for: duplicate timestamps, missing candles (gaps), and NaN values.
    Returns a dict of the findings (handy for tests).
    """
    exchange = get_exchange()  # only used to parse the timeframe string
    timeframe_ms = exchange.parse_timeframe(timeframe) * 1000
    expected_freq = pd.Timedelta(milliseconds=timeframe_ms)

    # Duplicates.
    duplicates = int(df.index.duplicated().sum())

    # NaNs anywhere in the OHLCV columns.
    nan_count = int(df[OHLCV_COLUMNS].isna().sum().sum())

    # Gaps: build the full expected timeline and see how many candles are missing.
    if len(df) > 1:
        full_range = pd.date_range(df.index.min(), df.index.max(), freq=expected_freq)
        missing = int(len(full_range) - len(df.index.unique()))
    else:
        missing = 0

    summary = {
        "rows": len(df),
        "start": df.index.min(),
        "end": df.index.max(),
        "duplicates": duplicates,
        "missing_candles": missing,
        "nan_values": nan_count,
    }

    print("  Data quality summary:")
    print(f"    Rows:             {summary['rows']:,}")
    print(f"    Date range:       {summary['start']}  ->  {summary['end']}")
    print(f"    Duplicate stamps: {summary['duplicates']}")
    print(f"    Missing candles:  {summary['missing_candles']} "
          f"(gaps in the timeline; some are normal exchange downtime)")
    print(f"    NaN values:       {summary['nan_values']}")
    if duplicates == 0 and nan_count == 0:
        print("    -> Looks clean. ✓")
    return summary


def to_backtesting_format(df: pd.DataFrame) -> pd.DataFrame:
    """Rename columns to the Capitalised names backtesting.py expects
    (Open/High/Low/Close/Volume) and drop the timezone from the index, because
    the engine prefers timezone-naive timestamps.
    """
    out = df.rename(
        columns={
            "open": "Open",
            "high": "High",
            "low": "Low",
            "close": "Close",
            "volume": "Volume",
        }
    ).copy()
    if out.index.tz is not None:
        out.index = out.index.tz_localize(None)
    return out


def split_in_out(df: pd.DataFrame, in_sample_end: str):
    """Split a DataFrame into (in_sample, out_of_sample) at `in_sample_end`.

    In-sample  = dates up to and including in_sample_end (used for tuning).
    Out-of-sample = everything after  (used to judge the strategy honestly).
    """
    cutoff = pd.Timestamp(in_sample_end)
    # Match the index's timezone so the comparison is valid either way.
    if df.index.tz is not None and cutoff.tz is None:
        cutoff = cutoff.tz_localize(df.index.tz)
    elif df.index.tz is None and cutoff.tz is not None:
        cutoff = cutoff.tz_localize(None)
    in_sample = df[df.index <= cutoff]
    out_sample = df[df.index > cutoff]
    return in_sample, out_sample
