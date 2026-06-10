"""Technical indicators, hand-rolled with pandas.

Why hand-rolled instead of a library like pandas-ta?
  * Fewer dependencies that can break on new Python/NumPy versions.
  * You can read exactly what each indicator does — nothing is hidden.
  * The same functions are reused by the live bot in a later phase.

Every function takes a pandas Series (a column of prices) and returns a new
Series of the same length. The first few values are NaN because an indicator
needs a "warm-up" window before it can be computed.
"""

import pandas as pd


def sma(series: pd.Series, period: int) -> pd.Series:
    """Simple Moving Average: the plain average of the last `period` values."""
    return series.rolling(window=period).mean()


def ema(series: pd.Series, period: int) -> pd.Series:
    """Exponential Moving Average: a moving average that weights recent values
    more heavily. `adjust=False` makes it match the standard trading formula.
    """
    return series.ewm(span=period, adjust=False).mean()


def _wilder_rma(series: pd.Series, period: int) -> pd.Series:
    """Wilder's smoothing (a.k.a. RMA) — the averaging method used by the
    original RSI and ATR formulas. It is an EMA with alpha = 1 / period.
    """
    return series.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()


def rsi(series: pd.Series, period: int = 14) -> pd.Series:
    """Relative Strength Index (Wilder's original definition).

    RSI oscillates between 0 and 100:
      * Below ~30 is often called "oversold" (price fell a lot, maybe a bounce).
      * Above ~70 is often called "overbought".
    """
    change = series.diff()                       # price change vs previous bar
    gain = change.clip(lower=0)                   # keep only the up moves
    loss = -change.clip(upper=0)                  # keep only the down moves (as +)

    avg_gain = _wilder_rma(gain, period)
    avg_loss = _wilder_rma(loss, period)

    # Relative Strength = average gain / average loss.
    rs = avg_gain / avg_loss
    rsi_values = 100 - (100 / (1 + rs))

    # When there were no losses at all, avg_loss is 0 -> RS is infinite -> RSI 100.
    rsi_values = rsi_values.where(avg_loss != 0, 100.0)
    return rsi_values


def atr(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
    """Average True Range — a measure of how much price typically moves per bar.

    We use it to place stop-losses a sensible distance away (further away when
    the market is volatile, closer when it is calm).
    """
    prev_close = close.shift(1)

    # "True Range" is the largest of these three distances:
    range_high_low = high - low
    range_high_prevclose = (high - prev_close).abs()
    range_low_prevclose = (low - prev_close).abs()

    true_range = pd.concat(
        [range_high_low, range_high_prevclose, range_low_prevclose], axis=1
    ).max(axis=1)

    return _wilder_rma(true_range, period)
