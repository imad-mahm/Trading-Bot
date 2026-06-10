"""The paper portfolio: cash, one position, fees, P&L, and a trade log.

This is pure accounting — no network, no strategy logic — so it's easy to test.
Each symbol has its OWN Portfolio so results never blur together.

Cost model (identical to the backtester): on every fill we charge a combined
commission = taker_fee + slippage on the traded notional. We fold slippage into
the commission exactly as the backtest did, so live and backtest stay comparable.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import pandas as pd


@dataclass
class Position:
    """One open long position."""
    units: float          # how much of the asset we hold
    entry_price: float    # fill price we bought at
    entry_time: str       # ISO timestamp
    stop_price: float     # ATR stop; if price hits this we exit
    entry_fee: float      # fee paid on entry (for P&L math)
    atr_at_entry: float   # ATR used to size the stop (for the record)

    def to_dict(self) -> dict:
        return self.__dict__.copy()

    @classmethod
    def from_dict(cls, d: dict) -> "Position":
        return cls(**d)


@dataclass
class Portfolio:
    symbol: str
    start_time: str
    start_equity: float
    cash: float
    position: Optional[Position] = None

    realized_pnl: float = 0.0
    fees_paid: float = 0.0
    trades: list = field(default_factory=list)

    total_periods: int = 0           # candles seen since start (for time-in-market)
    time_in_market_periods: int = 0  # candles where we held a position
    peak_equity: float = 0.0
    max_drawdown: float = 0.0        # most negative (equity/peak - 1), as a fraction

    # Engine bookkeeping (persisted so restarts are seamless).
    pending: Optional[dict] = None           # {"action","signal_time","atr"} or None
    last_processed_close: Optional[str] = None
    equity_snapshots: list = field(default_factory=list)  # [[iso, equity], ...]

    # ----- core accounting -----

    def equity(self, price: float) -> float:
        """Total account value = cash + (current value of any open position)."""
        if self.position is None:
            return self.cash
        return self.cash + self.position.units * price

    def unrealized_pnl(self, price: float) -> float:
        if self.position is None:
            return 0.0
        return (price - self.position.entry_price) * self.position.units

    def open_position(self, units, price, time_iso, stop_price, commission_rate, atr):
        """Buy `units` at `price`, paying the combined commission."""
        notional = units * price
        fee = notional * commission_rate
        self.cash -= notional + fee
        self.fees_paid += fee
        self.position = Position(
            units=units, entry_price=price, entry_time=time_iso,
            stop_price=stop_price, entry_fee=fee, atr_at_entry=atr,
        )

    def close_position(self, price, time_iso, commission_rate, reason) -> dict:
        """Sell the whole position at `price`, paying commission. Logs the trade
        and returns it. Net P&L includes BOTH the entry and exit fees."""
        pos = self.position
        notional = pos.units * price
        fee = notional * commission_rate
        self.cash += notional - fee
        self.fees_paid += fee

        net_pnl = (notional - fee) - (pos.units * pos.entry_price + pos.entry_fee)
        self.realized_pnl += net_pnl

        trade = {
            "symbol": self.symbol,
            "entry_time": pos.entry_time,
            "entry_price": pos.entry_price,
            "exit_time": time_iso,
            "exit_price": price,
            "units": pos.units,
            "stop_price": pos.stop_price,
            "reason": reason,                       # "signal" or "stop"
            "fees": pos.entry_fee + fee,
            "pnl": net_pnl,
            "return_pct": (price / pos.entry_price - 1) * 100,
        }
        self.trades.append(trade)
        self.position = None
        return trade

    # ----- equity / drawdown tracking -----

    def record_equity(self, price: float, now_iso: str) -> float:
        """Snapshot equity, update peak + max drawdown, and keep only the last
        24h of snapshots (used by the daily-loss limit)."""
        eq = self.equity(price)
        self.peak_equity = max(self.peak_equity, eq)
        if self.peak_equity > 0:
            self.max_drawdown = min(self.max_drawdown, eq / self.peak_equity - 1)

        self.equity_snapshots.append([now_iso, eq])
        cutoff = pd.Timestamp(now_iso) - pd.Timedelta(hours=24)
        self.equity_snapshots = [
            s for s in self.equity_snapshots if pd.Timestamp(s[0]) >= cutoff
        ]
        return eq

    def peak_equity_last_24h(self) -> float:
        if not self.equity_snapshots:
            return self.equity(self.position.entry_price) if self.position else self.cash
        return max(s[1] for s in self.equity_snapshots)

    def time_in_market_pct(self) -> float:
        if self.total_periods == 0:
            return 0.0
        return self.time_in_market_periods / self.total_periods * 100

    # ----- (de)serialisation -----

    def to_dict(self) -> dict:
        d = self.__dict__.copy()
        d["position"] = self.position.to_dict() if self.position else None
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "Portfolio":
        d = d.copy()
        pos = d.pop("position", None)
        port = cls(symbol=d.pop("symbol"), start_time=d.pop("start_time"),
                   start_equity=d.pop("start_equity"), cash=d.pop("cash"))
        for key, value in d.items():
            setattr(port, key, value)
        port.position = Position.from_dict(pos) if pos else None
        return port
