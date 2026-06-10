"""Crash-safe state persistence.

Everything the bot needs to resume after a restart — every portfolio, the global
halt flag, and bookkeeping cursors — lives in one JSON file. We write it with an
ATOMIC replace (write a temp file, then os.replace) so a crash mid-write can
never corrupt the real file: you either get the old complete file or the new
complete file, never a half-written one.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

from src.live.portfolio import Portfolio


class State:
    """In-memory state plus load/save to a JSON file."""

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.halted: bool = False
        self.halt_reason: str | None = None
        self.portfolios: dict[str, Portfolio] = {}

    # ----- persistence -----

    def save(self) -> None:
        """Atomically write the whole state to disk."""
        data = {
            "version": 1,
            "halted": self.halted,
            "halt_reason": self.halt_reason,
            "portfolios": {sym: p.to_dict() for sym, p in self.portfolios.items()},
        }
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(self.path.suffix + ".tmp")
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, self.path)  # atomic on all major OSes

    @classmethod
    def load(cls, path: str | Path) -> "State":
        """Load state from disk, or return a fresh empty state if none exists."""
        state = cls(path)
        if not state.path.exists():
            return state
        with open(state.path, "r", encoding="utf-8") as f:
            data = json.load(f)
        state.halted = data.get("halted", False)
        state.halt_reason = data.get("halt_reason")
        state.portfolios = {
            sym: Portfolio.from_dict(pd_)
            for sym, pd_ in data.get("portfolios", {}).items()
        }
        return state

    # ----- helpers -----

    def ensure_portfolio(self, symbol: str, start_time: str, starting_balance: float) -> Portfolio:
        """Return the portfolio for `symbol`, creating it on first use."""
        if symbol not in self.portfolios:
            self.portfolios[symbol] = Portfolio(
                symbol=symbol, start_time=start_time,
                start_equity=starting_balance, cash=starting_balance,
                peak_equity=starting_balance,
            )
        return self.portfolios[symbol]
