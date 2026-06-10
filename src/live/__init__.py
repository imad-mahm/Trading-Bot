"""Live paper-trading package (Phase 2).

Fake money, real live prices, no exchange account or API key. The engine imports
the SAME strategy signal logic, indicators, and risk.py used by the backtester,
so any gap between live and backtest results points to a bug — not to different
code paths.
"""
