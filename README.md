# Crypto Strategy Backtester — Phase 2 (Trend Following)

A small, readable framework that downloads **free** historical crypto data and
**rigorously** tests trading strategies against it. The whole point is honesty:
it bakes in the things that make most backtests lie (fees, slippage, lookahead
bias, over-fitting) so the results you see are closer to what would really happen.

The *strategy logic* and *risk rules* live in their own modules
(`src/strategies/`, `src/risk.py`, `src/indicators.py`) so the live bot in a
later phase can import the exact same code, unchanged.

> **The golden rule:** every strategy is measured against **Buy & Hold**. But in
> Phase 2 the goal isn't to beat its raw *return* — it's to **survive better**:
> a much shallower drawdown and a higher **Calmar ratio** (return per unit of pain),
> while trading rarely and paying little in fees.

### What changed since Phase 1

Phase 1 tested fast strategies on **hourly** candles. They lost badly: hundreds
of trades meant fees and slippage ate the account, and "buy the dip" mean
reversion is a poor fit for trending crypto. So Phase 2:

- **Switched to daily candles** and slower, **trend-following** ideas.
- **Deleted the RSI mean-reversion strategy** — ruled out by the data.
- Added a **Trend Filter**, a **Donchian breakout**, and a reusable **regime gate**.
- Upgraded the report with **time-in-market**, **Calmar**, and **fees-paid** columns.
- Kept **all** Phase 1 realism safeguards exactly as they were.

---

## 1. What you need

- **Python 3.10 or newer** (tested on 3.13).
- Internet the first time, to download Binance's public data (**no API key**).

No paid services, no databases. Data is cached as files on disk.

---

## 2. Setup (one time)

```powershell
python -m venv .venv                 # isolated environment
.\.venv\Scripts\Activate.ps1         # activate (macOS/Linux: source .venv/bin/activate)
pip install -r requirements.txt      # install the libraries
```

A virtual environment keeps these packages from clashing with other projects;
delete the `.venv/` folder anytime to reset. (Plain `pip install` also works.)

---

## 3. Quick start

```powershell
# A) Download + cache + quality-check the data (daily AND hourly). Run first.
python run.py download

# B) The Phase 2 main run: the full strategy matrix on daily candles
python run.py backtest --all --timeframe 1d

# C) Just one strategy family (still shown next to Buy & Hold)
python run.py backtest --strategy donchian_breakout --symbol BTC/USDT --timeframe 1d
```

| Flag | What it does |
|------|--------------|
| `--all` | Run every strategy variant on every symbol. |
| `--strategy NAME` | Run one family: `buy_and_hold`, `trend_filter`, `sma_crossover`, `donchian_breakout`. |
| `--symbol BTC/USDT` | One symbol instead of all. |
| `--timeframe 1d` \| `1h` | Candle size. Lookbacks are defined in **days** and auto-converted. |
| `--refresh` | Re-download data, ignoring the cache. |

Results print as tables (sorted by out-of-sample Calmar) **and** save to
`reports/` as CSVs + PNG charts.

---

## 4. Project layout

```
crypto-backtester/
├── config.yaml          # ALL settings. Lookbacks are in DAYS now.
├── run.py               # CLI: builds and runs the strategy matrix
├── data/                # cached downloads (gitignored)
├── reports/             # generated tables + charts (gitignored)
└── src/
    ├── data_loader.py   # download + cache + validate; day→candle conversion
    ├── indicators.py    # SMA, EMA, RSI, ATR (hand-written, commented)
    ├── risk.py          # position sizing + ATR stop (REUSED by the live bot)
    ├── backtest.py      # thin wrapper around the backtesting.py engine
    ├── report.py        # tables + equity/drawdown/regime charts
    └── strategies/
        ├── base.py            # shared risk sizing, ATR stop, + REGIME GATE
        ├── buy_and_hold.py    # the benchmark
        ├── trend_filter.py    # 200-day SMA: long above, cash below
        ├── sma_crossover.py   # 50/200 golden cross
        └── donchian_breakout.py  # N-day high in, M-day low out
```

---

## 5. The strategies, in plain language

### Buy & Hold *(benchmark)*
Buy once, never sell. Always 100% invested — so it also eats every bear market
in full. The yardstick everything else is compared to.

### Trend Filter *(the simplest trend follower)*
Be fully **long while price closes above its 200-day moving average; sit in cash
while below it.** That's the whole strategy. It trades rarely and sidesteps the
worst of bear markets by being in cash. The 200-day SMA *is* its stop-loss.
- **Buffer (optional):** require price to be e.g. 1% *beyond* the SMA before
  flipping. This dead-band reduces "whipsaw" — pointless flip-flopping when price
  hovers right on the line. We test it with buffer `0` and `1%`.

### SMA Crossover *(50/200 "golden cross")*
Go long when the 50-day SMA crosses above the 200-day SMA; exit when it crosses
back below. A slower, daily cousin of the Phase 1 version.

### Donchian Breakout *(classic "turtle" trend system)*
- **Buy** when price makes a new **N-day high** (it's breaking out upward).
- **Sell** when price makes a new **M-day low** (the move has rolled over).
- Keeps the ATR stop and 2%-risk sizing. We sweep N ∈ {20, 55}, M ∈ {10, 20}.

All strategies are **long-only, spot, no leverage, one position at a time.**

### The regime gate (applied to SMA Crossover and Donchian)
A reusable switch (in `src/strategies/base.py`) that wraps a strategy with the
trend filter's logic: while it's on, the strategy **may only open longs when
price is above the 200-day SMA, and is forced to exit the moment price closes
below it.** It lets a fast signal still fire, but only "with the tide." Each
gated strategy is run **both with and without** the gate, so you can see its
effect in isolation. On the equity charts, grey bands mark when the gate had the
strategy parked in cash.

---

## 6. How to read the report

Each run prints two tables per symbol — **IN-SAMPLE** and **OUT-OF-SAMPLE**
(see §7) — sorted by out-of-sample **Calmar**. Columns:

| Metric | Plain-English meaning | Good = |
|--------|----------------------|--------|
| **Return [%]** | Total profit/loss over the period. | higher |
| **CAGR [%]** | Return as a smooth *yearly* rate (annualised). | higher |
| **Max DD [%]** | Worst peak-to-valley drop. The "how much pain?" number. | closer to 0 |
| **Calmar** | **CAGR ÷ \|Max DD\|.** Return *per unit of pain* — our headline. The table is sorted by this. | higher |
| **Time in Mkt [%]** | Share of candles with an open position. Matching returns while invested *less* of the time is a real edge (less exposure to surprises). | context — lower for same return is better |
| **Sharpe** | Return per unit of volatility. >1 is good; negative means paid to lose. | higher |
| **# Trades** | How many round trips. Dozens = fine; hundreds = fee bleed. | dozens |
| **Fees [%]** | Total fees + slippage paid, as a % of starting equity. Makes the cost of over-trading explicit. | low (we target <5%) |
| **Final Equity** | Ending account value (starts at `cash`). | higher |
| **Beat B&H?** | `YES ✓` if the strategy's raw return beat Buy & Hold. | context |

**Why Calmar and time-in-market matter most here:** a strategy that returns a
bit less than Buy & Hold but with a -8% drawdown instead of -55%, while sitting
in cash 60% of the time, is *far* more survivable with real money. Raw return
(`Beat B&H?`) is the wrong lens for that; Calmar is the right one.

**Charts** (`reports/`):
- `*_oos_equity.png` — every strategy's equity vs Buy & Hold (dashed black).
- `*_oos_drawdown.png` — how far underwater each was, over time.
- `*_oos_regime.png` — the regime-gated strategies, with **grey bands showing
  when the 200-day filter had them in cash.**

---

## 7. The realism safeguards (unchanged — why this is trustworthy)

1. **Fees — 0.1% per trade** (Binance spot taker, `fees.taker_fee`), charged on
   entry *and* exit.
2. **Slippage — 0.05% per trade** (`fees.slippage`), added to the fee. A round
   trip costs ≈ **0.30%** (printed each run). This is why Phase 1's hundreds of
   trades were fatal — and why Phase 2 aims for dozens.
3. **No lookahead bias.** A signal from a candle's *close* can only trade at the
   **open of the next candle**. Enforced by the engine and **verified by tests**.
4. **In-sample / out-of-sample split.** Any tuning uses only data up to
   `in_sample_end` (2023-12-31). The period after is held back as the real exam.
   Both are printed; the in-sample table is shown in the *same row order* as the
   out-of-sample one so you can spot strategies that looked great while tuning
   and fell apart afterwards.
5. **No over-optimization.** The Donchian sweep uses tiny grids (2 values each),
   and **every** combo is shown out-of-sample, so degradation can't hide.

> **Daily lookbacks, honestly converted.** Lookbacks in `config.yaml` are in
> **days**. At run time they convert to candles for the chosen timeframe (a
> 200-day SMA = 200 candles on `1d`, or 4800 on `1h`). This prevents the classic
> mistake of reusing an hourly "200" as if it meant 200 days.

---

## 8. Risk management (shared with the future live bot)

In `src/risk.py`:
- **ATR stop-loss:** placed `2 × ATR(14)` below entry (wider when volatile).
- **2%-risk position sizing:** size is set from the stop distance so the dollar
  loss if stopped out is ≈ 2% of equity.

(The Trend Filter is the exception — its 200-day SMA *is* the exit, so it just
holds all-in when the trend is up. The regime-gated SMA/Donchian strategies use
both the ATR stop *and* the regime exit.)

---

## 9. The success bar for Phase 2

We are **not** trying to beat Buy & Hold's raw return. Out-of-sample, on **both**
symbols, a candidate worth promoting to Phase 2 paper trading should show:

- **Max drawdown meaningfully shallower than B&H** (B&H hit ~-51% on BTC, ~-67% on ETH).
- **Calmar above B&H's.**
- **Trade counts in the dozens, not hundreds.**
- **Fees under ~5% of starting equity.**

If one configuration clears that bar on both symbols out-of-sample, it's the
candidate. If nothing does, the sorted table makes that just as obvious.

---

## 10. Running the tests

```powershell
python -m pytest -q
```

Covers: the indicator maths (SMA/EMA/RSI/ATR), the **no-lookahead** rule, the
**Donchian** channel logic (entries on new highs, exits on new lows, prior-bar
window), and the **regime gate** (forced exit within one bar of the trend
flipping down, and reduced time in market).

---

## 11. Honest limitations

- **Whole-unit trading.** The engine buys whole units, so we use a large starting
  `cash` ($1,000,000) to make rounding negligible. Only **percentages** matter;
  absolute dollars are arbitrary.
- **Slippage is a flat 0.05% estimate**, not modelled per-order from the book.
- **ATR period stays 14 candles** on any timeframe (14 days on daily) — standard,
  and it only scales the stop.
- A few **missing candles** on hourly data is normal exchange downtime, not a bug.

---

## 12. What the Phase 2 run actually showed

Run `python run.py backtest --all --timeframe 1d` and read the **out-of-sample**
table (sorted by Calmar). The headline:

- **The Donchian breakout family cleared the success bar on both BTC and ETH
  out-of-sample** — e.g. `Donchian 20/10` and the regime-gated `Donchian 55/10
  + Regime`. Drawdowns of roughly **-5% to -12%** versus Buy & Hold's **-51%
  (BTC) / -67% (ETH)**, Calmar well above B&H, **dozens** of trades, and fees
  **under ~2%**. On ETH, where Buy & Hold actually *lost* money out-of-sample,
  these strategies stayed positive *and* far calmer.
- **The regime gate did its job:** it consistently shrank drawdowns and
  time-in-market, usually at a small cost to raw return — exactly the trade you
  want for survivability.
- **The plain Trend Filter and SMA crossover under-delivered** out-of-sample —
  useful as honest controls, not candidates.
- **Daily really is the better home.** Re-running Donchian on **hourly**
  (`--timeframe 1h`) roughly **doubled the trades and 5–10×'d the fees** (e.g.
  ~18% vs ~1.8% on BTC `20/10`) with deeper drawdowns and lower Calmar.

Bottom line: a slow, trend-following, regime-aware Donchian breakout on daily
candles is the first thing in this project that survives honest out-of-sample
testing on both coins — the natural candidate to carry into Phase 2 paper
trading. As always, the framework refuses to flatter: nothing beat Buy & Hold's
raw *return*, and the table says so plainly.

---
---

# Phase 2: Live Paper Trading (fake money, real prices)

Phase 2 takes the **exact** winning strategy from Phase 1.5 — daily Donchian
20/20 — and runs it **live against real market prices, but with fake money**. No
exchange account, no API keys, no real orders. The bot keeps its own simulated
portfolio and consumes free public price data.

**The one rule that makes this a valid experiment: reuse, don't rewrite.** The
live bot imports the *same* strategy signal, the *same* `indicators.py`, and the
*same* `risk.py` sizing + stop logic the backtester used. If live results drift
far from the backtest over the same period, that's a *bug*, not the market —
and the report is built to catch exactly that.

## A. Two-minute Telegram setup (free, optional)

You'll get a phone alert on every entry, exit, stop, halt, and a daily summary.

1. In Telegram, message **@BotFather**, send `/newbot`, follow the prompts, and
   copy the **bot token** it gives you.
2. Message your new bot once (say "hi") so it's allowed to message you.
3. Get your **chat id**: open
   `https://api.telegram.org/bot<YOUR_TOKEN>/getUpdates` in a browser and look
   for `"chat":{"id":...}`.
4. Put them in `config.yaml` under `live.telegram`, **or** (preferred, keeps
   secrets out of files) set environment variables `TELEGRAM_BOT_TOKEN` and
   `TELEGRAM_CHAT_ID`.

No token? The bot still runs fine — alerts just go to the log file instead.

## B. The commands

```powershell
python paper.py tick      # do ONE cycle and exit (used by the cron deployment)
python paper.py run       # loop forever (used by the always-on deployment)
python paper.py status    # equity, open positions, unrealized P&L
python paper.py report    # live vs backtest vs buy & hold
python paper.py halt --flatten   # emergency: stop entries (and sell everything)
python paper.py resume    # clear the halt and continue
python paper.py reset --yes      # wipe all paper state and start over
```

Settings live under the `live:` block in `config.yaml`: the strategy + params
(the Phase 1.5 winner), the symbols (BTC + ETH, each with its **own** $10,000
paper account), the timeframe (daily), the daily-loss limit, and Telegram keys.

## C. How a cycle works (and why it can't cheat)

1. Fetch recent candles via ccxt public endpoints. The still-forming candle is
   set aside — **we never act on an unfinished candle.**
2. Compute the signal from the latest *closed* candle, using the strategy's own
   functions. The simulated fill happens at the **open of the next candle** —
   the same no-lookahead rule as the backtester.
3. Fills pay the same combined cost as the backtest (0.1% fee + 0.05% slippage),
   so live and backtest stay comparable.
4. **Stops are polled between candles** (every `poll_seconds` in `run` mode, or
   every cron tick): if price touches the ATR stop, we exit at the stop price.
5. Every action — including "checked, no signal, holding" — is written to a
   rotating log (`logs/paper.log`).

## D. Crash safety (a bot that forgets is disqualified)

All state — positions, cash, pending orders, the last processed candle, halt
flag — is saved to `live_state.json` after every change, using an **atomic
write** (temp file + `os.replace`) so a crash mid-save can't corrupt it. On
startup the bot reloads everything, **fetches any candles it missed while down,
and processes them in order** before resuming. (Tested in
`tests/test_live_engine.py::test_catch_up_processes_all_missed_candles`.)

## E. The risk manager (same as backtest, plus live guards)

- **2% risk per trade + ATR stop on every position** — straight from `risk.py`.
- **Daily loss limit:** if equity falls more than `daily_loss_limit` (default
  5%) from its 24h peak, the bot **flattens and halts** until you `resume`.
- **Kill switch:** `paper.py halt` stops new entries (add `--flatten` to also
  sell now); `paper.py resume` restarts.
- **Data sanity guard:** if a candle jumps more than `data_anomaly_threshold`
  (default 30%) vs the prior one, or the data looks like garbage, the bot
  **skips the cycle and alerts** instead of trading on bad data.

## F. Reading the report

`python paper.py report` shows, per symbol:

| Column | Meaning |
|--------|---------|
| Live Return % / Live MaxDD % | The paper bot's actual performance since start. |
| **Backtest Return % / MaxDD %** | The backtester run over the **same window** — the expectation. |
| Buy&Hold Return % | The benchmark over the same window. |
| # Trades, Fees $, Time in Mkt %, Equity $ | The usual diagnostics. |

**The live-vs-backtest gap is the whole point.** A small gap = the bot faithfully
reproduces the strategy. A large, persistent gap = an implementation bug or an
unrealistic assumption somewhere. Watch that column above all.

## G. Deployment (pick one — both are free)

**Option 1 — Always-on (`paper.py run`).** Best for a spare PC, a Raspberry Pi,
or an **Oracle Cloud always-free VM**. Use the included
`deploy/paper-bot.service` systemd unit so it auto-restarts on crash or reboot:

```bash
sudo cp deploy/paper-bot.service /etc/systemd/system/
sudo systemctl daemon-reload && sudo systemctl enable --now paper-bot
journalctl -u paper-bot -f      # live logs
```

**Option 2 — Scheduled (`paper.py tick`).** Best if you don't want to run a
machine at all. The included `.github/workflows/paper-tick.yml` runs one cycle
every 30 minutes on **GitHub Actions** and commits the updated state file back to
the repo. Add `TELEGRAM_BOT_TOKEN` / `TELEGRAM_CHAT_ID` as repo secrets.
- *Why commit state instead of using the cache?* Committing is durable and
  auditable; the Actions cache can be evicted after ~7 days and would wipe the
  bot's memory. The tradeoff is a tiny `[skip ci]` commit each cycle.
- *Scheduling jitter:* GitHub cron is best-effort — runs can be delayed a few
  minutes or occasionally skipped.

**Recommendation for our daily strategy:** the daily Donchian makes at most one
decision per day, so **GitHub Actions scheduled mode is the better fit** —
zero machines to babysit, and 30-minute stop polling is plenty for a daily
system. Choose the always-on VM only if you later move to a faster timeframe or
want tighter stop polling.

## H. The run plan and what success looks like

- **Run untouched for 8–12 weeks minimum.** No tweaking parameters mid-run — if
  you change params, the clock **restarts** (otherwise you're just curve-fitting
  live).
- **Success =** live results roughly track the backtest expectation over the same
  period, every risk rule fires correctly when triggered, there are zero
  unhandled crashes, and the Calmar-vs-buy-and-hold picture looks like Phase 1.5.
- **Important nuance:** a *losing* 8 weeks does **not** by itself mean the bot is
  broken — markets have losing stretches, and the backtest had them too. What
  *does* mean it's broken is a **large live-vs-backtest divergence**. Judge the
  bot by faithfulness to its backtest, not by whether this particular stretch was
  green.

## I. Sample Telegram alerts

```
🟢 ENTER BTC/USDT
price $108.00  size 15.384615 (~$1,662)
stop $95.00  reason: new 3-candle high breakout

🛑 EXIT BTC/USDT (STOP-LOSS hit)
price $95.00  pnl $-204.68 (-12.04%)

⛔ DAILY LOSS LIMIT
BTC/USDT: equity $9,000 fell >5% from 24h peak $10,000. Trading halted.
```

## J. Live tests

`python -m pytest -q` also covers the live bot: portfolio accounting (fills,
fees, P&L), state save/reload round-trip, missed-candle catch-up, the stop-loss
trigger, and the daily-loss halt.
