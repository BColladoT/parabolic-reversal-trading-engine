"""Quickstart example: run a simplified V5 Relaxed backtest on sample data.

This is a self-contained demonstration of the strategy logic. The full
production engine in `src/` includes 3-tier scale-in, layered exits,
ATR-based stops, and absorption detection — this script implements the core
entry signal (VWAP extension + volume exhaustion + time window) on five
representative micro-cap setups.

Inputs:  data/sample/{SYMBOL}.parquet  (5 symbols, 30-day windows each)
Outputs: docs/images/example_equity_curve.png
         stdout: per-trade summary table

Run:     python scripts/run_example_backtest.py
"""
from __future__ import annotations

from datetime import time, timedelta
from pathlib import Path

import matplotlib.pyplot as plt
import polars as pl

ROOT = Path(__file__).resolve().parents[1]
SAMPLE = ROOT / "data" / "sample"
OUT_IMG = ROOT / "docs" / "images" / "example_equity_curve.png"

# Strategy parameters — simplified V5 Relaxed entry criteria.
GAIN_THRESHOLD_PCT = 60.0       # Day must be up 60% from open
VWAP_EXTENSION = 1.20           # Price >120% of session VWAP
VOLUME_EXHAUSTION = 0.60        # Current volume <60% of session peak so far
ENTRY_WINDOW = (time(9, 45), time(14, 30))   # ET
HARD_STOP_PCT = 0.02            # 2% above day high
FLATTEN_TIME = time(15, 25)     # ET
POSITION_NOTIONAL = 10_000.0    # $10K per trade


def to_et(df: pl.DataFrame) -> pl.DataFrame:
    """Add ET-aware timestamp columns. Source is naive UTC."""
    return df.with_columns(
        pl.col("timestamp").dt.replace_time_zone("UTC").dt.convert_time_zone("America/New_York").alias("ts_et")
    ).with_columns(
        pl.col("ts_et").dt.date().alias("trade_date"),
        pl.col("ts_et").dt.time().alias("clock_et"),
    )


def session_features(day: pl.DataFrame) -> pl.DataFrame:
    """Compute session-anchored VWAP, peak volume, day open/high. Anchored at 09:30 ET."""
    market_open = time(9, 30)
    rth = day.filter(pl.col("clock_et") >= market_open).sort("ts_et")
    if rth.is_empty():
        return rth
    rth = rth.with_columns(
        (pl.col("close") * pl.col("volume")).cum_sum().alias("_cum_pv"),
        pl.col("volume").cum_sum().alias("_cum_v"),
        pl.col("volume").cum_max().alias("peak_volume_so_far"),
        pl.col("high").cum_max().alias("day_high"),
    )
    day_open = rth["open"][0]
    return rth.with_columns(
        (pl.col("_cum_pv") / pl.col("_cum_v")).alias("session_vwap"),
        pl.lit(day_open).alias("day_open"),
        ((pl.col("high").cum_max() - day_open) / day_open * 100).alias("day_gain_pct"),
    ).drop(["_cum_pv", "_cum_v"])


def simulate_day(day: pl.DataFrame, symbol: str) -> dict | None:
    """Simulate one trading day. Returns trade dict or None if no entry fired."""
    feat = session_features(day)
    if feat.is_empty():
        return None
    win_start, win_end = ENTRY_WINDOW

    candidates = feat.filter(
        (pl.col("clock_et") >= win_start)
        & (pl.col("clock_et") <= win_end)
        & (pl.col("day_gain_pct") >= GAIN_THRESHOLD_PCT)
        & (pl.col("close") > VWAP_EXTENSION * pl.col("session_vwap"))
        & (pl.col("volume") < VOLUME_EXHAUSTION * pl.col("peak_volume_so_far"))
    )
    if candidates.is_empty():
        return None

    entry = candidates.row(0, named=True)
    entry_price = entry["close"]
    day_high_at_entry = entry["day_high"]
    hard_stop = day_high_at_entry * (1 + HARD_STOP_PCT)

    after = feat.filter(pl.col("ts_et") > entry["ts_et"])
    if after.is_empty():
        return None

    exit_price = None
    exit_reason = None
    exit_time = None
    for row in after.iter_rows(named=True):
        if row["clock_et"] >= FLATTEN_TIME:
            exit_price, exit_reason, exit_time = row["close"], "time_flatten", row["ts_et"]
            break
        if row["high"] >= hard_stop:
            exit_price, exit_reason, exit_time = hard_stop, "hard_stop", row["ts_et"]
            break
        if row["close"] <= row["session_vwap"]:
            exit_price, exit_reason, exit_time = row["close"], "vwap_target", row["ts_et"]
            break
    if exit_price is None:
        last = after.row(-1, named=True)
        exit_price, exit_reason, exit_time = last["close"], "session_end", last["ts_et"]

    shares = int(POSITION_NOTIONAL / entry_price)
    pnl = (entry_price - exit_price) * shares  # short

    return {
        "symbol": symbol,
        "date": str(entry["trade_date"]),
        "entry_time": entry["ts_et"].strftime("%H:%M ET"),
        "exit_time": exit_time.strftime("%H:%M ET"),
        "entry_price": entry_price,
        "exit_price": round(exit_price, 4),
        "shares": shares,
        "pnl": round(pnl, 2),
        "exit_reason": exit_reason,
        "day_gain_pct": round(entry["day_gain_pct"], 1),
        "vwap_ext": round(entry["close"] / entry["session_vwap"], 3),
    }


def run() -> list[dict]:
    trades: list[dict] = []
    for path in sorted(SAMPLE.glob("*.parquet")):
        symbol = path.stem
        df = to_et(pl.read_parquet(path))
        for trade_date, day in df.group_by("trade_date", maintain_order=True):
            t = simulate_day(day, symbol)
            if t:
                trades.append(t)
    return trades


def summarize(trades: list[dict]) -> None:
    if not trades:
        print("No trades fired on the sample data.")
        return

    print(f"\n{'symbol':<7}{'date':<12}{'entry':<11}{'exit':<11}{'gain%':>7}{'vwap_x':>9}{'shares':>8}{'pnl':>11}  reason")
    print("-" * 95)
    cum = 0.0
    for t in trades:
        cum += t["pnl"]
        print(f"{t['symbol']:<7}{t['date']:<12}{t['entry_time']:<11}{t['exit_time']:<11}"
              f"{t['day_gain_pct']:>7.1f}{t['vwap_ext']:>9.3f}{t['shares']:>8}"
              f"{t['pnl']:>11,.2f}  {t['exit_reason']}")
    print("-" * 95)

    n = len(trades)
    pnls = [t["pnl"] for t in trades]
    wins = sum(1 for p in pnls if p > 0)
    total = sum(pnls)
    win_rate = wins / n * 100
    print(f"\nTotal trades: {n}   Wins: {wins}   Win rate: {win_rate:.1f}%   Total P&L: ${total:,.2f}")
    print(f"Average trade: ${total/n:,.2f}   Best: ${max(pnls):,.2f}   Worst: ${min(pnls):,.2f}")

    # Equity curve
    OUT_IMG.parent.mkdir(parents=True, exist_ok=True)
    cum_pnl = []
    running = 0.0
    for p in pnls:
        running += p
        cum_pnl.append(running)
    fig, ax = plt.subplots(figsize=(10, 4.5), dpi=130)
    ax.plot(range(1, n + 1), cum_pnl, color="#1f77b4", lw=1.6, marker="o", markersize=3)
    ax.fill_between(range(1, n + 1), 0, cum_pnl, alpha=0.12, color="#1f77b4")
    ax.axhline(0, color="#444", lw=0.5)
    ax.set_title(f"Quickstart example — Simplified V5 Relaxed on 5 sample symbols  |  "
                 f"{n} trades, {win_rate:.0f}% win rate, ${total:,.0f} total")
    ax.set_xlabel("Trade #")
    ax.set_ylabel("Cumulative P&L ($)")
    ax.grid(alpha=0.25)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    fig.tight_layout()
    fig.savefig(OUT_IMG, bbox_inches="tight")
    plt.close(fig)
    print(f"\nWrote equity curve to {OUT_IMG.relative_to(ROOT)}")


if __name__ == "__main__":
    print("Running simplified V5 Relaxed backtest on sample data...")
    print(f"Sample data: {SAMPLE.relative_to(ROOT)}")
    trades = run()
    summarize(trades)
