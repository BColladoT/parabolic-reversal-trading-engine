"""Generate README header charts from V5 Relaxed backtest trade log.

Inputs:  reports/cached_parallel_backtest/combined_trades.csv
Outputs: docs/images/equity_curve.png
         docs/images/pnl_distribution.png
"""
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
TRADES = ROOT / "reports" / "cached_parallel_backtest" / "combined_trades.csv"
OUT = ROOT / "docs" / "images"
OUT.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(TRADES)
df = df[df["strategy"] == "v5_relaxed"].copy()
df["date"] = pd.to_datetime(df["date"])
df = df.sort_values("date").reset_index(drop=True)
df["cum_pnl"] = df["pnl"].cumsum()

n = len(df)
total = df["pnl"].sum()
win_rate = (df["pnl"] > 0).mean() * 100
gross_win = df.loc[df["pnl"] > 0, "pnl"].sum()
gross_loss = -df.loc[df["pnl"] < 0, "pnl"].sum()
pf = gross_win / gross_loss if gross_loss > 0 else float("inf")

# Equity curve
fig, ax = plt.subplots(figsize=(11, 5.5), dpi=140)
ax.plot(df["date"], df["cum_pnl"], color="#1f77b4", lw=1.6)
ax.fill_between(df["date"], 0, df["cum_pnl"], alpha=0.12, color="#1f77b4")
ax.axhline(0, color="#444", lw=0.5)
ax.set_title(
    f"V5 Relaxed — Cumulative P&L  |  {n} trades, {win_rate:.1f}% win rate, "
    f"${total:,.0f} total, PF {pf:.2f}",
    fontsize=12, pad=12,
)
ax.set_xlabel("Trade date")
ax.set_ylabel("Cumulative P&L ($)")
ax.grid(alpha=0.25)
for spine in ("top", "right"):
    ax.spines[spine].set_visible(False)
fig.tight_layout()
fig.savefig(OUT / "equity_curve.png", bbox_inches="tight")
plt.close(fig)

# P&L distribution
fig, ax = plt.subplots(figsize=(11, 5.5), dpi=140)
bins = 50
ax.hist(df["pnl"], bins=bins, color="#2ca02c", alpha=0.65, edgecolor="white", label="Wins")
ax.hist(df.loc[df["pnl"] < 0, "pnl"], bins=bins, color="#d62728", alpha=0.65,
        edgecolor="white", label="Losses")
ax.axvline(0, color="#444", lw=0.7)
ax.axvline(df["pnl"].mean(), color="#ff7f0e", lw=1.4, ls="--",
           label=f"Mean ${df['pnl'].mean():,.0f}")
ax.set_title(
    f"V5 Relaxed — Per-Trade P&L Distribution  |  median ${df['pnl'].median():,.0f}, "
    f"max win ${df['pnl'].max():,.0f}, max loss ${df['pnl'].min():,.0f}",
    fontsize=12, pad=12,
)
ax.set_xlabel("Trade P&L ($)")
ax.set_ylabel("Frequency")
ax.legend(loc="upper left", frameon=False)
ax.grid(alpha=0.25)
for spine in ("top", "right"):
    ax.spines[spine].set_visible(False)
fig.tight_layout()
fig.savefig(OUT / "pnl_distribution.png", bbox_inches="tight")
plt.close(fig)

print(f"trades: {n}  total: ${total:,.2f}  win_rate: {win_rate:.1f}%  PF: {pf:.2f}")
print(f"wrote {OUT / 'equity_curve.png'}")
print(f"wrote {OUT / 'pnl_distribution.png'}")
