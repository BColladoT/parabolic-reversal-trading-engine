"""Conditional edge estimator. Reads the trade journal, returns Kelly-capped stats.

Public API (consumed by RiskManager.calculate_position_size in Agent A3):
    compute_edge(features=None, lookback_days=365, min_samples_conditional=20)
        -> EdgeStats
    consecutive_losses(lookback_trades=10) -> int

EdgeStats fields:
    n_trades        — sample size used (overall or conditional slice)
    win_rate        — fraction of winning trades; 0.0 on empty journal
    avg_win_r       — mean R-multiple on winners (>= 0)
    avg_loss_r      — mean R-multiple on losers (<= 0)
    expected_r      — win_rate*avg_win_r + (1-win_rate)*avg_loss_r
    kelly_fraction  — fractional Kelly bet size, clamped to [0, KELLY_CAP=0.25]
    used_fallback   — True when the conditional slice was too small and we
                      returned overall stats instead.

Conditional slicing buckets each feature into pre-defined ranges (see _bucket).
When a feature is supplied via the `features` kwarg and the matching slice has
at least min_samples_conditional trades, conditional stats are returned;
otherwise the overall stats are returned with used_fallback=True.
"""
from dataclasses import dataclass
from datetime import date, timedelta
from typing import Optional

import polars as pl

from src.risk.trade_journal import read_trades

KELLY_CAP = 0.25  # Quarter Kelly per CLAUDE.md risk policy


@dataclass(frozen=True)
class EdgeStats:
    n_trades: int
    win_rate: float
    avg_win_r: float
    avg_loss_r: float
    expected_r: float
    kelly_fraction: float
    used_fallback: bool


_EMPTY = EdgeStats(0, 0.0, 0.0, 0.0, 0.0, 0.0, True)


def _kelly(win_rate: float, avg_win_r: float, avg_loss_r: float) -> float:
    """Fractional Kelly. avg_loss_r is negative; we use its magnitude."""
    if avg_win_r <= 0 or avg_loss_r >= 0:
        return 0.0
    b = avg_win_r / abs(avg_loss_r)  # win-to-loss payoff ratio
    if b <= 0:
        return 0.0
    f = win_rate - (1.0 - win_rate) / b
    return max(0.0, min(KELLY_CAP, f))


def _bucket(value: float, key: str) -> tuple[float, float]:
    """Return (low, high) bucket bounds for a feature value."""
    buckets = {
        "feat_vwap_extension": [0.0, 0.15, 0.25, 0.40, 10.0],
        "feat_volume_ratio":   [0.0, 2.0, 5.0, 10.0, 1e9],
        "feat_atr_pct":        [0.0, 0.01, 0.02, 0.05, 1.0],
        "feat_time_of_day_min":[0.0, 30.0, 90.0, 240.0, 1e9],
        "feat_day_of_week":    [-0.5, 0.5, 1.5, 2.5, 3.5, 4.5],
        "feat_factors_count":  [-0.5, 1.5, 2.5, 3.5, 1e9],
    }
    edges = buckets.get(key, [-1e18, 1e18])
    for i in range(len(edges) - 1):
        if edges[i] <= value < edges[i + 1]:
            return (edges[i], edges[i + 1])
    return (edges[-2], edges[-1])


def _stats_from(df: pl.DataFrame) -> tuple[int, float, float, float]:
    if df.is_empty():
        return 0, 0.0, 0.0, 0.0
    n = df.shape[0]
    wins = df.filter(pl.col("win"))
    losses = df.filter(~pl.col("win"))
    win_rate = wins.shape[0] / n
    avg_win = wins["r_multiple"].mean() if wins.shape[0] else 0.0
    avg_loss = losses["r_multiple"].mean() if losses.shape[0] else 0.0
    return n, float(win_rate), float(avg_win or 0.0), float(avg_loss or 0.0)


def compute_edge(
    features: Optional[dict[str, float]] = None,
    lookback_days: Optional[int] = 365,
    min_samples_conditional: int = 20,
) -> EdgeStats:
    today = date.today()
    # lookback_days=None means "all available data" — used by inspect tools and
    # by callers that want backfilled historical priors regardless of age.
    if lookback_days is None:
        df = read_trades()
    else:
        df = read_trades(date_from=today - timedelta(days=lookback_days))
    if df.is_empty():
        return _EMPTY

    overall_n, w_overall, win_r_overall, loss_r_overall = _stats_from(df)
    overall_expected = w_overall * win_r_overall + (1 - w_overall) * loss_r_overall
    overall_kelly = _kelly(w_overall, win_r_overall, loss_r_overall)
    overall = EdgeStats(
        overall_n, w_overall, win_r_overall, loss_r_overall,
        overall_expected, overall_kelly, used_fallback=True,
    )

    if not features:
        return overall

    sliced = df
    for key, val in features.items():
        if key not in df.columns:
            continue
        lo, hi = _bucket(val, key)
        sliced = sliced.filter((pl.col(key) >= lo) & (pl.col(key) < hi))

    if sliced.shape[0] < min_samples_conditional:
        return overall

    n, w, win_r, loss_r = _stats_from(sliced)
    expected = w * win_r + (1 - w) * loss_r
    kelly = _kelly(w, win_r, loss_r)
    return EdgeStats(n, w, win_r, loss_r, expected, kelly, used_fallback=False)


def consecutive_losses(lookback_trades: int = 10) -> int:
    df = read_trades()
    if df.is_empty():
        return 0
    tail = df.sort("exit_time").tail(lookback_trades)
    losses = 0
    for win in reversed(tail["win"].to_list()):
        if win:
            break
        losses += 1
    return losses
