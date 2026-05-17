"""Pure helpers that convert engine audit records into journal-schema dicts.

This module is the adapter layer between :class:`TickBacktestEngineV5`
``audit_records`` (a flat list of entry/exit decisions) and the
``TRADE_RECORD_SCHEMA`` dicts expected by :mod:`src.risk.trade_journal`.

Everything in here is pure: ``trades_from_engine_result`` and
``to_journal_record`` perform no I/O; ``atr_pct_at`` operates on a
caller-supplied Polars DataFrame (no file reads).  Dependencies are
restricted to Polars + the stdlib.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Any

import polars as pl


@dataclass(frozen=True)
class BacktestTrade:
    """One closed trade (entry -> exit) reconstructed from engine audit records."""

    entry_time: datetime
    exit_time: datetime
    entry_price: float        # weighted average across legs
    exit_price: float
    shares: int               # total shares closed by the exit
    pnl: float                # short-side: (entry_avg - exit_price) * shares
    exit_reason: str
    vwap_extension: float     # first leg's value (initial setup quality)
    volume_ratio: float       # first leg's value
    factors_count: float      # number of exhaustion criteria at entry (0 if engine doesn't expose)


def _safe_float(value: Any, default: float = 0.0) -> float:
    """Best-effort float coercion. Returns ``default`` for None / non-numeric."""
    if value is None:
        return default
    try:
        out = float(value)
    except (TypeError, ValueError):
        return default
    # Guard against NaN propagation; NaN != NaN.
    if out != out:  # noqa: PLR0124 — intentional NaN check
        return default
    return out


def _first_numeric(rec: Any, *attrs: str, default: float = 0.0) -> float:
    """Return the first attribute on ``rec`` that coerces to a real number.

    Used to be tolerant of slight naming drift across engine versions
    (``criteria_met`` vs ``confirming_factors`` vs ``factors_count``) and of
    MagicMock-shaped objects in unit tests, which auto-create attrs that
    cannot be cast to float.
    """
    for name in attrs:
        if not hasattr(rec, name):
            continue
        val = getattr(rec, name)
        if val is None:
            continue
        try:
            out = float(val)
        except (TypeError, ValueError):
            continue
        if out != out:  # NaN
            continue
        return out
    return default


def trades_from_engine_result(result: Any, setup_symbol: str) -> list[BacktestTrade]:
    """Pair entry audit records with their exit records into closed-trade tuples.

    Audit records are ordered. Entries accumulate into a buffer; an exit
    closes the buffered legs and emits one :class:`BacktestTrade`.  This
    tolerates multi-leg scale-ins (entry -> add -> exit).

    Parameters
    ----------
    result:
        Anything with an ``audit_records`` iterable attribute.  Each record
        must expose ``timestamp``, ``price``, ``shares`` and (for exits)
        ``exit_reason``.  Optional: ``vwap_extension``, ``volume_ratio``,
        ``confirming_factors``/``criteria_met``/``factors_count``.
    setup_symbol:
        Symbol associated with the setup (forwarded by the caller, retained
        in the signature for parity with the contract; not currently stored
        in the returned tuple because the journal record carries it).
    """
    del setup_symbol  # reserved for future per-symbol pairing tweaks

    audit = getattr(result, "audit_records", None) or []
    trades: list[BacktestTrade] = []
    open_legs: list[Any] = []

    for rec in audit:
        if getattr(rec, "exit_reason", None):  # exit
            if not open_legs:
                continue  # exit without preceding entry — skip defensively
            total_shares = sum(int(leg.shares) for leg in open_legs)
            if total_shares <= 0:
                open_legs.clear()
                continue
            weighted_entry = (
                sum(float(leg.price) * int(leg.shares) for leg in open_legs)
                / total_shares
            )
            first = open_legs[0]
            trades.append(
                BacktestTrade(
                    entry_time=first.timestamp,
                    exit_time=rec.timestamp,
                    entry_price=weighted_entry,
                    exit_price=float(rec.price),
                    shares=total_shares,
                    # Short side: profit when exit_price < weighted_entry.
                    pnl=(weighted_entry - float(rec.price)) * total_shares,
                    exit_reason=str(rec.exit_reason),
                    vwap_extension=_safe_float(getattr(first, "vwap_extension", 0.0)),
                    volume_ratio=_safe_float(getattr(first, "volume_ratio", 0.0)),
                    factors_count=_first_numeric(
                        first,
                        "confirming_factors",
                        "criteria_met",
                        "factors_count",
                        default=0.0,
                    ),
                )
            )
            open_legs.clear()
        else:  # entry leg
            shares = _safe_float(getattr(rec, "shares", 0))
            if int(shares) > 0:
                open_legs.append(rec)

    return trades


def atr_pct_at(bars: pl.DataFrame, ts: datetime, window: int = 14) -> float:
    """ATR-as-percent over ``window`` 1-min bars ending at ``ts``.

    Uses the simple ``mean(high - low)`` true-range proxy (no prev-close
    gap component) — sufficient for journal feature snapshots and avoids
    pulling in numba. Returns ``0.0`` when there are fewer than ``window``
    bars available or when the anchor close is non-positive; downstream
    consumers tolerate the 0.0 sentinel.
    """
    if bars.is_empty():
        return 0.0
    window_bars = bars.filter(pl.col("timestamp") <= ts).tail(window)
    if window_bars.shape[0] < window:
        return 0.0
    tr = (window_bars["high"] - window_bars["low"]).mean()
    close_at = float(window_bars["close"].item(-1))
    if close_at <= 0:
        return 0.0
    return float(tr) / close_at
