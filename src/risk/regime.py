"""Daily market-regime side-table for conditional edge estimation.

Data source: yfinance (^VIX close, SPY close). Storage: a single parquet at
``data/regime/regime_history.parquet`` (path override via ``REGIME_DIR`` env).

The trade journal schema is intentionally NOT touched: regime is joined at
query time on the trade's entry-date, so historical journal records remain
valid as the regime table is rewritten.

Labels:
    vix < 20  AND spy_trend >= 0  ->  "risk_on"
    vix >= 25 OR  spy_trend < 0   ->  "risk_off"
    otherwise                     ->  "neutral"

spy_trend rules (vs a 50-day SMA, min_periods=1 so early rows still classify):
    +1 if SPY close > SMA50 by more than 0.5%
    -1 if SPY close < SMA50 by more than 0.5%
     0 otherwise (within +-0.5% band)
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from datetime import date as date_cls
from pathlib import Path
from typing import Optional

import polars as pl


REGIME_DIR_ENV = "REGIME_DIR"

REGIME_SCHEMA: dict[str, pl.DataType] = {
    "date": pl.Date,
    "vix_level": pl.Float64,
    "spy_trend": pl.Int64,
    "label": pl.Utf8,
}


@dataclass(frozen=True)
class Regime:
    date: date_cls
    vix_level: float
    spy_trend: int
    label: str


def _regime_dir() -> Path:
    return Path(os.environ.get(REGIME_DIR_ENV, "data/regime"))


def _regime_path() -> Path:
    return _regime_dir() / "regime_history.parquet"


def _download_yf(ticker: str, start: date_cls, end: date_cls):
    """Thin wrapper around yfinance to make mocking easy in tests.

    Returns a pandas DataFrame (yfinance's native shape). Tests patch this
    function directly so the real package is never called.
    """
    import yfinance as yf
    return yf.download(
        ticker,
        start=str(start),
        end=str(end),
        progress=False,
        auto_adjust=False,
    )


def _classify(vix: float, spy_trend: int) -> str:
    if vix < 20 and spy_trend >= 0:
        return "risk_on"
    if vix >= 25 or spy_trend < 0:
        return "risk_off"
    return "neutral"


def _to_close_series(pd_df):
    """Extract a 1-D pandas Series of closes from yfinance's DataFrame.

    yfinance sometimes returns a MultiIndex column frame ((Close, ^VIX) etc.).
    ``squeeze()`` collapses the trivial second dim for a single ticker.
    """
    if pd_df is None:
        return None
    if pd_df.empty:
        return None
    close = pd_df["Close"]
    # If MultiIndex columns gave us a DataFrame back, squeeze to Series.
    if hasattr(close, "squeeze"):
        close = close.squeeze()
    return close


def fetch_regime_history(start: date_cls, end: date_cls) -> pl.DataFrame:
    """Pull VIX (^VIX) + SPY closes from yfinance and classify each trading day."""
    vix_pd = _download_yf("^VIX", start, end)
    spy_pd = _download_yf("SPY", start, end)
    vix_close = _to_close_series(vix_pd)
    spy_close = _to_close_series(spy_pd)
    if vix_close is None or spy_close is None:
        return pl.DataFrame(schema=REGIME_SCHEMA)
    if len(vix_close) == 0 or len(spy_close) == 0:
        return pl.DataFrame(schema=REGIME_SCHEMA)

    spy_sma50 = spy_close.rolling(window=50, min_periods=1).mean()

    # Intersection on date so we don't synthesize rows for half-holidays.
    vix_dates = {idx.date() if hasattr(idx, "date") else idx for idx in vix_close.index}

    rows: list[dict] = []
    for idx in spy_close.index:
        d = idx.date() if hasattr(idx, "date") else idx
        if d not in vix_dates:
            continue
        try:
            vix_v = float(vix_close.loc[idx])
            spy_v = float(spy_close.loc[idx])
            sma_v = float(spy_sma50.loc[idx])
        except Exception:
            continue
        deviation = (spy_v - sma_v) / sma_v if sma_v > 0 else 0.0
        if deviation > 0.005:
            trend = 1
        elif deviation < -0.005:
            trend = -1
        else:
            trend = 0
        rows.append({
            "date": d,
            "vix_level": vix_v,
            "spy_trend": trend,
            "label": _classify(vix_v, trend),
        })
    if not rows:
        return pl.DataFrame(schema=REGIME_SCHEMA)
    return pl.DataFrame(rows, schema=REGIME_SCHEMA)


def write_regime_history(df: pl.DataFrame) -> Path:
    """Write/overwrite ``data/regime/regime_history.parquet``."""
    p = _regime_path()
    p.parent.mkdir(parents=True, exist_ok=True)
    df.write_parquet(p)
    return p


def read_regime_history() -> pl.DataFrame:
    """Read full regime history. Returns empty DF with REGIME_SCHEMA if absent."""
    p = _regime_path()
    if not p.exists():
        return pl.DataFrame(schema=REGIME_SCHEMA)
    return pl.read_parquet(p)


def regime_for_date(d: date_cls) -> Optional[Regime]:
    """Lookup a single date. Returns None if not present."""
    df = read_regime_history()
    if df.is_empty():
        return None
    row = df.filter(pl.col("date") == d)
    if row.is_empty():
        return None
    r = row.row(0, named=True)
    return Regime(
        date=r["date"],
        vix_level=float(r["vix_level"]),
        spy_trend=int(r["spy_trend"]),
        label=r["label"],
    )
