"""Replay historical parabolic setups through TickBacktestEngineV5 and
populate the trade journal so Kelly sizing has real priors on day 1.

Idempotent: re-runs skip setups already present in the journal (by
symbol+entry_date). Designed to be interruptible and resumable.
"""
from __future__ import annotations

import argparse
import pickle
import sys
from datetime import date, datetime
from pathlib import Path
from typing import Optional

import polars as pl

# Force UTF-8 on stdout/stderr so warning messages with non-ASCII chars
# (e.g. Polars error strings containing 'μs') don't crash on Windows cp1252.
for _stream in (sys.stdout, sys.stderr):
    if hasattr(_stream, "reconfigure"):
        try:
            _stream.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass

from src.backtest.tick_backtest_engine_v5 import TickBacktestEngineV5
from src.risk.trade_journal import append_trade, read_trades
from src.scripts.backfill_helpers import (
    BacktestTrade,  # noqa: F401  (re-exported for callers that want the type)
    to_journal_record,
    trades_from_engine_result,
)


def _bars_path_for(bars_dir: Path, symbol: str) -> Optional[Path]:
    """Locate the first parquet matching ``{SYMBOL}_1min_*.parquet`` in bars_dir."""
    if not bars_dir.exists():
        return None
    matches = sorted(bars_dir.glob(f"{symbol}_1min_*.parquet"))
    return matches[0] if matches else None


def _already_journaled(symbol: str, setup_date: date) -> bool:
    """Check whether the journal already has any trade for (symbol, setup_date)."""
    df = read_trades(date_from=setup_date, date_to=setup_date)
    if df.is_empty():
        return False
    return symbol in df["symbol"].to_list()


def _initial_risk_per_share(audit_records) -> float:
    """Best-effort risk-per-share estimate from the first entry leg.

    Prefers an explicit ``stop_loss`` attribute on the audit record. Falls back
    to 1% of entry price (a conservative default that keeps r_multiple bounded).
    """
    for rec in audit_records:
        if not getattr(rec, "exit_reason", None) and int(getattr(rec, "shares", 0)) > 0:
            price = float(rec.price)
            stop = getattr(rec, "stop_loss", None)
            if stop:
                return max(abs(float(stop) - price), 0.01)
            return max(price * 0.01, 0.01)  # 1% default
    return 0.01


def _normalize_setup_date(d) -> date:
    """ParabolicSetup.date may be a string, datetime, or date — normalize to date."""
    if isinstance(d, str):
        return date.fromisoformat(d)
    if isinstance(d, datetime):
        return d.date()
    return d


def backfill(
    pickle_path: Path = Path("data/cache/setups/setups_relaxed_full_2019_2024.pkl"),
    bars_dir: Path = Path("data/cache/1min_extended"),
    max_setups: Optional[int] = None,
    skip_existing: bool = True,
) -> dict:
    """Replay every setup in the pickle through the engine, journaling trades.

    Returns a stats dict with keys ``setups_processed``, ``setups_skipped``,
    and ``trades_written``.
    """
    pickle_path = Path(pickle_path)
    bars_dir = Path(bars_dir)

    with pickle_path.open("rb") as f:
        setups = pickle.load(f)
    if max_setups is not None:
        setups = setups[:max_setups]

    engine = TickBacktestEngineV5()
    stats = {"setups_processed": 0, "setups_skipped": 0, "trades_written": 0}

    for setup in setups:
        symbol = setup.symbol
        setup_date = _normalize_setup_date(setup.date)

        if skip_existing and _already_journaled(symbol, setup_date):
            stats["setups_skipped"] += 1
            continue

        bars_path = _bars_path_for(bars_dir, symbol)
        if bars_path is None:
            stats["setups_skipped"] += 1
            continue

        bars = pl.read_parquet(bars_path).filter(
            pl.col("timestamp").dt.date() == setup_date
        )
        if bars.is_empty():
            stats["setups_skipped"] += 1
            continue

        # Engine signature: run_tick_backtest(symbol: str, date: datetime, verbose)
        # It fetches its own tick data internally via tick_fetcher; the parquet
        # bars we loaded above are only used for our ATR-at-entry computation.
        engine_date = datetime.combine(setup_date, datetime.min.time())
        result = engine.run_tick_backtest(symbol, engine_date, verbose=False)
        audit = getattr(result, "audit_records", []) or []
        risk_per_share = _initial_risk_per_share(audit)
        trades = trades_from_engine_result(result, symbol)

        for trade in trades:
            try:
                append_trade(
                    to_journal_record(trade, symbol, bars, risk_per_share)
                )
                stats["trades_written"] += 1
            except Exception as e:  # pragma: no cover - defensive log path
                # Best-effort: log via print (script context) and continue.
                print(
                    f"[warn] append_trade failed for {symbol}@{trade.entry_time}: {e}"
                )

        stats["setups_processed"] += 1

    return stats


def main() -> None:
    p = argparse.ArgumentParser(
        description=(
            "Replay historical parabolic setups through TickBacktestEngineV5 "
            "and populate the trade journal."
        )
    )
    p.add_argument(
        "--pickle",
        type=Path,
        default=Path("data/cache/setups/setups_relaxed_full_2019_2024.pkl"),
        help="Path to the pickled list of ParabolicSetup objects.",
    )
    p.add_argument(
        "--bars-dir",
        type=Path,
        default=Path("data/cache/1min_extended"),
        help="Directory containing per-symbol 1-min parquet bar files.",
    )
    p.add_argument(
        "--max-setups",
        type=int,
        default=None,
        help="Cap on number of setups to process (useful for smoke runs).",
    )
    p.add_argument(
        "--no-skip-existing",
        dest="skip_existing",
        action="store_false",
        help="Re-run setups that already have journal rows (default: skip them).",
    )
    p.set_defaults(skip_existing=True)
    args = p.parse_args()

    stats = backfill(
        pickle_path=args.pickle,
        bars_dir=args.bars_dir,
        max_setups=args.max_setups,
        skip_existing=args.skip_existing,
    )
    print(f"Backfill complete: {stats}")


if __name__ == "__main__":
    main()
