"""CLI: print rolling win rate + edge stats by feature slice.

Usage:
    python -m src.scripts.inspect_journal
    python -m src.scripts.inspect_journal --feature vwap_extension
    python -m src.scripts.inspect_journal --feature volume_ratio
    python -m src.scripts.inspect_journal --regime risk_on
"""
from __future__ import annotations

import argparse
import math
from typing import Optional

import polars as pl

from src.risk.edge_estimator import compute_edge, consecutive_losses
from src.risk.trade_journal import read_trades


def _print_overall(regime_label: Optional[str] = None) -> None:
    # Pass lookback_days=None so historical backfill data (often >1yr old)
    # is included in the displayed stats. Live sizing uses a tighter default.
    e = compute_edge(lookback_days=None, regime_label=regime_label)
    suffix = f" ({regime_label})" if regime_label else ""
    print(f"=== Overall edge{suffix} ===")
    print(f"n_trades         : {e.n_trades}")
    if e.n_trades == 0:
        print("(journal is empty - run backfill or trade some live setups first)")
        print(f"consecutive_loss : {consecutive_losses()}")
        return
    if math.isnan(e.win_rate_ci_low):
        print(f"win_rate         : {e.win_rate:.2%}  (95% CI: n/a)")
    else:
        print(f"win_rate         : {e.win_rate:.2%}  (95% CI: {e.win_rate_ci_low:.2%} - {e.win_rate_ci_high:.2%})")
    print(f"avg_win_r        : {e.avg_win_r:+.2f}")
    print(f"avg_loss_r       : {e.avg_loss_r:+.2f}")
    if math.isnan(e.expected_r_ci_low):
        print(f"expected_r       : {e.expected_r:+.3f}  (95% CI: n/a)")
    else:
        print(f"expected_r       : {e.expected_r:+.3f}  (95% CI: {e.expected_r_ci_low:+.2f} - {e.expected_r_ci_high:+.2f})")
    print(f"kelly_fraction   : {e.kelly_fraction:.3f} (quarter-Kelly cap = 0.25)")
    print(f"consecutive_loss : {consecutive_losses()}")


def _print_by_slice(feature: str) -> None:
    df = read_trades()
    if df.is_empty():
        print(f"No trades in journal yet — cannot slice by {feature}.")
        return
    col = f"feat_{feature}"
    if col not in df.columns:
        print(f"Unknown feature: {feature}")
        print(f"Available: {[c.removeprefix('feat_') for c in df.columns if c.startswith('feat_')]}")
        return
    # Bucket into 4 quantiles for the report
    buckets = df.select(
        [
            pl.col(col),
            pl.col(col).qcut(4, allow_duplicates=True).alias("bucket"),
            pl.col("win"),
            pl.col("r_multiple"),
        ]
    )
    summary = (
        buckets.group_by("bucket")
        .agg(
            [
                pl.len().alias("n"),
                pl.col("win").mean().alias("win_rate"),
                pl.col("r_multiple").mean().alias("avg_r"),
            ]
        )
        .sort("bucket")
    )
    print(f"=== Edge by {feature} ===")
    # Render as plain ASCII so the CLI works on cp1252 Windows consoles.
    # Polars' default __repr__ uses box-drawing chars which crash on those.
    rows = summary.to_dicts()
    if not rows:
        print("(no buckets)")
        return
    header = f"{'bucket':<24} {'n':>8} {'win_rate':>10} {'avg_r':>10}"
    print(header)
    print("-" * len(header))
    for row in rows:
        bucket = str(row.get("bucket", ""))
        n = row.get("n", 0)
        wr = row.get("win_rate")
        ar = row.get("avg_r")
        wr_s = f"{wr:>10.2%}" if wr is not None else f"{'n/a':>10}"
        ar_s = f"{ar:>+10.2f}" if ar is not None else f"{'n/a':>10}"
        print(f"{bucket:<24} {n:>8} {wr_s} {ar_s}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Inspect rolling edge stats from the trade journal."
    )
    parser.add_argument(
        "--feature",
        default=None,
        help="Feature to slice by (e.g. vwap_extension, volume_ratio, atr_pct)",
    )
    parser.add_argument(
        "--regime",
        default=None,
        help="Filter to a specific regime label (risk_on / risk_off / neutral)",
    )
    args = parser.parse_args()
    _print_overall(regime_label=args.regime)
    if args.feature:
        print()
        _print_by_slice(args.feature)


if __name__ == "__main__":
    main()
