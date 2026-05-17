"""Compare 2+ WFO training runs side-by-side.

Reads each run's wfo_results.json and prints a one-row-per-metric, one-
column-per-run table. Pure stdlib + Polars (no torch). Run on any machine.

Usage:
    python -m src.scripts.compare_training_runs \
        models/wfo_baseline/wfo_results.json \
        models/wfo_shaped/wfo_results.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def _load(p: Path) -> dict | None:
    if not p.exists():
        print(f"[warn] missing: {p}")
        return None
    return json.loads(p.read_text())


def _aggregate_total_pnl(payload: dict) -> float:
    folds = payload.get("per_fold_results", [])
    return sum(float(f.get("test_metrics", {}).get("total_test_pnl", 0.0)) for f in folds)


def _aggregate_mean_winrate(payload: dict) -> float:
    folds = payload.get("per_fold_results", [])
    rates = [float(f.get("test_metrics", {}).get("win_rate", 0.0)) for f in folds if f.get("test_metrics")]
    return sum(rates) / len(rates) if rates else float("nan")


def _aggregate_total_trades(payload: dict) -> int:
    folds = payload.get("per_fold_results", [])
    return sum(int(f.get("test_metrics", {}).get("total_trades", 0)) for f in folds)


def compare(paths: list[Path]) -> int:
    runs: list[tuple[str, dict | None]] = []
    for p in paths:
        runs.append((Path(p).parent.name or Path(p).stem, _load(Path(p))))
    valid = [(name, p) for name, p in runs if p is not None]
    if not valid:
        return 1

    col_w = 22
    header = f"{'metric':<22}" + "".join(f"{name[:col_w-1]:>{col_w}}" for name, _ in valid)
    print(header)
    print("-" * len(header))

    rows = [
        ("total_test_pnl", lambda p: f"{_aggregate_total_pnl(p):>{col_w}.2f}"),
        ("mean_win_rate",  lambda p: f"{_aggregate_mean_winrate(p):>{col_w}.4f}"),
        ("total_trades",   lambda p: f"{_aggregate_total_trades(p):>{col_w}d}"),
        ("n_folds",        lambda p: f"{len(p.get('per_fold_results', [])):>{col_w}d}"),
    ]
    for label, fn in rows:
        line = f"{label:<22}" + "".join(fn(p) for _, p in valid)
        print(line)
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("paths", nargs="+", type=Path,
                    help="2+ paths to wfo_results.json files to compare.")
    args = ap.parse_args()
    return compare(args.paths)


if __name__ == "__main__":
    raise SystemExit(main())
