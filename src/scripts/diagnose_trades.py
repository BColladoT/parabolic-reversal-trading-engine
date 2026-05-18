"""Trade-level diagnostic on a completed reward-weight sweep.

Pools trades.jsonl across all runs, groups by weight, computes per-weight
win/loss/MFE/MAE/symbol patterns. Pure stdlib + Polars + numpy. Read-only.

Usage:
    python -m src.scripts.diagnose_trades models/sweep_2026-05-18/
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Dict, List

import numpy as np


def _load_trades(sweep_root: Path) -> Dict[float, List[dict]]:
    """Group trades by weight (parsed from parent dir name wW.WW_sN)."""
    by_weight: Dict[float, List[dict]] = defaultdict(list)
    for trades_path in Path(sweep_root).glob("w*_s*/trades.jsonl"):
        try:
            weight = float(trades_path.parent.name.split("_")[0][1:])
        except (ValueError, IndexError):
            continue
        with trades_path.open(encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    by_weight[weight].append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    return dict(by_weight)


def _summarize_one_weight(trades: List[dict]) -> dict:
    """Compute the per-weight summary dict."""
    if not trades:
        return {"n": 0}
    winners = [t for t in trades if t.get("win")]
    losers = [t for t in trades if not t.get("win")]
    pnls = np.array([float(t.get("pnl", 0.0)) for t in trades])
    win_pnls = np.array([float(t.get("pnl", 0.0)) for t in winners])
    loss_pnls = np.array([float(t.get("pnl", 0.0)) for t in losers])
    bars_held = np.array([float(t.get("bars_held", 0)) for t in trades])
    win_mfe = np.array([float(t.get("mfe", 0.0)) for t in winners])
    win_mae = np.array([float(t.get("mae", 0.0)) for t in winners])
    loss_mfe = np.array([float(t.get("mfe", 0.0)) for t in losers])
    loss_mae = np.array([float(t.get("mae", 0.0)) for t in losers])

    # Per-symbol scorecard
    per_symbol_pnl: Dict[str, List[float]] = defaultdict(list)
    for t in trades:
        per_symbol_pnl[t.get("symbol", "?")].append(float(t.get("pnl", 0.0)))
    qualifying = {sym: vals for sym, vals in per_symbol_pnl.items() if len(vals) >= 5}
    by_mean = sorted(qualifying.items(), key=lambda kv: np.mean(kv[1]))
    worst = by_mean[:5]
    best = by_mean[-5:][::-1]

    def _safe_mean(arr: np.ndarray) -> float:
        return float(arr.mean()) if arr.size else 0.0

    win_mean_abs = abs(_safe_mean(win_pnls))
    loss_mean_abs = abs(_safe_mean(loss_pnls)) or 1e-9
    return {
        "n": len(trades),
        "total_pnl": float(pnls.sum()),
        "win_rate": float(len(winners) / len(trades)) if trades else 0.0,
        "mean_pnl": _safe_mean(pnls),
        "mean_winner_pnl": _safe_mean(win_pnls),
        "mean_loser_pnl": _safe_mean(loss_pnls),
        "expectancy_ratio": win_mean_abs / loss_mean_abs,
        "mean_bars_held_all": _safe_mean(bars_held),
        "median_bars_held": float(np.median(bars_held)) if bars_held.size else 0.0,
        "mean_winner_mfe": _safe_mean(win_mfe),
        "mean_winner_mae": _safe_mean(win_mae),
        "mean_loser_mfe": _safe_mean(loss_mfe),
        "mean_loser_mae": _safe_mean(loss_mae),
        "best_symbols": [(s, float(np.mean(v)), len(v)) for s, v in best],
        "worst_symbols": [(s, float(np.mean(v)), len(v)) for s, v in worst],
    }


def analyze(sweep_root: Path) -> str:
    by_weight = _load_trades(Path(sweep_root))
    if not by_weight:
        return "## Trade-level diagnosis\n\n_(No trades.jsonl files found.)_\n"

    lines: List[str] = []
    lines.append("## Trade-level diagnosis\n")
    lines.append(f"**Sweep:** `{sweep_root}`")
    total_trades = sum(len(v) for v in by_weight.values())
    lines.append(f"**Total trades pooled:** {total_trades:,} across {len(by_weight)} weights")
    lines.append("")

    lines.append("### Per-weight summary\n")
    lines.append("| weight | n_trades | win_rate | mean_pnl | mean_winner | mean_loser | expectancy_ratio | median_bars |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|")
    summaries = {w: _summarize_one_weight(t) for w, t in sorted(by_weight.items())}
    for w, s in summaries.items():
        lines.append(
            f"| {w:.2f} | {s['n']:,} | {s['win_rate']:.1%} | "
            f"${s['mean_pnl']:+.2f} | ${s['mean_winner_pnl']:+.2f} | "
            f"${s['mean_loser_pnl']:+.2f} | {s['expectancy_ratio']:.2f} | "
            f"{s['median_bars_held']:.0f} |"
        )
    lines.append("")

    lines.append("### MFE / MAE patterns (winners vs losers, per weight)\n")
    lines.append("| weight | winner_mean_mfe | winner_mean_mae | loser_mean_mfe | loser_mean_mae |")
    lines.append("|---|---:|---:|---:|---:|")
    for w, s in summaries.items():
        lines.append(
            f"| {w:.2f} | ${s['mean_winner_mfe']:+.2f} | "
            f"${s['mean_winner_mae']:+.2f} | ${s['mean_loser_mfe']:+.2f} | "
            f"${s['mean_loser_mae']:+.2f} |"
        )
    lines.append("")

    lines.append("### Best / worst symbols (>= 5 trades, pooled across all weights)\n")
    all_trades = [t for trades in by_weight.values() for t in trades]
    overall = _summarize_one_weight(all_trades)
    lines.append("Top 5 symbols (best mean pnl):")
    for sym, mean_pnl, n in overall["best_symbols"]:
        lines.append(f"  - {sym}: mean ${mean_pnl:+.2f} over {n} trades")
    lines.append("\nBottom 5 symbols (worst mean pnl):")
    for sym, mean_pnl, n in overall["worst_symbols"]:
        lines.append(f"  - {sym}: mean ${mean_pnl:+.2f} over {n} trades")
    lines.append("")

    lines.append("### Interpretation\n")
    avg_expectancy = float(np.mean([s["expectancy_ratio"] for s in summaries.values()]))
    avg_winner_mfe = float(np.mean([s["mean_winner_mfe"] for s in summaries.values()]))
    avg_winner_pnl = float(np.mean([s["mean_winner_pnl"] for s in summaries.values()]))
    mfe_capture = avg_winner_pnl / avg_winner_mfe if avg_winner_mfe > 0 else float("nan")
    interp: List[str] = []
    if avg_expectancy < 1.0:
        interp.append(
            f"- **Negative expectancy structure**: mean_winner / |mean_loser| = {avg_expectancy:.2f} "
            f"(< 1.0). Winners are smaller than losers on average; even at 50% win rate this loses money."
        )
    else:
        interp.append(
            f"- **Positive expectancy structure**: mean_winner / |mean_loser| = {avg_expectancy:.2f}. "
            f"Win rate is the bottleneck, not trade asymmetry."
        )
    if mfe_capture < 0.5 and not np.isnan(mfe_capture):
        interp.append(
            f"- **Premature exits on winners**: agent captures ${avg_winner_pnl:.2f} out of an "
            f"available ${avg_winner_mfe:.2f} MFE ({mfe_capture:.0%}). Exit policy leaves money on the table."
        )
    elif mfe_capture > 0.8:
        interp.append(
            f"- **Strong winner capture**: agent gets {mfe_capture:.0%} of available MFE on winners. "
            f"Exit timing on profitable trades is good."
        )
    lines.extend(interp)
    lines.append("")
    return "\n".join(lines)


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("sweep_root", type=Path)
    args = p.parse_args()
    print(analyze(args.sweep_root))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
