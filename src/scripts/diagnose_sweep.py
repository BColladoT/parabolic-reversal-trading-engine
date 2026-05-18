"""Top-level CLI that runs all three sweep diagnostics and concatenates them.

Usage:
    python -m src.scripts.diagnose_sweep models/sweep_2026-05-18/
    python -m src.scripts.diagnose_sweep models/sweep_2026-05-18/ --output docs/sweep_diagnosis_2026-05-18.md
"""
from __future__ import annotations

import argparse
from pathlib import Path

from src.scripts.diagnose_trades import analyze as analyze_trades
from src.scripts.diagnose_training import analyze as analyze_training
from src.scripts.diagnose_masks import analyze as analyze_masks


def diagnose(sweep_root: Path) -> str:
    parts = [
        "# Sweep Diagnosis Report",
        "",
        f"**Sweep root:** `{sweep_root}`",
        "",
        "Auto-generated from `diagnose_trades`, `diagnose_training`, and `diagnose_masks`.",
        "See `docs/sweep_diagnosis_2026-05-18.md` for the curated read of these findings.",
        "",
        "---",
        "",
        analyze_trades(sweep_root),
        "---",
        "",
        analyze_training(sweep_root),
        "---",
        "",
        analyze_masks(sweep_root),
    ]
    return "\n".join(parts)


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("sweep_root", type=Path)
    p.add_argument("--output", type=Path, default=None)
    args = p.parse_args()
    report = diagnose(args.sweep_root)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(report, encoding="utf-8")
        print(f"Report written to: {args.output}")
    else:
        print(report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
