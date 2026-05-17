"""Diagnostic: what would live sizing do today, given the current journal?

Synthesizes 5 representative entry signals varying VWAP extension and
volume ratio, runs each through RiskManager.calculate_position_size with
dry_run=True against a stub 10K-equity account, and prints the resulting
(kelly, dd_mod, shares, $risk) table.

Usage:
    python -m src.scripts.observe_live_sizing
"""
from __future__ import annotations

import os
import tempfile
from unittest.mock import MagicMock


def main() -> int:
    # Route daily-state persistence to a throwaway temp file so this diagnostic
    # does NOT read/write the live trading engine's state file. The trade
    # journal (TRADE_JOURNAL_DIR) is intentionally left alone so we observe
    # sizing against the *current* journal.
    tmp_state = tempfile.NamedTemporaryFile(
        prefix="observe_live_sizing_", suffix=".json", delete=False
    )
    tmp_state.close()
    os.environ["DAILY_STATE_PATH"] = tmp_state.name

    # Import after env override so RiskManager picks up the temp path.
    from src.risk.position_manager import RiskManager

    stub_account = {"equity": 10000.0, "cash": 10000.0, "buying_power": 50000.0}
    stub_client = MagicMock()
    stub_client.get_account = MagicMock(return_value=stub_account)
    stub_client.close_all_positions = MagicMock(return_value={"success": True})

    rm = RiskManager(alpaca_client=stub_client)
    # Force a known equity baseline regardless of any restored state.
    rm.account_equity = 10000.0
    rm.daily_pnl = 0.0
    rm.daily_stats["daily_loss_limit_hit"] = False

    scenarios = [
        {"vwap_extension": 0.5, "volume_ratio": 2.0, "atr_pct": 0.02,
         "time_of_day_min": 30.0, "day_of_week": 2.0, "factors_count": 2.0},
        {"vwap_extension": 1.0, "volume_ratio": 3.5, "atr_pct": 0.03,
         "time_of_day_min": 60.0, "day_of_week": 2.0, "factors_count": 3.0},
        {"vwap_extension": 1.5, "volume_ratio": 5.0, "atr_pct": 0.04,
         "time_of_day_min": 90.0, "day_of_week": 2.0, "factors_count": 4.0},
        {"vwap_extension": 2.0, "volume_ratio": 7.0, "atr_pct": 0.05,
         "time_of_day_min": 120.0, "day_of_week": 2.0, "factors_count": 4.0},
        {"vwap_extension": 2.5, "volume_ratio": 10.0, "atr_pct": 0.06,
         "time_of_day_min": 150.0, "day_of_week": 2.0, "factors_count": 5.0},
    ]
    header = (
        f"{'vwap_ext':>9} {'vol_ratio':>10} {'n_cond':>7} "
        f"{'kelly':>7} {'dd_mod':>7} {'shares':>7} {'$risk':>9}"
    )
    print(header)
    print("-" * len(header))
    for s in scenarios:
        result = rm.calculate_position_size(
            symbol="PROBE",
            entry_price=10.0,
            atr=0.5,
            vwap=8.0,
            day_high=12.0,
            features=s,
            dry_run=True,
        )
        if not result.get("valid", False):
            print(
                f"{s['vwap_extension']:>9.2f} {s['volume_ratio']:>10.1f}   "
                f"[invalid: {result.get('reason')}]"
            )
            continue
        print(
            f"{s['vwap_extension']:>9.2f} {s['volume_ratio']:>10.1f} "
            f"{result.get('edge_n_trades', 0):>7} "
            f"{result.get('kelly_fraction', 0.0):>7.3f} "
            f"{result.get('dd_modifier', 1.0):>7.2f} "
            f"{result.get('shares', 0):>7} "
            f"{result.get('total_risk', 0.0):>9.2f}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
