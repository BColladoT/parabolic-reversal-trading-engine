"""Daily watchdog: detect when recent edge falls below the long baseline.

Computes a pooled z-score of (recent_win_rate - baseline_win_rate) using
the binomial standard error of the difference. Negative z-score = recent
worse than baseline. Alerts via Slack when z < -sigma_threshold AND we
have at least min_recent_samples trades in the recent window (to avoid
firing on noise).

Run as: python -m src.scripts.edge_decay_watchdog
Recommended: schedule daily after market close.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import date as date_cls, timedelta

from src.risk.trade_journal import read_trades
from src.utils.alerting import send_alert


@dataclass(frozen=True)
class DecayCheck:
    decayed: bool
    recent_n: int
    recent_win_rate: float
    baseline_n: int
    baseline_win_rate: float
    z_score: float
    sigma_threshold: float
    reason: str


def _binomial_se_diff(p_recent: float, n_recent: int, p_base: float, n_base: int) -> float:
    var = (p_recent * (1 - p_recent)) / max(n_recent, 1) + (p_base * (1 - p_base)) / max(n_base, 1)
    return math.sqrt(var) if var > 0 else float("nan")


def _safe_winrate(df) -> float:
    """Return mean(win) as float, or NaN if column missing/empty."""
    if df.is_empty():
        return float("nan")
    m = df["win"].mean()
    if m is None:
        return float("nan")
    return float(m)


def check_decay(
    recent_days: int = 30,
    baseline_days: int = 365,
    min_recent_samples: int = 10,
    sigma_threshold: float = 2.0,
) -> DecayCheck:
    """Compare last ``recent_days`` win rate against the older baseline window.

    Args:
        recent_days: Size of the recent window (days back from today).
        baseline_days: Size of the full lookback. The baseline window is
            (today - baseline_days .. today - recent_days - 1), i.e. it
            EXCLUDES the recent window so the two samples are disjoint.
        min_recent_samples: Minimum number of recent trades required before
            we even consider firing an alert.
        sigma_threshold: Decay is flagged when z-score < -sigma_threshold.

    Returns:
        DecayCheck with ``decayed=True`` iff the z-score crosses the
        threshold AND the recent sample is large enough to trust.
    """
    today = date_cls.today()
    recent_from = today - timedelta(days=recent_days)
    baseline_from = today - timedelta(days=baseline_days)
    baseline_to = recent_from - timedelta(days=1)

    recent_df = read_trades(date_from=recent_from)
    baseline_df = read_trades(date_from=baseline_from, date_to=baseline_to)

    recent_n = recent_df.shape[0]
    baseline_n = baseline_df.shape[0]

    if recent_n == 0:
        return DecayCheck(
            decayed=False,
            recent_n=0,
            recent_win_rate=float("nan"),
            baseline_n=baseline_n,
            baseline_win_rate=_safe_winrate(baseline_df),
            z_score=float("nan"),
            sigma_threshold=sigma_threshold,
            reason="no recent trades to evaluate",
        )

    p_recent = _safe_winrate(recent_df)

    if recent_n < min_recent_samples:
        return DecayCheck(
            decayed=False,
            recent_n=recent_n,
            recent_win_rate=p_recent,
            baseline_n=baseline_n,
            baseline_win_rate=_safe_winrate(baseline_df),
            z_score=float("nan"),
            sigma_threshold=sigma_threshold,
            reason=f"insufficient recent samples ({recent_n} < {min_recent_samples})",
        )

    if baseline_n == 0:
        return DecayCheck(
            decayed=False,
            recent_n=recent_n,
            recent_win_rate=p_recent,
            baseline_n=0,
            baseline_win_rate=float("nan"),
            z_score=float("nan"),
            sigma_threshold=sigma_threshold,
            reason="no baseline trades to compare against",
        )

    p_base = _safe_winrate(baseline_df)
    se = _binomial_se_diff(p_recent, recent_n, p_base, baseline_n)
    if se and not math.isnan(se) and se > 0:
        z = (p_recent - p_base) / se
    else:
        z = 0.0

    decayed = z < -sigma_threshold
    return DecayCheck(
        decayed=decayed,
        recent_n=recent_n,
        recent_win_rate=p_recent,
        baseline_n=baseline_n,
        baseline_win_rate=p_base,
        z_score=z,
        sigma_threshold=sigma_threshold,
        reason=(
            "recent win rate collapsed vs baseline"
            if decayed
            else "within noise tolerance"
        ),
    )


def _fmt_pct(x: float) -> str:
    return f"{x:.2%}" if not math.isnan(x) else "n/a"


def _fmt_float(x: float) -> str:
    return f"{x:.2f}" if not math.isnan(x) else "n/a"


def main() -> None:
    """CLI entry point: print the DecayCheck; send Slack alert when decayed."""
    c = check_decay()
    print("=== Edge decay check ===")
    print(f"recent_n={c.recent_n}  recent_wr={_fmt_pct(c.recent_win_rate)}")
    print(f"baseline_n={c.baseline_n}  baseline_wr={_fmt_pct(c.baseline_win_rate)}")
    print(f"z_score={_fmt_float(c.z_score)}  threshold=-{c.sigma_threshold}")
    print(f"verdict: {'DECAYED' if c.decayed else 'OK'}  ({c.reason})")
    if c.decayed:
        title = "Edge decay detected"
        body = (
            f"Recent win rate {_fmt_pct(c.recent_win_rate)} (n={c.recent_n}) is "
            f"{abs(c.z_score):.1f} sigma below baseline {_fmt_pct(c.baseline_win_rate)} "
            f"(n={c.baseline_n}). Investigate before continuing live trading."
        )
        send_alert(title, body, "critical")


if __name__ == "__main__":
    main()
