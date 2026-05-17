"""Pre-flight checks before starting live paper trading.

Run via:  python -m src.scripts.preflight_paper_trade
Exit 0 if all 8 checks pass; 1 otherwise.

Each check is exposed as a private function for unit-test mocking.
"""
from __future__ import annotations

import os
import sys
import tempfile
from datetime import date
from pathlib import Path


_STUB_VALUES = {"ci-stub-key", "ci-stub-secret", "your_api_key_here", "your_secret_here", ""}


def _check_credentials() -> tuple[bool, str]:
    k = os.getenv("ALPACA_API_KEY", "")
    s = os.getenv("ALPACA_SECRET", "") or os.getenv("ALPACA_SECRET_KEY", "")
    if not k or not s:
        return False, "ALPACA_API_KEY and ALPACA_SECRET must both be set"
    if k in _STUB_VALUES or s in _STUB_VALUES:
        return False, "credentials look like CI stub values"
    if len(k) < 8 or len(s) < 8:
        return False, "credentials suspiciously short"
    return True, "credentials present"


def _check_alpaca_auth() -> tuple[bool, str]:
    try:
        import urllib.request
        import json
        req = urllib.request.Request(
            "https://paper-api.alpaca.markets/v2/account",
            headers={
                "APCA-API-KEY-ID": os.getenv("ALPACA_API_KEY", ""),
                "APCA-API-SECRET-KEY": os.getenv("ALPACA_SECRET", "") or os.getenv("ALPACA_SECRET_KEY", ""),
            },
        )
        with urllib.request.urlopen(req, timeout=8) as r:
            payload = json.loads(r.read())
        equity = payload.get("equity")
        return True, f"alpaca auth ok (equity={equity})"
    except Exception as e:
        return False, f"alpaca auth failed: {e}"


def _check_market_day(today: date | None = None) -> tuple[bool, str]:
    d = today or date.today()
    if d.weekday() >= 5:
        return False, f"{d} is a weekend"
    holidays = {(1, 1), (7, 4), (12, 25)}
    if (d.month, d.day) in holidays:
        return False, f"{d} is a fixed-date holiday"
    return True, f"{d} is a market day"


def _check_journal_writeable() -> tuple[bool, str]:
    journal_dir = Path(os.getenv("TRADE_JOURNAL_DIR", "data/trade_journal"))
    try:
        journal_dir.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(dir=journal_dir, delete=True):
            pass
    except OSError as e:
        return False, f"journal dir not writeable: {e}"
    return True, f"journal dir writeable ({journal_dir})"


def _check_daily_state() -> tuple[bool, str]:
    state_path = Path("data/state/daily_state.json")
    if not state_path.exists():
        return True, "daily_state.json absent - fresh start"
    try:
        import json
        json.loads(state_path.read_text())
    except Exception as e:
        return False, f"daily_state.json present but unreadable: {e}"
    return True, "daily_state.json loadable"


def _check_regime_fresh(max_age_days: int = 7) -> tuple[bool, str]:
    try:
        from src.risk.regime import read_regime_history
    except Exception as e:
        return False, f"regime module import failed: {e}"
    df = read_regime_history()
    if df.is_empty():
        return False, "regime table empty or missing - run backfill_regime"
    latest = df["date"].max()
    age = (date.today() - latest).days
    if age > max_age_days:
        return False, f"regime table stale ({age} days old; max {max_age_days})"
    return True, f"regime current (latest {latest}, {age}d old)"


def _check_slack_webhook() -> tuple[bool, str]:
    url = os.getenv("SLACK_WEBHOOK_URL", "")
    if not url:
        return True, "SLACK_WEBHOOK_URL not set - alerts disabled (ok)"
    try:
        import urllib.request, json
        req = urllib.request.Request(
            url,
            data=json.dumps({"text": "preflight ping"}).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=5) as r:
            if r.status == 200:
                return True, "slack webhook responding"
            return False, f"slack webhook returned status {r.status}"
    except Exception as e:
        return False, f"slack webhook failed: {e}"


def _check_logs_writeable() -> tuple[bool, str]:
    log_dir = Path(os.getenv("LOG_DIR", "logs"))
    try:
        log_dir.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(dir=log_dir, delete=True):
            pass
    except OSError as e:
        return False, f"log dir not writeable: {e}"
    return True, f"log dir writeable ({log_dir})"


# Names of check functions. Looked up via module globals at run time so that
# unit tests can monkey-patch e.g. ``_check_alpaca_auth`` and have ``main``
# see the patched version.
_CHECK_NAMES = [
    ("credentials",        "_check_credentials"),
    ("alpaca_auth",        "_check_alpaca_auth"),
    ("market_day",         "_check_market_day"),
    ("journal_writeable",  "_check_journal_writeable"),
    ("daily_state",        "_check_daily_state"),
    ("regime_fresh",       "_check_regime_fresh"),
    ("slack_webhook",      "_check_slack_webhook"),
    ("logs_writeable",     "_check_logs_writeable"),
]


def main() -> int:
    mod = sys.modules[__name__]
    fails = 0
    for name, fn_name in _CHECK_NAMES:
        fn = getattr(mod, fn_name)
        ok, msg = fn()
        status = "PASS" if ok else "FAIL"
        print(f"[{status}] {name:<20} {msg}")
        if not ok:
            fails += 1
    print()
    if fails == 0:
        print("READY - all 8 checks passed.")
        return 0
    print(f"BLOCKED - {fails} issue(s); fix and re-run.")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
