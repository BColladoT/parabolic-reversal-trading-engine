"""Pytest config: sys.path setup + Alpaca cred stubs for CI.

The cred stubs MUST run before any test file imports anything from ``src.*``
because ``src.utils.config`` reads ``os.getenv("ALPACA_API_KEY", "")`` at
module-load time and caches the result in the global ``CONFIG``, and
``src.backtest.__init__`` constructs singleton ``BacktestEngine`` /
``DataFetcher`` instances at import that crash on empty creds.

In CI those env vars don't exist. conftest.py loads earlier than any test
file, so setting stubs here is the only reliable fix. setdefault leaves
real creds untouched on dev/prod environments that have them in .env.
"""
import os
import sys
from pathlib import Path

os.environ.setdefault("ALPACA_API_KEY", "ci-stub-key")
os.environ.setdefault("ALPACA_SECRET", "ci-stub-secret")
os.environ.setdefault("ALPACA_SECRET_KEY", "ci-stub-secret")

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
