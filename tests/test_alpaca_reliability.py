"""Unit tests for AlpacaClient reliability primitives.

All tests mock the broker — no network calls.

Note: the plan suggested ``monkeypatch.setattr("src.utils.config.CONFIG", MagicMock())``
but doing that before importing AlpacaClient breaks the logger setup
(``CONFIG.logging.file`` becomes a MagicMock used as a filename). Since the
methods under test don't touch CONFIG, the patch is unnecessary — we let the
real CONFIG load and use ``__new__`` to skip ``__init__`` entirely.
"""
import time
from unittest.mock import patch, MagicMock

from src.data.alpaca_client import AlpacaClient


def test_is_feed_stale_true_when_no_recent_messages():
    c = AlpacaClient.__new__(AlpacaClient)
    c.last_message_time = time.time() - 60
    assert c.is_feed_stale(max_age_s=30) is True


def test_is_feed_stale_false_when_fresh():
    c = AlpacaClient.__new__(AlpacaClient)
    c.last_message_time = time.time() - 5
    assert c.is_feed_stale(max_age_s=30) is False
