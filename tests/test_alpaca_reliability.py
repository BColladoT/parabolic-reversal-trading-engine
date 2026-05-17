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


def test_reconnect_delay_doubles_with_jitter():
    """compute_backoff should grow exponentially with attempts and stay <= cap."""
    from src.data.alpaca_client import compute_backoff
    delays = [compute_backoff(attempt=i, base=1.0, cap=60.0) for i in range(6)]
    # All delays are non-negative and bounded by cap.
    for d in delays:
        assert 0.0 <= d <= 60.0
    # Attempt 0 → up to base * 2^0 = 1.0 (with full jitter)
    assert delays[0] <= 2.0
    # Attempt 5 → capped at 60
    assert delays[5] <= 60.0
    # Upper bound grows monotonically: take a hundred samples and compare maxima
    # rather than rely on a single sample (jitter can fool a single-sample check).
    import random
    random.seed(0)
    upper_attempt0 = max(compute_backoff(0, base=1.0, cap=60.0) for _ in range(200))
    upper_attempt5 = max(compute_backoff(5, base=1.0, cap=60.0) for _ in range(200))
    assert upper_attempt5 > upper_attempt0


def test_poll_fill_returns_filled_state():
    c = AlpacaClient.__new__(AlpacaClient)
    fake_order = MagicMock(status="filled", filled_qty="100", filled_avg_price="9.95")
    c.trading_client = MagicMock()
    c.trading_client.get_order_by_id.return_value = fake_order
    result = c.poll_fill("order-123", timeout_s=1)
    assert result == {"status": "filled", "filled_qty": 100, "filled_avg_price": 9.95}


def test_poll_fill_timeout_returns_partial_state():
    c = AlpacaClient.__new__(AlpacaClient)
    fake_order = MagicMock(status="partially_filled", filled_qty="50", filled_avg_price="9.90")
    c.trading_client = MagicMock()
    c.trading_client.get_order_by_id.return_value = fake_order
    result = c.poll_fill("order-123", timeout_s=1)
    assert result["status"] == "partially_filled"
    assert result["filled_qty"] == 50


def test_submit_short_order_uses_provided_client_order_id():
    c = AlpacaClient.__new__(AlpacaClient)
    c.trading_client = MagicMock()
    c.trading_client.submit_order.return_value = MagicMock(id="x", status="new")
    c.submit_short_order("AMC", qty=100, limit_price=10.0, client_order_id="my-key-1")
    sent = c.trading_client.submit_order.call_args.kwargs["order_data"]
    assert sent.client_order_id == "my-key-1"


def test_submit_short_order_autogenerates_client_order_id():
    c = AlpacaClient.__new__(AlpacaClient)
    c.trading_client = MagicMock()
    c.trading_client.submit_order.return_value = MagicMock(id="x", status="new")
    c.submit_short_order("AMC", qty=100, limit_price=10.0)
    sent = c.trading_client.submit_order.call_args.kwargs["order_data"]
    assert sent.client_order_id is not None
    assert sent.client_order_id.startswith("AMC-")
