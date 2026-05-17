"""Tests for src.utils.alerting — Slack-compatible webhook sender."""
from unittest.mock import patch, MagicMock


def test_send_alert_returns_false_when_no_webhook(monkeypatch):
    monkeypatch.delenv("SLACK_WEBHOOK_URL", raising=False)
    from src.utils.alerting import send_alert
    assert send_alert("test", "body") is False


def test_send_alert_posts_when_webhook_set(monkeypatch):
    monkeypatch.setenv("SLACK_WEBHOOK_URL", "https://hooks.slack.com/services/X/Y/Z")
    with patch("urllib.request.urlopen") as m:
        ctx = MagicMock()
        ctx.status = 200
        m.return_value.__enter__.return_value = ctx
        from src.utils.alerting import send_alert
        assert send_alert("title", "body", level="critical") is True
        assert m.called


def test_send_alert_swallows_network_errors(monkeypatch):
    monkeypatch.setenv("SLACK_WEBHOOK_URL", "https://hooks.slack.com/services/X/Y/Z")
    with patch("urllib.request.urlopen", side_effect=OSError("network down")):
        from src.utils.alerting import send_alert
        assert send_alert("title", "body") is False  # no raise


def test_send_alert_returns_false_on_non_2xx(monkeypatch):
    monkeypatch.setenv("SLACK_WEBHOOK_URL", "https://hooks.slack.com/services/X/Y/Z")
    with patch("urllib.request.urlopen") as m:
        ctx = MagicMock()
        ctx.status = 500
        m.return_value.__enter__.return_value = ctx
        from src.utils.alerting import send_alert
        assert send_alert("title", "body") is False


def test_send_alert_payload_includes_level_and_title(monkeypatch):
    monkeypatch.setenv("SLACK_WEBHOOK_URL", "https://hooks.slack.com/services/X/Y/Z")
    with patch("urllib.request.urlopen") as m:
        ctx = MagicMock()
        ctx.status = 200
        m.return_value.__enter__.return_value = ctx
        from src.utils.alerting import send_alert
        assert send_alert("circuit-breaker", "daily_pnl=-25000", level="critical") is True
        # urlopen called with a Request whose .data contains our serialized payload
        req = m.call_args.args[0]
        body = req.data.decode("utf-8")
        assert "CRITICAL" in body
        assert "circuit-breaker" in body
        assert "daily_pnl=-25000" in body
