"""Lightweight alerting: posts to a Slack-compatible webhook. Never raises.

Reads SLACK_WEBHOOK_URL from the environment. If unset, send_alert is a no-op
that returns False so callers can use it unconditionally without wiring config.

Uses stdlib urllib.request to avoid adding a new dependency.
"""
import json
import os
import urllib.request

from src.utils.logger import logger


def send_alert(title: str, body: str, level: str = "critical") -> bool:
    """Post an alert to the Slack-compatible webhook in SLACK_WEBHOOK_URL.

    Args:
        title: Short headline for the alert.
        body: Longer detail (wrapped in a code block).
        level: Severity tag (default "critical"). Free-form; uppercased in payload.

    Returns:
        True on 2xx response from the webhook.
        False if SLACK_WEBHOOK_URL is unset, or if any error (network, HTTP,
        serialization) occurred. This function never raises.
    """
    url = os.environ.get("SLACK_WEBHOOK_URL")
    if not url:
        return False

    try:
        payload = json.dumps(
            {"text": f"*[{level.upper()}] {title}*\n```{body}```"}
        ).encode("utf-8")

        req = urllib.request.Request(
            url,
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=5) as resp:
            ok = 200 <= resp.status < 300
            if not ok:
                logger.warning("alerting webhook non-2xx", status=resp.status)
            return ok
    except Exception as e:  # noqa: BLE001 — alerting must never raise
        logger.warning("alerting webhook failed", error=str(e))
        return False
