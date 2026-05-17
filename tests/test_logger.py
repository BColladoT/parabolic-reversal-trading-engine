"""Tests for src.utils.logger — LOG_LEVEL env var override."""
import logging
import importlib


def _reload_logger_module():
    """Clear the singleton and reload the logger module to pick up env changes."""
    import src.utils.logger as lg
    lg.TradingLogger._instance = None
    # Also clear any existing handlers on the named logger so reload doesn't stack them
    existing = logging.getLogger("parabolic_reversal")
    existing.handlers = []
    importlib.reload(lg)
    return lg


def test_log_level_respects_env(monkeypatch):
    monkeypatch.setenv("LOG_LEVEL", "DEBUG")
    lg = _reload_logger_module()
    assert lg.logger.logger.level == logging.DEBUG


def test_log_level_default_info(monkeypatch):
    monkeypatch.delenv("LOG_LEVEL", raising=False)
    lg = _reload_logger_module()
    assert lg.logger.logger.level == logging.INFO


def test_log_level_invalid_falls_back_to_info(monkeypatch):
    monkeypatch.setenv("LOG_LEVEL", "NOT_A_REAL_LEVEL")
    lg = _reload_logger_module()
    assert lg.logger.logger.level == logging.INFO
