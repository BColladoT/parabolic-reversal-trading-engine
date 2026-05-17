from unittest.mock import patch


def test_preflight_fails_when_creds_are_stubs(monkeypatch):
    monkeypatch.setenv("ALPACA_API_KEY", "ci-stub-key")
    monkeypatch.setenv("ALPACA_SECRET", "ci-stub-secret")
    from src.scripts.preflight_paper_trade import _check_credentials
    ok, msg = _check_credentials()
    assert ok is False
    assert "stub" in msg.lower()


def test_preflight_passes_when_creds_real(monkeypatch):
    monkeypatch.setenv("ALPACA_API_KEY", "AK_real_looking_key")
    monkeypatch.setenv("ALPACA_SECRET", "real_looking_secret_value")
    from src.scripts.preflight_paper_trade import _check_credentials
    ok, _ = _check_credentials()
    assert ok is True


def test_preflight_market_day_rejects_weekend():
    from src.scripts.preflight_paper_trade import _check_market_day
    from datetime import date
    ok, msg = _check_market_day(today=date(2024, 1, 6))  # Saturday
    assert ok is False


def test_preflight_market_day_accepts_weekday():
    from src.scripts.preflight_paper_trade import _check_market_day
    from datetime import date
    ok, _ = _check_market_day(today=date(2024, 1, 9))  # Tuesday
    assert ok is True


def test_preflight_market_day_rejects_christmas():
    from src.scripts.preflight_paper_trade import _check_market_day
    from datetime import date
    ok, _ = _check_market_day(today=date(2024, 12, 25))
    assert ok is False


def test_preflight_journal_writeable(tmp_path, monkeypatch):
    monkeypatch.setenv("TRADE_JOURNAL_DIR", str(tmp_path))
    from src.scripts.preflight_paper_trade import _check_journal_writeable
    ok, _ = _check_journal_writeable()
    assert ok is True


def test_preflight_regime_fresh(tmp_path, monkeypatch):
    monkeypatch.setenv("REGIME_DIR", str(tmp_path))
    from src.risk.regime import write_regime_history, REGIME_SCHEMA
    from datetime import date, timedelta
    import polars as pl
    write_regime_history(pl.DataFrame({
        "date": [date.today() - timedelta(days=1)],
        "vix_level": [18.0], "spy_trend": [1], "label": ["risk_on"],
    }, schema=REGIME_SCHEMA))
    from src.scripts.preflight_paper_trade import _check_regime_fresh
    ok, _ = _check_regime_fresh()
    assert ok is True


def test_preflight_regime_stale_fails(tmp_path, monkeypatch):
    monkeypatch.setenv("REGIME_DIR", str(tmp_path))
    from src.risk.regime import write_regime_history, REGIME_SCHEMA
    from datetime import date, timedelta
    import polars as pl
    write_regime_history(pl.DataFrame({
        "date": [date.today() - timedelta(days=30)],
        "vix_level": [18.0], "spy_trend": [1], "label": ["risk_on"],
    }, schema=REGIME_SCHEMA))
    from src.scripts.preflight_paper_trade import _check_regime_fresh
    ok, msg = _check_regime_fresh(max_age_days=7)
    assert ok is False


def test_preflight_regime_missing_fails(tmp_path, monkeypatch):
    monkeypatch.setenv("REGIME_DIR", str(tmp_path))
    from src.scripts.preflight_paper_trade import _check_regime_fresh
    ok, msg = _check_regime_fresh()
    assert ok is False
    assert "empty" in msg.lower() or "missing" in msg.lower()


def test_main_returns_nonzero_when_any_check_fails(monkeypatch, tmp_path, capsys):
    monkeypatch.setenv("ALPACA_API_KEY", "ci-stub-key")
    monkeypatch.setenv("ALPACA_SECRET", "ci-stub-secret")
    monkeypatch.setenv("TRADE_JOURNAL_DIR", str(tmp_path))
    monkeypatch.setenv("REGIME_DIR", str(tmp_path))
    from src.scripts.preflight_paper_trade import main
    rc = main()
    captured = capsys.readouterr()
    assert rc == 1
    assert "BLOCKED" in captured.out


def test_main_returns_zero_when_all_pass(monkeypatch, tmp_path, capsys):
    monkeypatch.setenv("ALPACA_API_KEY", "AK_real_long_enough_value")
    monkeypatch.setenv("ALPACA_SECRET", "real_secret_value_blahblah")
    monkeypatch.setenv("TRADE_JOURNAL_DIR", str(tmp_path / "journal"))
    monkeypatch.setenv("REGIME_DIR", str(tmp_path / "regime"))
    monkeypatch.setenv("LOG_DIR", str(tmp_path / "logs"))
    monkeypatch.delenv("SLACK_WEBHOOK_URL", raising=False)
    # Seed fresh regime
    from datetime import date
    import polars as pl
    from src.risk.regime import write_regime_history, REGIME_SCHEMA
    write_regime_history(pl.DataFrame({
        "date": [date.today()], "vix_level": [18.0], "spy_trend": [1], "label": ["risk_on"],
    }, schema=REGIME_SCHEMA))
    with patch("src.scripts.preflight_paper_trade._check_alpaca_auth", return_value=(True, "ok")):
        with patch("src.scripts.preflight_paper_trade._check_market_day", return_value=(True, "ok")):
            from src.scripts.preflight_paper_trade import main
            rc = main()
    captured = capsys.readouterr()
    assert rc == 0
    assert "READY" in captured.out
