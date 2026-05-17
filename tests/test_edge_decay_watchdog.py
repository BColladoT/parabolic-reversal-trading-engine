"""Tests for src/scripts/edge_decay_watchdog.py.

Use tmp_path + monkeypatch.setenv("TRADE_JOURNAL_DIR", ...) to isolate the
journal per test. Mock send_alert to avoid real HTTP.
"""
from datetime import datetime, timedelta


def _seed(tmp_path, monkeypatch, n_wins_recent, n_losses_recent, n_wins_old, n_losses_old):
    monkeypatch.setenv("TRADE_JOURNAL_DIR", str(tmp_path))
    from src.risk.trade_journal import append_trade
    today = datetime.now()

    def _row(t, pnl):
        return {
            "symbol": "X", "entry_time": t, "exit_time": t + timedelta(minutes=5),
            "entry_price": 10.0, "exit_price": 9.5, "shares": 100, "side": "short",
            "pnl": pnl, "r_multiple": 1.0 if pnl > 0 else -1.0,
            "hold_seconds": 300, "exit_reason": "tp1", "win": pnl > 0,
            "feat_vwap_extension": 0.2, "feat_volume_ratio": 3.0, "feat_atr_pct": 0.02,
            "feat_time_of_day_min": 60.0, "feat_day_of_week": 0.0, "feat_factors_count": 3.0,
        }

    for i in range(n_wins_recent):
        append_trade(_row(today - timedelta(days=5, minutes=i), 50.0))
    for i in range(n_losses_recent):
        append_trade(_row(today - timedelta(days=5, minutes=100 + i), -50.0))
    for i in range(n_wins_old):
        append_trade(_row(today - timedelta(days=200, minutes=i), 50.0))
    for i in range(n_losses_old):
        append_trade(_row(today - timedelta(days=200, minutes=100 + i), -50.0))


def test_check_decay_not_decayed_when_empty(tmp_path, monkeypatch):
    monkeypatch.setenv("TRADE_JOURNAL_DIR", str(tmp_path))
    from src.scripts.edge_decay_watchdog import check_decay
    c = check_decay()
    assert c.decayed is False
    assert c.recent_n == 0


def test_check_decay_not_decayed_when_recent_below_threshold_samples(tmp_path, monkeypatch):
    # Only 5 recent trades — below min_recent_samples=10
    _seed(tmp_path, monkeypatch, 1, 4, 60, 40)
    from src.scripts.edge_decay_watchdog import check_decay
    c = check_decay(min_recent_samples=10)
    assert c.decayed is False
    assert c.recent_n == 5
    assert "insufficient" in c.reason.lower()


def test_check_decay_flags_when_recent_winrate_collapses(tmp_path, monkeypatch):
    # Baseline 100 trades at 60% wins; recent 30 trades at 10% wins -> sharp decay
    _seed(tmp_path, monkeypatch, n_wins_recent=3, n_losses_recent=27,
          n_wins_old=60, n_losses_old=40)
    from src.scripts.edge_decay_watchdog import check_decay
    c = check_decay(min_recent_samples=10, sigma_threshold=2.0)
    assert c.decayed is True
    assert c.z_score < -2.0
    assert c.recent_win_rate < c.baseline_win_rate


def test_check_decay_not_flagged_when_recent_matches_baseline(tmp_path, monkeypatch):
    _seed(tmp_path, monkeypatch, n_wins_recent=18, n_losses_recent=12,
          n_wins_old=60, n_losses_old=40)
    from src.scripts.edge_decay_watchdog import check_decay
    c = check_decay()
    assert c.decayed is False
    assert c.z_score > -2.0


def test_main_sends_alert_when_decayed(tmp_path, monkeypatch):
    _seed(tmp_path, monkeypatch, 3, 27, 60, 40)
    from unittest.mock import patch
    with patch("src.scripts.edge_decay_watchdog.send_alert") as mock_alert:
        from src.scripts.edge_decay_watchdog import main
        main()
        mock_alert.assert_called_once()
        assert "decay" in mock_alert.call_args.args[0].lower()


def test_main_does_not_alert_when_healthy(tmp_path, monkeypatch):
    _seed(tmp_path, monkeypatch, 18, 12, 60, 40)
    from unittest.mock import patch
    with patch("src.scripts.edge_decay_watchdog.send_alert") as mock_alert:
        from src.scripts.edge_decay_watchdog import main
        main()
        mock_alert.assert_not_called()
