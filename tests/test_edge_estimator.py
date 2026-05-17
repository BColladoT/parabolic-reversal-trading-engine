from datetime import datetime, date, timedelta
import polars as pl
import pytest
import math


def _seed(tmp_path, monkeypatch, rows: list[dict]):
    monkeypatch.setenv("TRADE_JOURNAL_DIR", str(tmp_path))
    from src.risk.trade_journal import append_trade
    for r in rows:
        append_trade(r)


def _trade(pnl: float, r: float, ext: float = 0.18, dow: float = 0.0,
           t: datetime = datetime(2026, 5, 17, 10, 30)) -> dict:
    return {
        "symbol": "X", "entry_time": t, "exit_time": t + timedelta(minutes=15),
        "entry_price": 10.0, "exit_price": 9.5, "shares": 100, "side": "short",
        "pnl": pnl, "r_multiple": r, "hold_seconds": 900,
        "exit_reason": "tp1", "win": pnl > 0,
        "feat_vwap_extension": ext, "feat_volume_ratio": 3.0, "feat_atr_pct": 0.02,
        "feat_time_of_day_min": 60.0, "feat_day_of_week": dow, "feat_factors_count": 3.0,
    }


def test_compute_edge_empty_journal_returns_safe_zero(tmp_path, monkeypatch):
    monkeypatch.setenv("TRADE_JOURNAL_DIR", str(tmp_path))
    from src.risk.edge_estimator import compute_edge
    e = compute_edge()
    assert e.n_trades == 0
    assert e.kelly_fraction == 0.0
    assert e.used_fallback is True


def test_compute_edge_overall_stats(tmp_path, monkeypatch):
    rows = [_trade(100, 1.0)] * 6 + [_trade(-50, -1.0)] * 4
    _seed(tmp_path, monkeypatch, rows)
    from src.risk.edge_estimator import compute_edge
    e = compute_edge()
    assert e.n_trades == 10
    assert e.win_rate == pytest.approx(0.6)
    assert e.avg_win_r == pytest.approx(1.0)
    assert e.avg_loss_r == pytest.approx(-1.0)
    assert e.expected_r == pytest.approx(0.2)
    # Kelly: f* = win_rate/loss_payoff - (1-win_rate)/win_payoff
    # with payoff=1 each side: f* = 0.6 - 0.4 = 0.2; clamped to <=0.25
    assert e.kelly_fraction == pytest.approx(0.2)


def test_compute_edge_kelly_clamped_at_quarter(tmp_path, monkeypatch):
    # Wildly positive edge → Kelly would be huge, must clamp to 0.25
    rows = [_trade(100, 1.0)] * 90 + [_trade(-50, -1.0)] * 10
    _seed(tmp_path, monkeypatch, rows)
    from src.risk.edge_estimator import compute_edge
    e = compute_edge()
    assert e.kelly_fraction == pytest.approx(0.25)


def test_compute_edge_conditional_slice_uses_features(tmp_path, monkeypatch):
    # 25 winners in the "high vwap_extension" bucket, 5 losers
    high_rows = [_trade(100, 1.0, ext=0.30)] * 25 + [_trade(-50, -1.0, ext=0.30)] * 5
    # Mixed in low bucket
    low_rows = [_trade(100, 1.0, ext=0.10)] * 5 + [_trade(-50, -1.0, ext=0.10)] * 25
    _seed(tmp_path, monkeypatch, high_rows + low_rows)
    from src.risk.edge_estimator import compute_edge
    e_high = compute_edge(features={"feat_vwap_extension": 0.30})
    assert e_high.used_fallback is False
    assert e_high.win_rate > 0.7
    e_low = compute_edge(features={"feat_vwap_extension": 0.10})
    assert e_low.win_rate < 0.3


def test_compute_edge_falls_back_when_slice_too_small(tmp_path, monkeypatch):
    rows = [_trade(100, 1.0, ext=0.18)] * 50
    _seed(tmp_path, monkeypatch, rows)
    from src.risk.edge_estimator import compute_edge
    e = compute_edge(features={"feat_vwap_extension": 0.99}, min_samples_conditional=20)
    assert e.used_fallback is True
    assert e.n_trades == 50  # overall


def test_consecutive_losses_counts_tail(tmp_path, monkeypatch):
    rows = [
        _trade(100, 1.0, t=datetime(2026, 5, 17, 10, 0)),
        _trade(-50, -1.0, t=datetime(2026, 5, 17, 11, 0)),
        _trade(-50, -1.0, t=datetime(2026, 5, 17, 12, 0)),
        _trade(-50, -1.0, t=datetime(2026, 5, 17, 13, 0)),
    ]
    _seed(tmp_path, monkeypatch, rows)
    from src.risk.edge_estimator import consecutive_losses
    assert consecutive_losses() == 3


def test_consecutive_losses_zero_when_last_is_win(tmp_path, monkeypatch):
    rows = [_trade(-50, -1.0, t=datetime(2026, 5, 17, 10, 0)),
            _trade(100, 1.0, t=datetime(2026, 5, 17, 11, 0))]
    _seed(tmp_path, monkeypatch, rows)
    from src.risk.edge_estimator import consecutive_losses
    assert consecutive_losses() == 0
