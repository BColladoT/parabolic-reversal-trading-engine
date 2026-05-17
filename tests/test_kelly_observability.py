"""Tests for Agent A2 — features kwarg plumbing + structured INFO sizing log + diagnostic CLI."""
import inspect
import os
from unittest.mock import MagicMock, patch

# Set Alpaca cred stubs BEFORE imports — conftest does this too, but being defensive
os.environ.setdefault("ALPACA_API_KEY", "ci-stub-key")
os.environ.setdefault("ALPACA_SECRET", "ci-stub-secret")
os.environ.setdefault("ALPACA_SECRET_KEY", "ci-stub-secret")


def test_calculate_position_size_accepts_features_kwarg():
    from src.risk.position_manager import RiskManager
    sig = inspect.signature(RiskManager.calculate_position_size)
    assert "features" in sig.parameters
    assert sig.parameters["features"].default is None


def test_calculate_position_size_accepts_dry_run_kwarg():
    from src.risk.position_manager import RiskManager
    sig = inspect.signature(RiskManager.calculate_position_size)
    assert "dry_run" in sig.parameters
    assert sig.parameters["dry_run"].default is False


def test_features_are_forwarded_to_compute_edge():
    from src.risk.position_manager import RiskManager
    stub_account = MagicMock(equity="10000", cash="10000", buying_power="10000")
    stub_client = MagicMock(get_account=lambda: stub_account)
    rm = RiskManager(alpaca_client=stub_client)
    sentinel_features = {"vwap_extension": 1.23, "volume_ratio": 4.5}
    with patch("src.risk.edge_estimator.compute_edge") as mock_edge:
        mock_edge.return_value = MagicMock(
            n_trades=50, win_rate=0.7, expected_r=2.0, kelly_fraction=0.1, used_fallback=False,
        )
        with patch("src.risk.edge_estimator.consecutive_losses", return_value=0):
            rm.calculate_position_size(
                symbol="X", entry_price=10.0, atr=0.5, vwap=8.0,
                day_high=12.0, features=sentinel_features,
            )
    call_kwargs = mock_edge.call_args.kwargs
    assert call_kwargs.get("features") == sentinel_features


def test_main_engine_passes_signal_features_to_sizing():
    """Source-level check: main_engine.py passes features=signal.features (or getattr) to calculate_position_size."""
    from pathlib import Path
    src = Path("src/main_engine.py").read_text()
    assert "calculate_position_size" in src
    # Either explicit `features=signal.features` or `features=getattr(signal, "features"...`
    assert "features=" in src and ("signal.features" in src or 'features", None' in src)


def test_observe_live_sizing_smoke(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("TRADE_JOURNAL_DIR", str(tmp_path))
    from src.scripts.observe_live_sizing import main
    rc = main()
    captured = capsys.readouterr()
    assert rc == 0
    assert "vwap_ext" in captured.out or "kelly" in captured.out
