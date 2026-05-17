import json
from pathlib import Path


def _fixture_run(tmp_path: Path, label: str, total_pnl: float, win_rate: float) -> Path:
    run_dir = tmp_path / label
    run_dir.mkdir()
    payload = {
        "run_name": label,
        "per_fold_results": [
            {
                "fold": 0,
                "test_metrics": {
                    "total_test_pnl": total_pnl,
                    "win_rate": win_rate,
                    "mean_episode_pnl": total_pnl / 10,
                    "total_trades": 50,
                },
            }
        ],
    }
    (run_dir / "wfo_results.json").write_text(json.dumps(payload))
    return run_dir / "wfo_results.json"


def test_compare_two_runs_prints_table(tmp_path, capsys):
    a = _fixture_run(tmp_path, "shaped_w0", 1000.0, 0.6)
    b = _fixture_run(tmp_path, "shaped_w0p5", 1500.0, 0.7)
    from src.scripts.compare_training_runs import compare
    compare([a, b])
    captured = capsys.readouterr()
    assert "shaped_w0" in captured.out
    assert "shaped_w0p5" in captured.out
    assert "1000" in captured.out or "1,000" in captured.out or "1000.00" in captured.out
    assert "1500" in captured.out or "1,500" in captured.out or "1500.00" in captured.out


def test_compare_handles_missing_file(tmp_path, capsys):
    from src.scripts.compare_training_runs import compare
    missing = tmp_path / "nope.json"
    compare([missing])
    captured = capsys.readouterr()
    assert "not found" in captured.out.lower() or "missing" in captured.out.lower() or "warn" in captured.out.lower()


def test_compare_aggregates_pnl_across_folds(tmp_path, capsys):
    """Multi-fold runs sum total_test_pnl in the aggregate row."""
    run_dir = tmp_path / "multi"
    run_dir.mkdir()
    payload = {
        "run_name": "multi",
        "per_fold_results": [
            {"fold": 0, "test_metrics": {"total_test_pnl": 500.0, "win_rate": 0.6, "total_trades": 50}},
            {"fold": 1, "test_metrics": {"total_test_pnl": 700.0, "win_rate": 0.65, "total_trades": 60}},
            {"fold": 2, "test_metrics": {"total_test_pnl": 300.0, "win_rate": 0.55, "total_trades": 40}},
        ],
    }
    p = run_dir / "wfo_results.json"
    p.write_text(json.dumps(payload))
    from src.scripts.compare_training_runs import compare
    compare([p])
    captured = capsys.readouterr()
    # 500 + 700 + 300 = 1500
    assert "1500" in captured.out or "1500.00" in captured.out


def test_compare_training_runs_no_torch_import():
    """Sanity check: the module must not transitively import torch."""
    import importlib, sys
    sys.modules.pop("torch", None)  # ensure clean state
    importlib.import_module("src.scripts.compare_training_runs")
    assert "torch" not in sys.modules, "compare_training_runs must not import torch"
