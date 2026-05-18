"""Pure unit tests for mfe_evaporation_penalty in src/rl/env.py.

AST-extraction pattern: src/rl/env.py imports gym/torch at module scope so
a normal import would require the full RL stack. We parse env.py with the
ast module, locate the helper definition, and exec just that node into a
fresh namespace. Mirrors test_env_reward_rmultiple.py and
test_env_cover_fraction.py.
"""
from __future__ import annotations

import ast
from pathlib import Path

import pytest


def _load_helper_torch_free():
    env_path = (
        Path(__file__).resolve().parents[1] / "src" / "rl" / "env.py"
    )
    source = env_path.read_text(encoding="utf-8")
    tree = ast.parse(source)
    func_node = next(
        (
            n for n in tree.body
            if isinstance(n, ast.FunctionDef) and n.name == "mfe_evaporation_penalty"
        ),
        None,
    )
    if func_node is None:
        raise RuntimeError(
            "mfe_evaporation_penalty not found at module scope in src/rl/env.py"
        )
    module = ast.Module(body=[func_node], type_ignores=[])
    ns: dict = {}
    exec(compile(module, str(env_path), "exec"), ns)
    return ns["mfe_evaporation_penalty"]


mfe_evaporation_penalty = _load_helper_torch_free()


def test_zero_max_penalty_returns_zero():
    """Default max_penalty=0.0 is a backward-compatible no-op."""
    assert mfe_evaporation_penalty(unrealized_pnl=10.0, mfe_peak=100.0, max_penalty=0.0) == 0.0
    assert mfe_evaporation_penalty(unrealized_pnl=-50.0, mfe_peak=100.0, max_penalty=0.0) == 0.0


def test_at_peak_no_penalty():
    """When current PnL matches the peak, no penalty (haven't evaporated anything)."""
    assert mfe_evaporation_penalty(unrealized_pnl=100.0, mfe_peak=100.0, max_penalty=0.5) == 0.0


def test_above_peak_no_penalty():
    """If current somehow exceeds tracked peak, no penalty (peak hasn't been updated yet)."""
    out = mfe_evaporation_penalty(unrealized_pnl=120.0, mfe_peak=100.0, max_penalty=0.5)
    assert out == 0.0


def test_half_evaporation_half_penalty():
    """Current=50, peak=100, max=0.5 -> evap=0.5 -> penalty=-0.25."""
    out = mfe_evaporation_penalty(unrealized_pnl=50.0, mfe_peak=100.0, max_penalty=0.5)
    assert out == pytest.approx(-0.25)


def test_full_evaporation_max_penalty():
    """Current=0, peak=100, max=0.5 -> evap=1.0 -> penalty=-0.5."""
    out = mfe_evaporation_penalty(unrealized_pnl=0.0, mfe_peak=100.0, max_penalty=0.5)
    assert out == pytest.approx(-0.5)


def test_went_negative_capped_at_max():
    """Current went below zero (deeper than full evaporation) - penalty caps at -max."""
    out = mfe_evaporation_penalty(unrealized_pnl=-30.0, mfe_peak=100.0, max_penalty=0.5)
    assert out == pytest.approx(-0.5)


def test_mfe_never_positive_no_penalty():
    """Trade was never profitable (mfe_peak <= 0) - penalty disabled entirely."""
    assert mfe_evaporation_penalty(unrealized_pnl=-50.0, mfe_peak=0.0, max_penalty=0.5) == 0.0
    assert mfe_evaporation_penalty(unrealized_pnl=-50.0, mfe_peak=-10.0, max_penalty=0.5) == 0.0


def test_class_method_exists_and_routes_to_helper():
    """Source-level: _compute_mfe_evaporation_penalty exists and calls the helper."""
    env_path = (
        Path(__file__).resolve().parents[1] / "src" / "rl" / "env.py"
    )
    src = env_path.read_text(encoding="utf-8")
    assert "def _compute_mfe_evaporation_penalty" in src
    assert "mfe_evaporation_penalty(" in src


def test_step_reward_sum_includes_mfe_evap():
    """Source-level: step() adds mfe_evap penalty into the reward path."""
    env_path = (
        Path(__file__).resolve().parents[1] / "src" / "rl" / "env.py"
    )
    src = env_path.read_text(encoding="utf-8")
    assert "_compute_mfe_evaporation_penalty" in src


def test_config_has_mfe_evap_penalty_max_field():
    """Source-level: EnvironmentConfig has the new field with default 0.0."""
    env_path = (
        Path(__file__).resolve().parents[1] / "src" / "rl" / "env.py"
    )
    src = env_path.read_text(encoding="utf-8")
    assert "mfe_evaporation_penalty_max" in src
