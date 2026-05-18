"""Pure unit tests for the scale-out COVER helper in src/rl/env.py.

The helper itself is torch-free, but ``src/rl/env.py`` imports torch/gym at
module scope, so a normal ``from src.rl.env import scale_out_cover_target``
cannot run in CI without the full RL stack. To keep these tests honest and
self-contained, we parse ``src/rl/env.py`` with the ``ast`` module, locate the
``scale_out_cover_target`` function definition, and exec just that node into
a fresh namespace. No torch/gym/ray import is triggered.

Mirrors the pattern of tests/test_env_reward_rmultiple.py.
"""
from __future__ import annotations

import ast
from pathlib import Path

import pytest


def _load_helper_torch_free():
    """Parse src/rl/env.py and exec only the ``scale_out_cover_target`` def."""
    env_path = (
        Path(__file__).resolve().parents[1] / "src" / "rl" / "env.py"
    )
    source = env_path.read_text(encoding="utf-8")
    tree = ast.parse(source)
    func_node = next(
        (
            n for n in tree.body
            if isinstance(n, ast.FunctionDef) and n.name == "scale_out_cover_target"
        ),
        None,
    )
    if func_node is None:
        raise RuntimeError(
            "scale_out_cover_target not found at module scope in src/rl/env.py"
        )
    module = ast.Module(body=[func_node], type_ignores=[])
    ns: dict = {}
    exec(compile(module, str(env_path), "exec"), ns)
    return ns["scale_out_cover_target"]


scale_out_cover_target = _load_helper_torch_free()


def test_cover_50pct_closes_half_position():
    """action=+0.5 -> target = current * (1 - 0.5) = half the position remaining."""
    target = scale_out_cover_target(current_position_value=-2000.0, desired_exposure_fraction=0.5)
    assert target == pytest.approx(-1000.0)


def test_cover_100pct_fully_closes():
    """action=+1.0 -> target = 0 (full close), preserving prior behavior at max action."""
    target = scale_out_cover_target(current_position_value=-2000.0, desired_exposure_fraction=1.0)
    assert target == pytest.approx(0.0)


def test_cover_just_above_threshold_is_small_partial():
    """action=+0.06 (just above the +0.05 threshold) -> ~6% partial cover, not full."""
    target = scale_out_cover_target(current_position_value=-1000.0, desired_exposure_fraction=0.06)
    # -1000 * (1 - 0.06) = -940
    assert target == pytest.approx(-940.0, abs=1.0)


def test_cover_action_above_one_clamps_to_full_close():
    """Defensive: action > 1.0 (shouldn't happen post-clip but) clamps to full close."""
    target = scale_out_cover_target(current_position_value=-1000.0, desired_exposure_fraction=1.5)
    assert target == pytest.approx(0.0)


def test_cover_with_zero_position_returns_zero():
    """No position -> no-op."""
    target = scale_out_cover_target(current_position_value=0.0, desired_exposure_fraction=0.5)
    assert target == pytest.approx(0.0)


def test_cover_negative_action_treated_as_zero_fraction():
    """Defensive: if somehow called with negative action (shouldn't, but) -> no cover."""
    target = scale_out_cover_target(current_position_value=-1000.0, desired_exposure_fraction=-0.2)
    # Should clamp to fraction=0, leaving position unchanged
    assert target == pytest.approx(-1000.0)


def test_compute_cover_target_method_exists_on_env_class():
    """Source-level check: ParabolicReversalEnv has _compute_cover_target method that calls the helper."""
    env_path = (
        Path(__file__).resolve().parents[1] / "src" / "rl" / "env.py"
    )
    src = env_path.read_text(encoding="utf-8")
    assert "def _compute_cover_target" in src, "method missing from env"
    assert "scale_out_cover_target" in src, "helper not referenced inside env"


def test_step_uses_cover_target_for_action_type_1():
    """Source-level check: step() routes action_type==1 to _compute_cover_target."""
    env_path = (
        Path(__file__).resolve().parents[1] / "src" / "rl" / "env.py"
    )
    src = env_path.read_text(encoding="utf-8")
    # Must have a branch like 'elif action_type == 1' that calls _compute_cover_target
    assert "_compute_cover_target" in src
    # The COVER branch should reference the action_type == 1 case explicitly
    assert "action_type == 1" in src, "step() must explicitly handle COVER as action_type==1"
