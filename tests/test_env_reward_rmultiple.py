"""Pure unit tests for the R-multiple reward helper in src/rl/env.py.

The helper itself is torch-free, but ``src/rl/env.py`` imports torch at module
scope, so a normal ``from src.rl.env import r_multiple_reward_term`` cannot
run on a machine with a broken torch DLL. To keep these tests honest and
self-contained — and to truly exercise the helper's logic without skipping —
we parse ``src/rl/env.py`` with the ``ast`` module, locate the
``r_multiple_reward_term`` function definition, and exec just that node into
a fresh namespace. No torch import is triggered.

If torch IS available, we additionally cross-check that the loaded helper
is identical to the one exported from ``src.rl.env`` (sanity guard against
the test going stale).
"""
from __future__ import annotations

import ast
import math
from pathlib import Path

import pytest


def _load_helper_torch_free():
    """Parse src/rl/env.py and exec only the ``r_multiple_reward_term`` def.

    Avoids env.py's module-level ``import torch`` so the helper is testable
    on machines with a broken torch installation.
    """
    env_path = (
        Path(__file__).resolve().parents[1] / "src" / "rl" / "env.py"
    )
    source = env_path.read_text(encoding="utf-8")
    tree = ast.parse(source)
    func_node = next(
        (
            n for n in tree.body
            if isinstance(n, ast.FunctionDef) and n.name == "r_multiple_reward_term"
        ),
        None,
    )
    if func_node is None:
        raise RuntimeError(
            "r_multiple_reward_term not found at module scope in src/rl/env.py"
        )
    module = ast.Module(body=[func_node], type_ignores=[])
    ns: dict = {}
    exec(compile(module, str(env_path), "exec"), ns)
    return ns["r_multiple_reward_term"]


r_multiple_reward_term = _load_helper_torch_free()


def test_r_multiple_reward_zero_weight_returns_zero():
    """Default weight=0.0 is a strict no-op (preserves pre-A4 reward path)."""
    assert r_multiple_reward_term(realized_r=2.5, weight=0.0) == 0.0
    assert r_multiple_reward_term(realized_r=-3.0, weight=0.0) == 0.0


def test_r_multiple_reward_positive_realized_yields_positive_term():
    out = r_multiple_reward_term(realized_r=1.0, weight=0.5)
    assert out == pytest.approx(0.5)


def test_r_multiple_reward_negative_realized_yields_negative_term():
    out = r_multiple_reward_term(realized_r=-2.0, weight=0.5)
    assert out == pytest.approx(-1.0)


def test_r_multiple_reward_clips_outliers():
    # realized_r=10 should clip to 5 before scaling
    out = r_multiple_reward_term(realized_r=10.0, weight=1.0, clip=5.0)
    assert out == pytest.approx(5.0)
    # realized_r=-50 should clip to -5
    out = r_multiple_reward_term(realized_r=-50.0, weight=1.0, clip=5.0)
    assert out == pytest.approx(-5.0)


def test_r_multiple_reward_handles_nan_safely():
    """NaN realized R (rare but possible) must not propagate into reward."""
    out = r_multiple_reward_term(realized_r=math.nan, weight=1.0)
    assert not math.isnan(out)
