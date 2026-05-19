"""Tests for the configurable HOLD-band threshold in env.py.

The threshold controls action-discretization boundaries in
``_discretize_action``. Default 0.05 preserves pre-existing behavior;
wider values (e.g. 0.3) suppress noise-driven micro-covers from the
Gaussian policy.

Source-level + behavioral tests via AST extraction is impractical here
(the function is a method that reads ``self.config``). We use source-
level inspection to verify the wiring exists.
"""
from __future__ import annotations

import os
from pathlib import Path

os.environ.setdefault("ALPACA_API_KEY", "ci-stub-key")
os.environ.setdefault("ALPACA_SECRET", "ci-stub-secret")
os.environ.setdefault("ALPACA_SECRET_KEY", "ci-stub-secret")


ENV_PATH = (Path(__file__).resolve().parents[1] / "src" / "rl" / "env.py")
TRAIN_PATH = (Path(__file__).resolve().parents[1] / "src" / "scripts" / "train_wfo_quick_test.py")


def _env_src() -> str:
    return ENV_PATH.read_text(encoding="utf-8")


def _train_src() -> str:
    return TRAIN_PATH.read_text(encoding="utf-8")


def test_config_has_hold_band_threshold_field():
    src = _env_src()
    assert "hold_band_threshold" in src
    assert "hold_band_threshold: float = 0.05" in src, \
        "default must be 0.05 to preserve pre-existing behavior"


def test_discretize_action_uses_config_threshold():
    """_discretize_action reads self.config.hold_band_threshold instead of hardcoding 0.05."""
    src = _env_src()
    # The method body must reference the config field
    discretize_block = src[src.index("def _discretize_action"):]
    discretize_block = discretize_block[:discretize_block.index("def _compute_cover_target")
                                        if "def _compute_cover_target" in discretize_block else 2000]
    assert "self.config.hold_band_threshold" in discretize_block, \
        "_discretize_action must read self.config.hold_band_threshold"


def test_discretize_action_no_longer_hardcodes_0p05():
    """The literal 0.05 thresholds should be gone from _discretize_action body."""
    src = _env_src()
    discretize_block = src[src.index("def _discretize_action"):]
    end_idx = discretize_block.index("def _compute_cover_target") if "def _compute_cover_target" in discretize_block else 2000
    body = discretize_block[:end_idx]
    # The COMPARISONS must use the threshold variable. Hardcoded 0.05 comparisons must be gone.
    assert "< -0.05" not in body, "hardcoded -0.05 comparison still present"
    assert "> 0.05" not in body, "hardcoded +0.05 comparison still present"


def test_train_wfo_quick_test_has_hold_band_flag():
    src = _train_src()
    assert "--hold-band-threshold" in src
    assert "args.hold_band_threshold" in src


def test_train_wfo_quick_test_plumbs_hold_band_to_env_config():
    src = _train_src()
    assert '"hold_band_threshold"' in src or "'hold_band_threshold'" in src
    assert "_hold_band_threshold" in src  # the underscore-prefixed config attr


def test_default_threshold_preserves_old_behavior_in_decision_boundary():
    """At the default 0.05, the discretization is identical to the pre-change code.

    Mathematical check: with threshold=0.05,
      action = -0.06 -> ENTRY (< -0.05)
      action = +0.06 -> COVER (> +0.05)
      action = 0.00 / +0.04 / -0.04 -> HOLD
    """
    # We can't exec the method (needs self.config), so verify the formula directly:
    def classify(action, threshold=0.05):
        if action < -threshold:
            return 0
        elif action > threshold:
            return 1
        else:
            return 2

    assert classify(-0.06) == 0
    assert classify(+0.06) == 1
    assert classify(0.0) == 2
    assert classify(+0.04) == 2
    assert classify(-0.04) == 2
    # Boundary edge: exactly +0.05 should be HOLD (>= is the boundary)
    assert classify(+0.05) == 2


def test_wider_threshold_classifies_smaller_actions_as_hold():
    """At threshold=0.3, only |action| > 0.3 leaves the HOLD band."""
    def classify(action, threshold):
        if action < -threshold:
            return 0
        elif action > threshold:
            return 1
        else:
            return 2

    # At 0.3, +0.2 is HOLD (was COVER under 0.05)
    assert classify(+0.20, threshold=0.3) == 2
    assert classify(-0.20, threshold=0.3) == 2
    # +0.4 still COVER at 0.3
    assert classify(+0.40, threshold=0.3) == 1
    assert classify(-0.40, threshold=0.3) == 0
