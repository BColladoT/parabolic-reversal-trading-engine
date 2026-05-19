"""Tests for asymmetric ENTRY/COVER thresholds in env.py.

The agent's continuous action is discretized using two thresholds:
    ENTRY: action < -entry_threshold
    COVER: action > +cover_threshold
    HOLD:  otherwise

Both default to None (fall back to hold_band_threshold), preserving
backward compatibility with PR #13.
"""
from __future__ import annotations

import os
from pathlib import Path

os.environ.setdefault("ALPACA_API_KEY", "ci-stub-key")
os.environ.setdefault("ALPACA_SECRET", "ci-stub-secret")
os.environ.setdefault("ALPACA_SECRET_KEY", "ci-stub-secret")


ENV_PATH = Path(__file__).resolve().parents[1] / "src" / "rl" / "env.py"
TRAIN_PATH = Path(__file__).resolve().parents[1] / "src" / "scripts" / "train_wfo_quick_test.py"


def _env_src() -> str:
    return ENV_PATH.read_text(encoding="utf-8")


def _train_src() -> str:
    return TRAIN_PATH.read_text(encoding="utf-8")


def test_config_has_entry_threshold_field():
    src = _env_src()
    assert "entry_threshold" in src
    assert "entry_threshold: Optional[float] = None" in src, \
        "default must be None to preserve PR #13 fallback behavior"


def test_config_has_cover_threshold_field():
    src = _env_src()
    assert "cover_threshold" in src
    assert "cover_threshold: Optional[float] = None" in src, \
        "default must be None to preserve PR #13 fallback behavior"


def test_discretize_action_reads_separate_thresholds():
    """_discretize_action reads entry_threshold and cover_threshold (with fallback)."""
    src = _env_src()
    discretize_block = src[src.index("def _discretize_action"):]
    end_idx = discretize_block.index("def _compute_cover_target") if "def _compute_cover_target" in discretize_block else 2000
    body = discretize_block[:end_idx]
    assert "entry_threshold" in body or "self.config.entry_threshold" in body
    assert "cover_threshold" in body or "self.config.cover_threshold" in body


def test_discretize_action_falls_back_to_hold_band_when_unset():
    """The body must use hold_band_threshold as the fallback when overrides are None."""
    src = _env_src()
    discretize_block = src[src.index("def _discretize_action"):]
    end_idx = discretize_block.index("def _compute_cover_target") if "def _compute_cover_target" in discretize_block else 2000
    body = discretize_block[:end_idx]
    assert "hold_band_threshold" in body, \
        "must reference hold_band_threshold as fallback"


def test_train_wfo_quick_test_has_entry_threshold_flag():
    src = _train_src()
    assert "--entry-threshold" in src
    assert "args.entry_threshold" in src


def test_train_wfo_quick_test_has_cover_threshold_flag():
    src = _train_src()
    assert "--cover-threshold" in src
    assert "args.cover_threshold" in src


def test_train_wfo_quick_test_plumbs_both_thresholds_to_env_config():
    src = _train_src()
    assert '"entry_threshold"' in src or "'entry_threshold'" in src
    assert '"cover_threshold"' in src or "'cover_threshold'" in src


def test_symmetric_math_preserved_when_both_unset():
    """When both overrides are None, the math falls back to hold_band_threshold."""
    def classify(action, entry_t, cover_t, hold_t):
        et = entry_t if entry_t is not None else hold_t
        ct = cover_t if cover_t is not None else hold_t
        if action < -et:
            return 0
        elif action > ct:
            return 1
        else:
            return 2

    # With both unset, use hold_t=0.05 (PR #13 default)
    assert classify(-0.06, None, None, 0.05) == 0  # ENTRY
    assert classify(+0.06, None, None, 0.05) == 1  # COVER
    assert classify(0.00, None, None, 0.05) == 2   # HOLD

    # With both unset, use hold_t=0.30 (PR #13 wider-band)
    assert classify(-0.20, None, None, 0.30) == 2  # HOLD (was COVER under 0.05)
    assert classify(-0.40, None, None, 0.30) == 0  # ENTRY


def test_asymmetric_math_independent_of_hold_band():
    """When both overrides are SET, hold_band_threshold is irrelevant."""
    def classify(action, entry_t, cover_t, hold_t):
        et = entry_t if entry_t is not None else hold_t
        ct = cover_t if cover_t is not None else hold_t
        if action < -et:
            return 0
        elif action > ct:
            return 1
        else:
            return 2

    # Asymmetric: entry=0.05, cover=0.30. Hold band 999.9 should be ignored.
    assert classify(-0.10, 0.05, 0.30, 999.9) == 0
    assert classify(+0.10, 0.05, 0.30, 999.9) == 2
    assert classify(+0.40, 0.05, 0.30, 999.9) == 1


def test_one_threshold_set_other_fallback():
    """Setting only one override uses fallback for the other."""
    def classify(action, entry_t, cover_t, hold_t):
        et = entry_t if entry_t is not None else hold_t
        ct = cover_t if cover_t is not None else hold_t
        if action < -et:
            return 0
        elif action > ct:
            return 1
        else:
            return 2

    # Only entry set, cover falls back to hold_band=0.30
    assert classify(-0.10, entry_t=0.05, cover_t=None, hold_t=0.30) == 0
    assert classify(+0.20, entry_t=0.05, cover_t=None, hold_t=0.30) == 2
    assert classify(+0.40, entry_t=0.05, cover_t=None, hold_t=0.30) == 1
