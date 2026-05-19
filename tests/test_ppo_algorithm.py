"""Tests for PPO algorithm wiring in train_wfo_quick_test.py.

PPO is a drop-in alternative to SAC at the algorithm level. Default behavior
(--algo sac) is preserved; --algo ppo dispatches to a PPO config builder that
mirrors the SAC builder structure but with PPO-specific hyperparameters.

Source-level tests only — no ray/torch imports. Follows the pattern in
test_env_asymmetric_thresholds.py and test_env_hold_band.py.
"""
from __future__ import annotations

import os
from pathlib import Path

os.environ.setdefault("ALPACA_API_KEY", "ci-stub-key")
os.environ.setdefault("ALPACA_SECRET", "ci-stub-secret")
os.environ.setdefault("ALPACA_SECRET_KEY", "ci-stub-secret")


TRAIN_PATH = Path(__file__).resolve().parents[1] / "src" / "scripts" / "train_wfo_quick_test.py"


def _train_src() -> str:
    return TRAIN_PATH.read_text(encoding="utf-8")


def test_imports_ppo_config():
    src = _train_src()
    assert "from ray.rllib.algorithms.ppo import PPOConfig" in src, \
        "PPOConfig import is required to build PPO algorithm"


def test_has_algo_flag():
    src = _train_src()
    assert "--algo" in src
    assert "choices=" in src or "choices = " in src
    assert "'sac'" in src or '"sac"' in src
    assert "'ppo'" in src or '"ppo"' in src


def test_algo_flag_defaults_to_sac():
    """Default must be 'sac' to preserve existing behavior."""
    src = _train_src()
    idx = src.index("--algo")
    window = src[idx:idx + 400]
    assert "default='sac'" in window or 'default="sac"' in window, \
        "--algo must default to 'sac' to preserve backward compatibility"


def test_has_ppo_config_helper():
    """A helper method must build the PPO config (parallel to create_sac_config)."""
    src = _train_src()
    assert "create_ppo_config" in src or "_build_ppo_config" in src or "def build_ppo_config" in src


def test_ppo_config_sets_clip_param():
    src = _train_src()
    assert "clip_param" in src, "PPO config must set clip_param"


def test_ppo_config_sets_num_sgd_iter():
    src = _train_src()
    assert "num_sgd_iter" in src, "PPO config must set num_sgd_iter"


def test_ppo_config_sets_entropy_coeff():
    src = _train_src()
    assert "entropy_coeff" in src, "PPO config must set entropy_coeff"


def test_algo_plumbed_through_config():
    """args.algo must be plumbed onto the config dataclass for fold-level dispatch."""
    src = _train_src()
    assert "args.algo" in src
    assert "config._algo" in src or "_algo = args.algo" in src


def test_ppo_skips_sac_callbacks():
    """JointTrainingCallback is SAC-specific (it freezes/unfreezes actor via
    optimizer_names). When algo=ppo, the callback must not be wired in.

    We check call-site occurrences (the `Callback(` form), not docstring or
    comment mentions of the name.
    """
    src = _train_src()
    # Call-sites: '{name}(' followed by a non-name char. Skip docstring/comment
    # mentions and import lines.
    for cb in ["JointTrainingCallback", "WarmupCallback"]:
        call_pat = cb + "("
        positions = [i for i in range(len(src)) if src.startswith(call_pat, i)]
        for idx in positions:
            preceding = src[max(0, idx - 800):idx]
            in_guard = (
                "args.algo" in preceding
                or "_algo" in preceding
                or "algo ==" in preceding
                or "algo !=" in preceding
                or "if algo" in preceding
            )
            assert in_guard, (
                f"SAC callback call-site '{cb}(' at offset {idx} is not gated "
                "by algo (no _algo/args.algo/algo== in preceding 800 chars)."
            )


def test_dispatch_branches_on_algo():
    """The trainer must dispatch to SAC OR PPO config builder based on --algo."""
    src = _train_src()
    # The PPO config must be referenced somewhere in the dispatch path
    assert "PPOConfig" in src or "create_ppo_config" in src or "_build_ppo_config" in src
    # And the dispatch must branch on algo
    assert (
        "algo == 'ppo'" in src or 'algo == "ppo"' in src
        or "algo == 'sac'" in src or 'algo == "sac"' in src
    )
