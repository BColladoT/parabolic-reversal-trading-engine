"""Tests for Discrete(7) action space mode in env.py.

When EnvironmentConfig.action_space_type == 'discrete', the env exposes a
Discrete(7) action space:
    0: HOLD (no change)
    1-3: ENTRY at desired_exposure_fraction in {-0.25, -0.50, -1.00}
    4-6: COVER fraction in {0.25, 0.50, 1.00}

Default (action_space_type == 'continuous') preserves the existing
Box(-1, 1) behavior used by SAC and PPO continuous.

Source-level + math tests only. Follows the pattern in
test_env_asymmetric_thresholds.py and test_ppo_algorithm.py.
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


def test_config_has_action_space_type_field():
    src = _env_src()
    assert "action_space_type" in src
    assert 'action_space_type: str = "continuous"' in src or \
           "action_space_type: str = 'continuous'" in src, \
        "default must be 'continuous' to preserve existing behavior"


def test_config_has_discrete_action_bins_field():
    src = _env_src()
    assert "discrete_action_bins" in src
    assert "discrete_action_bins: int = 7" in src, \
        "default must be 7 to match the documented bin layout"


def test_env_branches_action_space_on_type():
    """When action_space_type=='discrete', action_space must be Discrete(N).
    When 'continuous', it must remain Box(-1,1)."""
    src = _env_src()
    assert "gym.spaces.Discrete(" in src
    # The branch must reference the config field
    assert ('action_space_type == "discrete"' in src or
            "action_space_type == 'discrete'" in src), \
        "must branch the action_space construction on action_space_type"


def test_env_has_apply_discrete_action_helper():
    src = _env_src()
    assert "_apply_discrete_action" in src or "_decode_discrete_action" in src


def test_step_routes_on_action_space_type():
    """step() must branch on action_space_type so Discrete actions bypass
    the continuous _discretize_action path."""
    src = _env_src()
    step_idx = src.index("def step(self")
    next_def_idx = src.index("\n    def ", step_idx + 1)
    step_body = src[step_idx:next_def_idx]
    assert ('action_space_type == "discrete"' in step_body or
            "action_space_type == 'discrete'" in step_body or
            "_apply_discrete_action" in step_body or
            "_decode_discrete_action" in step_body), \
        "step() must route Discrete actions away from _discretize_action"


def test_action_mask_branches_on_action_space_type():
    """_compute_action_mask must return shape (N,) when discrete,
    (3,) when continuous."""
    src = _env_src()
    mask_idx = src.index("def _compute_action_mask")
    next_def_idx = src.index("\n    def ", mask_idx + 1)
    mask_body = src[mask_idx:next_def_idx]
    assert ('action_space_type == "discrete"' in mask_body or
            "action_space_type == 'discrete'" in mask_body), \
        "_compute_action_mask must branch on action_space_type"


def test_observation_space_mask_shape_branches():
    """The observation_space's action_mask field must match the action space
    size. Either branch on action_space_type, or use a variable derived from
    discrete_action_bins."""
    src = _env_src()
    # Search the obs_space construction block
    obs_idx = src.index("self.observation_space = gym.spaces.Dict")
    # Look at a wide enough window to capture mask_shape derivation
    window = src[max(0, obs_idx - 800):obs_idx + 400]
    assert ("mask_shape" in window or
            "discrete_action_bins" in window or
            'action_space_type == "discrete"' in window or
            "action_space_type == 'discrete'" in window), \
        "observation_space.action_mask shape must derive from action_space_type"


def test_train_wfo_quick_test_has_action_space_flag():
    src = _train_src()
    assert "--action-space" in src
    assert "args.action_space" in src
    assert "'continuous'" in src or '"continuous"' in src
    assert "'discrete'" in src or '"discrete"' in src


def test_train_wfo_quick_test_plumbs_action_space_type():
    src = _train_src()
    assert '"action_space_type"' in src or "'action_space_type'" in src


def test_train_wfo_quick_test_action_space_defaults_to_continuous():
    src = _train_src()
    idx = src.index("--action-space")
    window = src[idx:idx + 400]
    assert "default='continuous'" in window or 'default="continuous"' in window, \
        "--action-space must default to 'continuous' for backward compatibility"


def test_discrete_bin_semantics_math():
    """Math test for the 7-bin discrete action mapping.

    Bin layout:
        0: HOLD                       -> (action_type=2, magnitude=0.0)
        1: ENTRY exposure_fraction=-0.25 -> (0, -0.25)
        2: ENTRY exposure_fraction=-0.50 -> (0, -0.50)
        3: ENTRY exposure_fraction=-1.00 -> (0, -1.00)
        4: COVER 25% of position      -> (1, +0.25)
        5: COVER 50%                  -> (1, +0.50)
        6: COVER 100% (full exit)     -> (1, +1.00)
    """
    def decode(action_int):
        if action_int == 0:
            return (2, 0.0)
        elif action_int in (1, 2, 3):
            mags = {1: -0.25, 2: -0.50, 3: -1.00}
            return (0, mags[action_int])
        elif action_int in (4, 5, 6):
            mags = {4: 0.25, 5: 0.50, 6: 1.00}
            return (1, mags[action_int])
        else:
            raise ValueError(f"bin {action_int} out of range")

    assert decode(0) == (2, 0.0)
    assert decode(1) == (0, -0.25)
    assert decode(2) == (0, -0.50)
    assert decode(3) == (0, -1.00)
    assert decode(4) == (1, 0.25)
    assert decode(5) == (1, 0.50)
    assert decode(6) == (1, 1.00)


def test_continuous_default_preserved():
    """When action_space_type is default ('continuous'), Box(-1,1) survives."""
    src = _env_src()
    # The Box(...) construction must still exist (it's the default branch)
    assert "self.action_space = gym.spaces.Box(" in src
    assert "self.config.action_space_low" in src
