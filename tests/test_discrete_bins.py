"""Tests for variable discrete-action bin counts (Phase 4 prep).

Generalizes the Discrete(7) action decoder + mask to support
N in {3, 5, 7, 9, 11}. Init-time validation rejects unsupported N values.

The canonical bin layout (ascending magnitude) matches the existing N=7
convention enforced by ``test_env_discrete_action.test_discrete_bin_semantics_math``:

    N=3:  0=HOLD, 1=ENTRY-100%, 2=COVER+100%
    N=5:  0=HOLD, 1-2=ENTRY at {-0.50, -1.00}, 3-4=COVER at {0.50, 1.00}
    N=7:  0=HOLD, 1-3=ENTRY at {-0.25, -0.50, -1.00}, 4-6=COVER at {0.25, 0.50, 1.00}
    N=9:  0=HOLD, 1-4=ENTRY at {-0.25, -0.50, -0.75, -1.00}, 5-8=COVER at {0.25, 0.50, 0.75, 1.00}
    N=11: 0=HOLD, 1-5=ENTRY at {-0.10, -0.25, -0.50, -0.75, -1.00},
          6-10=COVER at {0.10, 0.25, 0.50, 0.75, 1.00}

Action-type encoding (unchanged): 0=ENTRY, 1=COVER, 2=HOLD.

Reuses the stub data provider pattern from tests/test_action_diagnostics.py.
"""
from __future__ import annotations

import os
from datetime import datetime, timedelta

import pytest

# Cred stubs (also set in conftest.py, but defensive for direct invocation).
os.environ.setdefault("ALPACA_API_KEY", "ci-stub-key")
os.environ.setdefault("ALPACA_SECRET", "ci-stub-secret")
os.environ.setdefault("ALPACA_SECRET_KEY", "ci-stub-secret")

pytest.importorskip("gymnasium")
try:
    import torch  # noqa: F401
except (ImportError, OSError) as e:  # pragma: no cover
    pytest.skip(f"torch unavailable: {e}", allow_module_level=True)

import numpy as np
import polars as pl

from src.rl.data_provider_hybrid import Bar
from src.rl.env import EnvironmentConfig, ParabolicReversalEnv


class _StubDataProvider:
    """Minimal in-memory data provider — copy of the helper in
    ``tests/test_action_diagnostics.py`` so this file stays self-contained."""

    def __init__(self, n_bars: int = 30):
        self.n_bars = n_bars
        self.current_bar_idx = 0
        self.start_bar_idx = 0
        self.current_source = "stub"
        self.current_symbol = "TEST"
        self.current_date = "2024-01-02"
        self.mode = "train"
        self.seed = None

        base = datetime(2024, 1, 2, 10, 0, 0)
        rows = []
        self._bars = []
        for i in range(n_bars):
            close = 10.0 + 0.05 * i
            ts = base + timedelta(minutes=i)
            bar = Bar(
                open=close - 0.01,
                high=close + 0.02,
                low=close - 0.02,
                close=close,
                volume=1000.0,
                vwap=close * 0.8,
                vwap_deviation=25.0,
                timestamp=ts,
            )
            self._bars.append(bar)
            rows.append({
                "open": float(bar.open),
                "high": float(bar.high),
                "low": float(bar.low),
                "close": float(bar.close),
                "volume": float(bar.volume),
                "vwap": float(bar.vwap),
                "vwap_dev": float(bar.vwap_deviation),
                "timestamp": ts,
            })
        self.current_data = pl.DataFrame(rows)

    def reset_episode(self) -> bool:
        self.current_bar_idx = 0
        return True

    def get_current_bar(self):
        if self.current_bar_idx >= len(self._bars):
            return None
        return self._bars[self.current_bar_idx]

    def advance(self):
        self.current_bar_idx += 1
        if self.current_bar_idx < len(self._bars):
            return self._bars[self.current_bar_idx]
        return None

    def is_done(self) -> bool:
        return self.current_bar_idx >= len(self._bars)

    def get_pre_decision_sequence(self, lookback: int = 60):
        return np.zeros((lookback, 5), dtype=np.float32)

    def get_current_bar_index(self) -> int:
        return self.current_bar_idx - self.start_bar_idx

    def get_total_bars(self) -> int:
        return len(self._bars) - self.start_bar_idx


def _make_env(config: EnvironmentConfig, n_bars: int = 25) -> ParabolicReversalEnv:
    provider = _StubDataProvider(n_bars=n_bars)
    return ParabolicReversalEnv(config=config, data_provider=provider)


# Action-type encoding (matches env._apply_discrete_action):
#   0 = ENTRY, 1 = COVER, 2 = HOLD.
# Bin layouts use ASCENDING magnitude to match the existing N=7 convention
# enforced by tests/test_env_discrete_action.test_discrete_bin_semantics_math.
@pytest.mark.parametrize("n_bins,expected_entries,expected_covers", [
    (3, [(0, -1.00)], [(1, 1.00)]),
    (5, [(0, -0.50), (0, -1.00)], [(1, 0.50), (1, 1.00)]),
    (7, [(0, -0.25), (0, -0.50), (0, -1.00)],
        [(1, 0.25), (1, 0.50), (1, 1.00)]),
    (9, [(0, -0.25), (0, -0.50), (0, -0.75), (0, -1.00)],
        [(1, 0.25), (1, 0.50), (1, 0.75), (1, 1.00)]),
    (11, [(0, -0.10), (0, -0.25), (0, -0.50), (0, -0.75), (0, -1.00)],
         [(1, 0.10), (1, 0.25), (1, 0.50), (1, 0.75), (1, 1.00)]),
])
def test_apply_discrete_action_for_each_supported_n(
    n_bins, expected_entries, expected_covers,
):
    """For every supported N, the decoder maps every bin to the right
    (action_type, magnitude) tuple.

    Bin 0 is always HOLD. Bins [1, 1+n_entry_bins) are ENTRY with ascending
    magnitudes. Bins [1+n_entry_bins, N) are COVER with ascending magnitudes.
    """
    env = _make_env(EnvironmentConfig(
        action_space_type="discrete",
        discrete_action_bins=n_bins,
        min_vwap_deviation_entry=10.0,
    ))
    env.reset()

    # HOLD is always bin 0.
    assert env._apply_discrete_action(0) == (2, 0.0), \
        f"N={n_bins}: bin 0 should be HOLD"

    # ENTRY bins.
    for i, expected in enumerate(expected_entries, start=1):
        got = env._apply_discrete_action(i)
        assert got == expected, \
            f"N={n_bins}: bin {i} expected {expected}, got {got}"

    # COVER bins.
    cover_start = 1 + len(expected_entries)
    for i, expected in enumerate(expected_covers, start=cover_start):
        got = env._apply_discrete_action(i)
        assert got == expected, \
            f"N={n_bins}: bin {i} expected {expected}, got {got}"


@pytest.mark.parametrize("n_bins", [3, 5, 7, 9, 11])
def test_action_space_size_matches_n_bins(n_bins):
    """gym.spaces.Discrete(N) and the action mask shape both track
    discrete_action_bins for every supported N."""
    env = _make_env(EnvironmentConfig(
        action_space_type="discrete",
        discrete_action_bins=n_bins,
        min_vwap_deviation_entry=10.0,
    ))
    env.reset()

    assert env.action_space.n == n_bins, \
        f"Action space size mismatch: expected Discrete({n_bins}), got {env.action_space.n}"

    mask = env._compute_action_mask()
    assert mask.shape == (n_bins,), \
        f"Mask shape mismatch: expected ({n_bins},), got {mask.shape}"


@pytest.mark.parametrize("n_bins", [3, 5, 7, 9, 11])
def test_action_mask_gates_all_bins(n_bins):
    """Sanity: the action mask's ENTRY and COVER ranges cover the entire
    non-HOLD slot range. No bin should be silently un-maskable.

    For a freshly-reset env with the stub provider, ENTRY should be allowed
    (in_entry_window=True, vwap_dev=25 > threshold, no position) and COVER
    should be disallowed (position is flat).
    """
    env = _make_env(EnvironmentConfig(
        action_space_type="discrete",
        discrete_action_bins=n_bins,
        min_vwap_deviation_entry=10.0,
    ))
    env.reset()

    n_entry_bins = (n_bins - 1) // 2
    n_cover_bins = (n_bins - 1) - n_entry_bins
    assert n_entry_bins + n_cover_bins == n_bins - 1, \
        "ENTRY + COVER bins must cover every non-HOLD slot"

    mask = env._compute_action_mask()
    # HOLD always allowed at reset.
    assert mask[0] == 1, f"N={n_bins}: HOLD bin should be unmasked at reset"
    # ENTRY range — every bin should be allowed (entry conditions met).
    for i in range(1, 1 + n_entry_bins):
        assert mask[i] == 1, \
            f"N={n_bins}: ENTRY bin {i} should be unmasked at reset (no position, in window, vwap_dev=25)"
    # COVER range — every bin should be MASKED (flat position).
    for i in range(1 + n_entry_bins, n_bins):
        assert mask[i] == 0, \
            f"N={n_bins}: COVER bin {i} should be masked when flat (no position to cover)"


@pytest.mark.parametrize("bad_n", [0, 1, 2, 4, 6, 8, 10, 12, 13, 15, 100, -1])
def test_unsupported_n_raises_at_init(bad_n):
    """Unsupported N values must be rejected at env construction.

    The supported set is {3, 5, 7, 9, 11}. Anything else — including
    even N (no clean HOLD bin), N < 3 (degenerate, < 1 ENTRY + 1 COVER + HOLD),
    and N > 11 (no defined magnitude ladder) — must raise.
    """
    with pytest.raises((ValueError, AssertionError), match="discrete_action_bins"):
        _make_env(EnvironmentConfig(
            action_space_type="discrete",
            discrete_action_bins=bad_n,
            min_vwap_deviation_entry=10.0,
        ))


def test_supported_n_does_not_raise():
    """Sanity counterpart: every supported N must construct without error."""
    for n in (3, 5, 7, 9, 11):
        env = _make_env(EnvironmentConfig(
            action_space_type="discrete",
            discrete_action_bins=n,
            min_vwap_deviation_entry=10.0,
        ))
        assert env.action_space.n == n


def test_unsupported_n_via_env_config_dict_also_raises():
    """env_config dict path (used by RLlib) must also validate N.

    When RLlib passes an EnvContext, the env's __init__ mutates
    self.config.discrete_action_bins via setattr() — that path bypasses
    EnvironmentConfig.__post_init__, so the env must re-validate after the
    override has been applied.
    """
    with pytest.raises((ValueError, AssertionError), match="discrete_action_bins"):
        provider = _StubDataProvider(n_bars=25)
        ParabolicReversalEnv(
            config={
                "action_space_type": "discrete",
                "discrete_action_bins": 6,
                "min_vwap_deviation_entry": 10.0,
            },
            data_provider=provider,
        )
