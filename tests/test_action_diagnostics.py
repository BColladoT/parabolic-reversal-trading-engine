"""Tests for Phase 1.1 RL env diagnostics: action histogram + time-in-position.

Verifies that ParabolicReversalEnv tracks per-step action selection and
time-in-position counters during an episode, and exposes them through
``get_episode_diagnostics()`` (renamed from ``get_episode_info`` in
Phase 1.1 fix I1 to avoid name collision with Gym's per-step ``info``).

These diagnostics are consumed downstream by the trainer (Phase 1.2) and
the bin-count sweep (Phase 4); without them the sweep would be unable
to detect policy collapse via action-distribution entropy.
"""
from __future__ import annotations

import os
from datetime import datetime, timedelta

import pytest

# Cred stubs (also set in conftest.py, but defensive for direct invocation).
os.environ.setdefault("ALPACA_API_KEY", "ci-stub-key")
os.environ.setdefault("ALPACA_SECRET", "ci-stub-secret")
os.environ.setdefault("ALPACA_SECRET_KEY", "ci-stub-secret")

# RL deps (gymnasium, torch) are optional extras; skip cleanly when absent.
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
    """Minimal in-memory data provider for unit tests.

    Returns a fixed sequence of Bar namedtuples with VWAP deviation high
    enough to keep entries unmasked during the entry window. Implements
    just the surface that ``ParabolicReversalEnv`` calls during reset()
    and step(): reset_episode, get_current_bar, advance, is_done,
    get_pre_decision_sequence, plus a few mode/source attributes the env
    pokes at for episode-end logging.
    """

    def __init__(self, n_bars: int = 30):
        self.n_bars = n_bars
        self.current_bar_idx = 0
        self.start_bar_idx = 0
        # Episode metadata fields the env reads via getattr
        self.current_source = "stub"
        self.current_symbol = "TEST"
        self.current_date = "2024-01-02"
        self.mode = "train"
        self.seed = None

        # Build a synthetic bar sequence: prices oscillate so VWAP-dev is
        # well above the entry threshold (set to 10.0% in _make_test_env
        # below). All bars fall inside the entry window 09:45-14:30 ET so
        # entries are not time-masked.
        base = datetime(2024, 1, 2, 10, 0, 0)
        rows = []
        self._bars = []
        for i in range(n_bars):
            # Hand-set vwap_deviation to 25.0% so entries are never
            # masked-out by the min_vwap_deviation_entry gate. See the
            # test env factory for the explicit 10.0% threshold (Phase
            # 1.1 fix I3: decouple from any future config default change).
            close = 10.0 + 0.05 * i
            ts = base + timedelta(minutes=i)
            bar = Bar(
                open=close - 0.01,
                high=close + 0.02,
                low=close - 0.02,
                close=close,
                volume=1000.0,
                vwap=close * 0.8,  # vwap below price -> positive deviation
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
        # The env runs Polars-DataFrame invariant checks on current_data
        # (e.g. dp.current_data.row(idx, named=True)), so this must be a
        # real DataFrame — not a list.
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
        # Return zeros — TCN-AE encoder accepts zero history at reset.
        return np.zeros((lookback, 5), dtype=np.float32)

    def get_current_bar_index(self) -> int:
        return self.current_bar_idx - self.start_bar_idx

    def get_total_bars(self) -> int:
        return len(self._bars) - self.start_bar_idx


def _make_test_env(n_bars: int = 25) -> ParabolicReversalEnv:
    # Phase 1.1 fix I3: pin min_vwap_deviation_entry=10.0 inside the test
    # so the stub provider's vwap_deviation=25.0 is unambiguously above
    # threshold REGARDLESS of any future change to the production default
    # (currently 15.0 in EnvironmentConfig). Without this pin, raising
    # the production default above 25.0 would silently degrade these
    # tests to "all entries masked" without any failed assertion.
    config = EnvironmentConfig(
        action_space_type="discrete",
        discrete_action_bins=7,
        min_vwap_deviation_entry=10.0,
    )
    provider = _StubDataProvider(n_bars=n_bars)
    env = ParabolicReversalEnv(config=config, data_provider=provider)
    return env


def test_action_histogram_populated_after_episode():
    """End-to-end: run a random-action episode and check shape/invariants.

    Phase 1.1 fix M1: this also serves as the reset-clears-diagnostics
    smoke test. We assert the diagnostics are populated after stepping,
    then reset() and assert they're cleared. The standalone reset test
    is renamed to ``test_reset_clears_diagnostics`` for clarity (kept
    minimal — does not duplicate the populate-then-check loop).
    """
    env = _make_test_env()
    obs, _ = env.reset()
    done = False
    while not done:
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
    final_info = env.get_episode_diagnostics()
    assert "action_histogram" in final_info
    assert len(final_info["action_histogram"]) == 7
    assert sum(final_info["action_histogram"].values()) == env.episode_step_count
    assert "mean_bars_in_position" in final_info
    assert "n_trades" in final_info


def test_reset_clears_diagnostics():
    """A fresh reset() must clear all diagnostic counters.

    Phase 1.1 fix M1: renamed from ``test_episode_info_reset_on_reset``
    and trimmed — keeps only the reset invariants (not the populate-
    then-check loop, which lives in the main test above).
    """
    env = _make_test_env()
    env.reset()
    # Take a few steps so counters populate.
    for _ in range(3):
        env.step(env.action_space.sample())
    info_after_steps = env.get_episode_diagnostics()
    assert sum(info_after_steps["action_histogram"].values()) > 0

    # Reset and confirm the histogram is back to zero.
    env.reset()
    info_after_reset = env.get_episode_diagnostics()
    assert sum(info_after_reset["action_histogram"].values()) == 0
    assert info_after_reset["n_trades"] == 0
    assert info_after_reset["n_bars"] == 0


# Bin layout for discrete_action_bins == 7 (see env._apply_discrete_action):
#   0: HOLD, 1: ENTRY-25%, 2: ENTRY-50%, 3: ENTRY-100%,
#   4: COVER-25%, 5: COVER-50%, 6: COVER-100% (full exit).
HOLD_BIN = 0
ENTRY_FULL_BIN = 3
COVER_FULL_BIN = 6


def test_n_trades_and_bars_in_position_for_deterministic_round_trip():
    """Drive a deterministic enter -> hold -> cover sequence and verify counters.

    Phase 1.1 fix I4: the random-action test above only checks that the
    diagnostic dict has the right SHAPE, not that the counters are
    CORRECT. This test pins a known sequence so off-by-one or
    miscount bugs surface as a failed numeric assertion.
    """
    env = _make_test_env(n_bars=40)  # plenty of room for entry + holds + cover
    env.reset()

    n_hold_bars = 5

    # Step 1: full ENTRY (bin 3). After this step the env is in a non-flat
    # short position and the in-position counter has incremented by 1
    # (entry-bar counts as 1 bar-in-position; matches the pre-fix semantics).
    env.step(ENTRY_FULL_BIN)
    assert env.current_position < 0, \
        f"Expected short position after ENTRY, got {env.current_position}"

    # Steps 2..(1+n_hold_bars): HOLD. Each step the in-position counter
    # ticks up by 1.
    for _ in range(n_hold_bars):
        env.step(HOLD_BIN)

    # Final step: full COVER (bin 6). The regular-cover path zeros position
    # via _execute_position_change (does NOT call _close_position), so the
    # n_trades increment happens in step()'s reward block. The flag-based
    # double-count guard should NOT fire here (since _close_position never
    # ran on this step).
    env.step(COVER_FULL_BIN)

    diag = env.get_episode_diagnostics()
    assert diag["n_trades"] == 1, \
        f"Expected exactly 1 round-trip, got {diag['n_trades']}"
    # mean_bars_in_position should be exactly the number of steps the
    # position was open. With the entry-bar counted as 1 and each hold
    # adding 1, that's 1 + n_hold_bars. The cover step is NOT counted
    # because the in-position increment branch is guarded by
    # ``current_position != 0`` and the position is already zero at that
    # check point on the cover step.
    expected_bars = 1 + n_hold_bars
    assert diag["mean_bars_in_position"] == expected_bars, \
        f"Expected {expected_bars} bars-in-position, got {diag['mean_bars_in_position']}"
    assert diag["median_bars_in_position"] == expected_bars, \
        f"Expected median {expected_bars}, got {diag['median_bars_in_position']}"
    # Step count: 1 entry + n_hold_bars holds + 1 cover.
    assert diag["n_bars"] == 1 + n_hold_bars + 1, \
        f"Expected {1 + n_hold_bars + 1} bars, got {diag['n_bars']}"


def test_histogram_step_count_invariant_survives_active_circuit_breaker():
    """Regression for the C1+C2 bugs.

    Before Phase 1.1 fix C1, ``episode_step_count`` was bumped AFTER the
    already-triggered circuit-breaker early-return in step(), so a step
    that hit that branch would record its action in the histogram but
    NOT bump ``episode_step_count`` — breaking the invariant
    ``sum(histogram.values()) == episode_step_count``. We force the
    early-return by flipping the flag manually, then call step() a few
    times and assert the invariant.
    """
    env = _make_test_env()
    env.reset()

    # Two normal steps so the histogram/step counter both have some
    # baseline > 0.
    env.step(HOLD_BIN)
    env.step(HOLD_BIN)

    # Manually trip the circuit breaker so subsequent step() calls take
    # the early-return branch (lines around env.py:862-870). We don't
    # care about the reward signal here — only the diagnostic counters.
    env.circuit_breaker_triggered = True

    # Three more steps. Each hits the early-return.
    env.step(HOLD_BIN)
    env.step(ENTRY_FULL_BIN)
    env.step(COVER_FULL_BIN)

    diag = env.get_episode_diagnostics()
    # Invariant: every action sampled was recorded in the histogram AND
    # bumped episode_step_count.
    assert sum(diag["action_histogram"].values()) == diag["n_bars"], (
        f"Invariant violated: histogram_sum={sum(diag['action_histogram'].values())} "
        f"!= n_bars={diag['n_bars']}"
    )
    assert diag["n_bars"] == 5, f"Expected 5 total steps, got {diag['n_bars']}"
