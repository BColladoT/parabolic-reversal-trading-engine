"""Tests for Phase 1.1 RL env diagnostics: action histogram + time-in-position.

Verifies that ParabolicReversalEnv tracks per-step action selection and
time-in-position counters during an episode, and exposes them through
``get_episode_info()``.

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
        # well above the entry threshold (15.0%). All bars fall inside the
        # entry window 09:45-14:30 ET so entries are not time-masked.
        base = datetime(2024, 1, 2, 10, 0, 0)
        rows = []
        self._bars = []
        for i in range(n_bars):
            # Hand-set vwap_deviation to 25.0% so entries are never
            # masked-out by the min_vwap_deviation_entry gate.
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


def _make_test_env() -> ParabolicReversalEnv:
    config = EnvironmentConfig(
        action_space_type="discrete",
        discrete_action_bins=7,
    )
    provider = _StubDataProvider(n_bars=25)
    env = ParabolicReversalEnv(config=config, data_provider=provider)
    return env


def test_action_histogram_populated_after_episode():
    env = _make_test_env()
    obs, _ = env.reset()
    done = False
    while not done:
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
    final_info = env.get_episode_info()
    assert "action_histogram" in final_info
    assert len(final_info["action_histogram"]) == 7
    assert sum(final_info["action_histogram"].values()) == env.episode_step_count
    assert "mean_bars_in_position" in final_info
    assert "n_trades" in final_info


def test_episode_info_reset_on_reset():
    """A fresh reset() must clear all diagnostic counters."""
    env = _make_test_env()
    env.reset()
    # Take a few steps so counters populate.
    for _ in range(3):
        env.step(env.action_space.sample())
    info_after_steps = env.get_episode_info()
    assert sum(info_after_steps["action_histogram"].values()) > 0

    # Reset and confirm the histogram is back to zero.
    env.reset()
    info_after_reset = env.get_episode_info()
    assert sum(info_after_reset["action_histogram"].values()) == 0
    assert info_after_reset["n_trades"] == 0
    assert info_after_reset["n_bars"] == 0
