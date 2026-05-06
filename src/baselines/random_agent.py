"""
Random Agent Baseline — Establishes the noise floor.

If the RL agent can't significantly beat random actions,
the learning signal is insufficient or the model hasn't converged.
"""

import numpy as np
from typing import Dict, Any, Optional


class RandomAgent:
    """Agent that takes uniformly random actions in [-1, 1]."""

    def __init__(self, seed: Optional[int] = None):
        self._rng = np.random.RandomState(seed)

    def reset(self):
        pass  # No state to reset

    def act(self, observation: np.ndarray, info: Dict[str, Any]) -> np.ndarray:
        """Return a random action in [-1, 1]."""
        return np.array([self._rng.uniform(-1.0, 1.0)])
