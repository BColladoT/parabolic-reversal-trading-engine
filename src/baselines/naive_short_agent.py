"""
Naive Short Agent — Simplest possible short strategy.

Enters maximum short at first opportunity (VWAP > threshold),
holds until end-of-day flatten. No exit logic, no stop loss,
no position management.

This is the "buy-and-hold" equivalent for short-selling.
If RL can't beat this, it hasn't learned timing or sizing.
"""

import numpy as np
from typing import Dict, Any


class NaiveShortAgent:
    """Enter short once, hold to close."""

    def __init__(self, entry_threshold: float = 20.0):
        self.entry_threshold = entry_threshold
        self.entered = False

    def reset(self):
        self.entered = False

    def act(self, observation: np.ndarray, info: Dict[str, Any]) -> np.ndarray:
        vwap_dev = info.get('vwap_deviation', 0.0)
        position = info.get('position', 0.0)

        # Already in position — stay maximally short
        if self.entered or position < 0:
            self.entered = True
            return np.array([-1.0])

        # Entry: first time VWAP deviation > threshold
        if vwap_dev > self.entry_threshold:
            self.entered = True
            return np.array([-1.0])

        # No entry signal yet — hold flat
        return np.array([0.0])
