"""
Collapse Detector for SAC training.

Monitors per-episode trade rate and action standard deviation to detect
policy collapse (all-HOLD attractor). Logs warnings after consecutive
check intervals with low diversity, and emits "COLLAPSE DETECTED" after
`consecutive_alerts` consecutive failures.

Wired into WarmupCallback.on_episode_end in train_wfo.py.
"""

import numpy as np
import logging

logger = logging.getLogger(__name__)


class CollapseDetector:
    def __init__(
        self,
        check_interval: int = 50,
        min_trade_rate: float = 0.05,
        min_action_std: float = 0.05,
        consecutive_alerts: int = 3,
    ):
        self.check_interval = check_interval
        self.min_trade_rate = min_trade_rate
        self.min_action_std = min_action_std
        self.consecutive_alerts = consecutive_alerts
        self.episode_count = 0
        self.trade_counts = []
        self.action_stds = []
        self.alert_count = 0

    def on_episode_end(self, num_trades: int, action_values) -> None:
        self.episode_count += 1
        self.trade_counts.append(num_trades)
        if action_values:
            self.action_stds.append(np.std(action_values))
        if self.episode_count % self.check_interval == 0:
            return self._check_collapse()
        return None

    def _check_collapse(self):
        recent_trades = self.trade_counts[-self.check_interval:]
        recent_stds = self.action_stds[-self.check_interval:]
        trade_rate = sum(1 for t in recent_trades if t > 0) / len(recent_trades)
        avg_action_std = np.mean(recent_stds) if recent_stds else 0
        alerts = []
        if trade_rate < self.min_trade_rate:
            alerts.append(f"Trade rate {trade_rate:.2%} < {self.min_trade_rate:.2%}")
        if avg_action_std < self.min_action_std:
            alerts.append(f"Action std {avg_action_std:.4f} < {self.min_action_std}")
        if alerts:
            self.alert_count += 1
            msg = f"COLLAPSE WARNING ({self.alert_count}/{self.consecutive_alerts}): " + "; ".join(alerts)
            if self.alert_count >= self.consecutive_alerts:
                return f"COLLAPSE DETECTED: {msg}. Recommend stopping training."
            return msg
        else:
            self.alert_count = 0
            return None
