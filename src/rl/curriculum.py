"""
Curriculum Learning Manager for RL Training

Three-phase curriculum that progressively increases difficulty:
  Phase 1 (Easy)   - only curated winners, loose drawdown, amplified rewards
  Phase 2 (Medium) - mixed data, standard drawdown
  Phase 3 (Full)   - full difficulty matching final evaluation conditions

The CurriculumManager is wired into WarmupCallback in train_wfo.py.
Phase transitions update the data provider csv_weight and the environment's
reward_scale / max_drawdown via update_curriculum_params().
"""

from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


@dataclass
class CurriculumConfig:
    # Phase 1: Easy — only curated winners, loose drawdown, amplified rewards
    phase1_csv_ratio: float = 1.0
    phase1_max_drawdown: float = -15000
    phase1_reward_scale: float = 2.0
    phase1_min_episodes: int = 200
    phase1_target_trade_rate: float = 0.3

    # Phase 2: Medium — mixed data, standard drawdown
    phase2_csv_ratio: float = 0.60
    phase2_max_drawdown: float = -10000
    phase2_reward_scale: float = 1.5
    phase2_min_episodes: int = 300
    phase2_target_win_rate: float = 0.35

    # Phase 3: Full difficulty
    phase3_csv_ratio: float = 0.50
    phase3_max_drawdown: float = -10000
    phase3_reward_scale: float = 1.0


class CurriculumManager:
    def __init__(self, config=None):
        self.config = config or CurriculumConfig()
        self.current_phase = 1
        self.episode_count = 0
        self.phase_metrics = {'trades': 0, 'episodes': 0, 'wins': 0}

    def should_advance(self):
        m = self.phase_metrics
        if self.current_phase == 1:
            if m['episodes'] < self.config.phase1_min_episodes:
                return False
            trade_rate = m['trades'] / max(m['episodes'], 1)
            return trade_rate >= self.config.phase1_target_trade_rate
        elif self.current_phase == 2:
            if m['episodes'] < self.config.phase2_min_episodes:
                return False
            win_rate = m['wins'] / max(m['trades'], 1)
            return win_rate >= self.config.phase2_target_win_rate
        return False

    def get_current_config(self):
        if self.current_phase == 1:
            return {
                'csv_ratio': self.config.phase1_csv_ratio,
                'max_drawdown': self.config.phase1_max_drawdown,
                'reward_scale': self.config.phase1_reward_scale,
            }
        elif self.current_phase == 2:
            return {
                'csv_ratio': self.config.phase2_csv_ratio,
                'max_drawdown': self.config.phase2_max_drawdown,
                'reward_scale': self.config.phase2_reward_scale,
            }
        else:
            return {
                'csv_ratio': self.config.phase3_csv_ratio,
                'max_drawdown': self.config.phase3_max_drawdown,
                'reward_scale': self.config.phase3_reward_scale,
            }

    def on_episode_end(self, num_trades: int, total_pnl: float, base_pnl: float = None) -> bool:
        """
        Record episode outcome and check for phase advancement.

        Args:
            num_trades: Number of trades executed in the episode.
            total_pnl: Episode PnL including any shaping bonuses (used for logging).
            base_pnl: True equity PnL (no bonuses). Defaults to total_pnl if not provided.
                      FIX 11: Phase 2 win gate uses base_pnl to prevent bonus farming.

        Returns:
            True if the phase just advanced, False otherwise.
        """
        if base_pnl is None:
            base_pnl = total_pnl
        self.phase_metrics['episodes'] += 1
        self.phase_metrics['trades'] += num_trades
        # FIX 11: Gate win on base_pnl (true equity) not total_pnl (which includes bonuses)
        if num_trades > 0 and base_pnl > 0:
            self.phase_metrics['wins'] += 1

        if self.should_advance():
            old_phase = self.current_phase
            self.current_phase += 1
            self.phase_metrics = {'trades': 0, 'episodes': 0, 'wins': 0}
            logger.info(
                f"CURRICULUM ADVANCE: Phase {old_phase} → Phase {self.current_phase}. "
                f"New config: {self.get_current_config()}"
            )
            return True  # Signal phase transition
        return False
