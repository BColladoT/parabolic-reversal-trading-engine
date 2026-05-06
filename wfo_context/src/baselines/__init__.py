"""
Baseline trading agents for comparison against RL.

This package implements deterministic and simple learning-based baselines
that establish floors for RL performance.
"""

from .rule_baseline import RuleBasedAgent

__all__ = ['RuleBasedAgent']
