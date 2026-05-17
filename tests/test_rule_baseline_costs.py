"""Tests for transaction-cost fairness in the rule-based baseline.

The RL environment (``src/rl/env.py``) charges 30bps per leg on each trade.
For RL-vs-rule comparison to be honest, the rule baseline must charge the
same. These tests pin the default to 30.0 and verify the cost is applied
to BOTH legs of a round-trip trade.
"""
from __future__ import annotations

import inspect

import pytest

from src.baselines import rule_baseline


def test_baseline_default_cost_is_30bps():
    """The entry-point ``run_baseline`` must default ``transaction_cost_bps`` to 30.0."""
    sig = inspect.signature(rule_baseline.run_baseline)
    assert "transaction_cost_bps" in sig.parameters
    assert sig.parameters["transaction_cost_bps"].default == 30.0


def test_baseline_zero_cost_matches_gross_pnl():
    """With transaction_cost_bps=0, PnL equals raw short-side gross PnL."""
    # Short 100 sh @ $10, cover @ $9.50 → gross = (10 - 9.50) * 100 = $50
    pnl = rule_baseline.run_baseline(
        entry_price=10.0,
        exit_price=9.50,
        shares=100,
        side="short",
        transaction_cost_bps=0.0,
    )
    assert pnl == pytest.approx(50.0)


def test_baseline_charges_transaction_cost_on_each_leg():
    """Default 30bps must charge cost on BOTH entry and exit legs.

    Round trip: short 100 sh @ $10, cover @ $9.50.
      gross_pnl  = (10.00 - 9.50) * 100 = $50.00
      entry_fee  = 10.00 * 100 * 0.0030 = $3.00
      exit_fee   =  9.50 * 100 * 0.0030 = $2.85
      net_pnl    = 50.00 - 3.00 - 2.85  = $44.15
    """
    gross = rule_baseline.run_baseline(
        entry_price=10.0,
        exit_price=9.50,
        shares=100,
        side="short",
        transaction_cost_bps=0.0,
    )
    net = rule_baseline.run_baseline(
        entry_price=10.0,
        exit_price=9.50,
        shares=100,
        side="short",
        transaction_cost_bps=30.0,
    )
    # Both legs charged → net strictly less than gross
    assert net < gross
    # Total cost ≈ 2 × bps × notional ≈ 2 × 0.003 × 10 × 100 = $6.00
    cost = gross - net
    assert cost == pytest.approx(6.0, abs=0.5)
    # Precise per-leg formula
    expected_entry_fee = 10.0 * 100 * (30.0 / 10_000.0)
    expected_exit_fee = 9.50 * 100 * (30.0 / 10_000.0)
    assert cost == pytest.approx(expected_entry_fee + expected_exit_fee, rel=1e-6)


def test_baseline_uses_30bps_by_default_when_kwarg_omitted():
    """Calling without transaction_cost_bps must apply the 30bps default."""
    net_explicit = rule_baseline.run_baseline(
        entry_price=10.0,
        exit_price=9.50,
        shares=100,
        side="short",
        transaction_cost_bps=30.0,
    )
    net_default = rule_baseline.run_baseline(
        entry_price=10.0,
        exit_price=9.50,
        shares=100,
        side="short",
    )
    assert net_default == pytest.approx(net_explicit, rel=1e-9)


def test_baseline_cost_scales_with_notional():
    """Larger share count → proportionally larger total cost."""
    pnl_small = rule_baseline.run_baseline(
        entry_price=10.0, exit_price=9.50, shares=100, side="short"
    )
    pnl_large = rule_baseline.run_baseline(
        entry_price=10.0, exit_price=9.50, shares=1000, side="short"
    )
    # Gross scales 10x; cost also scales 10x; so net should scale exactly 10x.
    assert pnl_large == pytest.approx(pnl_small * 10.0, rel=1e-9)
